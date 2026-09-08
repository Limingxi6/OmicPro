from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .model_registry import build_model
from .preprocess import FoldPreprocessor, ModalityStats


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ExperimentLayout:
    """Stable model/result paths shared by training and prediction commands."""

    run_name: str
    model_root: Path
    result_root: Path

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], config_path: str | Path | None = None) -> "ExperimentLayout":
        default_name = Path(config_path).stem if config_path is not None else "run"
        run_name = str(cfg.get("run_name") or default_name).strip()
        if not run_name:
            raise ValueError("run_name must be non-empty.")

        legacy_output = cfg.get("output_dir")
        config_dir = Path(config_path).expanduser().resolve().parent if config_path is not None else Path.cwd()

        def resolve_base(value: Any) -> Path:
            candidate = Path(str(value))
            return candidate if candidate.is_absolute() else (config_dir / candidate).resolve()

        result_base = resolve_base(cfg.get("result_dir") or legacy_output or "results")
        model_base = resolve_base(cfg.get("model_dir") or "models")
        return cls(
            run_name=run_name,
            model_root=model_base / run_name,
            result_root=result_base / run_name,
        )

    def checkpoint_path(self, trait: str, fold: int, model_name: str = "omicpro") -> Path:
        return self.model_root / f"k{int(fold)}" / str(trait) / f"{model_name}.pt"

    def fixed_checkpoint_path(
        self,
        trait: str,
        valid_fold: int,
        test_fold: int,
        model_name: str = "omicpro",
    ) -> Path:
        return (
            self.model_root
            / f"fixed_v{int(valid_fold)}_t{int(test_fold)}"
            / str(trait)
            / f"{model_name}.pt"
        )

    def fold_prediction_path(self, trait: str, fold: int) -> Path:
        return self.result_root / f"k{int(fold)}" / f"{trait}.csv"

    @property
    def summary_dir(self) -> Path:
        return self.result_root / "summary"

    @property
    def log_dir(self) -> Path:
        return self.result_root / "logs"

    def ensure_roots(self) -> None:
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)


def _series_to_payload(value: pd.Series | None) -> dict[str, list[Any]] | None:
    if value is None:
        return None
    return {
        "index": [str(x) for x in value.index.tolist()],
        "values": [float(x) for x in value.to_numpy(dtype=np.float64).tolist()],
    }


def _series_from_payload(value: Mapping[str, Any] | None) -> pd.Series | None:
    if value is None:
        return None
    return pd.Series(
        np.asarray(value["values"], dtype=np.float64),
        index=[str(x) for x in value["index"]],
        dtype=np.float64,
    )


def serialize_preprocessor(preprocessor: FoldPreprocessor) -> dict[str, Any]:
    if not preprocessor._is_fitted:
        raise RuntimeError("Cannot serialize an unfitted FoldPreprocessor.")

    stats: dict[str, Any] = {}
    for name, st in preprocessor.stats.items():
        stats[name] = {
            "columns": [str(x) for x in st.columns],
            "median": _series_to_payload(st.median),
            "var_selected_columns": (
                [str(x) for x in st.var_selected_columns] if st.var_selected_columns is not None else None
            ),
            "mean": _series_to_payload(st.mean),
            "std": _series_to_payload(st.std),
        }
    return {"topk": dict(preprocessor.topk), "stats": stats}


def restore_preprocessor(payload: Mapping[str, Any]) -> FoldPreprocessor:
    topk = dict(payload.get("topk", {}))
    preprocessor = FoldPreprocessor(
        topk_genotype=topk.get("genotype"),
        topk_expression=topk.get("expression"),
        topk_metabolites=topk.get("metabolites"),
    )
    for name in ("genotype", "expression", "metabolites"):
        raw = dict(payload["stats"][name])
        preprocessor.stats[name] = ModalityStats(
            columns=[str(x) for x in raw["columns"]],
            median=_series_from_payload(raw.get("median")),
            var_selected_columns=(
                [str(x) for x in raw["var_selected_columns"]]
                if raw.get("var_selected_columns") is not None
                else None
            ),
            mean=_series_from_payload(raw.get("mean")),
            std=_series_from_payload(raw.get("std")),
        )
    preprocessor._is_fitted = True
    return preprocessor


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_name: str,
    model_init: Mapping[str, Any],
    trait: str,
    preprocessor: FoldPreprocessor,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    fill_vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    split_ids: Mapping[str, list[str]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": str(model_name).strip().lower(),
        "model_init": dict(model_init),
        "model_state_dict": state,
        "trait": str(trait),
        "preprocessor": serialize_preprocessor(preprocessor),
        "target_standardizer": {
            "mean": np.asarray(target_mean, dtype=np.float64).tolist(),
            "std": np.asarray(target_std, dtype=np.float64).tolist(),
        },
        "fill_vectors": {
            "geno": np.asarray(fill_vectors[0], dtype=np.float32),
            "expr": np.asarray(fill_vectors[1], dtype=np.float32),
            "metab": np.asarray(fill_vectors[2], dtype=np.float32),
        },
        "split_ids": {str(k): [str(x) for x in v] for k, v in split_ids.items()},
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, FoldPreprocessor, dict[str, Any]]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload: {checkpoint_path}")
    version = int(payload.get("format_version", 0))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version={version}; expected {CHECKPOINT_FORMAT_VERSION}."
        )

    model = build_model(payload["model_name"], payload["model_init"], device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    preprocessor = restore_preprocessor(payload["preprocessor"])
    return model, preprocessor, payload
