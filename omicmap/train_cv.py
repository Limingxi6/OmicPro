from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data_utils import TRAIT_COLUMNS, load_and_align_data
from .metrics import regression_metrics_per_trait
from .model import MultiOmicsMultiTaskRegressor
from .preprocess import FoldPreprocessor

try:
    import yaml
except Exception as e:  # noqa: BLE001
    raise RuntimeError("请先安装 pyyaml: pip install pyyaml") from e


MISSING_CODE_MAP = {
    "none": 0,
    "geno": 1,
    "genotype": 1,
    "expr": 2,
    "expression": 2,
    "metab": 4,
    "metabolites": 4,
    "geno_expr": 3,
    "geno_metab": 5,
    "expr_metab": 6,
    "all": 7,
}


def _ratio_to_tag(x: float) -> str:
    return f"{float(x):.4f}".replace(".", "p")


def _clamp_ratio(x: Any, *, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x}")
    return max(0.0, min(1.0, v))


def _is_modality_ratio_dict(mapping: dict[str, Any]) -> bool:
    keys = {str(k).strip().lower() for k in mapping.keys()}
    return any(k in keys for k in ("geno", "genotype", "expr", "expression", "metab", "metabolites"))


def _resolve_ratio_by_modality(
    missing_cfg: dict[str, Any],
    split: str,
) -> dict[str, float] | None:
    raw = missing_cfg.get("ratio_by_modality", None)
    if not isinstance(raw, dict):
        return None
    entry = _get_split_value(raw, split, None)
    if not isinstance(entry, dict):
        # Allow flat syntax: ratio_by_modality: {geno: 0.1, expr: 0.2, metab: 0.3}
        if _is_modality_ratio_dict(raw):
            entry = raw
        else:
            return None
    return {
        "geno": _clamp_ratio(entry.get("geno", entry.get("genotype", 0.0)), name=f"missing.ratio_by_modality.{split}.geno"),
        "expr": _clamp_ratio(entry.get("expr", entry.get("expression", 0.0)), name=f"missing.ratio_by_modality.{split}.expr"),
        "metab": _clamp_ratio(entry.get("metab", entry.get("metabolites", 0.0)), name=f"missing.ratio_by_modality.{split}.metab"),
    }


def _build_codes_from_modality_ratios(
    sample_ids: Sequence[str],
    ratios: dict[str, float],
    *,
    split: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    n = len(sample_ids)
    codes = np.zeros(n, dtype=np.int64)
    specs = (
        ("geno", 1),
        ("expr", 2),
        ("metab", 4),
    )
    for name, bit in specs:
        ratio = _clamp_ratio(ratios.get(name, 0.0), name=f"ratio_by_modality.{split}.{name}")
        k = int(round(n * ratio))
        k = max(0, min(n, k))
        miss_mask = np.zeros(n, dtype=bool)
        if k > 0:
            idx = rng.choice(n, size=k, replace=False)
            miss_mask[idx] = True
        codes[miss_mask] = codes[miss_mask] | int(bit)
    return codes


@dataclass
class YStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray) -> "YStandardizer":
        mean = y.mean(axis=0)
        std = np.where(y.std(axis=0) < 1e-8, 1.0, y.std(axis=0))
        return cls(mean=mean, std=std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8").strip()) or {}
    return cfg


def with_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "seed": 42,
        "data_dir": "data",
        "output_dir": "outputs",
        "traits": TRAIT_COLUMNS,
        "validation_split": 0.1,
        "training": {
            "epochs": 200,
            "batch_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "use_aux_losses": True,
            "aux_loss_weights": {
                "a": 1.0,
                "b": 1.0,
                "c": 1.0,
            },
            "prompt_lr_mult": 3.0,
            "prompt_warmup_freeze_branch_epochs": 5,
            "patience": 25,
            "min_delta": 1e-5,
            "num_workers": 0,
        },
        "model": {
            "branch_emb_dim": 128,
            "fusion_hidden_dims": [256, 128],
            "dropout": 0.2,
            "use_batchnorm": True,
            "fusion_num_heads": 4,
            "fusion_layers": 6,
            "prompt_enable": True,
            "prompt_length": 12,
            "prompt_depth": 3,
            "num_missing_types": 8,
            "prompt_gate_init": 0.05,
            "prompt_gate_learnable": True,
            "prompt_gate_per_modality": True,
            "prompt_gate_use_missing_delta": True,
            "prompt_gate_missing_delta_scale": 0.4,
            "prompt_gate_missing_delta_init_no_missing": -0.04,
            "prompt_gate_missing_delta_init_missing": 0.12,
            "prompt_disable_on_complete": False,
            "prompt_complete_update_scale": 1.0,
            "mamba_layers": 2,
            "mamba_d_state": 16,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
        },
        "missing": {
            "enable": True,
            "ratio": {"train": 0.7, "valid": 0.7, "test": 0.7},
            "type": {"train": "mixed", "valid": "mixed", "test": "mixed"},
            "ratio_by_modality": {},
            "drop_missing_samples": False,
            "pattern_probs": {"geno": 1.0, "expr": 1.0, "metab": 1.0},
            "table_root": "outputs/missing_tables",
            "simulate_missing": False,
            "simulate_pool": [0, 1, 2, 4],
            "fill_strategy": "zero",
            "fill_value": 0.0,
        },
        "preprocess": {
            "topk_genotype": None,
            "topk_expression": None,
            "topk_metabolites": None,
        },
        "device": "auto",
        "single_trait_mode": True,
        "logging": {
            "log_prompt_gate": True,
        },
    }
    return _deep_merge(defaults, cfg)


def choose_device(cfg_device: str) -> torch.device:
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_train_valid(train_ids: Sequence[str], valid_ratio: float, seed: int, fold: int) -> tuple[list[str], list[str]]:
    ids = np.array(list(train_ids), dtype=object)
    perm = np.random.RandomState(seed + int(fold)).permutation(len(ids))
    n_valid = max(1, int(round(len(ids) * valid_ratio)))
    return ids[perm[n_valid:]].tolist(), ids[perm[:n_valid]].tolist()


def _get_split_value(mapping: Any, split: str, default: Any) -> Any:
    if not isinstance(mapping, dict):
        return default
    if split in mapping:
        return mapping[split]
    if split == "valid" and "val" in mapping:
        return mapping["val"]
    return default


def _sample_mixed_codes(n: int, rng: np.random.RandomState, pattern_probs: dict[str, float]) -> np.ndarray:
    codes, weights = [], []
    for k, v in pattern_probs.items():
        kk = str(k).strip().lower()
        if kk in MISSING_CODE_MAP and kk != "none" and float(v) > 0:
            codes.append(MISSING_CODE_MAP[kk])
            weights.append(float(v))
    if not codes:
        codes, weights = [1, 2, 4], [1.0, 1.0, 1.0]
    p = np.asarray(weights, dtype=np.float64)
    p = p / p.sum()
    return rng.choice(np.asarray(codes, dtype=np.int64), size=n, replace=True, p=p)


def _build_or_load_missing_codes(
    sample_ids: list[str],
    split: str,
    fold: int,
    missing_cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
    n = len(sample_ids)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    ratio_by_modality = _resolve_ratio_by_modality(missing_cfg, split)
    ratio = float(_get_split_value(missing_cfg.get("ratio", {}), split, 0.0))
    ratio = _clamp_ratio(ratio, name=f"missing.ratio.{split}")
    mtype = str(_get_split_value(missing_cfg.get("type", {}), split, "none")).strip().lower()
    table_root = Path(str(missing_cfg.get("table_root", "outputs/missing_tables")))
    table_root.mkdir(parents=True, exist_ok=True)
    sid_hash = hashlib.sha1("|".join(sample_ids).encode("utf-8")).hexdigest()[:12]
    if ratio_by_modality is not None:
        mode_tag = (
            "bymod_"
            f"g{_ratio_to_tag(ratio_by_modality['geno'])}_"
            f"e{_ratio_to_tag(ratio_by_modality['expr'])}_"
            f"m{_ratio_to_tag(ratio_by_modality['metab'])}"
        )
    else:
        mode_tag = f"{mtype}_{_ratio_to_tag(ratio)}"
    path = table_root / f"fold{fold}_{split}_{mode_tag}_seed{int(seed)}_{sid_hash}.npz"

    if path.exists():
        blob = np.load(path, allow_pickle=True)
        saved_ids = [str(x) for x in blob["sample_ids"].tolist()]
        saved_codes = blob["missing_codes"].astype(np.int64)
        if saved_ids == sample_ids and len(saved_codes) == n:
            return saved_codes

    rng = np.random.RandomState(seed + int(fold) * 1009 + {"train": 11, "valid": 17, "test": 23}.get(split, 31))
    if ratio_by_modality is not None:
        codes = _build_codes_from_modality_ratios(
            sample_ids,
            ratio_by_modality,
            split=split,
            rng=rng,
        )
    else:
        codes = np.zeros(n, dtype=np.int64)
        k = int(n * ratio)
        if k > 0 and mtype != "none":
            idx = rng.choice(n, size=k, replace=False)
            if mtype in {"mixed", "both"}:
                codes[idx] = _sample_mixed_codes(k, rng=rng, pattern_probs=missing_cfg.get("pattern_probs", {}))
            elif mtype in MISSING_CODE_MAP:
                codes[idx] = MISSING_CODE_MAP[mtype]
            else:
                raise ValueError(f"Unsupported missing.type[{split}]={mtype}")

    if ratio_by_modality is None and split == "train" and bool(missing_cfg.get("simulate_missing", False)):
        complete = np.where(codes == 0)[0]
        if complete.size > 0:
            pool = np.asarray(missing_cfg.get("simulate_pool", [0, 1, 2, 4]), dtype=np.int64)
            pool = pool[(pool >= 0) & (pool <= 7)]
            if pool.size == 0:
                pool = np.asarray([0, 1, 2, 4], dtype=np.int64)
            codes[complete] = rng.choice(pool, size=complete.size, replace=True)

    np.savez(path, sample_ids=np.asarray(sample_ids, dtype=object), missing_codes=codes)
    return codes


def _should_drop_missing_samples(missing_cfg: dict[str, Any]) -> bool:
    return bool(missing_cfg.get("drop_missing_samples", False))


def _filter_complete_samples(
    sample_ids: list[str],
    missing_codes: np.ndarray,
    split: str,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    codes = np.asarray(missing_codes, dtype=np.int64)
    if len(sample_ids) != int(codes.shape[0]):
        raise ValueError(
            f"sample_ids/missing_codes length mismatch for {split}: "
            f"{len(sample_ids)} vs {codes.shape[0]}"
        )
    keep = codes == 0
    if not bool(keep.any()):
        raise ValueError(
            f"drop_missing_samples removed every {split} sample. "
            "Reduce missing ratio or disable sample dropping."
        )
    kept_ids = [sid for sid, ok in zip(sample_ids, keep.tolist()) if ok]
    kept_codes = codes[keep]
    stats = {
        f"{split}_before_drop": int(len(sample_ids)),
        f"{split}_after_drop": int(len(kept_ids)),
        f"{split}_dropped": int(len(sample_ids) - len(kept_ids)),
    }
    return kept_ids, kept_codes, stats


def _resolve_fill_vectors(
    missing_cfg: dict[str, Any],
    xg: np.ndarray,
    xe: np.ndarray,
    xm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strategy = str(missing_cfg.get("fill_strategy", "zero")).strip().lower()
    if strategy == "mean":
        return np.mean(xg, axis=0), np.mean(xe, axis=0), np.mean(xm, axis=0)
    if strategy == "one":
        return np.ones(xg.shape[1]), np.ones(xe.shape[1]), np.ones(xm.shape[1])
    if strategy == "constant":
        fv = missing_cfg.get("fill_value", 0.0)
        if isinstance(fv, dict):
            vg, ve, vm = float(fv.get("geno", 0.0)), float(fv.get("expr", 0.0)), float(fv.get("metab", 0.0))
        else:
            vg = ve = vm = float(fv)
        return np.full(xg.shape[1], vg), np.full(xe.shape[1], ve), np.full(xm.shape[1], vm)
    return np.zeros(xg.shape[1]), np.zeros(xe.shape[1]), np.zeros(xm.shape[1])


def _apply_missing(
    xg: np.ndarray,
    xe: np.ndarray,
    xm: np.ndarray,
    codes: np.ndarray,
    fill_g: np.ndarray,
    fill_e: np.ndarray,
    fill_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    og, oe, om = np.array(xg, copy=True), np.array(xe, copy=True), np.array(xm, copy=True)
    mg, me, mm = (codes & 1) > 0, (codes & 2) > 0, (codes & 4) > 0
    if mg.any():
        og[mg, :] = fill_g
    if me.any():
        oe[me, :] = fill_e
    if mm.any():
        om[mm, :] = fill_m
    present = np.ones((len(codes), 3), dtype=np.float32)
    present[mg, 0], present[me, 1], present[mm, 2] = 0.0, 0.0, 0.0
    return og.astype(np.float32), oe.astype(np.float32), om.astype(np.float32), present


def make_loader(
    geno: np.ndarray,
    expr: np.ndarray,
    metab: np.ndarray,
    present: np.ndarray,
    missing_codes: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(geno, dtype=torch.float32).unsqueeze(1),
        torch.tensor(expr, dtype=torch.float32),
        torch.tensor(metab, dtype=torch.float32),
        torch.tensor(present, dtype=torch.float32),
        torch.tensor(missing_codes, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _safe_to_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, out_dim: int) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds, trues, losses = [], [], []
    loss_fn = nn.L1Loss()
    with torch.no_grad():
        for geno, expr, metab, present, missing_code, y in loader:
            out = model(
                geno.to(device),
                expr.to(device),
                metab.to(device),
                missing_code=missing_code.to(device),
            )
            loss = loss_fn(out, y.to(device))
            losses.append(float(loss.item()))
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    p = np.concatenate(preds, axis=0) if preds else np.empty((0, int(out_dim)), dtype=np.float32)
    t = np.concatenate(trues, axis=0) if trues else np.empty((0, int(out_dim)), dtype=np.float32)
    return p, t, float(np.mean(losses) if losses else 0.0)


def _get_prompt_gate_vector(
    model: nn.Module,
    missing_code: int | None = None,
    *,
    effective: bool = False,
) -> np.ndarray | None:
    getter_name = "get_effective_prompt_gate" if effective else "get_prompt_gate"
    getter = getattr(model, getter_name, None)
    if getter is None or not callable(getter):
        return None
    try:
        if effective and missing_code is None:
            return None
        if missing_code is None:
            gate = getter()
        else:
            device = next(model.parameters()).device
            code = torch.tensor([int(missing_code)], dtype=torch.long, device=device)
            gate = getter(code)
        if isinstance(gate, torch.Tensor):
            arr = gate.detach().cpu().reshape(-1).numpy().astype(np.float64)
        else:
            arr = np.asarray([float(gate)], dtype=np.float64)
        if arr.size == 0:
            return None
        if arr.size == 1:
            return np.repeat(arr, 3)
        if arr.size < 3:
            return np.pad(arr, (0, 3 - arr.size), mode="edge")
        return arr[:3]
    except Exception:
        return None


def _gate_stats_row(model: nn.Module) -> dict[str, float | None]:
    stats: dict[str, float | None] = {}

    def _fill(prefix: str, vec: np.ndarray | None) -> None:
        if vec is None:
            stats[prefix] = None
            stats[f"{prefix}_g"] = None
            stats[f"{prefix}_e"] = None
            stats[f"{prefix}_m"] = None
            return
        stats[prefix] = float(np.mean(vec))
        stats[f"{prefix}_g"] = float(vec[0])
        stats[f"{prefix}_e"] = float(vec[1])
        stats[f"{prefix}_m"] = float(vec[2])

    gate_now = _get_prompt_gate_vector(model, missing_code=None, effective=False)
    gate_code0 = _get_prompt_gate_vector(model, missing_code=0, effective=False)
    gate_code7 = _get_prompt_gate_vector(model, missing_code=7, effective=False)
    eff_code0 = _get_prompt_gate_vector(model, missing_code=0, effective=True)
    eff_code7 = _get_prompt_gate_vector(model, missing_code=7, effective=True)

    _fill("prompt_gate", gate_now)
    _fill("prompt_gate_code0", gate_code0)
    _fill("prompt_gate_code7", gate_code7)
    _fill("effective_prompt_gate_code0", eff_code0)
    _fill("effective_prompt_gate_code7", eff_code7)
    return stats


def _set_branch_trainable(model: nn.Module, trainable: bool) -> None:
    branch_names = ("genotype_branch", "expression_branch", "metabolites_branch")
    for name in branch_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = bool(trainable)


def _build_optimizer(model: nn.Module, tr_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    base_lr = float(tr_cfg["lr"])
    weight_decay = float(tr_cfg["weight_decay"])
    prompt_lr_mult = float(tr_cfg.get("prompt_lr_mult", 1.0))
    prompt_lr_mult = max(prompt_lr_mult, 0.0)
    prompt_keys = ("prompt_learner", "prompted_fusion", "prompt_gate")

    base_params: list[torch.nn.Parameter] = []
    prompt_params: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if any(k in name for k in prompt_keys):
            prompt_params.append(p)
        else:
            base_params.append(p)

    param_groups: list[dict[str, Any]] = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr})
    if prompt_params:
        param_groups.append({"params": prompt_params, "lr": base_lr * prompt_lr_mult})

    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def run_one_fold(
    fold: int,
    cfg: Dict[str, Any],
    aligned,
    train_ids: list[str],
    test_ids: list[str],
    device: torch.device,
    traits: list[str],
    output_dir: Path,
) -> Dict[str, Any]:
    if len(traits) != 1:
        raise ValueError(f"Single-regression mode requires exactly one trait, got {traits}")
    tr_cfg, model_cfg, prep_cfg = cfg["training"], cfg["model"], cfg["preprocess"]
    missing_cfg, missing_enabled = cfg["missing"], bool(cfg["missing"].get("enable", False))
    train_inner_ids, valid_ids = split_train_valid(train_ids, float(cfg["validation_split"]), int(cfg["seed"]), fold)
    drop_stats: dict[str, int] = {}

    if missing_enabled:
        train_codes = _build_or_load_missing_codes(train_inner_ids, "train", fold, missing_cfg, int(cfg["seed"]))
        valid_codes = _build_or_load_missing_codes(valid_ids, "valid", fold, missing_cfg, int(cfg["seed"]))
        test_codes = _build_or_load_missing_codes(test_ids, "test", fold, missing_cfg, int(cfg["seed"]))
    else:
        train_codes = np.zeros(len(train_inner_ids), dtype=np.int64)
        valid_codes = np.zeros(len(valid_ids), dtype=np.int64)
        test_codes = np.zeros(len(test_ids), dtype=np.int64)

    if _should_drop_missing_samples(missing_cfg):
        train_inner_ids, train_codes, stats = _filter_complete_samples(train_inner_ids, train_codes, "train")
        drop_stats.update(stats)
        valid_ids, valid_codes, stats = _filter_complete_samples(valid_ids, valid_codes, "valid")
        drop_stats.update(stats)
        test_ids, test_codes, stats = _filter_complete_samples(test_ids, test_codes, "test")
        drop_stats.update(stats)

    pre = FoldPreprocessor(
        topk_genotype=prep_cfg.get("topk_genotype"),
        topk_expression=prep_cfg.get("topk_expression"),
        topk_metabolites=prep_cfg.get("topk_metabolites"),
    )
    pre.fit(
        aligned.genotype.loc[train_inner_ids],
        aligned.expression.loc[train_inner_ids],
        aligned.metabolites.loc[train_inner_ids],
    )

    x_train_g = pre.transform_genotype(aligned.genotype.loc[train_inner_ids]).to_numpy(dtype=np.float32)
    x_train_e = pre.transform_expression(aligned.expression.loc[train_inner_ids]).to_numpy(dtype=np.float32)
    x_train_m = pre.transform_metabolites(aligned.metabolites.loc[train_inner_ids]).to_numpy(dtype=np.float32)
    y_train_raw = aligned.phenotype.loc[train_inner_ids, traits].to_numpy(dtype=np.float32)

    x_valid_g = pre.transform_genotype(aligned.genotype.loc[valid_ids]).to_numpy(dtype=np.float32)
    x_valid_e = pre.transform_expression(aligned.expression.loc[valid_ids]).to_numpy(dtype=np.float32)
    x_valid_m = pre.transform_metabolites(aligned.metabolites.loc[valid_ids]).to_numpy(dtype=np.float32)
    y_valid_raw = aligned.phenotype.loc[valid_ids, traits].to_numpy(dtype=np.float32)

    x_test_g = pre.transform_genotype(aligned.genotype.loc[test_ids]).to_numpy(dtype=np.float32)
    x_test_e = pre.transform_expression(aligned.expression.loc[test_ids]).to_numpy(dtype=np.float32)
    x_test_m = pre.transform_metabolites(aligned.metabolites.loc[test_ids]).to_numpy(dtype=np.float32)
    y_test_raw = aligned.phenotype.loc[test_ids, traits].to_numpy(dtype=np.float32)

    fill_g, fill_e, fill_m = _resolve_fill_vectors(missing_cfg, x_train_g, x_train_e, x_train_m)
    x_train_g, x_train_e, x_train_m, train_present = _apply_missing(x_train_g, x_train_e, x_train_m, train_codes, fill_g, fill_e, fill_m)
    x_valid_g, x_valid_e, x_valid_m, valid_present = _apply_missing(x_valid_g, x_valid_e, x_valid_m, valid_codes, fill_g, fill_e, fill_m)
    x_test_g, x_test_e, x_test_m, test_present = _apply_missing(x_test_g, x_test_e, x_test_m, test_codes, fill_g, fill_e, fill_m)

    y_std = YStandardizer.fit(y_train_raw)
    y_train = y_std.transform(y_train_raw).astype(np.float32)
    y_valid = y_std.transform(y_valid_raw).astype(np.float32)
    y_test = y_std.transform(y_test_raw).astype(np.float32)

    train_loader = make_loader(x_train_g, x_train_e, x_train_m, train_present, train_codes, y_train, int(tr_cfg["batch_size"]), True, int(tr_cfg["num_workers"]))
    valid_loader = make_loader(x_valid_g, x_valid_e, x_valid_m, valid_present, valid_codes, y_valid, int(tr_cfg["batch_size"]), False, int(tr_cfg["num_workers"]))
    test_loader = make_loader(x_test_g, x_test_e, x_test_m, test_present, test_codes, y_test, int(tr_cfg["batch_size"]), False, int(tr_cfg["num_workers"]))

    model = MultiOmicsMultiTaskRegressor(
        num_genotype_features=x_train_g.shape[1],
        num_expression_features=x_train_e.shape[1],
        num_metabolites_features=x_train_m.shape[1],
        num_tasks=1,
        branch_emb_dim=int(model_cfg["branch_emb_dim"]),
        fusion_hidden_dims=list(model_cfg["fusion_hidden_dims"]),
        dropout=float(model_cfg["dropout"]),
        use_batchnorm=bool(model_cfg["use_batchnorm"]),
        fusion_num_heads=int(model_cfg["fusion_num_heads"]),
        fusion_layers=int(model_cfg["fusion_layers"]),
        prompt_enable=bool(model_cfg["prompt_enable"]),
        prompt_length=int(model_cfg["prompt_length"]),
        prompt_depth=int(model_cfg["prompt_depth"]),
        num_missing_types=int(model_cfg["num_missing_types"]),
        prompt_gate_init=model_cfg.get("prompt_gate_init", 0.0),
        prompt_gate_learnable=bool(model_cfg.get("prompt_gate_learnable", True)),
        prompt_gate_per_modality=bool(model_cfg.get("prompt_gate_per_modality", True)),
        prompt_gate_use_missing_delta=bool(model_cfg.get("prompt_gate_use_missing_delta", True)),
        prompt_gate_missing_delta_scale=float(model_cfg.get("prompt_gate_missing_delta_scale", 0.2)),
        prompt_gate_missing_delta_init_no_missing=model_cfg.get("prompt_gate_missing_delta_init_no_missing", -0.04),
        prompt_gate_missing_delta_init_missing=model_cfg.get("prompt_gate_missing_delta_init_missing", 0.12),
        prompt_disable_on_complete=bool(model_cfg.get("prompt_disable_on_complete", True)),
        prompt_complete_update_scale=float(model_cfg.get("prompt_complete_update_scale", 0.0)),
        mamba_layers=int(model_cfg.get("mamba_layers", 2)),
        mamba_d_state=int(model_cfg.get("mamba_d_state", 16)),
        mamba_d_conv=int(model_cfg.get("mamba_d_conv", 4)),
        mamba_expand=int(model_cfg.get("mamba_expand", 2)),
    ).to(device)

    optimizer = _build_optimizer(model, tr_cfg)
    loss_fn = nn.L1Loss()
    use_aux_losses = bool(tr_cfg.get("use_aux_losses", True))
    aux_cfg = tr_cfg.get("aux_loss_weights", {})
    aux_w_g = float(aux_cfg.get("a", 1.0)) if use_aux_losses else 0.0
    aux_w_e = float(aux_cfg.get("b", 1.0)) if use_aux_losses else 0.0
    aux_w_m = float(aux_cfg.get("c", 1.0)) if use_aux_losses else 0.0
    best_val, best_state, wait = float("inf"), None, 0
    epochs, patience, min_delta = int(tr_cfg["epochs"]), int(tr_cfg["patience"]), float(tr_cfg["min_delta"])
    warmup_epochs = int(tr_cfg.get("prompt_warmup_freeze_branch_epochs", 0))
    prompt_enabled = bool(model_cfg.get("prompt_enable", False))
    gate_log_rows: list[dict[str, Any]] = []
    trait_tag = traits[0]

    for epoch_idx in range(1, epochs + 1):
        if prompt_enabled and warmup_epochs > 0:
            _set_branch_trainable(model, trainable=(epoch_idx > warmup_epochs))
        else:
            _set_branch_trainable(model, trainable=True)

        model.train()
        for geno, expr, metab, present, missing_code, yb in train_loader:
            optimizer.zero_grad()
            if use_aux_losses:
                out_main, out_geno, out_expr, out_metab = model(
                    geno.to(device),
                    expr.to(device),
                    metab.to(device),
                    missing_code=missing_code.to(device),
                    return_aux=True,
                )
            else:
                out_main = model(
                    geno.to(device),
                    expr.to(device),
                    metab.to(device),
                    missing_code=missing_code.to(device),
                    return_aux=False,
                )
            target = yb.to(device)
            main_loss = loss_fn(out_main, target)
            if use_aux_losses:
                geno_loss = loss_fn(out_geno, target)
                expr_loss = loss_fn(out_expr, target)
                metab_loss = loss_fn(out_metab, target)
                total_loss = main_loss + aux_w_g * geno_loss + aux_w_e * expr_loss + aux_w_m * metab_loss
            else:
                total_loss = main_loss
            total_loss.backward()
            optimizer.step()

        _, _, val_loss = evaluate_model(model, valid_loader, device, out_dim=1)
        is_best = val_loss < (best_val - min_delta)
        gate_log_rows.append(
            {
                "fold": int(fold),
                "trait": trait_tag,
                "epoch": int(epoch_idx),
                "prompt_enabled": int(prompt_enabled),
                "branch_trainable": int((not prompt_enabled) or (warmup_epochs <= 0) or (epoch_idx > warmup_epochs)),
                "val_loss": float(val_loss),
                "is_best": int(is_best),
            }
        )
        gate_log_rows[-1].update(_gate_stats_row(model))
        if is_best:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if bool(cfg.get("logging", {}).get("log_prompt_gate", True)):
        gate_log_dir = output_dir / "prompt_gate_logs"
        gate_log_dir.mkdir(parents=True, exist_ok=True)
        gate_log_path = gate_log_dir / f"{trait_tag}_fold_{fold}.csv"
        _safe_to_csv(pd.DataFrame(gate_log_rows), gate_log_path, index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_std, true_std, _ = evaluate_model(model, test_loader, device, out_dim=1)
    pred_test, true_test = y_std.inverse_transform(pred_std), y_std.inverse_transform(true_std)
    fold_metrics = regression_metrics_per_trait(true_test, pred_test, traits)
    for row in fold_metrics:
        row["fold"] = fold

    extra_df = pd.DataFrame(
        {
            "missing_code": test_codes.astype(np.int64),
            "present_geno": test_present[:, 0].astype(np.float32),
            "present_expr": test_present[:, 1].astype(np.float32),
            "present_metab": test_present[:, 2].astype(np.float32),
        }
    )
    fold_pred_df = pd.concat(
        [
            pd.DataFrame({"sample_id": test_ids, "fold": fold}),
            extra_df,
            pd.DataFrame(true_test, columns=[f"true_{t}" for t in traits]),
            pd.DataFrame(pred_test, columns=[f"pred_{t}" for t in traits]),
        ],
        axis=1,
    )
    return {
        "fold": fold,
        "metrics": fold_metrics,
        "test_predictions": fold_pred_df,
        "best_val_loss": best_val,
        "drop_stats": drop_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed 10-fold CV training for OmicMAP")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--trait_name", type=str, default=None, help="Run single trait (yd/tp/gn/kgw).")
    args = parser.parse_args()

    cfg = with_defaults(load_config(args.config))
    set_seed(int(cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_pred_dir = output_dir / "fold_predictions"
    fold_pred_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("missing", {}).get("enable", False)):
        table_root = Path(str(cfg["missing"].get("table_root", "outputs/missing_tables")))
        if not table_root.is_absolute():
            table_root = output_dir / table_root
        table_root.mkdir(parents=True, exist_ok=True)
        cfg["missing"]["table_root"] = str(table_root)

    device = choose_device(str(cfg["device"]))
    all_traits = list(cfg["traits"])
    trait_name = str(args.trait_name).strip() if args.trait_name is not None else None
    if trait_name is not None:
        if trait_name not in TRAIT_COLUMNS:
            raise ValueError(f"Unsupported trait_name={trait_name}, expected one of {TRAIT_COLUMNS}")
        all_traits = [trait_name]
        cfg["traits"] = all_traits

    aligned = load_and_align_data(data_dir=cfg["data_dir"], traits=all_traits)
    all_folds = sorted(pd.Series(aligned.fold_map.values).astype(int).unique().tolist())
    single_trait_mode = True
    fold_rows: List[Dict[str, Any]] = []
    fold_drop_rows: List[Dict[str, Any]] = []

    oof_by_trait: Dict[str, List[pd.DataFrame]] = {t: [] for t in all_traits}
    for trait in all_traits:
        trait_fold_dir = fold_pred_dir / trait
        trait_fold_dir.mkdir(parents=True, exist_ok=True)
        for fold in all_folds:
            train_ids = [sid for sid in aligned.sample_ids if int(aligned.fold_map.loc[sid]) != int(fold)]
            test_ids = [sid for sid in aligned.sample_ids if int(aligned.fold_map.loc[sid]) == int(fold)]
            result = run_one_fold(fold, cfg, aligned, train_ids, test_ids, device, [trait], output_dir)
            fold_rows.extend(result["metrics"])
            if result.get("drop_stats"):
                fold_drop_rows.append({"trait": trait, "fold": int(fold), **result["drop_stats"]})
            fold_pred_df = result["test_predictions"]
            oof_by_trait[trait].append(fold_pred_df)
            _safe_to_csv(fold_pred_df, trait_fold_dir / f"fold_{fold}_test_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_rows)
    fold_metrics_path = output_dir / "fold_metrics.csv"
    _safe_to_csv(fold_metrics_df, fold_metrics_path, index=False)

    summary: Dict[str, Any] = {
        "config": cfg,
        "num_folds": int(len(all_folds)),
        "traits": all_traits,
        "folds": all_folds,
        "single_trait_mode": single_trait_mode,
        "metrics_mean_by_trait": {},
        "metrics_se_by_trait": {},
        "overall_oof_metrics_by_trait": {},
        "drop_missing_samples": {
            "enabled": _should_drop_missing_samples(cfg.get("missing", {})),
            "by_fold": fold_drop_rows,
        },
    }
    for trait in all_traits:
        sub = fold_metrics_df[fold_metrics_df["trait"] == trait]
        n_folds = max(1, int(len(sub)))
        summary["metrics_mean_by_trait"][trait] = {
            "pearson": float(sub["pearson"].mean()),
            "pcc": float(sub["pcc"].mean()),
            "rmse": float(sub["rmse"].mean()),
            "r2": float(sub["r2"].mean()),
        }
        summary["metrics_se_by_trait"][trait] = {
            "pearson": float(sub["pearson"].std(ddof=0) / np.sqrt(n_folds)),
            "pcc": float(sub["pcc"].std(ddof=0) / np.sqrt(n_folds)),
            "rmse": float(sub["rmse"].std(ddof=0) / np.sqrt(n_folds)),
            "r2": float(sub["r2"].std(ddof=0) / np.sqrt(n_folds)),
        }

    for trait in all_traits:
        trait_oof_df = pd.concat(oof_by_trait[trait], axis=0, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
        trait_oof_path = output_dir / f"oof_predictions_{trait}.csv"
        _safe_to_csv(trait_oof_df, trait_oof_path, index=False)
        y_true = trait_oof_df[[f"true_{trait}"]].to_numpy(dtype=np.float32)
        y_pred = trait_oof_df[[f"pred_{trait}"]].to_numpy(dtype=np.float32)
        row = regression_metrics_per_trait(y_true, y_pred, [trait])[0]
        summary["overall_oof_metrics_by_trait"][trait] = {
                "pearson": float(row["pearson"]),
                "pcc": float(row["pcc"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]),
            }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Done] fold metrics: {fold_metrics_path}")
    print(f"[Done] oof predictions: {output_dir / 'oof_predictions_<trait>.csv'}")
    print(f"[Done] summary: {summary_path}")


if __name__ == "__main__":
    main()

