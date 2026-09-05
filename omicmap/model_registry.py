from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn

from .model import MultiOmicsMultiTaskRegressor


ModelBuilder = Callable[[Mapping[str, Any]], nn.Module]
_MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model constructor under a stable, CLI-facing name."""

    key = str(name).strip().lower()
    if not key:
        raise ValueError("Model name must be non-empty.")

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if key in _MODEL_BUILDERS:
            raise ValueError(f"Model is already registered: {key}")
        _MODEL_BUILDERS[key] = builder
        return builder

    return decorator


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_BUILDERS))


def build_model(
    name: str,
    init_kwargs: Mapping[str, Any],
    device: torch.device | str | None = None,
) -> nn.Module:
    """Build a registered model from checkpoint-safe initialization values."""

    key = str(name).strip().lower()
    if key not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available models: {available_models()}")
    model = _MODEL_BUILDERS[key](dict(init_kwargs))
    return model.to(device) if device is not None else model


def omicmap_init_kwargs(
    model_cfg: Mapping[str, Any],
    *,
    num_genotype_features: int,
    num_expression_features: int,
    num_metabolites_features: int,
    num_tasks: int = 1,
) -> dict[str, Any]:
    """Convert the experiment config into the model's explicit constructor contract."""

    return {
        "num_genotype_features": int(num_genotype_features),
        "num_expression_features": int(num_expression_features),
        "num_metabolites_features": int(num_metabolites_features),
        "num_tasks": int(num_tasks),
        "branch_emb_dim": int(model_cfg["branch_emb_dim"]),
        "fusion_hidden_dims": list(model_cfg["fusion_hidden_dims"]),
        "dropout": float(model_cfg["dropout"]),
        "use_batchnorm": bool(model_cfg["use_batchnorm"]),
        "fusion_num_heads": int(model_cfg["fusion_num_heads"]),
        "fusion_layers": int(model_cfg["fusion_layers"]),
        "prompt_enable": bool(model_cfg["prompt_enable"]),
        "prompt_length": int(model_cfg["prompt_length"]),
        "prompt_depth": int(model_cfg["prompt_depth"]),
        "num_missing_types": int(model_cfg["num_missing_types"]),
        "prompt_gate_init": model_cfg.get("prompt_gate_init", 0.0),
        "prompt_gate_learnable": bool(model_cfg.get("prompt_gate_learnable", True)),
        "prompt_gate_per_modality": bool(model_cfg.get("prompt_gate_per_modality", True)),
        "prompt_gate_use_missing_delta": bool(model_cfg.get("prompt_gate_use_missing_delta", True)),
        "prompt_gate_missing_delta_scale": float(model_cfg.get("prompt_gate_missing_delta_scale", 0.2)),
        "prompt_gate_missing_delta_init_no_missing": model_cfg.get(
            "prompt_gate_missing_delta_init_no_missing", -0.04
        ),
        "prompt_gate_missing_delta_init_missing": model_cfg.get(
            "prompt_gate_missing_delta_init_missing", 0.12
        ),
        "prompt_disable_on_complete": bool(model_cfg.get("prompt_disable_on_complete", True)),
        "prompt_complete_update_scale": float(model_cfg.get("prompt_complete_update_scale", 0.0)),
        "mamba_layers": int(model_cfg.get("mamba_layers", 2)),
        "mamba_d_state": int(model_cfg.get("mamba_d_state", 16)),
        "mamba_d_conv": int(model_cfg.get("mamba_d_conv", 4)),
        "mamba_expand": int(model_cfg.get("mamba_expand", 2)),
    }


@register_model("omicmap")
def _build_omicmap(init_kwargs: Mapping[str, Any]) -> nn.Module:
    return MultiOmicsMultiTaskRegressor(**dict(init_kwargs))
