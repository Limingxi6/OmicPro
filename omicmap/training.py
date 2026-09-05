from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class FitResult:
    best_val_loss: float
    best_state_dict: dict[str, torch.Tensor]
    history: list[dict[str, Any]]
    epochs_ran: int


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dim: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    total_abs_error = 0.0
    total_values = 0
    with torch.no_grad():
        for geno, expr, metab, _present, missing_code, y in loader:
            target = y.to(device)
            out = model(
                geno.to(device),
                expr.to(device),
                metab.to(device),
                missing_code=missing_code.to(device),
            )
            total_abs_error += float(torch.abs(out - target).sum().item())
            total_values += int(target.numel())
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())

    pred_array = np.concatenate(preds, axis=0) if preds else np.empty((0, int(out_dim)), dtype=np.float32)
    true_array = np.concatenate(trues, axis=0) if trues else np.empty((0, int(out_dim)), dtype=np.float32)
    mae = total_abs_error / max(1, total_values)
    return pred_array, true_array, float(mae)


def _get_prompt_gate_vector(
    model: nn.Module,
    missing_code: int | None = None,
    *,
    effective: bool = False,
) -> np.ndarray | None:
    getter = getattr(model, "get_effective_prompt_gate" if effective else "get_prompt_gate", None)
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
            arr = gate.detach().cpu().numpy()
            if arr.ndim > 1:
                arr = arr[0]
            return np.asarray(arr, dtype=np.float64).reshape(-1)
    except (RuntimeError, TypeError, ValueError):
        return None
    return None


def gate_stats_row(model: nn.Module) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    base = _get_prompt_gate_vector(model)
    for idx, name in enumerate(("geno", "expr", "metab")):
        if base is None or base.size == 0:
            row[f"gate_base_{name}"] = None
        else:
            row[f"gate_base_{name}"] = float(base[min(idx, base.size - 1)])
    for code in range(8):
        effective = _get_prompt_gate_vector(model, code, effective=True)
        for idx, name in enumerate(("geno", "expr", "metab")):
            key = f"gate_effective_code{code}_{name}"
            row[key] = (
                None
                if effective is None or effective.size == 0
                else float(effective[min(idx, effective.size - 1)])
            )
    return row


def set_branch_trainable(model: nn.Module, trainable: bool) -> None:
    for name in ("genotype_branch", "expression_branch", "metabolites_branch"):
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = bool(trainable)


def build_optimizer(model: nn.Module, training_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    base_lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg["weight_decay"])
    prompt_lr_mult = max(float(training_cfg.get("prompt_lr_mult", 1.0)), 0.0)
    prompt_keys = ("prompt_learner", "prompted_fusion", "prompt_gate")

    base_params: list[torch.nn.Parameter] = []
    prompt_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if any(key in name for key in prompt_keys):
            prompt_params.append(parameter)
        else:
            base_params.append(parameter)

    groups: list[dict[str, Any]] = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if prompt_params:
        groups.append({"params": prompt_params, "lr": base_lr * prompt_lr_mult})
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    training_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    *,
    history_context: Mapping[str, Any] | None = None,
) -> FitResult:
    """Shared single-target training loop for CV and explicit fixed splits."""

    optimizer = build_optimizer(model, training_cfg)
    loss_fn = nn.L1Loss()
    use_aux_losses = bool(training_cfg.get("use_aux_losses", True))
    aux_cfg = dict(training_cfg.get("aux_loss_weights", {}))
    aux_weights = (
        float(aux_cfg.get("geno", aux_cfg.get("a", 1.0))) if use_aux_losses else 0.0,
        float(aux_cfg.get("expr", aux_cfg.get("b", 1.0))) if use_aux_losses else 0.0,
        float(aux_cfg.get("metab", aux_cfg.get("c", 1.0))) if use_aux_losses else 0.0,
    )

    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg["patience"])
    min_delta = float(training_cfg["min_delta"])
    warmup_epochs = int(training_cfg.get("prompt_warmup_freeze_branch_epochs", 0))
    prompt_enabled = bool(model_cfg.get("prompt_enable", False))
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, Any]] = []
    context = dict(history_context or {})

    for epoch_idx in range(1, epochs + 1):
        branch_trainable = (not prompt_enabled) or warmup_epochs <= 0 or epoch_idx > warmup_epochs
        set_branch_trainable(model, trainable=branch_trainable)
        model.train()

        for geno, expr, metab, _present, missing_code, target in train_loader:
            optimizer.zero_grad()
            if use_aux_losses:
                main, pred_geno, pred_expr, pred_metab = model(
                    geno.to(device),
                    expr.to(device),
                    metab.to(device),
                    missing_code=missing_code.to(device),
                    return_aux=True,
                )
            else:
                main = model(
                    geno.to(device),
                    expr.to(device),
                    metab.to(device),
                    missing_code=missing_code.to(device),
                    return_aux=False,
                )

            target = target.to(device)
            total_loss = loss_fn(main, target)
            if use_aux_losses:
                total_loss = (
                    total_loss
                    + aux_weights[0] * loss_fn(pred_geno, target)
                    + aux_weights[1] * loss_fn(pred_expr, target)
                    + aux_weights[2] * loss_fn(pred_metab, target)
                )
            total_loss.backward()
            optimizer.step()

        _, _, val_loss = evaluate_model(model, valid_loader, device, out_dim=1)
        is_best = val_loss < (best_val - min_delta)
        row: dict[str, Any] = {
            **context,
            "epoch": int(epoch_idx),
            "prompt_enabled": int(prompt_enabled),
            "branch_trainable": int(branch_trainable),
            "val_loss": float(val_loss),
            "is_best": int(is_best),
        }
        row.update(gate_stats_row(model))
        history.append(row)

        if is_best:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return FitResult(
        best_val_loss=float(best_val),
        best_state_dict=best_state,
        history=history,
        epochs_ran=len(history),
    )
