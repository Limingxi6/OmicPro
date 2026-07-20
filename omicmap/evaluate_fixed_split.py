from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data_utils import TRAIT_COLUMNS, get_fixed_fold_split, load_and_align_data
from .metrics import regression_metrics_per_trait
from .model import MultiOmicsMultiTaskRegressor
from .preprocess import FoldPreprocessor
from .train_cv import (
    YStandardizer,
    _apply_missing,
    _build_optimizer,
    _build_or_load_missing_codes,
    _filter_complete_samples,
    _gate_stats_row,
    _resolve_fill_vectors,
    _safe_to_csv,
    _set_branch_trainable,
    _should_drop_missing_samples,
    choose_device,
    evaluate_model,
    load_config,
    make_loader,
    set_seed,
    with_defaults,
)


def _parse_csv_list(raw: str, cast_fn) -> list[Any]:
    vals: list[Any] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(cast_fn(part))
    return vals


def _resolve_output_dir(cfg: Dict[str, Any], output_dir_arg: str | None, valid_fold: int, test_fold: int) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    base = Path(str(cfg.get("output_dir", "outputs")))
    return base / f"fixed_split_v{valid_fold}_t{test_fold}"


def _build_model(
    model_cfg: Dict[str, Any],
    x_train_g: np.ndarray,
    x_train_e: np.ndarray,
    x_train_m: np.ndarray,
    device: torch.device,
) -> nn.Module:
    return MultiOmicsMultiTaskRegressor(
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


def _build_pred_df(
    sample_ids: list[str],
    split: str,
    valid_fold: int,
    test_fold: int,
    missing_codes: np.ndarray,
    present_mask: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait: str,
) -> pd.DataFrame:
    extra_df = pd.DataFrame(
        {
            "missing_code": missing_codes.astype(np.int64),
            "present_geno": present_mask[:, 0].astype(np.float32),
            "present_expr": present_mask[:, 1].astype(np.float32),
            "present_metab": present_mask[:, 2].astype(np.float32),
        }
    )
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "split": split,
                    "valid_fold": int(valid_fold),
                    "test_fold": int(test_fold),
                }
            ),
            extra_df,
            pd.DataFrame(y_true, columns=[f"true_{trait}"]),
            pd.DataFrame(y_pred, columns=[f"pred_{trait}"]),
        ],
        axis=1,
    )


def run_one_trait_fixed_split(
    cfg: Dict[str, Any],
    aligned,
    trait: str,
    split_ids: Dict[str, list[str]],
    valid_fold: int,
    test_fold: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    tr_cfg = cfg["training"]
    model_cfg = cfg["model"]
    prep_cfg = cfg["preprocess"]
    missing_cfg = cfg["missing"]
    missing_enabled = bool(cfg["missing"].get("enable", False))

    train_ids = split_ids["train"]
    valid_ids = split_ids["valid"]
    test_ids = split_ids["test"]
    original_split_counts = {
        "train": int(len(train_ids)),
        "valid": int(len(valid_ids)),
        "test": int(len(test_ids)),
    }
    drop_stats: dict[str, int] = {}

    if missing_enabled:
        cache_fold = int(test_fold)
        train_codes = _build_or_load_missing_codes(train_ids, "train", cache_fold, missing_cfg, int(cfg["seed"]))
        valid_codes = _build_or_load_missing_codes(valid_ids, "valid", cache_fold, missing_cfg, int(cfg["seed"]))
        test_codes = _build_or_load_missing_codes(test_ids, "test", cache_fold, missing_cfg, int(cfg["seed"]))
    else:
        train_codes = np.zeros(len(train_ids), dtype=np.int64)
        valid_codes = np.zeros(len(valid_ids), dtype=np.int64)
        test_codes = np.zeros(len(test_ids), dtype=np.int64)

    if _should_drop_missing_samples(missing_cfg):
        train_ids, train_codes, stats = _filter_complete_samples(train_ids, train_codes, "train")
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
        aligned.genotype.loc[train_ids],
        aligned.expression.loc[train_ids],
        aligned.metabolites.loc[train_ids],
    )

    x_train_g = pre.transform_genotype(aligned.genotype.loc[train_ids]).to_numpy(dtype=np.float32)
    x_train_e = pre.transform_expression(aligned.expression.loc[train_ids]).to_numpy(dtype=np.float32)
    x_train_m = pre.transform_metabolites(aligned.metabolites.loc[train_ids]).to_numpy(dtype=np.float32)
    y_train_raw = aligned.phenotype.loc[train_ids, [trait]].to_numpy(dtype=np.float32)

    x_valid_g = pre.transform_genotype(aligned.genotype.loc[valid_ids]).to_numpy(dtype=np.float32)
    x_valid_e = pre.transform_expression(aligned.expression.loc[valid_ids]).to_numpy(dtype=np.float32)
    x_valid_m = pre.transform_metabolites(aligned.metabolites.loc[valid_ids]).to_numpy(dtype=np.float32)
    y_valid_raw = aligned.phenotype.loc[valid_ids, [trait]].to_numpy(dtype=np.float32)

    x_test_g = pre.transform_genotype(aligned.genotype.loc[test_ids]).to_numpy(dtype=np.float32)
    x_test_e = pre.transform_expression(aligned.expression.loc[test_ids]).to_numpy(dtype=np.float32)
    x_test_m = pre.transform_metabolites(aligned.metabolites.loc[test_ids]).to_numpy(dtype=np.float32)
    y_test_raw = aligned.phenotype.loc[test_ids, [trait]].to_numpy(dtype=np.float32)

    fill_g, fill_e, fill_m = _resolve_fill_vectors(missing_cfg, x_train_g, x_train_e, x_train_m)
    x_train_g, x_train_e, x_train_m, train_present = _apply_missing(
        x_train_g, x_train_e, x_train_m, train_codes, fill_g, fill_e, fill_m
    )
    x_valid_g, x_valid_e, x_valid_m, valid_present = _apply_missing(
        x_valid_g, x_valid_e, x_valid_m, valid_codes, fill_g, fill_e, fill_m
    )
    x_test_g, x_test_e, x_test_m, test_present = _apply_missing(
        x_test_g, x_test_e, x_test_m, test_codes, fill_g, fill_e, fill_m
    )

    y_std = YStandardizer.fit(y_train_raw)
    y_train = y_std.transform(y_train_raw).astype(np.float32)
    y_valid = y_std.transform(y_valid_raw).astype(np.float32)
    y_test = y_std.transform(y_test_raw).astype(np.float32)

    train_loader = make_loader(
        x_train_g,
        x_train_e,
        x_train_m,
        train_present,
        train_codes,
        y_train,
        int(tr_cfg["batch_size"]),
        True,
        int(tr_cfg["num_workers"]),
    )
    valid_loader = make_loader(
        x_valid_g,
        x_valid_e,
        x_valid_m,
        valid_present,
        valid_codes,
        y_valid,
        int(tr_cfg["batch_size"]),
        False,
        int(tr_cfg["num_workers"]),
    )
    test_loader = make_loader(
        x_test_g,
        x_test_e,
        x_test_m,
        test_present,
        test_codes,
        y_test,
        int(tr_cfg["batch_size"]),
        False,
        int(tr_cfg["num_workers"]),
    )

    model = _build_model(model_cfg, x_train_g, x_train_e, x_train_m, device)
    optimizer = _build_optimizer(model, tr_cfg)
    loss_fn = nn.L1Loss()

    use_aux_losses = bool(tr_cfg.get("use_aux_losses", True))
    aux_cfg = tr_cfg.get("aux_loss_weights", {})
    aux_w_g = float(aux_cfg.get("a", 1.0)) if use_aux_losses else 0.0
    aux_w_e = float(aux_cfg.get("b", 1.0)) if use_aux_losses else 0.0
    aux_w_m = float(aux_cfg.get("c", 1.0)) if use_aux_losses else 0.0

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    epochs = int(tr_cfg["epochs"])
    patience = int(tr_cfg["patience"])
    min_delta = float(tr_cfg["min_delta"])
    warmup_epochs = int(tr_cfg.get("prompt_warmup_freeze_branch_epochs", 0))
    prompt_enabled = bool(model_cfg.get("prompt_enable", False))
    gate_log_rows: list[dict[str, Any]] = []

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
                "trait": trait,
                "epoch": int(epoch_idx),
                "valid_fold": int(valid_fold),
                "test_fold": int(test_fold),
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
        gate_log_path = gate_log_dir / f"{trait}_v{valid_fold}_t{test_fold}.csv"
        _safe_to_csv(pd.DataFrame(gate_log_rows), gate_log_path, index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_valid_std, true_valid_std, valid_loss = evaluate_model(model, valid_loader, device, out_dim=1)
    pred_test_std, true_test_std, test_loss = evaluate_model(model, test_loader, device, out_dim=1)
    pred_valid = y_std.inverse_transform(pred_valid_std)
    true_valid = y_std.inverse_transform(true_valid_std)
    pred_test = y_std.inverse_transform(pred_test_std)
    true_test = y_std.inverse_transform(true_test_std)

    valid_metrics = regression_metrics_per_trait(true_valid, pred_valid, [trait])[0]
    test_metrics = regression_metrics_per_trait(true_test, pred_test, [trait])[0]
    valid_metrics.update(
        {
            "split": "valid",
            "valid_fold": int(valid_fold),
            "test_fold": int(test_fold),
            "n_samples": int(len(valid_ids)),
            "mae": float(valid_loss),
        }
    )
    test_metrics.update(
        {
            "split": "test",
            "valid_fold": int(valid_fold),
            "test_fold": int(test_fold),
            "n_samples": int(len(test_ids)),
            "mae": float(test_loss),
        }
    )

    valid_pred_df = _build_pred_df(
        sample_ids=valid_ids,
        split="valid",
        valid_fold=valid_fold,
        test_fold=test_fold,
        missing_codes=valid_codes,
        present_mask=valid_present,
        y_true=true_valid,
        y_pred=pred_valid,
        trait=trait,
    )
    test_pred_df = _build_pred_df(
        sample_ids=test_ids,
        split="test",
        valid_fold=valid_fold,
        test_fold=test_fold,
        missing_codes=test_codes,
        present_mask=test_present,
        y_true=true_test,
        y_pred=pred_test,
        trait=trait,
    )

    return {
        "metrics_rows": [valid_metrics, test_metrics],
        "predictions_df": pd.concat([valid_pred_df, test_pred_df], axis=0, ignore_index=True),
        "best_val_loss": float(best_val),
        "n_train": int(len(train_ids)),
        "n_valid": int(len(valid_ids)),
        "n_test": int(len(test_ids)),
        "original_split_counts": original_split_counts,
        "drop_stats": drop_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed 8:1:1 evaluation with explicit validation/test folds")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--trait_name", type=str, default=None, help="Run single trait (yd/tp/gn/kgw).")
    parser.add_argument(
        "--traits",
        type=str,
        default=None,
        help="Comma-separated traits, e.g. yd,tp,gn,kgw. Overrides config.traits.",
    )
    parser.add_argument("--valid_fold", type=int, default=9, help="Validation fold id. Default: 9")
    parser.add_argument("--test_fold", type=int, default=10, help="Independent test fold id. Default: 10")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output dir override")
    args = parser.parse_args()

    if int(args.valid_fold) == int(args.test_fold):
        raise ValueError("valid_fold and test_fold must be different.")

    cfg = with_defaults(load_config(args.config))
    set_seed(int(cfg["seed"]))

    output_dir = _resolve_output_dir(cfg, args.output_dir, int(args.valid_fold), int(args.test_fold))
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)

    if bool(cfg.get("missing", {}).get("enable", False)):
        table_root = Path(str(cfg["missing"].get("table_root", "outputs/missing_tables")))
        if not table_root.is_absolute():
            table_root = output_dir / table_root
        table_root.mkdir(parents=True, exist_ok=True)
        cfg["missing"]["table_root"] = str(table_root)

    all_traits = list(cfg["traits"])
    if args.traits is not None and args.trait_name is not None:
        raise ValueError("Use either --traits or --trait_name, not both.")

    traits_raw = _parse_csv_list(args.traits, str) if args.traits is not None else None
    if traits_raw is not None:
        if not traits_raw:
            raise ValueError("--traits is empty. Example: --traits yd,tp,gn,kgw")
        traits: list[str] = []
        for t in traits_raw:
            tt = str(t).strip()
            if tt and tt not in traits:
                traits.append(tt)
        invalid = [t for t in traits if t not in TRAIT_COLUMNS]
        if invalid:
            raise ValueError(f"Unsupported traits in --traits: {invalid}, expected subset of {TRAIT_COLUMNS}")
        all_traits = traits
        cfg["traits"] = all_traits

    trait_name = str(args.trait_name).strip() if args.trait_name is not None else None
    if trait_name is not None:
        if trait_name not in TRAIT_COLUMNS:
            raise ValueError(f"Unsupported trait_name={trait_name}, expected one of {TRAIT_COLUMNS}")
        all_traits = [trait_name]
        cfg["traits"] = all_traits

    device = choose_device(str(cfg["device"]))
    aligned = load_and_align_data(data_dir=cfg["data_dir"], traits=all_traits)
    split_ids = get_fixed_fold_split(
        aligned.fold_map,
        test_fold=int(args.test_fold),
        valid_fold=int(args.valid_fold),
    )

    metrics_rows: List[Dict[str, Any]] = []
    best_val_loss_by_trait: Dict[str, float] = {}
    effective_sample_counts_by_trait: Dict[str, Dict[str, int]] = {}
    drop_stats_by_trait: Dict[str, Dict[str, int]] = {}
    split_pred_dir = output_dir / "split_predictions"
    split_pred_dir.mkdir(parents=True, exist_ok=True)

    for trait in all_traits:
        result = run_one_trait_fixed_split(
            cfg=cfg,
            aligned=aligned,
            trait=trait,
            split_ids=split_ids,
            valid_fold=int(args.valid_fold),
            test_fold=int(args.test_fold),
            device=device,
            output_dir=output_dir,
        )
        metrics_rows.extend(result["metrics_rows"])
        best_val_loss_by_trait[trait] = float(result["best_val_loss"])
        effective_sample_counts_by_trait[trait] = {
            "train": int(result["n_train"]),
            "valid": int(result["n_valid"]),
            "test": int(result["n_test"]),
        }
        drop_stats_by_trait[trait] = dict(result.get("drop_stats", {}))
        pred_path = split_pred_dir / f"{trait}_v{args.valid_fold}_t{args.test_fold}_predictions.csv"
        _safe_to_csv(result["predictions_df"], pred_path, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "split_metrics.csv"
    _safe_to_csv(metrics_df, metrics_path, index=False)

    summary: Dict[str, Any] = {
        "config": cfg,
        "traits": all_traits,
        "split": {
            "train_folds": sorted(
                set(pd.to_numeric(aligned.fold_map, errors="coerce").astype(int).unique().tolist())
                - {int(args.valid_fold), int(args.test_fold)}
            ),
            "valid_fold": int(args.valid_fold),
            "test_fold": int(args.test_fold),
        },
        "num_samples": {
            "train": int(len(split_ids["train"])),
            "valid": int(len(split_ids["valid"])),
            "test": int(len(split_ids["test"])),
        },
        "effective_num_samples_by_trait": effective_sample_counts_by_trait,
        "drop_missing_samples": {
            "enabled": _should_drop_missing_samples(cfg.get("missing", {})),
            "by_trait": drop_stats_by_trait,
        },
        "best_val_loss_by_trait": best_val_loss_by_trait,
        "metrics_by_trait": {},
        "metrics_mean_by_split": {},
    }

    for trait in all_traits:
        summary["metrics_by_trait"][trait] = {}
        sub = metrics_df[metrics_df["trait"] == trait]
        for split_name in ("valid", "test"):
            one = sub[sub["split"] == split_name]
            if one.empty:
                continue
            row = one.iloc[0]
            summary["metrics_by_trait"][trait][split_name] = {
                "pearson": float(row["pearson"]),
                "pcc": float(row["pcc"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]),
                "mae": float(row["mae"]),
            }

    for split_name in ("valid", "test"):
        sub = metrics_df[metrics_df["split"] == split_name]
        if sub.empty:
            continue
        summary["metrics_mean_by_split"][split_name] = {
            "pearson": float(sub["pearson"].mean()),
            "pcc": float(sub["pcc"].mean()),
            "rmse": float(sub["rmse"].mean()),
            "r2": float(sub["r2"].mean()),
            "mae": float(sub["mae"].mean()),
        }

    summary_path = output_dir / "summary_fixed_split.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Done] split metrics: {metrics_path}")
    print(f"[Done] split predictions dir: {split_pred_dir}")
    print(f"[Done] summary: {summary_path}")


if __name__ == "__main__":
    main()

