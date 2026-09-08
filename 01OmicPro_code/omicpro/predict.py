from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import load_checkpoint
from .data_utils import load_prediction_modalities


def _choose_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda") and torch.cuda.is_available():
        return torch.device(value)
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu")


def _load_missing_codes(
    sample_ids: list[str],
    missing_table: str | None,
    default_code: int,
) -> np.ndarray:
    if not 0 <= int(default_code) <= 7:
        raise ValueError("missing_code must be between 0 and 7.")
    if missing_table is None:
        return np.full(len(sample_ids), int(default_code), dtype=np.int64)

    table = pd.read_csv(missing_table)
    if "ID" in table.columns:
        id_col = "ID"
    elif "sample_id" in table.columns:
        id_col = "sample_id"
    else:
        id_col = str(table.columns[0])
    if "missing_code" not in table.columns:
        raise ValueError("Missing-code table must contain a 'missing_code' column.")
    table[id_col] = table[id_col].astype(str).str.strip()
    if table[id_col].duplicated().any():
        raise ValueError("Missing-code table contains duplicate sample IDs.")
    code_map = pd.to_numeric(table.set_index(id_col)["missing_code"], errors="raise").astype(int)
    missing_ids = [sid for sid in sample_ids if sid not in code_map.index]
    if missing_ids:
        raise ValueError(f"Missing-code table has no row for {len(missing_ids)} prediction samples.")
    codes = code_map.loc[sample_ids].to_numpy(dtype=np.int64)
    if np.any((codes < 0) | (codes > 7)):
        raise ValueError("Every missing_code must be between 0 and 7.")
    return codes


def predict_from_checkpoint(
    *,
    checkpoint: str | Path,
    data_dir: str | Path,
    output: str | Path,
    device: str = "auto",
    batch_size: int = 64,
    missing_code: int = 0,
    missing_table: str | None = None,
    geno_file: str = "Rice_geno_zhuanzhi.csv",
    expr_file: str = "Rice-Expression_zhaunzhi.csv",
    metab_file: str = "Rice_Metabolites_zhuanzhi.csv",
) -> Path:
    runtime_device = _choose_device(device)
    model, preprocessor, payload = load_checkpoint(checkpoint, device=runtime_device)
    aligned = load_prediction_modalities(
        data_dir,
        geno_file=geno_file,
        expr_file=expr_file,
        metab_file=metab_file,
    )
    sample_ids = aligned.sample_ids
    codes = _load_missing_codes(sample_ids, missing_table, missing_code)

    geno = preprocessor.transform_genotype(aligned.genotype).to_numpy(dtype=np.float32)
    expr = preprocessor.transform_expression(aligned.expression).to_numpy(dtype=np.float32)
    metab = preprocessor.transform_metabolites(aligned.metabolites).to_numpy(dtype=np.float32)

    fill = payload["fill_vectors"]
    geno[(codes & 1) > 0, :] = np.asarray(fill["geno"], dtype=np.float32)
    expr[(codes & 2) > 0, :] = np.asarray(fill["expr"], dtype=np.float32)
    metab[(codes & 4) > 0, :] = np.asarray(fill["metab"], dtype=np.float32)

    dataset = TensorDataset(
        torch.tensor(geno, dtype=torch.float32).unsqueeze(1),
        torch.tensor(expr, dtype=torch.float32),
        torch.tensor(metab, dtype=torch.float32),
        torch.tensor(codes, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_geno, batch_expr, batch_metab, batch_codes in loader:
            output_std = model(
                batch_geno.to(runtime_device),
                batch_expr.to(runtime_device),
                batch_metab.to(runtime_device),
                missing_code=batch_codes.to(runtime_device),
            )
            predictions.append(output_std.cpu().numpy())

    pred_std = np.concatenate(predictions, axis=0)
    standardizer = payload["target_standardizer"]
    target_mean = np.asarray(standardizer["mean"], dtype=np.float64)
    target_std = np.asarray(standardizer["std"], dtype=np.float64)
    pred = pred_std * target_std + target_mean
    model_name = str(payload["model_name"])

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ID": sample_ids,
            model_name: pred[:, 0],
            "trait": str(payload["trait"]),
            "missing_code": codes,
        }
    ).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict phenotypes from a saved OmicPro checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pt checkpoint")
    parser.add_argument("--data_dir", default="data", help="Directory containing the three omics CSV files")
    parser.add_argument("--output", required=True, help="Prediction CSV path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--missing_code", type=int, default=0, help="One bitmask code (0-7) for every sample")
    parser.add_argument("--missing_table", default=None, help="Optional CSV with ID and missing_code columns")
    parser.add_argument("--geno_file", default="Rice_geno_zhuanzhi.csv")
    parser.add_argument("--expr_file", default="Rice-Expression_zhaunzhi.csv")
    parser.add_argument("--metab_file", default="Rice_Metabolites_zhuanzhi.csv")
    args = parser.parse_args()

    output_path = predict_from_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        missing_code=args.missing_code,
        missing_table=args.missing_table,
        geno_file=args.geno_file,
        expr_file=args.expr_file,
        metab_file=args.metab_file,
    )
    print(f"[Done] predictions: {output_path}")


if __name__ == "__main__":
    main()
