from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data_utils import load_aligned_from_config

try:
    import yaml
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Please install pyyaml before checking a dataset.") from exc


def check_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate one configured dataset without starting model training."""

    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    traits = [str(value) for value in cfg.get("traits", [])]
    if not traits:
        raise ValueError(f"No phenotype traits are configured in {path}.")

    aligned = load_aligned_from_config(cfg, traits=traits)
    fold_counts = (
        pd.to_numeric(aligned.fold_map, errors="raise")
        .astype(int)
        .value_counts()
        .sort_index()
    )
    return {
        "config": str(path),
        "data_dir": str(cfg.get("data_dir", "data")),
        "samples": len(aligned.sample_ids),
        "features": {
            "genotype": int(aligned.genotype.shape[1]),
            "expression": int(aligned.expression.shape[1]),
            "metabolites": int(aligned.metabolites.shape[1]),
        },
        "traits": traits,
        "fold_counts": {str(int(key)): int(value) for key, value in fold_counts.items()},
        "missing_cells": {
            "genotype": int(aligned.genotype.isna().sum().sum()),
            "expression": int(aligned.expression.isna().sum().sum()),
            "metabolites": int(aligned.metabolites.isna().sum().sum()),
            "phenotype": int(aligned.phenotype.isna().sum().sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an OmicMAP dataset config")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    report = check_config(args.config)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"Dataset OK: {report['config']}")
    print(f"Samples: {report['samples']}")
    print(f"Features: {report['features']}")
    print(f"Traits ({len(report['traits'])}): {', '.join(report['traits'])}")
    print(f"Fold counts: {report['fold_counts']}")
    print(f"Missing cells: {report['missing_cells']}")


if __name__ == "__main__":
    main()
