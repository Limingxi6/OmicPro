# OmicPro model code

This directory is the runnable and installable OmicPro package. It separates data loading, preprocessing, model definition, training orchestration, checkpoints, and inference so each layer can be tested independently.

```text
01OmicPro_code/
|-- omicpro/        # importable package and command implementations
|-- configs/        # named experimental settings
|-- scripts/        # backwards-compatible, thin Python entry points
|-- tests/          # fast structural and import checks
|-- docs/           # data, method, and implementation documentation
|-- data/           # local/demo data mount; contents are ignored by Git
|-- environment.yml
|-- requirements.txt
`-- pyproject.toml
```

## Installation

From the repository root:

```bash
conda env create -f 01OmicPro_code/environment.yml
conda activate omicpro
pip install --no-deps -e 01OmicPro_code
```

Set `OMICPRO_DATA_DIR` to the directory containing your private dataset folders. The supplied Rice, Maize, and Rapeseed configs resolve their `data_dir` from that variable, rather than from an author-specific drive.

```powershell
$env:OMICPRO_DATA_DIR = "D:/your-private-omicpro-data"
```

## Main commands

Run the public synthetic demo without downloading private data:

```bash
omicpro-check-data --config 01OmicPro_code/configs/demo.yaml
omicpro-train --config 01OmicPro_code/configs/demo.yaml --trait_name demo_trait --folds 1
```

Use the following commands for a configured research dataset:

```bash
# Check file schemas, IDs, traits, and fold assignments without training.
omicpro-check-data --config 01OmicPro_code/configs/rice210_complete.yaml

# Cross-validation for a single trait.
omicpro-train --config 01OmicPro_code/configs/rice210_complete.yaml --trait_name yd

# Fixed validation/test-fold training.
omicpro-train-fixed --config 01OmicPro_code/configs/rice210_prompt_missing.yaml --trait_name yd --valid_fold 1 --test_fold 2

# Prediction needs only the three omics input matrices and a checkpoint.
omicpro-predict --checkpoint models/rice210_complete/k1/yd/omicpro.pt --data_dir D:/new-data --output results/new_data/yd_predictions.csv
```

The scripts under `scripts/` remain available for users who call Python files directly; the installed `omicpro-*` commands are the supported interface.

Read [DATA.md](docs/DATA.md) for CSV requirements, [ENVIRONMENT.md](docs/ENVIRONMENT.md) for the server3-compatible CUDA environment, [CODE_ORGANIZATION.md](docs/CODE_ORGANIZATION.md) for module boundaries, and [METHOD.md](docs/METHOD.md) for the model description.
