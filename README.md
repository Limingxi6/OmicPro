# OmicMAP

OmicMAP is a missing-aware prompt-guided multi-omics phenotype prediction framework. It models genomics, transcriptomics, and metabolomics inputs with modality-specific encoders, keeps a fixed token position for each omics modality, and uses learnable prompts to compensate missing modalities in latent space before cross-modal fusion.

![OmicMAP architecture](docs/figures/omicmap_architecture.svg)

## Key Ideas

- Fixed modality tokens for genomics (G), transcriptomics (T), and metabolomics (M).
- Zero-filled placeholders for missing modalities instead of deleting missing-token positions.
- Missing-aware prompts, including staged, common, dynamic, and deeper correlation prompts.
- Prompt-gated residual updates that strengthen missing-token compensation while limiting perturbation of observed modalities.
- Summary residual attention for compact cross-modal context fusion.
- Auxiliary single-modality losses during training to regularize modality encoders.

## Repository Layout

```text
.
|-- omicmap/                    # Importable package
|   |-- model.py               # OmicMAP architecture
|   |-- model_registry.py      # Stable model construction interface
|   |-- data_utils.py          # Data schema, alignment, and fold parsing
|   |-- preprocess.py          # Fold-fitted preprocessing
|   |-- training.py            # Shared training/evaluation engine
|   |-- artifacts.py           # Checkpoints and experiment layout
|   |-- train_cv.py            # Cross-validation pipeline
|   |-- train_fixed_split.py   # Explicit validation/test split pipeline
|   `-- predict.py             # Checkpoint-based inference
|-- configs/                    # Named, reproducible experiment configs
|-- scripts/                    # Thin compatibility wrappers
|-- docs/                       # Method/data documentation and figures
|-- data/                       # Optional local data mount; real data are not committed
|-- tests/                      # Lightweight import checks
|-- models/                     # Generated checkpoints (not committed)
|-- results/                    # Generated predictions, metrics, and logs (not committed)
|-- METHOD.md                   # Manuscript-style method description
`-- README.md
```

This follows the same useful separation seen in GEG2P and GEFormer: data contracts,
model implementation, training orchestration, prediction entry points, and stable
artifact paths are independent layers.

## Installation

Python 3.10+ is recommended. GPU training requires a PyTorch/CUDA environment compatible with `mamba-ssm`.

Conda users can reproduce the named environment directly:

```bash
conda env create -f environment.yml
conda activate omicmap
```

Alternatively, create a standard virtual environment:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Data

Real data stay outside the repository. Each experiment config declares its data
directory, logical file mapping, phenotype columns, and CV assignment column.
The current local configs use `F:/data/prompt`:

| Config | Dataset | Samples | Targets |
| --- | --- | ---: | ---: |
| `configs/rice210_complete.yaml` | Rice210 | 210 | 4 |
| `configs/maize368_complete.yaml` | Maize368 | 333 | 20 |
| `configs/rapeseed_kf_complete.yaml` | Rapeseed KF | 175 | 13 |
| `configs/rapeseed_yl_complete.yaml` | Rapeseed YL | 175 | 13 |

The rapeseed CV table contains repeated assignments (`cv_1` through `cv_10`),
so the selected repetition is explicit in `data.fold_column`. See
`docs/DATA.md` for the complete data contract. Check redistribution permissions
before publishing any original data files.

## Quick Check

After adding data files, run a one-epoch smoke test:

```bash
python -m omicmap.train_cv --config configs/quick_test.yaml --trait_name yd
```

Outputs are written to `results/quick_test/`, and reusable checkpoints are written
to `models/quick_test/`.

## Training

Complete-modality training for one target:

```bash
python -m omicmap.train_cv --config configs/rice210_complete.yaml --trait_name yd
```

Prompt-enabled missing-modality training for one target:

```bash
python -m omicmap.train_cv --config configs/rice210_prompt_missing.yaml --trait_name yd
```

Run only selected folds while debugging:

```bash
python -m omicmap.train_cv \
  --config configs/quick_test.yaml \
  --trait_name yd \
  --folds 1,2
```

The phenotype targets are read from `traits` in the selected config; they are no
longer hard-coded to the four rice traits.

Examples for the other crops:

```bash
python -m omicmap.train_cv --config configs/maize368_complete.yaml --trait_name Plantheight
python -m omicmap.train_cv --config configs/rapeseed_kf_complete.yaml --trait_name Fe
python -m omicmap.train_cv --config configs/rapeseed_yl_complete.yaml --trait_name Fe
```

## Explicit Fixed-Split Training and Evaluation

```bash
python -m omicmap.train_fixed_split \
  --config configs/rice210_prompt_missing.yaml \
  --trait_name yd \
  --valid_fold 1 \
  --test_fold 2
```

`omicmap.evaluate_fixed_split` remains available as a compatibility alias, but the
operation trains a model before evaluating it, so `train_fixed_split` is the
accurate command name.

## Prediction

Every checkpoint contains the model weights, constructor arguments, selected
feature order, preprocessing statistics, target standardization, missing-value
fill vectors, and training split IDs. Predict without phenotype or CV files:

```bash
python -m omicmap.predict \
  --checkpoint models/rice210_complete/k1/yd/omicmap.pt \
  --data_dir F:/data/prompt/Rice210 \
  --output results/rice210_complete/predictions/yd_k1.csv
```

For per-sample missing modalities, provide a CSV containing `ID` and
`missing_code` columns with bitmask values from 0 to 7.

## Artifact Layout

```text
models/<run_name>/
`-- k<fold>/<trait>/<model>.pt

results/<run_name>/
|-- k<fold>/<trait>.csv          # ID, model prediction, truth, missingness metadata
|-- logs/prompt_gate/            # Epoch-level diagnostics
`-- summary/                     # Metrics, OOF predictions, checkpoints, resolved config
```

The fold prediction table is deliberately model-column based so it can later be
consumed by a GEG2P-style ensemble without changing the training code.

## Method

The full method write-up is in `METHOD.md`. Data and code-organization notes are
provided in `docs/DATA.md` and `docs/CODE_ORGANIZATION.md`.

## License

No open-source license has been selected yet. Add the intended license before making the repository public or reusable.
