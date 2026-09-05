# Code Organization

## Design Goal

OmicMAP separates five contracts that should remain independently testable:

1. **Data contract**: file schemas, sample alignment, phenotype names, and CV folds.
2. **Model contract**: a stable registered name plus explicit constructor arguments.
3. **Training contract**: shared optimization, early stopping, and evaluation logic.
4. **Artifact contract**: everything required to reproduce inference from a checkpoint.
5. **Result contract**: one model column per fold/trait prediction table.

This combines GEG2P's strong experiment/result organization with GEFormer's clear
train/predict separation while avoiding platform-specific compiled model modules.

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `data_utils.py` | Read schemas, align IDs, and parse fixed folds |
| `preprocess.py` | Fit feature selection, imputation, and scaling on training data only |
| `model.py` / `mamba_net.py` | Neural architecture only |
| `model_registry.py` | Convert stable model names into model instances |
| `training.py` | Shared fit/evaluate engine for every split protocol |
| `artifacts.py` | Stable paths plus checkpoint save/load |
| `train_cv.py` | Fold orchestration and OOF summaries |
| `train_fixed_split.py` | Explicit validation/test fold orchestration |
| `predict.py` | Phenotype-free inference from a saved checkpoint |

## Checkpoint Contract

A checkpoint is not only a weight file. It contains:

- checkpoint format version;
- registered model name and explicit constructor arguments;
- `state_dict` weights;
- trait name;
- selected feature names and preprocessing statistics for every modality;
- target mean and standard deviation;
- missing-modality fill vectors;
- training, validation, and test sample IDs;
- run, fold, best validation loss, and completed epoch metadata.

Saving this information prevents a common failure mode in research repositories:
published weights that cannot be reused because their feature order or scaling is
unknown.

## Remaining Boundaries

- Baseline and ensemble models can be added through `model_registry.py`; only
  `omicmap` is registered in the current release.
- Hyperparameter search is intentionally not mixed into the final CV reporting
  loop. If added, it should use nested validation or a separate tuning split.
- Raw datasets are not versioned in Git. Dataset hashes and a formal validation
  report should be added when redistribution permissions and the release dataset
  are finalized.
