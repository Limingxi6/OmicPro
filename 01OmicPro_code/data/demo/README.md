# Public synthetic demo data

These CSV files are deterministic synthetic values created only to exercise the
OmicPro data contract and training pipeline. They are not real omics data and
must not be used for biological evaluation, benchmarking, or model claims.

Run from the repository root after installing the package:

```bash
omicpro-check-data --config 01OmicPro_code/configs/demo.yaml
omicpro-train --config 01OmicPro_code/configs/demo.yaml --trait_name demo_trait --folds 1
```

The run trains one epoch on the first of three fixed folds and writes outputs to
`models/demo/` and `results/demo/`.

Training uses the same `mamba-ssm` dependency as the full OmicPro model. Create
the documented Conda environment before running the training command; the data
validation command does not construct a model.
