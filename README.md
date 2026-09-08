# OmicPro: Multi-Omics Prediction Based on Prompt Learning with Incomplete Data

The repository is arranged around three distinct research workflows so model users do not need to navigate figure-production or application examples.

```text
.
|-- 01OmicPro_code/          # installable model package, configs, tests, and data contract
|-- 02Figure_code/           # manuscript figure source assets and reproduction notes
|-- 03OmicPro_application/   # checkpoint inference and custom-data recipes
|-- models/                  # local checkpoints; never committed
`-- results/                 # local predictions and metrics; never committed
```

## Start here

1. Read the model package guide in [01OmicPro_code](01OmicPro_code/README.md).
2. Create the environment and install the package from the repository root:

   ```bash
   conda env create -f 01OmicPro_code/environment.yml
   conda activate omicpro
   pip install --no-deps -e 01OmicPro_code
   ```

3. Run the included synthetic demo immediately, or follow [the data contract](01OmicPro_code/docs/DATA.md) and set the private data root once:

   ```bash
   omicpro-check-data --config 01OmicPro_code/configs/demo.yaml
   omicpro-train --config 01OmicPro_code/configs/demo.yaml --trait_name demo_trait --folds 1
   ```

   For the supplied research datasets, set the data root:

   ```powershell
   $env:OMICPRO_DATA_DIR = "D:/your-private-omicpro-data"
   ```

4. Validate a configured research dataset, then run a small cross-validation check:

   ```bash
   omicpro-check-data --config 01OmicPro_code/configs/rice210_complete.yaml
   omicpro-train --config 01OmicPro_code/configs/quick_test.yaml --trait_name yd
   ```

The data, trained weights, and generated results are intentionally excluded from Git. See [the application recipes](03OmicPro_application/README.md) for prediction on new samples and [the figure guide](02Figure_code/README.md) for manuscript assets.

The core environment is pinned to the tested Linux/CUDA 12.1 server baseline;
see [environment instructions](01OmicPro_code/docs/ENVIRONMENT.md) before
running GPU training.

## Repository conventions

- A YAML config is the source of truth for every experiment.
- `models/<run>/k<fold>/<trait>/` stores reusable checkpoints.
- `results/<run>/` stores fold predictions, metrics, resolved configuration, and diagnostic logs.
- Original datasets and checkpoint weights must be released only after their redistribution permissions are confirmed.

## License

No open-source license has been selected yet. Add the intended license before a public data, model, or code release.
