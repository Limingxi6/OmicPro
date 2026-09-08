# Prepare a custom training dataset

Create one folder per dataset and point a copied YAML config at it. The config must name five CSV files:

```text
my_dataset/
|-- genotype.csv
|-- expression.csv
|-- metabolites.csv
|-- phenotype.csv
`-- folds.csv
```

All five files require a sample-ID column. `phenotype.csv` must include each trait named in `traits`; `folds.csv` must contain either one fold-ID column or a true one-hot fold assignment. For repeated CV columns, set `data.fold_column` explicitly.

Before training, always run:

```bash
omicpro-check-data --config path/to/my_experiment.yaml
```

The full, validation-oriented schema is documented in [`01OmicPro_code/docs/DATA.md`](../../01OmicPro_code/docs/DATA.md).
