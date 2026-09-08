# Predict from a released checkpoint

Prediction requires a saved `*.pt` checkpoint plus genotype, expression, and metabolite CSVs. Each CSV needs a unique sample-ID column (`ID`, `sample_id`, or its first column); the three matrices are aligned by their shared IDs.

```bash
omicpro-predict --checkpoint models/<run>/k<fold>/<trait>/omicpro.pt --data_dir D:/new-omics-data --output results/<run>/predictions.csv
```

Use `--geno_file`, `--expr_file`, and `--metab_file` if your filenames differ from the defaults. To declare missing modalities per sample, provide `--missing_table` with `ID` and `missing_code` columns. The code is a bitmask: genotype = 1, expression = 2, metabolites = 4; for example, 3 means genotype and expression are both missing.

The checkpoint contains the model constructor arguments, selected feature order, preprocessing statistics, target scaling, and missing-value fill vectors. Do not preprocess new data separately before calling this command.
