# Data Format

OmicMAP expects five CSV files under `data/`.

| File | Purpose |
| --- | --- |
| `Rice_geno_zhuanzhi.csv` | Genotype feature matrix |
| `Rice-Expression_zhaunzhi.csv` | Transcriptomics/expression feature matrix |
| `Rice_Metabolites_zhuanzhi.csv` | Metabolomics feature matrix |
| `Rice-Phenotypes.csv` | Phenotype target table |
| `CVFs.csv` | Fixed fold assignment table |

Each file must contain a sample identifier column. The loader accepts common names such as `id`, `sample_id`, or `sampleid`; otherwise, it treats the first column as the sample identifier. Samples are aligned by the intersection of identifiers across all files.

`Rice-Phenotypes.csv` must contain one or more supported target columns:

```text
yd, tp, gn, kgw
```

`CVFs.csv` can use either a direct `sample_id + fold_id` layout or a multi-column fold indicator layout. Fold IDs are expected to represent a fixed 10-fold split.

The clean GitHub release intentionally does not include the original data files. Confirm data redistribution permissions before uploading real data to a public repository.
