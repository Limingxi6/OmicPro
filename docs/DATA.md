# Data Format

OmicMAP expects five logical CSV inputs. Their physical names and subdirectories
are declared in each YAML file under `data`.

| Config key | Purpose |
| --- | --- |
| `data.genotype` | Genotype feature matrix |
| `data.expression` | Transcriptomics/expression feature matrix |
| `data.metabolites` | Metabolomics or ASV feature matrix |
| `data.phenotype` | Phenotype target table |
| `data.folds` | Fixed fold assignment table |

Each file must contain a sample identifier column. The loader accepts common names such as `id`, `sample_id`, or `sampleid`; otherwise, it treats the first column as the sample identifier. Samples are aligned by the intersection of identifiers across all files.

The phenotype table must contain every column listed by `traits`. For example,
Rice210 uses:

```text
yd, tp, gn, kgw
```

The fold table may use a direct `sample_id + fold_id` layout or a true one-hot
layout. For repeated CV, where each column contains a complete set of fold IDs,
set `data.fold_column` explicitly. OmicMAP rejects ambiguous multi-column files
instead of guessing.

The current external datasets were checked without copying them into the
repository:

| Dataset | Genotype features | Expression features | Third-omics features | Shared samples |
| --- | ---: | ---: | ---: | ---: |
| Rice210 | 1,619 | 24,994 | 1,000 metabolites | 210 |
| Maize368 | 16,383 | 16,383 | 748 metabolites | 333 |
| Rapeseed KF | 50,000 | 17,006 | 203 ASVs | 175 |
| Rapeseed YL | 50,000 | 17,006 | 203 ASVs | 175 |

All five inputs within each configured dataset have a complete sample-ID
intersection and no duplicate IDs. The two trailing columns in the maize
phenotype file are completely empty and are intentionally excluded from
`traits`.

The clean GitHub release intentionally does not include the original data files. Confirm data redistribution permissions before uploading real data to a public repository.
