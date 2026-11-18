# Data

## Partitions & Strict Leakage Separation
| Partition | Path / Env | Trips | Role |
| :--- | :--- | :---: | :--- |
| **`train`** | `Student folder/train/` | `80` | Model fitting, 5-fold `GroupKFold` CV, Bayesian tuning, HMM parameter optimisation |
| **`student_test`** | `Student folder/student_test/` | `10` | Development evaluation, 5-model comparison, Phase 2 HMM ablation |
| **`evaluator_test`** | `evaluator_test/` *(git-ignored)* | `10` | Final unseen two-laptop live evaluation only (never used for training/tuning/selection; never committed to git) |

## Raw & Processed Schemas
- **Raw trip files** (whitespace-separated, no headers): `.track` (`lon, lat, timestamp`), `.nodes` (`lon, lat`), `.arcs` (`from_node_id, to_node_id`), `.route` (`arc_id` sequence).
- **Processed manifest**: See `data/processed/schema_manifest.json` (`20` leakage-free features in `FEATURE_COLUMNS` + `8` metadata columns + `label`).
- Derived parquet caches and raw downloads stay git-ignored (`data/raw/*`, `data/processed/*`).
