# Data

## Partitions & Leakage Separation

| Partition | Path / Env | Trips | Role |
| :--- | :--- | :---: | :--- |
| **`train`** | `Student folder/train/` | `80` | Model fitting, 5-fold `GroupKFold` CV, GP-EI tuning, Optuna HMM optimisation |
| **`student_test`** | `Student folder/student_test/` | `10` | Held-out Phase 1 comparison and Phase 2 HMM/recovery evaluation |
| **`evaluator_test`** | `evaluator_test/` *(git-ignored)* | `10` | Reserved live two-laptop evaluation only (never committed or used for model/parameter tuning) |

## Processed Assets

- [processed/schema_manifest.json](processed/schema_manifest.json): Canonical `20`-feature schema (`FEATURE_COLUMNS`) and partition summary manifest.
- Generated `.parquet` candidate feature caches (`train_k6_features.parquet`, `student_test_k6_features.parquet`) are stored locally in `data/processed/` and git-ignored (`data/processed/*`).
