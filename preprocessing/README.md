# Data Loading & Spatial Preprocessing

Leakage-free raw data ingestion, cleaning diagnostics, local tangent-plane metric projection, adaptive `cKDTree` candidate generation, and 20-feature extraction.

## Modules

| File | Role |
| :--- | :--- |
| [spatial_pipeline.py](spatial_pipeline.py) | **Canonical production pipeline** (`extract_trip_candidates_and_features`, `build_partition_dataset`): local-meter projection, `K=6` adaptive `cKDTree` candidates (`99.32%` `student_test` coverage), 20 leakage-free features (`FEATURE_COLUMNS`), and `route=None` isolation for inference |
| [data_loader.py](data_loader.py) | Raw `.track`, `.nodes`, `.arcs`, and `.route` parser preserving zero-based indices and trip IDs |
| [data_cleaning.py](data_cleaning.py) | GPS step velocity, duplicate observation, gap flagging, and road network geometry checks |
| [road_network.py](road_network.py) | Directed arc table with segment lengths and bearings |
| [candidate_generation.py](candidate_generation.py) | Radius-based equirectangular `cKDTree` candidate generator |
| [feature_engineering.py](feature_engineering.py) | Kinematic and arc-geometry feature computation (`route`-free) |
| [label_generation.py](label_generation.py) | Candidate reference-route label assignment for offline evaluation |
| [data_exploration.py](data_exploration.py) | Read-only partition inventory and anomaly auditing |
| [run_pipeline.py](run_pipeline.py) | Standalone CLI preprocessing runner writing CSV tables to `data/processed/` |

## Production Verification

```powershell
python -m pytest -v tests/test_data_loader.py tests/test_preprocessing.py
```
