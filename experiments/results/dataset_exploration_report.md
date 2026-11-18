# Dataset Exploration Report

- **Inspection mode**: Read-only audit across all partitions (`train` = 80 trips, `student_test` = 10 trips, `evaluator_test` = 10 reserved unseen trips).
- **File parsing verification**: All `.track`, `.nodes`, `.arcs`, and `.route` files are headerless, whitespace-separated records. Track row counts (`1 Hz` GPS observations) do **not** match `.route` row counts (traversed road arcs) and are never aligned positionally.

## Partition Inventory & Completeness

| Partition | Trips | `.track` | `.nodes` | `.arcs` | `.route` | Total GPS Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | `80` | `80` | `80` | `80` | `80` | `208,417` |
| **`student_test`** | `10` | `10` | `10` | `10` | `10` | `19,160` |
| **`evaluator_test`** *(git-ignored, server-only)* | `10` | `10` | `10` | `10` | `10` | *(Evaluated strictly live server-side)* |

*(Note: Initial exploration flagged missing `00000000.nodes` and `00000000.arcs` in `student_test`, which were restored via PR #3 so all `360 / 360` train & student-test files are complete.)*

## Empirical Data Quality & Anomaly Audit (`train` & `student_test`)

1. **Missing Values (`NaN`) & Timestamp Ordering**:
   - `0` missing/NaN entries and `0` non-monotonic timestamp steps ($\Delta t \le 0$) across all 90 trips.
   - `96.50%` of `train` steps (`201,053 / 208,337`) and `98.62%` of `student_test` steps (`18,886 / 19,150`) are exactly `1.0 s`; `28 / 80` train trips exhibit GPS gaps `> 30 s` (max `153,064.0 s`), and `2 / 10` test trips exhibit gaps `> 30 s` (max `210.0 s` in trip `00000065`).
2. **Coordinate System & Scale**:
   - Trajectories span `34.7°N` to `61.7°N` latitude; longitude degree spacing shrinks by $\cos(\text{lat})$. Both `preprocessing/spatial_pipeline.py` and `preprocessing/feature_engineering.py` apply local equirectangular tangent-plane projection in meters.
3. **Degenerate Geometry & Stationary GPS**:
   - Audited `100` zero-length arcs in `train` (`6` in `student_test`) and `6` stationary zero-displacement GPS steps in `train`. Protected via `v_len2 > 1e-12` guards and neutral heading assignment when `< 0.25 m/s` or `< 0.5 m`.
4. **Road-Network Graph & Reference `.route` Continuity**:
   - `79 / 80` train networks and `10 / 10` test networks contain multiple clipped bounding-box components (`0` isolated nodes).
   - `70 / 80` train reference routes (`13 / 31,256` transitions) and `9 / 10` test routes (`1 / 3,467` transitions) are 100% connected.
   - Two training trips (`00000033`: `122.4 m` median distance to `.route`; `00000054`: `138.5 m` median distance) contain large parallel GPS offsets, creating `66.7%` and `74.8%` candidate miss rates at `K=6`.
