# PATHSNAP Phase I — Final Data Engineering Report

## Dataset overview

The supplied dataset contains 80 training trips and 10 student-test trips. Each trip uses headerless whitespace-separated `.track`, `.nodes`, `.arcs`, and `.route` files. Original trip IDs, observation order, node IDs, and arc IDs were preserved. Student-test trip `00000000` was excluded from network-dependent processing because its `.nodes` and `.arcs` files are absent.

## Processing methodology

GPS records were retained and annotated with duplicate, coordinate, timestamp, gap, movement, speed, and cleaning-status fields. Network records were loaded as a directed graph representation; segment lengths use a local geographic projection approximation in metres, and bearings are retained. Candidates were found using a 50-metre configurable search radius, a spatial KD-tree prefilter, and exact point-to-segment projection. Features include candidate distance, nearest point, projection fraction, GPS time/movement/speed/direction, segment length/bearing, bearing difference, and distances to segment endpoints. Reference routes were never aligned to GPS rows by index. A candidate received label `1` only when its arc ID appeared in the ordered reference route; unsupported candidates remain explicitly `unresolved` with missing label values.

## Generated files and schemas

| File | Rows | Columns |
|---|---:|---:|
| `data/processed/train_features.csv` | 1,784,403 | 24 |
| `data/processed/student_test_features.csv` | 187,571 | 24 |
| `data/processed/train_candidate_labels.csv` | 1,784,403 | 12 |
| `data/processed/student_test_candidate_labels.csv` | 187,571 | 12 |

Feature columns: `trip_id`, `observation_id`, `timestamp`, GPS coordinates, candidate arc ID, candidate distance, nearest-point coordinates, projection fraction, trajectory features, and road geometry features. Label columns preserve the candidate join identifiers and add `label` and `label_status`.

## Coverage and labels

Coverage denominator is the number of candidate records for generated candidate datasets. Training produced 1,784,403 candidate records, of which 541,437 were validated positives and 1,242,966 were unresolved. Student-test produced 187,571 candidate records, of which 68,765 were validated positives and 118,806 were unresolved. No unsupported negative labels were fabricated. Trip `00000000` contributed no candidates because its network files are missing.

## Known limitations

- Labels are conservative route-membership references, not full monotonic sequence-alignment labels.
- Unresolved labels must be handled explicitly by downstream model-training code.
- The 50-metre radius is configurable and should be tuned using training-only diagnostics.
- Student-test labels are present only where reference route files and network data permitted candidate generation; they must not be used for model selection.

## Reproduction

```powershell
python -m preprocessing.run_pipeline --dataset-root ".\Student folder" --output-dir ".\data\processed" --radius-m 50
```
