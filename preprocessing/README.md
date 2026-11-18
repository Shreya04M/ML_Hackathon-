# Preprocessing integration handoff

This package converts raw PathSnap trip files into candidate-level feature and label tables. It does not train a model or perform route reconstruction.

## Input layout

`PATHSNAP_DATA_ROOT` must point to a dataset directory containing optional `train/` and `student_test/` partitions. Each trip is an immediate child directory:

```text
<dataset-root>/<partition>/<trip-id>/
    <trip-id>.track
    <trip-id>.nodes
    <trip-id>.arcs
    <trip-id>.route       # normally available for train trips
```

The loader preserves the trip directory name as `trip_id`. Files are whitespace separated and have no header.

## Parsed schemas

| File | Columns | Meaning |
|---|---|---|
| `.track` | `longitude`, `latitude`, `timestamp` | Ordered GPS observations |
| `.nodes` | `node_id`, `longitude`, `latitude` | Node coordinates; `node_id` is the zero-based row index |
| `.arcs` | `arc_id`, `from_node_id`, `to_node_id` | Directed road arcs; `arc_id` is the zero-based row index |
| `.route` | `route_position`, `arc_id` | Reference route sequence; `route_position` is the zero-based row index |

`observation_id` is added as the zero-based row index of each track. Cleaning also derives `time_diff_s`, `movement_distance_m`, `speed_mps`, duplicate flags, validity flags, and `clean`.

## Candidate identifiers and schema

Candidates are generated per observation using a local equirectangular projection and a `cKDTree` over road-segment midpoints. The default perpendicular-distance radius is 50 metres (`radius_m`).

Each candidate row contains:

| Column | Meaning |
|---|---|
| `observation_id` | GPS observation owning the candidate |
| `timestamp` | Observation timestamp |
| `gps_longitude`, `gps_latitude` | Original GPS position |
| `candidate_arc_id` | Road arc being considered; primary candidate identifier |
| `candidate_distance_m` | Distance from GPS point to projected arc point |
| `nearest_longitude`, `nearest_latitude` | Projected point on the candidate arc |
| `projection_fraction` | Clamped position on the arc, from 0 to 1 |

The road-network table uses `arc_id` for the same road identity. Candidate records should be identified by `(trip_id, observation_id, candidate_arc_id)`; a candidate arc can recur at multiple observations.

## Feature schema

`extract_features(observations, candidates, road_network)` returns the candidate columns plus:

```text
time_diff_s
movement_distance_m
speed_mps
movement_bearing_deg
arc_id
length_m
bearing_deg
start_lon
start_lat
end_lon
end_lat
bearing_difference_deg
distance_to_start_m
distance_to_end_m
```

The pipeline inserts `trip_id` as the first column. The feature extractor does not use reference-route membership or evaluator labels.

## Label schema and semantics

`generate_labels(candidates, route)` appends:

| Column | Values | Meaning |
|---|---|---|
| `label` | `1` or nullable `Int64` | `1` when `candidate_arc_id` occurs in the reference route; nullable when unresolved |
| `label_status` | `validated_positive` or `unresolved` | Whether the candidate received a validated positive label |

This implementation is conservative: it does **not** assign `0` to non-route candidates. It reports `negative_labels: 0` and leaves non-members unresolved. If a trip has no route, all labels are unresolved.

## Output files

The pipeline writes:

```text
data/processed/train_features.csv
data/processed/train_candidate_labels.csv
data/processed/student_test_features.csv
data/processed/student_test_candidate_labels.csv
```

It also returns cleaning, candidate-coverage, and label summary records. The current runner creates the output and results directories but does not write those returned summaries to a separate report file.

## Entry point

Configure `.env` at the repository root:

```text
PATHSNAP_DATA_ROOT=<dataset-root>
```

Run from the repository root:

```powershell
python -m preprocessing.run_pipeline
```

Useful overrides:

```powershell
python -m preprocessing.run_pipeline `
  --dataset-root <dataset-root> `
  --output-dir data/processed `
  --results-dir experiments/results `
  --radius-m 50
```

Programmatic entry point:

```python
from preprocessing.run_pipeline import run

summary = run(
    dataset_root="<dataset-root>",
    output_dir="data/processed",
    results_dir="experiments/results",
    radius_m=50.0,
)
```

The runner processes `train` and `student_test` independently, skips trips without a track, and records zero candidate coverage for trips missing either `.nodes` or `.arcs`.
