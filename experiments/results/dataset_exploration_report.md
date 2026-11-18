# Dataset exploration report

- Dataset root: `C:\Users\SHREYA\Desktop\SEM 5\ML\ML_Hackathon-\Student folder`
- Dataset directory: detected and accessible
- Inspection mode: read-only; raw files were not modified, moved, renamed, or copied.
- File parsing observation: representative files are whitespace/tab-delimited and headerless.

## Partition inventory

| Partition | Trips found | Guideline | `.track` | `.nodes` | `.arcs` | `.route` |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 80 | 80 | 80 | 80 | 80 | 80 |
| `student_test` | 10 | 10 | 10 | 9 | 9 | 10 |

The student-testing partition has one trip without `.nodes` and `.arcs` (trip `00000000`). This discrepancy needs attention before any network-based processing. No evaluator data outside the supplied dataset was used.

## Representative training trip

Trip `00000001`:

| File | Rows | Columns | Header | First record |
|---|---:|---:|---|---|
| `.track` | 1,315 | 3 | No | `8.595203 49.844053 1.000000` |
| `.nodes` | 88,809 | 2 | No | `8.581679 49.857880` |
| `.arcs` | 165,790 | 2 | No | `0 281` |
| `.route` | 525 | 1 | No | `2641` |

The observed columns are consistent with longitude/latitude/timestamp, node coordinates, arc endpoint IDs, and ordered route arc IDs. IDs are preserved as raw strings by the exploration code. Track-row count and route-row count differ, so they are not treated as positionally aligned.

## Initial data-quality concerns

- Confirm the intended semantics and units of the third `.track` field; it appears numeric and sequential in the sample.
- Validate coordinate ranges, timestamp monotonicity, duplicate rows, and missing values during the later cleaning stage.
- Resolve whether the student-testing trip missing network files is intentionally evaluator-only or incomplete input.
- Define how route arc IDs map to `.arcs` rows before labels or features are prepared.
