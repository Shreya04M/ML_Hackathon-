"""Dataset inventory and format exploration for the Member 1 workflow.

The functions in this module are read-only. They inspect files as raw text and
use pandas only for type, shape, missing-value, and sample-record summaries.
"""

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import FILE_TYPES, discover_partitions, discover_trips, files_for_trip


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a whitespace/tab-delimited file with no assumed header."""
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python", dtype=str)


def _summary(path: Path) -> dict[str, Any]:
    frame = _read_raw(path)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return {
        "file": str(path),
        "header_detected": False,
        "columns": int(frame.shape[1]),
        "rows": int(frame.shape[0]),
        "dtypes": [str(dtype) for dtype in numeric.dtypes],
        "missing_values": int(frame.isna().sum().sum()),
        "sample": frame.head(3).to_dict(orient="records"),
        "numeric_min": numeric.min(numeric_only=True).tolist(),
        "numeric_max": numeric.max(numeric_only=True).tolist(),
    }


def explore_dataset(dataset_root: Path) -> dict[str, Any]:
    """Inventory partitions/trips and inspect one representative trip per partition."""
    result: dict[str, Any] = {"dataset_root": str(dataset_root), "exists": dataset_root.is_dir(), "partitions": {}}
    if not dataset_root.is_dir():
        return result
    for partition in discover_partitions(dataset_root):
        trips = discover_trips(partition)
        partition_result: dict[str, Any] = {"trip_count": len(trips), "trip_ids": [p.name for p in trips], "file_counts": Counter(), "representative": None}
        for trip in trips:
            grouped = files_for_trip(trip)
            for kind, paths in grouped.items():
                partition_result["file_counts"][kind] += len(paths)
        if trips:
            representative = trips[0]
            grouped = files_for_trip(representative)
            partition_result["representative"] = {"trip_id": representative.name, "files": {kind: _summary(paths[0]) for kind, paths in grouped.items() if paths}}
        result["partitions"][partition.name] = partition_result
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    """Render a concise exploration report from measured metadata."""
    lines = ["# Dataset exploration report", "", f"- Dataset root: `{summary['dataset_root']}`", f"- Accessible: **{summary['exists']}**", "", "## Partitions"]
    for name, partition in summary.get("partitions", {}).items():
        expected = 80 if name == "train" else 10 if name == "student_test" else "not specified"
        lines += [f"### `{name}`", f"- Trips found: **{partition['trip_count']}** (guideline: {expected})", f"- File counts: `{dict(partition['file_counts'])}`"]
        representative = partition.get("representative")
        if not representative:
            continue
        lines += [f"- Representative trip: `{representative['trip_id']}`", "", "| Type | Rows | Columns | Header | Missing values | Sample |", "|---|---:|---:|---|---:|---|"]
        for kind in FILE_TYPES:
            info = representative["files"].get(kind)
            if info:
                sample = "; ".join(str(row) for row in info["sample"])
                lines.append(f"| `{kind}` | {info['rows']} | {info['columns']} | {info['header_detected']} | {info['missing_values']} | `{sample}` |")
            else:
                lines.append(f"| `{kind}` | — | — | — | — | **missing** |")
        lines += ["", "Initial observations: files were inspected as headerless whitespace/tab-delimited records; IDs were retained as strings. GPS rows and route rows have separate counts and must not be aligned positionally without a later contract. Further quality review is required for coordinate ranges, timestamp semantics, duplicate records, and network connectivity.", ""]
    return "\n".join(lines)
