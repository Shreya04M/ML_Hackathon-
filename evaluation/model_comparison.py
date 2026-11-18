
"""Model comparison and experiment recording for PathSnap."""

from typing import Any
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd


def build_model_comparison(results: Any) -> pd.DataFrame:
    """Build a comparison table from measured CV results."""

    rows = []

    for result in results:
        row = {
            "model": result["model"],
            "variant": result.get("variant", "untuned"),
            "n_splits": result["n_splits"],
        }

        row.update(result["summary"])
        rows.append(row)

    comparison = pd.DataFrame(rows)

    if comparison.empty:
        return comparison

    comparison = comparison.sort_values(
        by=["mean_macro_f1", "mean_accuracy"],
        ascending=False
    ).reset_index(drop=True)

    comparison.insert(0, "rank", range(1, len(comparison) + 1))

    return comparison


def save_comparison(
    comparison: pd.DataFrame,
    filename: str = "model_comparison.csv"
) -> Path:
    """Save the comparison table as a CSV file."""

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    comparison.to_csv(output_path, index=False)

    return output_path


def save_experiment(
    result: dict,
    experiment_name: str
) -> Path:
    """Save complete experiment results as a JSON record."""

    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)

    record = {
        "experiment_name": experiment_name,
        "timestamp": timestamp.isoformat(),
        "results": result,
    }

    filename = (
        f"{experiment_name}_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
    )

    output_path = log_dir / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=4)

    return output_path