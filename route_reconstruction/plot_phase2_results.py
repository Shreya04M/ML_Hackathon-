"""Generate actual comparison graphs from measured Phase 2 result files."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
OUTPUT_DIR = Path("experiments/results")


def _read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        output_dir / "phase2_overall_metrics.json"
        if (output_dir / "phase2_overall_metrics.json").is_file()
        else output_dir / "overall_metrics.json"
    )
    metrics = _read_json(metrics_path)
    previous_hmm = _read_json(
        ROOT / "experiments/results/optimised_reconstruction/overall_metrics.json"
    )
    recovery = metrics["recovery"]
    previous_constrained = metrics[
        "previous_constrained_recovery_counts"
    ]
    previous_unconstrained = metrics[
        "previous_unconstrained_recovery_counts"
    ]

    observation_labels = [
        "Classifier",
        "Previous optimized HMM",
        "Improved optimized HMM",
    ]
    values = {
        "Observation accuracy": [
            metrics["observation_metrics"]["classifier_accuracy"],
            previous_hmm["hmm_accuracy"],
            metrics["observation_metrics"]["hmm_accuracy"],
        ],
        "Direct continuity": [
            metrics["observation_metrics"]["classifier_continuity"],
            previous_hmm["hmm_continuity"],
            metrics["observation_metrics"]["hmm_continuity"],
        ],
    }

    fig, axis = plt.subplots(figsize=(9, 5))
    x = range(len(observation_labels))
    width = 0.36
    axis.bar(
        [value - width / 2 for value in x],
        values["Observation accuracy"],
        width,
        label="Observation accuracy",
    )
    axis.bar(
        [value + width / 2 for value in x],
        values["Direct continuity"],
        width,
        label="Direct continuity",
    )
    axis.set_xticks(list(x), observation_labels, rotation=10, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction")
    axis.set_title("Observation-level prediction quality")
    axis.legend()
    fig.tight_layout()
    fig.savefig(graphs_dir / "observation_metrics.png", dpi=160)
    plt.close(fig)

    recovery_labels = [
        "Unconstrained baseline",
        "Previous constrained",
        "Improved constrained",
    ]
    recovery_series = {
        "Inserted arcs": [
            previous_unconstrained["inserted_arcs"],
            previous_constrained["inserted_arcs"],
            recovery["inserted_arcs"],
        ],
        "Disconnected / rejected pairs": [
            previous_unconstrained["disconnected_pairs"],
            previous_constrained["rejected_connections"],
            recovery["rejected_connections"],
        ],
        "Route sections": [
            previous_unconstrained["route_sections"],
            previous_constrained["route_sections"],
            recovery["route_sections"],
        ],
    }

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    for axis, (metric_name, series) in zip(axes[:3], recovery_series.items()):
        axis.bar(recovery_labels, series, color=["#7d8ca3", "#d99b42", "#3b8f71"])
        axis.set_title(metric_name)
        axis.tick_params(axis="x", labelrotation=18)
        axis.grid(axis="y", alpha=0.25)
    axes[3].bar(
        ["Previous constrained", "Improved constrained"],
        [
            previous_constrained["implausible_transitions"],
            recovery["implausible_transitions"],
        ],
        color=["#d99b42", "#3b8f71"],
    )
    axes[3].set_title("Physically implausible")
    axes[3].tick_params(axis="x", labelrotation=18)
    axes[3].grid(axis="y", alpha=0.25)
    fig.suptitle("Intermediate road recovery outcomes (10 trips)")
    fig.tight_layout()
    fig.savefig(graphs_dir / "recovery_comparison.png", dpi=160)
    plt.close(fig)

    route_quality = metrics["route_quality_macro_average"]
    constrained_quality = metrics[
        "previous_constrained_route_quality_macro_average"
    ]
    unconstrained_quality = metrics[
        "previous_unconstrained_route_quality_macro_average"
    ]
    quality_labels = [
        "Sequence LCS F1",
        "Road-overlap F1",
    ]
    quality_series = {
        "Unconstrained baseline": [
            unconstrained_quality["sequence_lcs_f1"],
            unconstrained_quality["road_overlap_f1"],
        ],
        "Previous constrained": [
            constrained_quality["sequence_lcs_f1"],
            constrained_quality["road_overlap_f1"],
        ],
        "Improved constrained": [
            route_quality["sequence_lcs_f1"],
            route_quality["road_overlap_f1"],
        ],
    }

    fig, axis = plt.subplots(figsize=(9, 5))
    positions = range(len(quality_labels))
    width = 0.25
    for offset, (label, series) in enumerate(quality_series.items()):
        axis.bar(
            [position + (offset - 1) * width for position in positions],
            series,
            width,
            label=label,
        )
    axis.set_xticks(list(positions), quality_labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro-average fraction")
    axis.set_title("Recovered route correctness against .route references")
    axis.legend()
    fig.tight_layout()
    fig.savefig(graphs_dir / "route_quality.png", dpi=160)
    plt.close(fig)

    old_runtime = previous_hmm["runtime_seconds"]
    runtime_labels = [
        "Previous HMM pipeline",
        "Improved HMM pipeline",
        "Legacy unconstrained recovery",
        "Improved constrained recovery",
    ]
    runtime_values = [
        old_runtime,
        metrics["runtime_seconds"]["hmm_pipeline_total"],
        metrics["runtime_seconds"]["legacy_unconstrained_recovery_total"],
        metrics["runtime_seconds"]["constrained_recovery_total"],
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(runtime_labels, runtime_values, color="#526f91")
    axis.set_ylabel("Seconds")
    axis.set_title("Measured inference and recovery runtime")
    axis.tick_params(axis="x", labelrotation=15)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(graphs_dir / "runtime_comparison.png", dpi=160)
    plt.close(fig)

    pd.DataFrame({
        "metric": runtime_labels,
        "runtime_seconds": runtime_values,
    }).to_csv(output_dir / "phase2_runtime_comparison.csv", index=False)
    print(f"Saved four comparison graphs to {graphs_dir}")


if __name__ == "__main__":
    main()
