"""
PathSnap - Student-Test Route Reconstruction
Member 3 - Phase 2
"""

import json
import time
from pathlib import Path

import pandas as pd

from route_reconstruction.hmm import (
    reconstruct_route,
    evaluate_route_continuity,
)


INPUT_PATH = Path(
    "models/artifacts/results/"
    "m3_handoff_student_test_probabilities.csv.gz"
)

OUTPUT_DIR = Path(
    "experiments/results/route_reconstruction"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading PathSnap candidate probabilities...")

    df = pd.read_csv(INPUT_PATH)

    print("Candidate records:", len(df))
    print("Total trips:", df["trip_id"].nunique())
    print(
        "GPS observations:",
        df[["trip_id", "obs_id"]]
        .drop_duplicates()
        .shape[0],
    )

    all_routes = []
    per_trip_results = []

    start_time = time.perf_counter()

    for trip_id, trip_df in df.groupby("trip_id"):

        print(f"\nProcessing Trip {trip_id}...")

        trip_start = time.perf_counter()

        route = reconstruct_route(
            candidates=trip_df,
            transition_weight=1.0,
        )

        elapsed = time.perf_counter() - trip_start

        metrics = evaluate_route_continuity(route)

        metrics["trip_id"] = int(trip_id)
        metrics["runtime_seconds"] = elapsed

        per_trip_results.append(metrics)
        all_routes.append(route)

        print("Observations:", len(route))

        print(
            "Classifier accuracy:",
            metrics["classifier_accuracy"],
        )

        print(
            "HMM accuracy:",
            metrics["hmm_accuracy"],
        )

        print(
            "Classifier continuity:",
            metrics["classifier_continuity"],
        )

        print(
            "HMM continuity:",
            metrics["hmm_continuity"],
        )

        print(f"Runtime: {elapsed:.2f} seconds")

    # Combine all reconstructed routes

    final_routes = pd.concat(
        all_routes,
        ignore_index=True,
    )

    overall_metrics = evaluate_route_continuity(
        final_routes
    )

    overall_metrics["runtime_seconds"] = (
        time.perf_counter() - start_time
    )

    # Save reconstructed route predictions

    final_routes.to_csv(
        OUTPUT_DIR / "reconstructed_routes.csv",
        index=False,
    )

    # Save individual trip metrics

    pd.DataFrame(per_trip_results).to_csv(
        OUTPUT_DIR / "per_trip_comparison.csv",
        index=False,
    )

    # Save overall evaluation results

    with open(
        OUTPUT_DIR / "overall_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            overall_metrics,
            file,
            indent=4,
        )

    # Display final results

    print("\n========== FINAL HMM RESULTS ==========")

    for key, value in overall_metrics.items():
        print(f"{key}: {value}")

    print("\nResults saved in:", OUTPUT_DIR)

    print("\nRoute reconstruction completed successfully!")


if __name__ == "__main__":
    main()
