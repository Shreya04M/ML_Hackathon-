"""
PathSnap - Distance-Based HMM Experiment
Member 3 - Phase 2
"""

import json
import time
from pathlib import Path

import pandas as pd

from route_reconstruction.road_graph import load_road_graph
from route_reconstruction.distance_transition import load_gps_track

from route_reconstruction.hmm import (
    reconstruct_route,
    evaluate_route_continuity,
)


INPUT_PATH = Path(
    "models/artifacts/results/"
    "m3_handoff_student_test_probabilities.csv.gz"
)

OUTPUT_DIR = Path(
    "experiments/results/distance_reconstruction"
)

BETA = 30.0
TRANSITION_WEIGHT = 1.0


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_PATH)

    print("Candidate records:", len(df))
    print("Total trips:", df.trip_id.nunique())

    all_routes = []
    per_trip_results = []

    start_time = time.perf_counter()

    for trip_id, trip_df in df.groupby("trip_id"):

        print(
            f"\nProcessing Trip {trip_id}...",
            flush=True,
        )

        trip_start = time.perf_counter()

        graph = load_road_graph(trip_id)
        track = load_gps_track(trip_id)

        route = reconstruct_route(
            candidates=trip_df,
            road_network=graph,
            gps_track=track,
            transition_mode="distance",
            transition_weight=TRANSITION_WEIGHT,
            beta=BETA,
        )

        metrics = evaluate_route_continuity(route)

        metrics["trip_id"] = int(trip_id)

        metrics["runtime_seconds"] = (
            time.perf_counter() - trip_start
        )

        all_routes.append(route)
        per_trip_results.append(metrics)

        print("Observations:", len(route))

        print(
            "Classifier accuracy:",
            metrics["classifier_accuracy"],
        )

        print(
            "Distance HMM accuracy:",
            metrics["hmm_accuracy"],
        )

        print(
            "Classifier continuity:",
            metrics["classifier_continuity"],
        )

        print(
            "Distance HMM continuity:",
            metrics["hmm_continuity"],
        )

        print(
            "Runtime:",
            round(metrics["runtime_seconds"], 2),
            "seconds",
        )

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

    overall_metrics["beta"] = BETA

    overall_metrics["transition_weight"] = (
        TRANSITION_WEIGHT
    )

    final_routes.to_csv(
        OUTPUT_DIR / "reconstructed_routes.csv",
        index=False,
    )

    pd.DataFrame(per_trip_results).to_csv(
        OUTPUT_DIR / "per_trip_comparison.csv",
        index=False,
    )

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

    print("\n===== DISTANCE HMM RESULTS =====")

    for key, value in overall_metrics.items():
        print(f"{key}: {value}")

    print("\nResults saved in:", OUTPUT_DIR)

    print("Distance reconstruction completed!")


if __name__ == "__main__":
    main()
