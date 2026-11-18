"""
PathSnap - Full Route Recovery Experiment
Member 3 - Phase 2
"""

import json
from pathlib import Path

import pandas as pd

from route_reconstruction.road_graph import load_road_graph

from route_reconstruction.route_recovery import (
    recover_continuous_route,
)


INPUT_PATH = Path(
    "experiments/results/optimised_reconstruction/"
    "reconstructed_routes.csv"
)

OUTPUT_DIR = Path(
    "experiments/results/optimised_route_recovery"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_PATH)

    all_routes = {}
    summaries = []

    for trip_id, trip in df.groupby("trip_id"):

        print(f"\nRecovering Trip {trip_id}...")

        trip = trip.sort_values("obs_id")

        graph = load_road_graph(trip_id)

        predicted_arcs = (
            trip["hmm_arc_id"]
            .astype(int)
            .tolist()
        )

        result = recover_continuous_route(
            predicted_arcs,
            graph,
        )

        # Verify connectivity inside each section

        for section in result["route_sections"]:

            for first, second in zip(
                section[:-1], section[1:]
            ):

                if (
                    graph.to_nodes[first]
                    != graph.from_nodes[second]
                ):
                    raise ValueError(
                        "Invalid recovered connection"
                    )

        all_routes[str(trip_id)] = result

        summary = {
            "trip_id": int(trip_id),
            "original_predictions":
                result["original_predictions"],
            "recovered_arc_count":
                result["recovered_arc_count"],
            "inserted_arcs":
                result["inserted_arcs"],
            "repeated_arcs_removed":
                result["repeated_arcs_removed"],
            "disconnected_pairs":
                len(result["disconnected_pairs"]),
            "route_sections":
                len(result["route_sections"]),
            "is_continuous":
                result["is_continuous"],
        }

        summaries.append(summary)

        for key, value in summary.items():
            print(f"{key}: {value}")

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUTPUT_DIR / "route_recovery_summary.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "recovered_routes.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_routes,
            file,
            indent=4,
        )

    print("\n===== FINAL ROUTE RECOVERY RESULTS =====")

    print("Total trips:", len(summary_df))

    print(
        "Fully connected trips:",
        summary_df["is_continuous"].sum(),
    )

    print(
        "Total intermediate arcs inserted:",
        summary_df["inserted_arcs"].sum(),
    )

    print(
        "Total repeated arcs removed:",
        summary_df["repeated_arcs_removed"].sum(),
    )

    print(
        "Remaining disconnected pairs:",
        summary_df["disconnected_pairs"].sum(),
    )

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
