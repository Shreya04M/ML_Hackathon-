"""
PathSnap - HMM Validation Cache
Member 3 - Phase 2
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from route_reconstruction.road_graph import (
    load_road_graph,
    haversine_distance,
)

from route_reconstruction.distance_transition import (
    load_gps_track,
    candidate_road_distance,
)

from route_reconstruction.emission_probability import (
    calculate_emission_probabilities,
)


INPUT_PATH = Path(
    "experiments/results/hmm_validation/"
    "training_oof_probabilities.csv.gz"
)

CACHE_DIR = Path(
    "experiments/results/hmm_validation/cache"
)

TOP_K = 6


def prepare_trip(trip_id, candidates):

    graph = load_road_graph(
        trip_id,
        partition="train",
    )

    track = load_gps_track(
        trip_id,
        partition="train",
    )

    # Remove duplicate candidate roads

    candidates = (
        candidates.sort_values(
            "pred_prob",
            ascending=False,
            kind="stable",
        )
        .drop_duplicates(
            ["trip_id", "obs_id", "candidate_arc_id"]
        )
    )

    df = calculate_emission_probabilities(
        candidates=candidates
    )

    df = df.sort_values(
        ["obs_id", "candidate_arc_id"]
    )

    groups = list(df.groupby("obs_id"))

    n = len(groups)

    arc_ids = np.full(
        (n, TOP_K), -1, dtype=np.int64
    )

    from_nodes = np.full(
        (n, TOP_K), -1, dtype=np.int64
    )

    to_nodes = np.full(
        (n, TOP_K), -1, dtype=np.int64
    )

    emissions = np.full(
        (n, TOP_K), -np.inf, dtype=float
    )

    distance_errors = np.full(
        (max(n - 1, 0), TOP_K, TOP_K),
        np.inf,
        dtype=float,
    )

    references = np.full(
        n, -1, dtype=np.int64
    )

    observation_ids = np.zeros(
        n, dtype=np.int64
    )

    candidate_counts = np.zeros(
        n, dtype=np.int64
    )

    for i, (obs_id, group) in enumerate(groups):

        obs_id = int(obs_id)
        k = len(group)

        if k > TOP_K:
            raise ValueError("Too many candidates")

        if obs_id < 0 or obs_id >= len(track):
            raise ValueError("Invalid observation ID")

        ids = group["candidate_arc_id"].to_numpy(
            dtype=int
        )

        if ids.min() < 0 or ids.max() >= len(graph.arcs):
            raise ValueError("Invalid candidate arc ID")

        endpoints = group[
            ["from_node_id", "to_node_id"]
        ].to_numpy(dtype=int)

        if not np.array_equal(
            graph.arcs[ids],
            endpoints,
        ):
            raise ValueError("Road endpoint mismatch")

        if not np.allclose(
            group["timestamp"].to_numpy(dtype=float),
            track[obs_id, 2],
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("Timestamp mismatch")

        arc_ids[i, :k] = ids

        from_nodes[i, :k] = endpoints[:, 0]

        to_nodes[i, :k] = endpoints[:, 1]

        emissions[i, :k] = group[
            "emission_log_prob"
        ].to_numpy()

        references[i] = int(
            group["ref_arc_id"].iloc[0]
        )

        observation_ids[i] = obs_id

        candidate_counts[i] = k

    # Precompute GPS-road distance differences

    for t in range(n - 1):

        gps1 = track[observation_ids[t], :2]
        gps2 = track[observation_ids[t + 1], :2]

        gps_distance = haversine_distance(
            gps1, gps2
        )

        for i in range(candidate_counts[t]):

            for j in range(candidate_counts[t + 1]):

                road_distance = candidate_road_distance(
                    graph,
                    arc_ids[t, i],
                    arc_ids[t + 1, j],
                    gps1,
                    gps2,
                )

                if np.isfinite(road_distance):

                    distance_errors[t, i, j] = abs(
                        road_distance - gps_distance
                    )

    output_path = CACHE_DIR / (
        f"trip_{int(trip_id):08d}.npz"
    )

    np.savez_compressed(
        output_path,
        arc_ids=arc_ids,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        emissions=emissions,
        distance_errors=distance_errors,
        references=references,
        observation_ids=observation_ids,
        candidate_counts=candidate_counts,
    )

    return n


def main():

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_PATH)

    start_time = time.perf_counter()

    total_trips = df.trip_id.nunique()
    total_observations = 0

    for index, (trip_id, candidates) in enumerate(
        df.groupby("trip_id"),
        start=1,
    ):

        print(
            f"Processing trip {trip_id} "
            f"({index}/{total_trips})",
            flush=True,
        )

        trip_start = time.perf_counter()

        n = prepare_trip(
            trip_id,
            candidates,
        )

        total_observations += n

        print(
            f"Cached {n} observations in "
            f"{time.perf_counter() - trip_start:.2f}s",
            flush=True,
        )

    print("\n===== HMM CACHE COMPLETED =====")

    print("Trips:", total_trips)
    print("Observations:", total_observations)

    print(
        "Runtime:",
        round(time.perf_counter() - start_time, 2),
        "seconds",
    )

    print("Cache directory:", CACHE_DIR)


if __name__ == "__main__":
    main()
