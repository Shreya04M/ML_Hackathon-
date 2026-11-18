"""
PathSnap - HMM Validation Cache
Member 3 - Phase 2
"""

import argparse
import json
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
    project_fraction,
)

from route_reconstruction.emission_probability import (
    calculate_emission_probabilities,
)


INPUT_PATH = Path(
    "experiments/results/hmm_validation/"
    "training_oof_probabilities.csv.gz"
)

CACHE_DIR = Path(
    "route_reconstruction/experiments/phase2_improved/"
    "hmm_validation/cache"
)

TOP_K = 6
BACKWARD_TOLERANCE_M = 30.0


def prepare_trip(trip_id, candidates, cache_dir=None):

    cache_dir = Path(cache_dir or CACHE_DIR)

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
                    backward_tolerance_m=BACKWARD_TOLERANCE_M,
                )

                if np.isfinite(road_distance):

                    distance_errors[t, i, j] = abs(
                        road_distance - gps_distance
                    )

    output_path = cache_dir / (
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


def rebuild_cache_from_compatible_source(source_cache_dir, cache_dir):
    """Rebuild the cache while reusing unaffected training distances."""

    source_cache_dir = Path(source_cache_dir)
    cache_dir = Path(cache_dir)
    files = sorted(source_cache_dir.glob("trip_*.npz"))

    if len(files) != 80:
        raise ValueError(
            f"Expected 80 source cache files, found {len(files)}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    total_observations = 0
    corrected_transitions = 0

    for index, source_path in enumerate(files, start=1):
        trip_id = int(source_path.stem.split("_", 1)[1])
        graph = load_road_graph(trip_id, partition="train")
        track = load_gps_track(trip_id, partition="train")

        with np.load(source_path, allow_pickle=False) as data:
            trip = {key: data[key] for key in data.files}

        observation_ids = trip["observation_ids"]
        counts = trip["candidate_counts"]
        arc_ids = trip["arc_ids"]
        distance_errors = trip["distance_errors"].copy()

        if len(observation_ids) != len(counts):
            raise ValueError(f"Invalid cached observations in {source_path}")

        if (observation_ids < 0).any() or (
            observation_ids >= len(track)
        ).any():
            raise ValueError(f"Invalid cached observation IDs in {source_path}")

        for t in range(len(observation_ids) - 1):
            gps1 = track[observation_ids[t], :2]
            gps2 = track[observation_ids[t + 1], :2]
            timestamp1 = track[observation_ids[t], 2]
            timestamp2 = track[observation_ids[t + 1], 2]

            if (
                not np.isfinite(timestamp1)
                or not np.isfinite(timestamp2)
                or timestamp2 <= timestamp1
            ):
                distance_errors[t, :, :] = np.inf
                continue

            gps_distance = float(haversine_distance(gps1, gps2))

            for i in range(int(counts[t])):
                first_arc = int(arc_ids[t, i])
                for j in range(int(counts[t + 1])):
                    if first_arc != int(arc_ids[t + 1, j]):
                        continue

                    u, v = graph.arcs[first_arc]
                    first_fraction = project_fraction(
                        gps1, graph.nodes[u], graph.nodes[v]
                    )
                    second_fraction = project_fraction(
                        gps2, graph.nodes[u], graph.nodes[v]
                    )
                    signed_distance = (
                        second_fraction - first_fraction
                    ) * graph.arc_lengths[first_arc]

                    if (
                        signed_distance < 0
                        and abs(signed_distance)
                        <= BACKWARD_TOLERANCE_M
                    ):
                        distance_errors[t, i, j] = gps_distance
                        corrected_transitions += 1

        trip["distance_errors"] = distance_errors
        np.savez_compressed(cache_dir / source_path.name, **trip)
        total_observations += len(observation_ids)

        print(
            f"Rebuilt {source_path.name} "
            f"({index}/{len(files)}): {len(observation_ids)} observations",
            flush=True,
        )

    if total_observations != 208417:
        raise ValueError(
            f"Expected 208,417 training observations, found {total_observations}"
        )

    metadata = {
        "source_cache_dir": str(source_cache_dir),
        "training_trips": len(files),
        "observations": total_observations,
        "corrected_same_arc_jitter_transitions": corrected_transitions,
        "backward_tolerance_m": BACKWARD_TOLERANCE_M,
        "unchanged_values": (
            "All non-jitter transition errors and remaining arrays were "
            "reused from the compatible training cache."
        ),
    }
    with (cache_dir / "cache_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("Corrected same-arc jitter transitions:", corrected_transitions)
    print("Cache directory:", cache_dir)


def main(
    input_path=INPUT_PATH,
    cache_dir=CACHE_DIR,
    reuse_cache_dir=None,
):

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if reuse_cache_dir is not None:
        rebuild_cache_from_compatible_source(
            source_cache_dir=reuse_cache_dir,
            cache_dir=cache_dir,
        )
        return

    df = pd.read_csv(input_path)

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
            cache_dir=cache_dir,
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

    print("Cache directory:", cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=INPUT_PATH,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
    )
    parser.add_argument(
        "--reuse-cache-dir",
        type=Path,
        default=None,
        help=(
            "Reuse mathematically unchanged distances from an 80-trip cache "
            "and recalculate only affected entries."
        ),
    )
    arguments = parser.parse_args()
    main(
        input_path=arguments.input_path,
        cache_dir=arguments.cache_dir,
        reuse_cache_dir=arguments.reuse_cache_dir,
    )
