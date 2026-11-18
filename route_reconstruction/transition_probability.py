"""
PathSnap - HMM Transition Probabilities
Member 3 - Route Reconstruction
"""

import numpy as np
import pandas as pd
from route_reconstruction.distance_transition import (
    distance_transition_score,
)

EPSILON = 1e-12


def calculate_transition_probabilities(
    candidates,
    road_network=None,
    **kwargs,
):
    """
    Calculate connectivity-based transition matrices
    between consecutive GPS observations.
    """

    required_columns = {
        "trip_id",
        "obs_id",
        "candidate_arc_id",
        "from_node_id",
        "to_node_id",
    }

    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if candidates.empty:
        raise ValueError("Candidate dataset is empty")

    if candidates[list(required_columns)].isna().any().any():
        raise ValueError("Missing candidate or node IDs")

    same_arc_weight = kwargs.get("same_arc_weight", 1.0)
    connected_weight = kwargs.get("connected_weight", 0.9)
    reverse_weight = kwargs.get("reverse_weight", 0.05)
    disconnected_weight = kwargs.get(
        "disconnected_weight", 0.01
    )

    weights = np.asarray([
        same_arc_weight,
        connected_weight,
        reverse_weight,
        disconnected_weight,
    ], dtype=float)

    if not np.isfinite(weights).all() or (
        (weights <= 0) | (weights > 1)
    ).any():
        raise ValueError("Invalid transition weights")
    transition_mode = kwargs.get(
        "transition_mode", "connectivity"
    )

    gps_track = kwargs.get("gps_track")
    beta = float(kwargs.get("beta", 30.0))

    if transition_mode not in (
        "connectivity", "distance"
    ):
        raise ValueError("Invalid transition mode")

    if transition_mode == "distance":

        if road_network is None or gps_track is None:
            raise ValueError(
                "Road graph and GPS track are required"
            )

        if candidates["trip_id"].nunique() != 1:
            raise ValueError(
                "Process one trip at a time in distance mode"
            )

        ids = candidates["candidate_arc_id"].to_numpy(dtype=int)
        obs = candidates["obs_id"].to_numpy(dtype=int)

        if ids.min() < 0 or ids.max() >= len(road_network.arcs):
            raise ValueError("Invalid candidate arc IDs")

        if obs.min() < 0 or obs.max() >= len(gps_track):
            raise ValueError("Invalid GPS observation IDs")

        if not np.array_equal(
            road_network.arcs[ids],
            candidates[["from_node_id", "to_node_id"]].to_numpy()
        ):
            raise ValueError("Road endpoints do not match")

        if "timestamp" in candidates.columns:
            if not np.allclose(
                gps_track[obs, 2],
                candidates["timestamp"].to_numpy(),
                rtol=0,
                atol=1e-6,
            ):
                raise ValueError("GPS timestamps do not match")
    transitions = {}

    df = candidates.sort_values(
        ["trip_id", "obs_id", "candidate_arc_id"]
    )

    for trip_id, trip_df in df.groupby("trip_id"):

        observation_groups = list(
            trip_df.groupby("obs_id", sort=True)
        )

        trip_transitions = []

        for t in range(len(observation_groups) - 1):

            current_obs, current = observation_groups[t]
            next_obs, following = observation_groups[t + 1]

            current_arcs = current[
                "candidate_arc_id"
            ].to_numpy()[:, None]

            next_arcs = following[
                "candidate_arc_id"
            ].to_numpy()[None, :]

            current_from = current[
                "from_node_id"
            ].to_numpy()[:, None]

            current_to = current[
                "to_node_id"
            ].to_numpy()[:, None]

            next_from = following[
                "from_node_id"
            ].to_numpy()[None, :]

            next_to = following[
                "to_node_id"
            ].to_numpy()[None, :]

            # Default: no direct connection

            matrix = np.full(
                (len(current), len(following)),
                disconnected_weight,
                dtype=float,
            )

            # Direct forward connection

            connected = current_to == next_from

            matrix[connected] = connected_weight

            # Reverse direction

            reverse = (
                (current_from == next_to)
                & (current_to == next_from)
            )

            matrix[reverse] = reverse_weight

            # Same road segment

            same_arc = current_arcs == next_arcs

            matrix[same_arc] = same_arc_weight

                     # Distance-based transition mode

            if transition_mode == "distance":

                gps1 = gps_track[int(current_obs), :2]
                gps2 = gps_track[int(next_obs), :2]

                for i, arc1 in enumerate(
                    current_arcs.ravel()
                ):
                    for j, arc2 in enumerate(
                        next_arcs.ravel()
                    ):

                        matrix[i, j] = distance_transition_score(
                            graph=road_network,
                            previous_arc=arc1,
                            next_arc=arc2,
                            previous_gps=gps1,
                            next_gps=gps2,
                            beta=beta,
                        )

            else:

                # Preserve baseline normalisation

                matrix = matrix / matrix.sum(
                    axis=1, keepdims=True
                )

            log_matrix = np.log(
                np.clip(matrix, EPSILON, 1.0)
            )

            trip_transitions.append({
                "from_obs_id": current_obs,
                "to_obs_id": next_obs,
                "from_candidates": current_arcs.ravel(),
                "to_candidates": next_arcs.ravel(),
                "transition_prob": matrix,
                "transition_log_prob": log_matrix,
            })

        transitions[trip_id] = trip_transitions

    return transitions
