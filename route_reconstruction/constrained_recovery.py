"""Physically constrained intermediate road-segment recovery."""

import numpy as np

from route_reconstruction.distance_transition import (
    candidate_road_distance,
    project_fraction,
)
from route_reconstruction.road_graph import haversine_distance


def recover_constrained_route(
    predicted_arcs,
    graph,
    gps_track,
    observation_ids=None,
    max_speed_mps=60.0,
    gps_tolerance_m=30.0,
    backward_tolerance_m=30.0,
):
    """Recover directed paths only when their distance fits the time gap.

    Rejected transitions start separate route sections; no alternate road
    connection is fabricated.
    """

    arcs = np.asarray(predicted_arcs, dtype=int)
    track = np.asarray(gps_track, dtype=float)

    if arcs.ndim != 1 or len(arcs) == 0:
        raise ValueError("Predicted route must be a non-empty sequence")

    if (arcs < 0).any() or (arcs >= len(graph.arcs)).any():
        raise ValueError("Invalid arc IDs")

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError("GPS track must contain longitude, latitude, timestamp")

    if observation_ids is None:
        observation_ids = np.arange(len(arcs))

    obs = np.asarray(observation_ids, dtype=int)

    if obs.ndim != 1 or len(obs) != len(arcs):
        raise ValueError("Observation alignment error")

    if (obs < 0).any() or (obs >= len(track)).any():
        raise ValueError("Invalid observation IDs")

    limits = np.asarray(
        [max_speed_mps, gps_tolerance_m, backward_tolerance_m],
        dtype=float,
    )

    if (
        not np.isfinite(limits).all()
        or max_speed_mps <= 0
        or gps_tolerance_m < 0
        or backward_tolerance_m < 0
    ):
        raise ValueError("Invalid recovery constraints")

    sections = []
    current_section = [int(arcs[0])]
    inserted_arcs = 0
    repeated_arcs_removed = 0
    rejected_connections = []

    for i in range(1, len(arcs)):
        previous_arc = int(arcs[i - 1])
        current_arc = int(arcs[i])

        gps1 = track[obs[i - 1], :2]
        gps2 = track[obs[i], :2]
        dt = float(track[obs[i], 2] - track[obs[i - 1], 2])
        allowed_distance = None
        gps_distance = None
        road_distance = float("inf")
        reason = None

        if not np.isfinite(dt) or dt <= 0:
            reason = "invalid_timestamp"
        elif not np.isfinite(gps1).all() or not np.isfinite(gps2).all():
            reason = "invalid_gps"
        else:
            allowed_distance = (
                max_speed_mps * dt + gps_tolerance_m
            )
            if not np.isfinite(allowed_distance):
                reason = "invalid_timestamp"
            else:
                gps_distance = float(haversine_distance(gps1, gps2))
                road_distance = candidate_road_distance(
                    graph,
                    previous_arc,
                    current_arc,
                    gps1,
                    gps2,
                    backward_tolerance_m=backward_tolerance_m,
                )

                if not np.isfinite(road_distance):
                    reason = "unreachable"
                elif road_distance > allowed_distance:
                    reason = "implausible_distance"

        if reason is not None:
            rejected_connections.append({
                "obs_id": int(obs[i]),
                "previous_arc": previous_arc,
                "current_arc": current_arc,
                "reason": reason,
                "road_distance_m": (
                    float(road_distance)
                    if np.isfinite(road_distance)
                    else None
                ),
                "gps_distance_m": gps_distance,
                "allowed_distance_m": allowed_distance,
                "time_gap_s": dt,
            })
            sections.append(current_section)
            current_section = [current_arc]
            continue

        if previous_arc == current_arc:
            u, v = graph.arcs[previous_arc]
            first_fraction = project_fraction(
                gps1, graph.nodes[u], graph.nodes[v]
            )
            second_fraction = project_fraction(
                gps2, graph.nodes[u], graph.nodes[v]
            )
            backward_distance = (
                first_fraction - second_fraction
            ) * graph.arc_lengths[previous_arc]

            if backward_distance <= backward_tolerance_m:
                repeated_arcs_removed += 1
                continue

        previous_end = int(graph.to_nodes[previous_arc])
        current_start = int(graph.from_nodes[current_arc])
        middle_distance, middle_arcs = graph.shortest_path(
            previous_end, current_start
        )

        if not np.isfinite(middle_distance):
            rejected_connections.append({
                "obs_id": int(obs[i]),
                "previous_arc": previous_arc,
                "current_arc": current_arc,
                "reason": "unreachable",
                "road_distance_m": float(road_distance),
                "gps_distance_m": gps_distance,
                "allowed_distance_m": allowed_distance,
                "time_gap_s": dt,
            })
            sections.append(current_section)
            current_section = [current_arc]
            continue

        current_section.extend(middle_arcs)
        inserted_arcs += len(middle_arcs)
        current_section.append(current_arc)

    sections.append(current_section)

    recovered_arcs = [
        arc for section in sections for arc in section
    ]

    return {
        "recovered_arcs": recovered_arcs,
        "route_sections": sections,
        "original_predictions": len(arcs),
        "recovered_arc_count": len(recovered_arcs),
        "inserted_arcs": inserted_arcs,
        "repeated_arcs_removed": repeated_arcs_removed,
        "rejected_connections": rejected_connections,
        "is_continuous": len(rejected_connections) == 0,
    }
