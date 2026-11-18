"""
PathSnap - Distance-Based Transition Scoring
Member 3 - Phase 2
"""

from pathlib import Path

import numpy as np

from route_reconstruction.road_graph import (
    haversine_distance,
)


def load_gps_track(
    trip_id,
    partition="student_test",
    data_root="Student folder",
):
    """Load GPS coordinates and timestamps."""

    trip_name = f"{int(trip_id):08d}"

    track_path = (
        Path(data_root)
        / partition
        / trip_name
        / f"{trip_name}.track"
    )

    track = np.loadtxt(
        track_path,
        dtype=float,
        ndmin=2,
    )

    if track.shape[1] != 3:
        raise ValueError("Invalid GPS track format")

    if not np.isfinite(track).all():
        raise ValueError("Invalid GPS coordinates")

    return track


def project_fraction(point, start, end):
    """Estimate the GPS position along a road."""

    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    scale = np.array([
        np.cos(np.radians(point[1])),
        1.0,
    ])

    road_vector = (end - start) * scale
    gps_vector = (point - start) * scale

    denominator = np.dot(
        road_vector, road_vector
    )

    if denominator <= 0:
        return 0.0

    fraction = (
        np.dot(gps_vector, road_vector)
        / denominator
    )

    return float(np.clip(fraction, 0.0, 1.0))


def candidate_road_distance(
    graph,
    previous_arc,
    next_arc,
    previous_gps,
    next_gps,
    backward_tolerance_m=30.0,
):
    """Estimate directed travel distance between candidates."""

    previous_arc = int(previous_arc)
    next_arc = int(next_arc)

    if (
        previous_arc < 0
        or previous_arc >= len(graph.arcs)
        or next_arc < 0
        or next_arc >= len(graph.arcs)
    ):
        raise ValueError("Invalid candidate arc ID")

    if (
        not np.isfinite(backward_tolerance_m)
        or backward_tolerance_m < 0
    ):
        raise ValueError("Invalid backward tolerance")

    previous_gps = np.asarray(previous_gps, dtype=float)
    next_gps = np.asarray(next_gps, dtype=float)

    if (
        previous_gps.shape != (2,)
        or next_gps.shape != (2,)
        or not np.isfinite(previous_gps).all()
        or not np.isfinite(next_gps).all()
    ):
        raise ValueError("Invalid GPS coordinates")

    u1, v1 = graph.arcs[previous_arc]
    u2, v2 = graph.arcs[next_arc]

    length1 = graph.arc_lengths[previous_arc]
    length2 = graph.arc_lengths[next_arc]

    fraction1 = project_fraction(
        previous_gps,
        graph.nodes[u1],
        graph.nodes[v1],
    )

    fraction2 = project_fraction(
        next_gps,
        graph.nodes[u2],
        graph.nodes[v2],
    )

    if previous_arc == next_arc:
        signed_distance = (fraction2 - fraction1) * length1

        if signed_distance >= 0:
            return float(signed_distance)

        if abs(signed_distance) <= backward_tolerance_m:
            return 0.0

    # Travel through the directed road network

    middle_distance = graph.shortest_distance(
        v1, u2
    )

    if not np.isfinite(middle_distance):
        return float("inf")

    remaining_distance = (
        1.0 - fraction1
    ) * length1

    entering_distance = fraction2 * length2

    return float(
        remaining_distance
        + middle_distance
        + entering_distance
    )


def distance_transition_log_score(
    graph,
    previous_arc,
    next_arc,
    previous_gps,
    next_gps,
    beta=30.0,
    backward_tolerance_m=30.0,
):
    """Return an unclipped log score; unreachable transitions are impossible."""

    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("Beta must be positive")

    gps_distance = float(
        haversine_distance(previous_gps, next_gps)
    )

    road_distance = candidate_road_distance(
        graph,
        previous_arc,
        next_arc,
        previous_gps,
        next_gps,
        backward_tolerance_m=backward_tolerance_m,
    )

    if not np.isfinite(road_distance):
        return float("-inf")

    return float(-abs(road_distance - gps_distance) / beta)


def distance_transition_score(
    graph,
    previous_arc,
    next_arc,
    previous_gps,
    next_gps,
    beta=30.0,
    minimum_score=1e-12,
    backward_tolerance_m=30.0,
):
    """Score a transition using road and GPS distances."""

    if (
        not np.isfinite(minimum_score)
        or minimum_score <= 0
        or minimum_score > 1
    ):
        raise ValueError("Minimum score must be in (0, 1]")

    log_score = distance_transition_log_score(
        graph,
        previous_arc,
        next_arc,
        previous_gps,
        next_gps,
        beta=beta,
        backward_tolerance_m=backward_tolerance_m,
    )

    if not np.isfinite(log_score):
        return minimum_score

    return float(np.exp(max(log_score, np.log(minimum_score))))
