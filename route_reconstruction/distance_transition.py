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
):
    """Estimate directed travel distance between candidates."""

    previous_arc = int(previous_arc)
    next_arc = int(next_arc)

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

    # Movement forward along the same road

    if (
        previous_arc == next_arc
        and fraction2 >= fraction1
    ):
        return float(
            (fraction2 - fraction1) * length1
        )

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


def distance_transition_score(
    graph,
    previous_arc,
    next_arc,
    previous_gps,
    next_gps,
    beta=30.0,
    minimum_score=1e-12,
):
    """Score a transition using road and GPS distances."""

    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("Beta must be positive")

    gps_distance = float(
        haversine_distance(
            previous_gps,
            next_gps,
        )
    )

    road_distance = candidate_road_distance(
        graph,
        previous_arc,
        next_arc,
        previous_gps,
        next_gps,
    )

    if not np.isfinite(road_distance):
        return minimum_score

    difference = abs(
        road_distance - gps_distance
    )

    score = np.exp(-difference / beta)

    return float(
        np.clip(score, minimum_score, 1.0)
    )
