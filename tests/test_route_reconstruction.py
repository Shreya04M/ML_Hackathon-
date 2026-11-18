"""Focused tests for PathSnap HMM route reconstruction and constrained recovery."""

import numpy as np
import pandas as pd

from route_reconstruction.constrained_recovery import (
    recover_constrained_route,
)
from route_reconstruction.distance_transition import (
    candidate_road_distance,
    distance_transition_log_score,
)
from route_reconstruction.hmm import reconstruct_route
from route_reconstruction.road_graph import RoadGraph
from route_reconstruction.route_metrics import (
    evaluate_route_sequence,
    load_reference_route,
)
from route_reconstruction.route_recovery import recover_continuous_route
from route_reconstruction.transition_probability import (
    calculate_transition_probabilities,
)
from route_reconstruction.viterbi import viterbi_decode


def make_graph():
    nodes = np.asarray([
        [0.000, 0.000],
        [0.001, 0.000],
        [0.002, 0.000],
        [0.010, 0.000],
        [0.011, 0.000],
    ])
    arcs = np.asarray([
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4],
    ])
    return RoadGraph(nodes, arcs)


def test_same_road_forward_movement_uses_projected_distance():
    graph = make_graph()
    distance = candidate_road_distance(
        graph, 0, 0, [0.0002, 0.0], [0.0008, 0.0]
    )

    assert 60 < distance < 75


def test_small_same_road_backward_jitter_does_not_require_a_loop():
    graph = make_graph()
    distance = candidate_road_distance(
        graph, 0, 0, [0.0005, 0.0], [0.0004, 0.0]
    )
    strict_distance = candidate_road_distance(
        graph,
        0,
        0,
        [0.0005, 0.0],
        [0.0004, 0.0],
        backward_tolerance_m=5.0,
    )
    recovered = recover_constrained_route(
        [0, 0],
        graph,
        np.asarray([
            [0.0005, 0.0, 1.0],
            [0.0004, 0.0, 2.0],
        ]),
    )

    assert distance == 0.0
    assert strict_distance > 100
    assert recovered["rejected_connections"] == []
    assert recovered["route_sections"] == [[0]]
    assert recovered["repeated_arcs_removed"] == 1


def test_significant_same_road_reverse_movement_uses_directed_loop():
    graph = make_graph()
    distance = candidate_road_distance(
        graph, 0, 0, [0.0008, 0.0], [0.0002, 0.0]
    )
    recovered = recover_constrained_route(
        [0, 0],
        graph,
        np.asarray([
            [0.0008, 0.0, 1.0],
            [0.0002, 0.0, 2.0],
        ]),
    )

    assert distance > 100
    assert recovered["rejected_connections"][0]["reason"] == (
        "implausible_distance"
    )
    assert len(recovered["route_sections"]) == 2


def test_direct_connection_and_intermediate_recovery():
    graph = make_graph()
    direct = recover_continuous_route(
        [0, 1],
        graph,
        gps_track=np.asarray([
            [0.0008, 0.0, 1.0],
            [0.0012, 0.0, 2.0],
        ]),
    )
    intermediate = recover_constrained_route(
        [0, 2],
        graph,
        np.asarray([
            [0.0005, 0.0, 1.0],
            [0.0018, 0.0, 11.0],
        ]),
    )

    assert direct["route_sections"] == [[0, 1]]
    assert direct["inserted_arcs"] == 0
    assert intermediate["route_sections"] == [[0, 1, 2]]
    assert intermediate["inserted_arcs"] == 1


def test_unreachable_and_unrealistic_detours_are_separate_sections():
    graph = make_graph()
    track = np.asarray([
        [0.0005, 0.0, 1.0],
        [0.0105, 0.0, 2.0],
    ])
    unreachable = recover_constrained_route([0, 3], graph, track)
    unrealistic = recover_constrained_route(
        [0, 2],
        graph,
        track,
        max_speed_mps=1.0,
        gps_tolerance_m=0.0,
    )

    assert unreachable["rejected_connections"][0]["reason"] == (
        "unreachable"
    )
    assert len(unreachable["route_sections"]) == 2
    assert unrealistic["rejected_connections"][0]["reason"] == (
        "implausible_distance"
    )
    assert len(unrealistic["route_sections"]) == 2


def test_log_transition_scores_separate_long_detours_from_unreachable_paths():
    graph = make_graph()
    long_detour = distance_transition_log_score(
        graph,
        0,
        2,
        [0.0002, 0.0],
        [0.0002, 0.0],
        beta=1.0,
    )
    unreachable = distance_transition_log_score(
        graph,
        0,
        3,
        [0.0002, 0.0],
        [0.0105, 0.0],
        beta=1.0,
    )

    assert np.isfinite(long_detour)
    assert long_detour < np.log(1e-12)
    assert np.isneginf(unreachable)


def test_distance_transition_matrix_keeps_unreachable_edges_impossible():
    graph = make_graph()
    candidates = pd.DataFrame([
        [1, 0, 0, 0, 1],
        [1, 1, 3, 3, 4],
    ], columns=[
        "trip_id",
        "obs_id",
        "candidate_arc_id",
        "from_node_id",
        "to_node_id",
    ])
    track = np.asarray([
        [0.0005, 0.0, 1.0],
        [0.0105, 0.0, 2.0],
    ])
    transition = calculate_transition_probabilities(
        candidates,
        road_network=graph,
        transition_mode="distance",
        gps_track=track,
    )[1][0]

    assert transition["transition_prob"][0, 0] == 0.0
    assert np.isneginf(transition["transition_log_prob"][0, 0])


def test_invalid_timestamps_split_the_route():
    graph = make_graph()
    for invalid_timestamp in [np.nan, 1.0, 0.0]:
        recovered = recover_constrained_route(
            [0, 1],
            graph,
            np.asarray([
                [0.0008, 0.0, 1.0],
                [0.0012, 0.0, invalid_timestamp],
            ]),
        )

        assert recovered["rejected_connections"][0]["reason"] == (
            "invalid_timestamp"
        )
        assert recovered["route_sections"] == [[0], [1]]


def test_viterbi_decodes_and_restarts_at_impossible_transitions():
    decoded = viterbi_decode(
        emissions=[
            np.log([0.9, 0.1]),
            np.log([0.1, 0.9]),
            np.log([0.2, 0.8]),
        ],
        transitions=[
            np.asarray([
                [float("-inf"), float("-inf")],
                [float("-inf"), float("-inf")],
            ]),
            np.log(np.asarray([
                [0.9, 0.1],
                [0.1, 0.9],
            ])),
        ],
    )

    assert decoded["best_path"] == [0, 1, 1]
    assert decoded["segments"] == [(0, 1), (1, 3)]

    classifier_only = viterbi_decode(
        emissions=[np.log([0.8, 0.2]), np.log([0.1, 0.9])],
        transitions=[np.full((2, 2), float("-inf"))],
        transition_weight=0.0,
    )
    assert classifier_only["best_path"] == [0, 1]


def test_duplicate_candidates_and_ground_truth_free_live_inference():
    graph = make_graph()
    candidates = pd.DataFrame([
        [1, 0, 0, 1, 0.2],
        [1, 0, 0, 1, 0.8],
        [1, 1, 0, 0, 0.7],
        [1, 1, 1, 1, 0.3],
    ], columns=[
        "trip_id",
        "obs_id",
        "candidate_arc_id",
        "from_node_id",
        "pred_prob",
    ])
    candidates["to_node_id"] = [1, 1, 1, 2]

    route = reconstruct_route(
        candidates=candidates,
        road_network=graph,
        transition_mode="connectivity",
        transition_weight=1.0,
    )

    assert route["hmm_arc_id"].tolist() == [0, 0]
    assert "ref_arc_id" not in route.columns
    assert len(route) == 2


def test_route_reference_format_and_sequence_overlap(tmp_path):
    route_path = tmp_path / "trip.route"
    route_path.write_text("0\n1\n2\n", encoding="utf-8")
    reference = load_reference_route(route_path)
    metrics = evaluate_route_sequence([0, 1, 3], reference)

    assert reference == [0, 1, 2]
    assert metrics["sequence_edit_distance"] == 1
    assert metrics["road_overlap_precision"] == 2 / 3
    assert metrics["road_overlap_recall"] == 2 / 3
