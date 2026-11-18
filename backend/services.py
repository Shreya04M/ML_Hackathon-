"""Service-layer integration for PathSnap (Laptop 1 FastAPI Backend).

Executes the live end-to-end PathSnap pipeline for any approved trip in
``evaluator_test`` or ``student_test`` without exposing reference ``.route``
ground truth to feature extraction, ML inference, HMM decoding, or the UI:
    1. Load raw .track, .nodes, .arcs (strictly route=None during inference).
    2. Extract 20 spatial/kinematic/topological features (K=6 candidates).
    3. Run live inference using the Bayesian-tuned HistGradientBoosting model.
    4. Run distance-based HMM + Viterbi decoding and constrained route recovery.
    5. Compute server-side accuracy & route overlap metrics post-prediction.
    6. Emit structured terminal logs on Laptop 1 for evaluator verification.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import folium
import numpy as np
import pandas as pd

from evaluation.route_evaluation import evaluate_route
from models import FEATURE_COLUMNS, load_final_model, predict_candidate_probabilities
from preprocessing.spatial_pipeline import (
    associate_reference_arcs,
    compute_gps_kinematics,
    extract_trip_candidates_and_features,
    project_lonlat_to_meters,
)
from route_reconstruction.hmm import reconstruct_route
from route_reconstruction.road_graph import RoadGraph
from route_reconstruction.route_metrics import evaluate_route_sequence
from route_reconstruction.route_recovery import recover_continuous_route

logger = logging.getLogger("pathsnap.api.inference")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_data_dir() -> Path:
    """Resolve active evaluation directory (env override > evaluator_test > student_test)."""
    env_dir = os.environ.get("PATHSNAP_TRIP_DIR")
    if env_dir:
        return Path(env_dir)
    eval_dir = PROJECT_ROOT / "evaluator_test"
    if eval_dir.is_dir():
        return eval_dir
    return PROJECT_ROOT / "Student folder" / "student_test"


DATA_DIR = _resolve_data_dir()

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "selected_best_model_tuned.joblib"
    if (PROJECT_ROOT / "artifacts" / "models" / "selected_best_model_tuned.joblib").is_file()
    else PROJECT_ROOT / "models" / "artifacts" / "models" / "selected_best_model_tuned.joblib"
)

HMM_PARAM_PATH = (
    PROJECT_ROOT / "artifacts" / "hmm" / "best_hmm_parameters.json"
    if (PROJECT_ROOT / "artifacts" / "hmm" / "best_hmm_parameters.json").is_file()
    else PROJECT_ROOT
    / "route_reconstruction"
    / "experiments"
    / "phase2_improved"
    / "hmm_optimisation"
    / "best_hmm_parameters.json"
)


def list_available_trips() -> list[str]:
    """Discover valid trips from the active dataset directory."""
    data_dir = _resolve_data_dir()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Trip dataset not available at {data_dir}")

    trip_ids: list[str] = []
    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue
        trip_id = folder.name
        required_files = [
            folder / f"{trip_id}.track",
            folder / f"{trip_id}.nodes",
            folder / f"{trip_id}.arcs",
        ]
        if all(file.is_file() for file in required_files):
            trip_ids.append(trip_id)

    return sorted(trip_ids)


@lru_cache(maxsize=1)
def _load_classifier() -> Any:
    """Load the persisted Bayesian-tuned HistGradientBoosting pipeline once."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {MODEL_PATH}")
    logger.info("[MODEL LOAD] Loaded tuned classifier artifact: %s", MODEL_PATH)
    return load_final_model(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_hmm_params() -> dict[str, float]:
    """Load optimised HMM emission/transition parameters."""
    if HMM_PARAM_PATH.is_file():
        data = json.loads(HMM_PARAM_PATH.read_text(encoding="utf-8"))
        return {
            "beta": float(data.get("beta", 30.0)),
            "transition_weight": float(data.get("transition_weight", 1.948614369534382)),
            "backward_tolerance_m": float(data.get("backward_tolerance_m", 30.0)),
        }
    return {"beta": 30.0, "transition_weight": 1.948614369534382, "backward_tolerance_m": 30.0}


def _compute_server_side_pointwise_accuracy(
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    reference_route: list[int],
    hmm_arc_ids: np.ndarray,
) -> float:
    """Evaluate observation-level accuracy strictly server-side after inference."""
    ref_lat = float(np.mean(track[:, 1]))
    track_xy = project_lonlat_to_meters(track[:, 0], track[:, 1], ref_lat)
    nodes_xy = project_lonlat_to_meters(nodes[:, 0], nodes[:, 1], ref_lat)
    p1 = nodes_xy[arcs[:, 0]]
    p2 = nodes_xy[arcs[:, 1]]
    vec = p2 - p1
    v_len2 = np.sum(vec**2, axis=1)
    arc_lengths = np.sqrt(v_len2)
    v_len2_safe = np.where(v_len2 > 1e-12, v_len2, 1.0)
    arc_valid_dir = arc_lengths > 1e-3
    arc_heading = np.where(arc_valid_dir, np.arctan2(vec[:, 1], vec[:, 0]), 0.0)
    _, _, gps_heading, gps_hdg_valid = compute_gps_kinematics(track_xy, track[:, 2])
    ref_arcs = associate_reference_arcs(
        track_xy=track_xy,
        heading=gps_heading,
        heading_valid=gps_hdg_valid,
        p1=p1,
        v=vec,
        v_len2_safe=v_len2_safe,
        arc_heading=arc_heading,
        arc_valid_dir=arc_valid_dir,
        route_arcs=np.asarray(reference_route, dtype=np.int64),
    )
    return float(np.mean(hmm_arc_ids == ref_arcs))


def predict_trip(trip_id: str) -> dict[str, Any]:
    """Run live feature extraction, ML inference, HMM/Viterbi decoding, and evaluation."""
    data_dir = _resolve_data_dir()
    if trip_id not in list_available_trips():
        raise ValueError(f"Invalid trip ID: {trip_id}")

    started_at = perf_counter()
    trip_dir = data_dir / trip_id
    track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64, ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64, ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)

    # Strictly set route=None so reference labels never enter features or predictions.
    X_df, _, meta_df = extract_trip_candidates_and_features(
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=None,
        trip_id=trip_id,
        top_k=6,
    )

    classifier = _load_classifier()
    probs = predict_candidate_probabilities(classifier, X_df[FEATURE_COLUMNS])

    candidates_for_hmm = meta_df[
        ["trip_id", "obs_id", "timestamp", "candidate_arc_id", "from_node_id", "to_node_id"]
    ].copy()
    candidates_for_hmm["pred_prob"] = probs
    candidates_for_hmm["pred_log_prob"] = np.log(np.clip(probs, 1e-15, 1.0))

    graph = RoadGraph(nodes, arcs)
    hmm_params = _load_hmm_params()
    reconstructed = reconstruct_route(
        candidates=candidates_for_hmm,
        road_network=graph,
        gps_track=track,
        transition_mode="distance",
        transition_weight=hmm_params["transition_weight"],
        beta=hmm_params["beta"],
        backward_tolerance_m=hmm_params["backward_tolerance_m"],
    ).sort_values("obs_id")

    hmm_obs_arcs = reconstructed["hmm_arc_id"].to_numpy(dtype=np.int64)
    obs_ids = reconstructed["obs_id"].to_numpy(dtype=np.int64)
    recovery = recover_continuous_route(
        hmm_obs_arcs.tolist(),
        graph,
        gps_track=track,
        observation_ids=obs_ids,
        max_speed_mps=60.0,
        gps_tolerance_m=30.0,
        backward_tolerance_m=hmm_params["backward_tolerance_m"],
    )
    predicted_route = [int(a) for a in recovery["recovered_arcs"]]

    # Server-side evaluation only (ground truth never leaves Laptop 1).
    metrics: dict[str, Any] | None = None
    reference_file = trip_dir / f"{trip_id}.route"
    if reference_file.is_file():
        reference_route = [
            int(line.strip())
            for line in reference_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metrics = evaluate_route(predicted_route, reference_route)
        metrics["accuracy"] = _compute_server_side_pointwise_accuracy(
            track=track,
            nodes=nodes,
            arcs=arcs,
            reference_route=reference_route,
            hmm_arc_ids=hmm_obs_arcs,
        )
        seq_metrics = evaluate_route_sequence(predicted_route, reference_route, graph=graph)
        metrics["sequence_lcs_f1"] = seq_metrics.get("sequence_lcs_f1")
        metrics["road_overlap_f1"] = seq_metrics.get("road_overlap_f1")

    elapsed_s = perf_counter() - started_at
    map_html = generate_trip_map(trip_id, predicted_route)

    logger.info(
        "[POST /predict] data_dir=%s | trip_id=%s | gps_obs=%d | candidates=%d | "
        "model=%s | hmm_obs_arcs=%d | recovered_arcs=%d | acc=%.4f | route_f1=%.4f | latency=%.3fs",
        data_dir.name,
        trip_id,
        len(track),
        len(X_df),
        MODEL_PATH.name,
        len(hmm_obs_arcs),
        len(predicted_route),
        float(metrics["accuracy"]) if metrics and metrics.get("accuracy") is not None else -1.0,
        float(metrics["route_f1"]) if metrics else -1.0,
        elapsed_s,
    )

    return {
        "trip_id": trip_id,
        "status": "completed",
        "predicted_route": predicted_route,
        "metrics": metrics,
        "execution_time": elapsed_s,
        "map_html": map_html,
    }


def generate_trip_map(
    trip_id: str,
    predicted_route: list[int] | None = None,
) -> str:
    """Generate a Folium map with road network, raw GPS, and predicted route (no ground truth)."""
    data_dir = _resolve_data_dir()
    if trip_id not in list_available_trips():
        raise ValueError(f"Invalid trip ID: {trip_id}")

    trip_dir = data_dir / trip_id
    track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64, ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64, ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)

    coordinates = [[float(lat), float(lon)] for lon, lat in track[:, :2]]
    if not coordinates:
        raise ValueError("No GPS coordinates found")

    route_map = folium.Map(location=coordinates[0], zoom_start=14, tiles="OpenStreetMap")

    road_segments = [
        [
            [float(nodes[s, 0]), float(nodes[s, 1])],
            [float(nodes[e, 0]), float(nodes[e, 1])],
        ]
        for s, e in arcs
    ]

    folium.GeoJson(
        {"type": "MultiLineString", "coordinates": road_segments},
        name="Road Network",
        style_function=lambda _feat: {"color": "#808080", "weight": 1.5, "opacity": 0.45},
    ).add_to(route_map)

    folium.PolyLine(
        coordinates,
        color="#2563eb",
        weight=3.5,
        opacity=0.8,
        tooltip="Raw GPS Trajectory",
    ).add_to(route_map)

    folium.Marker(coordinates[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker(coordinates[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(route_map)

    if predicted_route is not None:
        predicted_segments = []
        for arc_id in predicted_route:
            if not 0 <= arc_id < len(road_segments):
                raise ValueError(f"Invalid predicted arc ID: {arc_id}")
            predicted_segments.append(road_segments[arc_id])

        if predicted_segments:
            folium.GeoJson(
                {"type": "MultiLineString", "coordinates": predicted_segments},
                name="Predicted Reconstructed Route",
                style_function=lambda _feat: {"color": "#dc2626", "weight": 5, "opacity": 0.9},
                tooltip="Predicted Reconstructed Route",
            ).add_to(route_map)

    route_map.fit_bounds(coordinates)
    folium.LayerControl().add_to(route_map)
    return route_map.get_root().render()
