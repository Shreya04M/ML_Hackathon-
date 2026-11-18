"""Unit tests for spatial candidate generation, metric features, and cleaning."""
from pathlib import Path
import numpy as np

from preprocessing import FEATURE_COLUMNS, extract_trip_candidates_and_features
from preprocessing.data_cleaning import clean_gps_data
from preprocessing.data_loader import load_trip
from preprocessing.feature_engineering import extract_features
from preprocessing.candidate_generation import generate_candidates
from preprocessing.road_network import load_road_network


def test_canonical_spatial_pipeline_trip_00000052() -> None:
    trip_dir = Path("Student folder/student_test/00000052")
    raw = load_trip(trip_dir)
    X_df, y, meta_df = extract_trip_candidates_and_features(
        track=raw["track"].to_numpy(dtype=np.float64),
        nodes=raw["nodes"][["longitude", "latitude"]].to_numpy(dtype=np.float64),
        arcs=raw["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64),
        route=None,
        trip_id="00000052",
        top_k=6,
    )
    assert y is None
    assert list(X_df.columns) == FEATURE_COLUMNS
    assert len(X_df) == len(raw["track"]) * 6
    assert not X_df.isna().any().any()
    assert (meta_df["ref_arc_id"] == -1).all()


def test_clean_and_legacy_feature_pipeline() -> None:
    trip_dir = Path("Student folder/student_test/00000052")
    raw = load_trip(trip_dir)
    obs = clean_gps_data(raw["track"].iloc[:20])
    net = load_road_network(trip_dir / "00000052.nodes", trip_dir / "00000052.arcs")
    cands = generate_candidates(obs, net, radius_m=50.0)
    feats = extract_features(obs, cands, net)
    assert "bearing_difference_deg" in feats.columns
    assert "distance_to_start_m" in feats.columns
