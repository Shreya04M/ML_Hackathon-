"""Leakage-safe candidate-level feature extraction."""
from typing import Any
import numpy as np
import pandas as pd


def extract_features(observations: Any, candidates: Any, road_network: Any) -> Any:
    """Extract model-ready features without evaluator-label leakage."""
    o = observations.copy()
    o["time_diff_s"] = o.timestamp.diff()
    cos_lat = np.cos(np.radians(o.latitude.to_numpy(dtype=float)))
    dlon_m = np.diff(o.longitude, prepend=np.nan) * 111320.0 * cos_lat
    dlat_m = np.diff(o.latitude, prepend=np.nan) * 110540.0
    o["movement_bearing_deg"] = (np.degrees(np.arctan2(dlon_m, dlat_m)) + 360.0) % 360.0

    out = candidates.merge(
        o[["observation_id", "time_diff_s", "movement_distance_m", "speed_mps", "movement_bearing_deg"]],
        on="observation_id",
        how="left",
    )
    seg = road_network["segments"][["arc_id", "length_m", "bearing_deg", "start_lon", "start_lat", "end_lon", "end_lat"]]
    out = out.merge(seg, left_on="candidate_arc_id", right_on="arc_id", how="left")
    out["bearing_difference_deg"] = abs((out.movement_bearing_deg - out.bearing_deg + 180.0) % 360.0 - 180.0)

    cand_cos = np.cos(np.radians(out.gps_latitude.to_numpy(dtype=float)))
    out["distance_to_start_m"] = np.hypot(
        (out.gps_longitude - out.start_lon) * 111320.0 * cand_cos,
        (out.gps_latitude - out.start_lat) * 110540.0,
    )
    out["distance_to_end_m"] = np.hypot(
        (out.gps_longitude - out.end_lon) * 111320.0 * cand_cos,
        (out.gps_latitude - out.end_lat) * 110540.0,
    )
    return out
