"""Leakage-safe candidate-level feature extraction."""
from typing import Any
import numpy as np
import pandas as pd

def extract_features(observations: Any, candidates: Any, road_network: Any) -> Any:
    """Extract model-ready features without evaluator-label leakage."""
    o=observations.copy(); o["time_diff_s"]=o.timestamp.diff(); o["movement_bearing_deg"]=(np.degrees(np.arctan2(np.diff(o.longitude,prepend=np.nan),np.diff(o.latitude,prepend=np.nan)))+360)%360
    out=candidates.merge(o[["observation_id","time_diff_s","movement_distance_m","speed_mps","movement_bearing_deg"]],on="observation_id",how="left")
    seg=road_network["segments"][["arc_id","length_m","bearing_deg","start_lon","start_lat","end_lon","end_lat"]]
    out=out.merge(seg,left_on="candidate_arc_id",right_on="arc_id",how="left"); out["bearing_difference_deg"]=abs((out.movement_bearing_deg-out.bearing_deg+180)%360-180); out["distance_to_start_m"]=np.hypot((out.gps_longitude-out.start_lon)*111320,(out.gps_latitude-out.start_lat)*110540); out["distance_to_end_m"]=np.hypot((out.gps_longitude-out.end_lon)*111320,(out.gps_latitude-out.end_lat)*110540)
    return out
