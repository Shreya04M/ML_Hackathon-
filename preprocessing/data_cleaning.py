"""Non-destructive validation and cleaning diagnostics."""
from typing import Any
import numpy as np
import pandas as pd

def clean_gps_data(observations: Any) -> Any:
    return validate_gps(observations)[0]

def validate_gps(df: pd.DataFrame, max_speed_mps: float = 100.0, max_gap_s: float = 300.0):
    x = df.copy(); x.insert(0, "observation_id", range(len(x)))
    x["duplicate_observation"] = x.duplicated(keep=False)
    x["invalid_coordinate"] = ~x.longitude.between(-180, 180) | ~x.latitude.between(-90, 90)
    x["time_diff_s"] = x.timestamp.diff(); x["nonpositive_time"] = x.time_diff_s.le(0).fillna(False)
    x["timestamp_gap"] = x.time_diff_s.gt(max_gap_s).fillna(False)
    lon, lat = np.radians(x.longitude), np.radians(x.latitude)
    x["movement_distance_m"] = np.hypot(np.diff(lon, prepend=np.nan) * 6371008.8 * np.cos(lat), np.diff(lat, prepend=np.nan) * 6371008.8)
    x["speed_mps"] = x.movement_distance_m / x.time_diff_s.replace(0, np.nan)
    x["implausible_speed"] = x.speed_mps.gt(max_speed_mps).fillna(False)
    x["clean"] = ~(x[["invalid_coordinate", "nonpositive_time", "implausible_speed"]].any(axis=1))
    return x, x[["observation_id", "duplicate_observation", "invalid_coordinate", "nonpositive_time", "timestamp_gap", "implausible_speed", "clean"]]

def validate_network(nodes: pd.DataFrame, arcs: pd.DataFrame) -> dict:
    ids = set(nodes.node_id.dropna().astype(int)); a = arcs.dropna(subset=["from_node_id", "to_node_id"])
    missing = a[~a.from_node_id.astype(int).isin(ids) | ~a.to_node_id.astype(int).isin(ids)]
    used = set(a.from_node_id.astype(int)) | set(a.to_node_id.astype(int))
    return {"node_count": len(nodes), "arc_count": len(arcs), "invalid_coordinates": int((~nodes.longitude.between(-180,180) | ~nodes.latitude.between(-90,90)).sum()), "missing_endpoint_arcs": len(missing), "duplicate_arcs": int(arcs.duplicated(["from_node_id", "to_node_id"]).sum()), "isolated_nodes": len(ids-used)}

def validate_route(route: pd.DataFrame, arc_count: int) -> dict:
    invalid = route.arc_id.isna() | ~route.arc_id.between(0, arc_count-1)
    return {"route_rows": len(route), "invalid_arc_ids": int(invalid.sum()), "unique_arcs": int(route.arc_id.nunique(dropna=True))}
