"""Directed road-network preparation with geodesic geometry."""
from pathlib import Path
from typing import Any
import math
import pandas as pd
import numpy as np

def load_road_network(nodes_path: Path | str, arcs_path: Path | str) -> dict[str, Any]:
    """Load nodes and arcs without assuming an unverified schema."""
    nodes = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["longitude", "latitude"])
    arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
    nodes.insert(0, "node_id", range(len(nodes))); arcs.insert(0, "arc_id", range(len(arcs)))
    a=nodes.rename(columns={"node_id":"from_node_id","longitude":"start_lon","latitude":"start_lat"})
    b=nodes.rename(columns={"node_id":"to_node_id","longitude":"end_lon","latitude":"end_lat"})
    seg=arcs.merge(a[["from_node_id","start_lon","start_lat"]],on="from_node_id",how="left",indicator="from_ok")
    seg=seg.merge(b[["to_node_id","end_lon","end_lat"]],on="to_node_id",how="left",indicator="to_ok")
    valid=seg[seg.from_ok=="both"].loc[seg.to_ok=="both"].copy()
    lat=np.radians((valid.start_lat+valid.end_lat)/2); dx=(valid.end_lon-valid.start_lon)*111320*np.cos(lat); dy=(valid.end_lat-valid.start_lat)*110540
    valid["min_lon"]=valid[["start_lon","end_lon"]].min(axis=1); valid["max_lon"]=valid[["start_lon","end_lon"]].max(axis=1); valid["min_lat"]=valid[["start_lat","end_lat"]].min(axis=1); valid["max_lat"]=valid[["start_lat","end_lat"]].max(axis=1); valid["length_m"]=np.hypot(dx,dy); valid["bearing_deg"]=(np.degrees(np.arctan2(dx,dy))+360)%360
    return {"nodes":nodes,"arcs":arcs,"segments":valid.drop(columns=["from_ok","to_ok"]),"invalid_arc_ids":sorted(set(arcs.arc_id)-set(valid.arc_id))}
