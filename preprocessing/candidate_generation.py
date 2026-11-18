"""Candidate road-segment generation using local equirectangular projection."""
from typing import Any
import math
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def generate_candidates(observations: Any, road_network: Any, **kwargs: Any) -> Any:
    """Generate candidate segments per GPS observation."""
    radius_m = float(kwargs.get("radius_m", 50.0)); seg = road_network["segments"]
    out=[]
    seg_xy=np.c_[seg.start_lon.to_numpy()*111320,seg.start_lat.to_numpy()*110540]
    mid_xy=np.c_[((seg.start_lon+seg.end_lon)/2).to_numpy()*111320,((seg.start_lat+seg.end_lat)/2).to_numpy()*110540]
    tree=cKDTree(mid_xy)
    for o in observations.itertuples(index=False):
        px,py=o.longitude*111320,o.latitude*110540
        idx=tree.query_ball_point([px,py], radius_m+100.0)
        for j in idx:
            s=seg.iloc[j]
            lat=math.radians(o.latitude); scale=111320*math.cos(lat); x1,y1=s.start_lon*scale,s.start_lat*110540; x2,y2=s.end_lon*scale,s.end_lat*110540; px,py=o.longitude*scale,o.latitude*110540
            dx,dy=x2-x1,y2-y1; t=max(0,min(1,((px-x1)*dx+(py-y1)*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0; qx,qy=x1+t*dx,y1+t*dy; d=math.hypot(px-qx,py-qy)
            if d<=radius_m: out.append({"observation_id":o.observation_id,"timestamp":o.timestamp,"gps_longitude":o.longitude,"gps_latitude":o.latitude,"candidate_arc_id":s.arc_id,"candidate_distance_m":d,"nearest_longitude":qx/scale,"nearest_latitude":qy/110540,"projection_fraction":t})
    return pd.DataFrame(out, columns=["observation_id","timestamp","gps_longitude","gps_latitude","candidate_arc_id","candidate_distance_m","nearest_longitude","nearest_latitude","projection_fraction"])
