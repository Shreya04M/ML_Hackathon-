"""Reproducible PathSnap preprocessing pipeline; never trains models."""
from pathlib import Path
import argparse
import pandas as pd
from .config import get_dataset_root
from .data_loader import discover_trips, load_trip
from .data_cleaning import validate_gps, validate_network, validate_route
from .road_network import load_road_network
from .candidate_generation import generate_candidates
from .feature_engineering import extract_features
from .label_generation import generate_labels

def run(dataset_root=None, output_dir="data/processed", results_dir="experiments/results", radius_m=50.0):
    root=Path(dataset_root) if dataset_root else get_dataset_root(); out=Path(output_dir); res=Path(results_dir); out.mkdir(parents=True,exist_ok=True); res.mkdir(parents=True,exist_ok=True)
    all_features={"train":[],"student_test":[]}; all_labels={"train":[],"student_test":[]}; coverage=[]; cleaning=[]; labels=[]
    for part in ("train","student_test"):
        p=root/part
        for trip_dir in discover_trips(p) if p.exists() else []:
            trip_id=trip_dir.name; d=load_trip(trip_dir); track=d["track"]
            if track is None: continue
            gps, flags=validate_gps(track); gps.insert(0,"trip_id",trip_id); flags.insert(0,"trip_id",trip_id); cleaning.append({"partition":part,"trip_id":trip_id,"observations":len(gps),"invalid":int((~gps.clean).sum()),"duplicates":int(gps.duplicate_observation.sum()),"timestamp_gaps":int(gps.timestamp_gap.sum())})
            if d["nodes"] is None or d["arcs"] is None:
                coverage.append({"partition":part,"trip_id":trip_id,"total_observations":len(gps),"observations_with_candidates":0,"observations_without_candidates":len(gps),"average_candidates":0,"network_available":False}); continue
            net=load_road_network(next((trip_dir.glob("*.nodes"))),next((trip_dir.glob("*.arcs"))))
            cand=generate_candidates(gps,net,radius_m=radius_m); cand.insert(0,"trip_id",trip_id); feats=extract_features(gps,cand,net); all_features[part].append(feats)
            lab,stat=generate_labels(cand,d["route"]); all_labels[part].append(lab); stat.update({"partition":part,"trip_id":trip_id}); labels.append(stat); n=len(gps); coverage.append({"partition":part,"trip_id":trip_id,"total_observations":n,"observations_with_candidates":cand.observation_id.nunique(),"observations_without_candidates":n-cand.observation_id.nunique(),"average_candidates":len(cand)/n if n else 0,"network_available":True})
    for part in ("train","student_test"):
        pd.concat(all_features[part],ignore_index=True).to_csv(out/f"{part}_features.csv",index=False) if all_features[part] else pd.DataFrame().to_csv(out/f"{part}_features.csv",index=False)
        pd.concat(all_labels[part],ignore_index=True).to_csv(out/f"{part}_candidate_labels.csv",index=False) if all_labels[part] else pd.DataFrame().to_csv(out/f"{part}_candidate_labels.csv",index=False)
    return {"cleaning":cleaning,"coverage":coverage,"labels":labels}

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root"); ap.add_argument("--output-dir",default="data/processed"); ap.add_argument("--results-dir",default="experiments/results"); ap.add_argument("--radius-m",type=float,default=50.0); a=ap.parse_args(); run(a.dataset_root,a.output_dir,a.results_dir,a.radius_m)
