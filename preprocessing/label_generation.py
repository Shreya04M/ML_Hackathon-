"""Conservative candidate labels based on reference-route membership."""
import pandas as pd

def generate_labels(candidates: pd.DataFrame, route: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    if route is None or route.empty or candidates.empty: return candidates.assign(label=pd.NA, label_status="unresolved"), {"validated_labels":0,"unresolved":len(candidates)}
    route_ids=list(route.arc_id.dropna().astype(int)); rset=set(route_ids); x=candidates.copy(); x["label"]=x.candidate_arc_id.map(lambda a: 1 if a in rset else pd.NA).astype("Int64"); x["label_status"]=x.label.map({1:"validated_positive"}).fillna("unresolved"); return x,{"validated_labels":int(x.label.notna().sum()),"positive_labels":int((x.label==1).sum()),"negative_labels":0,"unresolved":int(x.label.isna().sum())}
