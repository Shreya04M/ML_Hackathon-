"""Binary candidate labels based on reference-route membership."""
import pandas as pd

def generate_labels(candidates: pd.DataFrame, route: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Assign binary labels when a reference route is available.

    A candidate is positive when its arc occurs in the reference route and
    negative otherwise. Trips without a route remain unresolved because there
    is no ground truth from which to derive either class.
    """
    if route is None or route.empty or candidates.empty:
        result = candidates.assign(label=pd.Series(pd.NA, index=candidates.index, dtype="Int64"), label_status="unresolved")
        return result, {"validated_labels": 0, "positive_labels": 0, "negative_labels": 0, "unresolved": len(candidates)}

    route_ids = set(route.arc_id.dropna().astype(int))
    result = candidates.copy()
    result["label"] = result["candidate_arc_id"].map(lambda arc_id: int(arc_id in route_ids)).astype("Int64")
    result["label_status"] = result["label"].map({1: "validated_positive", 0: "validated_negative"})
    return result, {
        "validated_labels": int(result["label"].notna().sum()),
        "positive_labels": int((result["label"] == 1).sum()),
        "negative_labels": int((result["label"] == 0).sum()),
        "unresolved": int(result["label"].isna().sum()),
    }
