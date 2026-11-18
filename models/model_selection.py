"""Group-aware cross-validation, 12-metric evaluation, and model selection engine.

Implements leakage-free trip-level GroupKFold cross-validation, candidate-level
and observation/route-level map-matching metrics, consolidated Section 3.1
comparison tables, and multi-criteria final model selection for PathSnap.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold


def _predict_positive_proba(model: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted binary labels and smooth positive-class probabilities."""
    if hasattr(model, "predict_proba"):
        proba_mat = np.asarray(model.predict_proba(X), dtype=np.float64)
        classes = list(getattr(model, "classes_", [0, 1]))
        pos_col = classes.index(1) if 1 in classes else -1
        y_proba = proba_mat[:, pos_col]
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        y_proba = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    else:
        y_proba = np.asarray(model.predict(X), dtype=np.float64)

    y_pred = (y_proba >= 0.5).astype(np.int64)
    return y_pred, np.clip(y_proba, 1e-15, 1.0 - 1e-15)


def _compute_route_level_metrics(
    meta_df: pd.DataFrame,
    y_proba: np.ndarray,
    road_graphs: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compute observation-level Top-1 accuracy, miss rate, route continuity, and sequence F1."""
    required = {"trip_id", "obs_id", "candidate_arc_id", "ref_arc_id"}
    if not required.issubset(meta_df.columns):
        return {
            "top1_accuracy": None,
            "candidate_miss_rate": None,
            "top1_accuracy_given_hit": None,
            "route_continuity": None,
            "sequence_f1_mean": None,
            "per_trip_top1_mean": None,
            "per_trip_top1_std": None,
            "n_observations": None,
            "n_trips": int(meta_df["trip_id"].nunique()) if "trip_id" in meta_df.columns else None,
        }

    work = pd.DataFrame(
        {
            "trip_id": meta_df["trip_id"].to_numpy(),
            "obs_id": meta_df["obs_id"].to_numpy(dtype=np.int64),
            "candidate_arc_id": meta_df["candidate_arc_id"].to_numpy(dtype=np.int64),
            "ref_arc_id": meta_df["ref_arc_id"].to_numpy(dtype=np.int64),
            "prob": y_proba,
        }
    )
    if "is_ref_in_candidates" in meta_df.columns:
        work["is_ref_in_candidates"] = meta_df["is_ref_in_candidates"].to_numpy(dtype=bool)
    else:
        work["obs_hit"] = work["candidate_arc_id"] == work["ref_arc_id"]
        hit_map = work.groupby(["trip_id", "obs_id"], sort=False)["obs_hit"].transform("any")
        work["is_ref_in_candidates"] = hit_map.to_numpy(dtype=bool)

    has_nodes = {"from_node_id", "to_node_id"}.issubset(meta_df.columns)
    if has_nodes:
        work["from_node_id"] = meta_df["from_node_id"].to_numpy(dtype=np.int64)
        work["to_node_id"] = meta_df["to_node_id"].to_numpy(dtype=np.int64)

    # Preserve input order on exact probability ties so lower-distance candidates win ties.
    work["_orig_order"] = np.arange(len(work), dtype=np.int64)
    work_sorted = work.sort_values(
        by=["trip_id", "obs_id", "prob", "_orig_order"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top1 = work_sorted.drop_duplicates(subset=["trip_id", "obs_id"], keep="first").reset_index(drop=True)

    n_obs = len(top1)
    hits = top1["is_ref_in_candidates"].to_numpy(dtype=bool)
    correct = (top1["candidate_arc_id"] == top1["ref_arc_id"]).to_numpy(dtype=bool)

    top1_acc = float(np.mean(correct))
    miss_rate = float(1.0 - np.mean(hits))
    top1_given_hit = float(np.mean(correct[hits])) if np.any(hits) else 0.0

    # Populate node endpoints from road_graphs if not directly attached to meta_df.
    if not has_nodes and road_graphs:
        from_arr = np.full(n_obs, -1, dtype=np.int64)
        to_arr = np.full(n_obs, -2, dtype=np.int64)
        for trip_id, idxs in top1.groupby("trip_id", sort=False).groups.items():
            g = road_graphs.get(str(trip_id))
            if g is not None:
                cands = top1.loc[idxs, "candidate_arc_id"].to_numpy(dtype=np.int64)
                from_arr[idxs] = g["from_node"][cands]
                to_arr[idxs] = g["to_node"][cands]
        top1["from_node_id"] = from_arr
        top1["to_node_id"] = to_arr
        has_nodes = True

    trip_ids_arr = top1["trip_id"].to_numpy()
    pred_arcs = top1["candidate_arc_id"].to_numpy(dtype=np.int64)
    ref_arcs = top1["ref_arc_id"].to_numpy(dtype=np.int64)

    continuity: float | None = None
    if has_nodes and n_obs > 1:
        from_nodes = top1["from_node_id"].to_numpy(dtype=np.int64)
        to_nodes = top1["to_node_id"].to_numpy(dtype=np.int64)
        same_trip = trip_ids_arr[:-1] == trip_ids_arr[1:]
        if np.any(same_trip):
            same_arc = pred_arcs[:-1] == pred_arcs[1:]
            direct_edge = to_nodes[:-1] == from_nodes[1:]
            valid_steps = same_trip
            connected = (same_arc | direct_edge)[valid_steps]
            continuity = float(np.mean(connected))

    # Per-trip Top-1 accuracy & deduplicated route-arc sequence F1.
    per_trip_top1: list[float] = []
    per_trip_seq_f1: list[float] = []
    for _, grp in top1.groupby("trip_id", sort=False):
        p_seq = grp["candidate_arc_id"].to_numpy(dtype=np.int64)
        r_seq = grp["ref_arc_id"].to_numpy(dtype=np.int64)
        per_trip_top1.append(float(np.mean(p_seq == r_seq)))

        p_set = set(p_seq.tolist())
        r_set = set(r_seq.tolist())
        inter = len(p_set & r_set)
        denom = len(p_set) + len(r_set)
        per_trip_seq_f1.append((2.0 * inter / denom) if denom > 0 else 0.0)

    return {
        "top1_accuracy": top1_acc,
        "candidate_miss_rate": miss_rate,
        "top1_accuracy_given_hit": top1_given_hit,
        "route_continuity": continuity,
        "sequence_f1_mean": float(np.mean(per_trip_seq_f1)) if per_trip_seq_f1 else None,
        "per_trip_top1_mean": float(np.mean(per_trip_top1)) if per_trip_top1 else None,
        "per_trip_top1_std": float(np.std(per_trip_top1)) if per_trip_top1 else None,
        "n_observations": int(n_obs),
        "n_trips": int(len(per_trip_top1)),
    }


def evaluate_model(
    model: Any,
    X: Any,
    y: Any,
    meta_df: pd.DataFrame | None = None,
    road_graphs: dict[str, dict[str, np.ndarray]] | None = None,
    train_time_s: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute all candidate-level, observation-level, route-level, and timing metrics."""
    y_true = np.asarray(y, dtype=np.int64)

    t0 = time.perf_counter()
    y_pred, y_proba = _predict_positive_proba(model, X)
    inference_time_s = float(time.perf_counter() - t0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    metrics: dict[str, Any] = {
        "n_candidates": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "train_time_s": float(train_time_s) if train_time_s is not None else None,
        "inference_time_s": inference_time_s,
    }

    if meta_df is not None:
        route_metrics = _compute_route_level_metrics(meta_df, y_proba, road_graphs=road_graphs)
        metrics.update(route_metrics)
        n_trips = route_metrics.get("n_trips")
        n_obs = route_metrics.get("n_observations")
        metrics["inference_ms_per_trip"] = (
            float(1000.0 * inference_time_s / n_trips) if n_trips else None
        )
        metrics["inference_ms_per_1k_obs"] = (
            float(1000.0 * inference_time_s / (n_obs / 1000.0)) if n_obs else None
        )
    else:
        metrics.update(
            {
                "top1_accuracy": None,
                "candidate_miss_rate": None,
                "top1_accuracy_given_hit": None,
                "route_continuity": None,
                "sequence_f1_mean": None,
                "per_trip_top1_mean": None,
                "per_trip_top1_std": None,
                "n_observations": None,
                "n_trips": None,
                "inference_ms_per_trip": None,
                "inference_ms_per_1k_obs": None,
            }
        )

    if kwargs.get("return_probabilities"):
        metrics["_y_proba"] = y_proba
    return metrics


def run_grouped_cross_validation(
    model_factory: Callable[[], Any],
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame | None = None,
    road_graphs: dict[str, dict[str, np.ndarray]] | None = None,
    n_splits: int = 5,
    on_fold_complete: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run leakage-free trip-grouped GroupKFold cross-validation on training trips."""
    y_arr = np.asarray(y, dtype=np.int64)
    groups_arr = np.asarray(groups)
    gkf = GroupKFold(n_splits=n_splits)

    oof_proba = np.zeros(len(y_arr), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []

    is_df = isinstance(X, pd.DataFrame)

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y_arr, groups_arr), start=1):
        X_tr = X.iloc[tr_idx] if is_df else X[tr_idx]
        X_val = X.iloc[val_idx] if is_df else X[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        meta_val = meta_df.iloc[val_idx].reset_index(drop=True) if meta_df is not None else None

        # Verify strict trip isolation across fold split.
        tr_trips = set(np.unique(groups_arr[tr_idx]).tolist())
        val_trips = set(np.unique(groups_arr[val_idx]).tolist())
        assert tr_trips.isdisjoint(val_trips), "Trip leakage detected across CV folds!"

        estimator = model_factory()
        t_fit0 = time.perf_counter()
        estimator.fit(X_tr, y_tr)
        fit_s = float(time.perf_counter() - t_fit0)

        fold_metrics = evaluate_model(
            estimator,
            X_val,
            y_val,
            meta_df=meta_val,
            road_graphs=road_graphs,
            train_time_s=fit_s,
            return_probabilities=True,
        )
        oof_proba[val_idx] = fold_metrics.pop("_y_proba")
        fold_metrics["fold"] = fold_idx
        fold_metrics["n_val_trips"] = len(val_trips)
        fold_records.append(fold_metrics)

        if on_fold_complete is not None:
            on_fold_complete(fold_idx, n_splits, fold_metrics)

    scalar_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "top1_accuracy",
        "candidate_miss_rate",
        "route_continuity",
        "sequence_f1_mean",
        "train_time_s",
        "inference_time_s",
    ]
    summary: dict[str, Any] = {"n_splits": n_splits, "folds": fold_records}
    for key in scalar_keys:
        vals = [f[key] for f in fold_records if f.get(key) is not None]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None

    # Also compute pooled out-of-fold evaluation across all 80 training trips.
    if meta_df is not None:
        oof_route = _compute_route_level_metrics(meta_df, oof_proba, road_graphs=road_graphs)
    else:
        oof_route = {}
    summary["oof_pooled_route_metrics"] = oof_route
    summary["_oof_proba"] = oof_proba
    return summary


def compare_models(results: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the consolidated Section 3.1 comparison DataFrame and compute selection ranking."""
    records = list(results.values()) if isinstance(results, dict) else list(results)
    rows: list[dict[str, Any]] = []
    for entry in records:
        cv = entry["cv"]
        st = entry["student_test"]
        tr = entry["train_fit"]
        rows.append(
            {
                "Model Key": entry["key"],
                "Model": entry["name"],
                "Algorithm Family": entry["algorithm_family"],
                "Key Configuration": entry["key_configuration"],
                "CV Macro F1-score": round(cv["f1_macro_mean"], 4),
                "CV Macro F1 Std": round(cv["f1_macro_std"], 4),
                "Top-1 Accuracy": round(st["top1_accuracy"], 4) if st["top1_accuracy"] is not None else "N/A",
                "Route continuity": round(st["route_continuity"], 4) if st["route_continuity"] is not None else "N/A",
                "Student-Test Macro F1-score": round(st["f1_macro"], 4),
                "Log Loss": round(st["log_loss"], 4),
                "Inference Time (s)": round(st["inference_time_s"], 4),
                "Inference (ms/trip)": round(st["inference_ms_per_trip"], 2) if st["inference_ms_per_trip"] is not None else "N/A",
                "Train Time (s)": round(tr["train_time_s"], 3),
                "Train Macro F1-score": round(tr["f1_macro"], 4),
                "Overfit Gap (Train-CV F1)": round(tr["f1_macro"] - cv["f1_macro_mean"], 4),
                "Student-Test Accuracy": round(st["accuracy"], 4),
                "Student-Test Precision (Macro)": round(st["precision_macro"], 4),
                "Student-Test Recall (Macro)": round(st["recall_macro"], 4),
                "Sequence F1": round(st["sequence_f1_mean"], 4) if st["sequence_f1_mean"] is not None else "N/A",
                "Candidate Miss Rate": round(st["candidate_miss_rate"], 4) if st["candidate_miss_rate"] is not None else "N/A",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["Student-Test Macro F1-score", "Log Loss", "CV Macro F1-score"],
            ascending=[False, True, False],
        ).reset_index(drop=True)
    return df


def select_best_model(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Select the optimal model using all six Section 6 criteria from the guidelines.

    Balances (1) map-matching generalisation (Student-Test Top-1 Accuracy, Sequence F1,
    Macro F1, CV Macro F1), (2) probability calibration (Log Loss for HMM emissions),
    (3) route continuity, (4) CV fold stability, (5) fast API inference latency, and
    (6) model complexity & Bayesian tuning cost (fit speed).
    """
    best_key: str | None = None
    best_score = -1e9
    scored_candidates: list[dict[str, Any]] = []

    log_losses = [r["student_test"]["log_loss"] for r in results.values()]
    max_ll = max(log_losses) if log_losses else 1.0
    min_ll = min(log_losses) if log_losses else 0.0
    ll_span = max(max_ll - min_ll, 1e-6)

    inf_times = [r["student_test"]["inference_time_s"] for r in results.values()]
    max_inf = max(inf_times) if inf_times else 1.0

    train_times = [r["train_fit"]["train_time_s"] or 0.0 for r in results.values()]
    max_tr = max(train_times) if train_times else 1.0

    for key, entry in results.items():
        cv_f1 = entry["cv"]["f1_macro_mean"]
        cv_std = entry["cv"]["f1_macro_std"]
        st_f1 = entry["student_test"]["f1_macro"]
        st_top1 = entry["student_test"]["top1_accuracy"] or 0.0
        st_seq = entry["student_test"].get("sequence_f1_mean") or st_f1
        st_cont = entry["student_test"]["route_continuity"] or 0.0
        st_ll = entry["student_test"]["log_loss"]
        st_inf = entry["student_test"]["inference_time_s"]
        tr_time = entry["train_fit"]["train_time_s"] or 0.0

        norm_ll_quality = 1.0 - (st_ll - min_ll) / ll_span
        norm_inf_speed = 1.0 - min(st_inf / max(max_inf, 1e-6), 1.0)
        norm_fit_efficiency = 1.0 - min(tr_time / max(max_tr, 1e-6), 1.0)
        stability = max(0.0, 1.0 - 10.0 * cv_std)

        composite = (
            0.25 * st_top1
            + 0.20 * st_seq
            + 0.18 * st_f1
            + 0.14 * cv_f1
            + 0.10 * norm_ll_quality
            + 0.05 * st_cont
            + 0.05 * norm_fit_efficiency
            + 0.02 * norm_inf_speed
            + 0.01 * stability
        )
        scored_candidates.append({"key": key, "name": entry["name"], "composite_score": round(composite, 5)})
        if composite > best_score:
            best_score = composite
            best_key = key

    scored_candidates.sort(key=lambda d: d["composite_score"], reverse=True)
    return {
        "selected_key": best_key,
        "selected_name": results[best_key]["name"] if best_key else None,
        "ranking": scored_candidates,
    }


def create_bayesian_cv_objective(
    model_factory: Callable[..., Any],
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    metric: str = "f1_macro",
) -> Callable[[dict[str, Any]], float]:
    """Create a GroupKFold CV objective strictly for Bayesian hyperparameter optimisation.

    Designed exclusively for Bayesian surrogate optimisers (e.g., GP-EI or Optuna
    TPE) in the PathSnap tuning pipeline without grid or random search.
    """
    y_arr = np.asarray(y, dtype=np.int64)
    groups_arr = np.asarray(groups)
    splits = list(GroupKFold(n_splits=n_splits).split(X, y_arr, groups_arr))
    is_df = isinstance(X, pd.DataFrame)

    def objective(params: dict[str, Any]) -> float:
        fold_scores: list[float] = []
        for tr_idx, val_idx in splits:
            X_tr = X.iloc[tr_idx] if is_df else X[tr_idx]
            X_val = X.iloc[val_idx] if is_df else X[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

            estimator = model_factory(**params)
            estimator.fit(X_tr, y_tr)
            y_pred, y_proba = _predict_positive_proba(estimator, X_val)
            if metric == "f1_macro":
                fold_scores.append(float(f1_score(y_val, y_pred, average="macro", zero_division=0)))
            elif metric == "neg_log_loss":
                fold_scores.append(-float(log_loss(y_val, y_proba, labels=[0, 1])))
            else:
                raise ValueError(f"Unsupported Bayesian CV objective metric: {metric}")
        return float(np.mean(fold_scores))

    return objective
