"""End-to-end training orchestrator, artifact persistence, and Member 3/4 handoffs.

Trains the five core PathSnap algorithm families on the 80 training trips using
leakage-free 5-fold trip-grouped cross-validation (GroupKFold), evaluates all models
on the 10 held-out student-testing trips, generates all Section 3.3 visualisations,
persists compressed model artifacts and experiment logs, and exports calibrated
candidate probabilities for Member 3's Bayesian Optimisation & HMM/Viterbi pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from models.decision_tree import get_decision_tree_spec
from models.gradient_boosting import (
    extract_training_history as extract_hgb_history,
    get_gradient_boosting_spec,
)
from models.knn import get_knn_spec
from models.logistic_regression import get_logistic_regression_spec
from models.mlp import extract_training_history as extract_mlp_history, get_mlp_spec
from models.model_selection import (
    _predict_positive_proba,
    compare_models,
    evaluate_model,
    run_grouped_cross_validation,
    select_best_model,
)
from models.plots import generate_all_plots
from models.random_forest import get_random_forest_spec
from models.schema_adapter import (
    DEFAULT_RAW_ROOT,
    FEATURE_COLUMNS,
    build_partition_dataset,
    load_feature_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("models") / "artifacts"


def get_model_specs(include_mlp: bool = False) -> list[dict[str, Any]]:
    """Return specifications for the five core Phase 1 models (and optionally MLP)."""
    specs = [
        get_logistic_regression_spec(),
        get_knn_spec(),
        get_decision_tree_spec(),
        get_random_forest_spec(),
        get_gradient_boosting_spec(),
    ]
    if include_mlp:
        specs.append(get_mlp_spec())
    return specs


def save_final_model(model: Any, path: str | Path) -> None:
    """Persist a trained model pipeline cleanly with compression."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path, compress=3)


def load_final_model(path: str | Path) -> Any:
    """Load a persisted Member 2 model pipeline from disk."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Saved model artifact not found: {in_path}")
    return joblib.load(in_path)


def predict_candidate_probabilities(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict positive-class candidate probabilities P(match=1 | x) for Member 3/4."""
    _, proba = _predict_positive_proba(model, X)
    return proba


def _extract_feature_importance_vector(
    model_key: str,
    model: Any,
    X_sample: pd.DataFrame,
    y_sample: np.ndarray,
) -> np.ndarray | None:
    """Extract normalized feature importance or absolute coefficient magnitudes."""
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_, dtype=np.float64)
    if hasattr(clf, "coef_"):
        return np.abs(np.asarray(clf.coef_, dtype=np.float64).ravel())
    if model_key == "hist_gradient_boosting":
        perm = permutation_importance(
            model,
            X_sample,
            y_sample,
            scoring="f1_macro",
            n_repeats=3,
            random_state=42,
            n_jobs=-1,
        )
        return np.maximum(perm.importances_mean, 0.0)
    return None


def _compute_per_trip_breakdown(
    meta_df: pd.DataFrame,
    y_proba: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Compute per-trip Top-1 accuracy, miss rate, and continuity across student_test trips."""
    work = meta_df[["trip_id", "obs_id", "candidate_arc_id", "from_node_id", "to_node_id", "ref_arc_id", "is_ref_in_candidates"]].copy()
    work["prob"] = y_proba
    work["_ord"] = np.arange(len(work))
    top1 = (
        work.sort_values(["trip_id", "obs_id", "prob", "_ord"], ascending=[True, True, False, True], kind="mergesort")
        .drop_duplicates(["trip_id", "obs_id"], keep="first")
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for trip_id, grp in top1.groupby("trip_id", sort=True):
        grp = grp.sort_values("obs_id")
        pred = grp["candidate_arc_id"].to_numpy()
        ref = grp["ref_arc_id"].to_numpy()
        hits = grp["is_ref_in_candidates"].to_numpy(dtype=bool)
        fn = grp["from_node_id"].to_numpy()
        tn = grp["to_node_id"].to_numpy()
        cont = float(np.mean((pred[:-1] == pred[1:]) | (tn[:-1] == fn[1:]))) if len(pred) > 1 else 1.0
        rows.append(
            {
                "model": model_name,
                "trip_id": str(trip_id),
                "n_obs": int(len(grp)),
                "top1_accuracy": round(float(np.mean(pred == ref)), 4),
                "candidate_miss_rate": round(float(1.0 - np.mean(hits)), 4),
                "route_continuity": round(cont, 4),
            }
        )
    return pd.DataFrame(rows)


def _format_markdown_comparison_table(df: pd.DataFrame) -> str:
    """Render the exact Deliverables.pdf Section 3.1 markdown comparison table."""
    cols = [
        "Model",
        "Algorithm Family",
        "Key Configuration",
        "CV Macro F1-score",
        "Top-1 Accuracy",
        "Route continuity",
        "Student-Test Macro F1-score",
        "Log Loss",
        "Inference Time (s)",
    ]
    sub = df[cols].copy()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_all_experiments(
    data_root: Path | str = DEFAULT_RAW_ROOT,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    train_source: Path | str | None = None,
    test_source: Path | str | None = None,
    top_k: int = 6,
    n_splits: int = 5,
    include_mlp: bool = False,
) -> dict[str, Any]:
    """Execute full Member 2 model training, grouped CV, evaluation, plotting, and export."""
    artifacts_root = Path(artifacts_dir)
    models_dir = artifacts_root / "models"
    results_dir = artifacts_root / "results"
    plots_dir = artifacts_root / "plots"
    for d in (models_dir, results_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets (either Member 1 feature files or standalone spatial extraction).
    if train_source is not None and test_source is not None:
        logger.info("Loading Member 1 external datasets: train=%s, test=%s", train_source, test_source)
        X_train, y_train, meta_train = load_feature_dataset(train_source)
        X_test, y_test, meta_test = load_feature_dataset(test_source)
        train_graphs, test_graphs = {}, {}
    else:
        logger.info("Building/loading spatial candidate datasets (K=%d) from %s", top_k, data_root)
        X_train, y_train, meta_train, train_graphs = build_partition_dataset(
            partition="train",
            data_root=data_root,
            top_k=top_k,
            cache_dir=artifacts_root / "cache",
            use_cache=True,
        )
        X_test, y_test, meta_test, test_graphs = build_partition_dataset(
            partition="student_test",
            data_root=data_root,
            top_k=top_k,
            cache_dir=artifacts_root / "cache",
            use_cache=True,
        )

    groups_train = meta_train["trip_id"].to_numpy()
    feature_names = list(X_train.columns)
    logger.info(
        "Dataset ready | train=%s (trips=%d, pos=%.2f%%) | student_test=%s (trips=%d, pos=%.2f%%)",
        X_train.shape,
        pd.Series(groups_train).nunique(),
        100.0 * float(np.mean(y_train)),
        X_test.shape,
        meta_test["trip_id"].nunique(),
        100.0 * float(np.mean(y_test)),
    )

    specs = get_model_specs(include_mlp=include_mlp)
    all_results: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    importances_by_model: dict[str, np.ndarray] = {}
    per_trip_frames: list[pd.DataFrame] = []
    student_test_probas: dict[str, np.ndarray] = {}

    # Fixed 15,000-row sample of student_test for fast permutation importance when needed.
    rng = np.random.default_rng(42)
    sample_n = min(15_000, len(X_test))
    sample_idx = np.sort(rng.choice(len(X_test), size=sample_n, replace=False))
    X_imp_sample = X_test.iloc[sample_idx]
    y_imp_sample = y_test[sample_idx]

    for spec in specs:
        key = spec["key"]
        name = spec["name"]
        logger.info("=== Running 5-fold GroupKFold CV & Student-Test Evaluation: %s ===", name)

        cv_res = run_grouped_cross_validation(
            model_factory=spec["factory"],
            X=X_train,
            y=y_train,
            groups=groups_train,
            meta_df=meta_train,
            road_graphs=train_graphs,
            n_splits=n_splits,
        )
        cv_res.pop("_oof_proba", None)

        # Fit final untuned model on all 80 training trips.
        model = spec["factory"]()
        t_fit0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time_s = float(time.perf_counter() - t_fit0)

        train_fit_metrics = evaluate_model(
            model,
            X_train,
            y_train,
            meta_df=meta_train,
            road_graphs=train_graphs,
            train_time_s=train_time_s,
        )

        st_metrics = evaluate_model(
            model,
            X_test,
            y_test,
            meta_df=meta_test,
            road_graphs=test_graphs,
            train_time_s=train_time_s,
            return_probabilities=True,
        )
        st_proba = st_metrics.pop("_y_proba")
        student_test_probas[key] = st_proba

        if {"from_node_id", "to_node_id"}.issubset(meta_test.columns):
            per_trip_frames.append(_compute_per_trip_breakdown(meta_test, st_proba, name))

        # Record genuine iterative history only when supported by the algorithm.
        if key == "hist_gradient_boosting":
            hist = extract_hgb_history(model)
            if hist:
                histories[name] = hist
        elif key == "mlp":
            hist = extract_mlp_history(model)
            if hist:
                histories[name] = hist

        imp = _extract_feature_importance_vector(key, model, X_imp_sample, y_imp_sample)
        if imp is not None:
            importances_by_model[name] = imp

        model_path = models_dir / f"{key}.joblib"
        save_final_model(model, model_path)
        trained_models[key] = model

        logger.info(
            "%s | CV Macro-F1=%.4f (±%.4f) | Test Top-1=%.4f | Test Macro-F1=%.4f | LogLoss=%.4f | Continuity=%.4f | Inf=%.3fs",
            name,
            cv_res["f1_macro_mean"],
            cv_res["f1_macro_std"],
            st_metrics["top1_accuracy"] or 0.0,
            st_metrics["f1_macro"],
            st_metrics["log_loss"],
            st_metrics["route_continuity"] or 0.0,
            st_metrics["inference_time_s"],
        )

        all_results[key] = {
            "key": key,
            "name": name,
            "algorithm_family": spec["algorithm_family"],
            "key_configuration": spec["key_configuration"],
            "default_params": spec["default_params"],
            "loss_objective": spec["loss_objective"],
            "produces_training_history": spec["produces_training_history"],
            "model_artifact_path": str(model_path),
            "cv": cv_res,
            "train_fit": train_fit_metrics,
            "student_test": st_metrics,
        }

    # 2. Compare models and select the winner based on multi-criteria generalisation & calibration.
    comparison_df = compare_models(all_results)
    selection = select_best_model(all_results)
    selected_key = selection["selected_key"]
    selected_model = trained_models[selected_key]

    best_model_path = models_dir / "selected_best_model.joblib"
    save_final_model(selected_model, best_model_path)

    selected_meta = {
        "selected_key": selected_key,
        "selected_name": selection["selected_name"],
        "ranking": selection["ranking"],
        "feature_columns": feature_names,
        "top_k": top_k,
        "config": all_results[selected_key]["default_params"],
        "cv_summary": {
            "f1_macro_mean": all_results[selected_key]["cv"]["f1_macro_mean"],
            "f1_macro_std": all_results[selected_key]["cv"]["f1_macro_std"],
            "log_loss_mean": all_results[selected_key]["cv"]["log_loss_mean"],
            "top1_accuracy_mean": all_results[selected_key]["cv"]["top1_accuracy_mean"],
        },
        "student_test_summary": all_results[selected_key]["student_test"],
    }
    with open(models_dir / "selected_model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(selected_meta, fh, indent=2)

    # 3. Export comparison tables, detailed JSON logs, per-trip diagnostics, and M3 handoff probabilities.
    comparison_df.to_csv(results_dir / "section_3_1_model_comparison.csv", index=False)
    md_table = _format_markdown_comparison_table(comparison_df)
    (results_dir / "section_3_1_model_comparison.md").write_text(md_table + "\n", encoding="utf-8")

    if per_trip_frames:
        per_trip_df = pd.concat(per_trip_frames, ignore_index=True)
        per_trip_df.to_csv(results_dir / "per_trip_student_test_breakdown.csv", index=False)

    handoff_df = meta_test.copy()
    handoff_df["label"] = y_test
    best_proba = student_test_probas[selected_key]
    handoff_df["pred_prob"] = best_proba
    handoff_df["pred_log_prob"] = np.log(np.clip(best_proba, 1e-15, 1.0))
    handoff_df.to_csv(results_dir / "m3_handoff_student_test_probabilities.csv.gz", index=False, compression="gzip")

    detailed_payload = {
        "dataset_summary": {
            "train_rows": int(len(X_train)),
            "train_trips": int(pd.Series(groups_train).nunique()),
            "train_positive_rate": float(np.mean(y_train)),
            "student_test_rows": int(len(X_test)),
            "student_test_trips": int(meta_test["trip_id"].nunique()),
            "student_test_positive_rate": float(np.mean(y_test)),
            "feature_columns": feature_names,
            "top_k": top_k,
        },
        "selection": selection,
        "models": all_results,
        "training_histories": histories,
    }
    with open(results_dir / "detailed_experiment_results.json", "w", encoding="utf-8") as fh:
        json.dump(detailed_payload, fh, indent=2)

    # 4. Generate all compulsory Section 3.3 visualisations.
    saved_plots = generate_all_plots(
        results=all_results,
        histories=histories,
        feature_names=feature_names,
        importances_by_model=importances_by_model,
        plots_dir=plots_dir,
    )
    logger.info("Saved %d Section 3.3 visualisations to %s", len(saved_plots), plots_dir)
    logger.info("Selected Best Model: %s (%s)", selection["selected_name"], best_model_path)

    return {
        "comparison_table": comparison_df,
        "selection": selection,
        "results": all_results,
        "plots": saved_plots,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PathSnap Member 2 Model Experimentation & Selection")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_RAW_ROOT), help="Path to Student folder raw trips")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Output directory for models/results/plots")
    parser.add_argument("--train-source", type=str, default=None, help="Optional Member 1 processed train feature file")
    parser.add_argument("--test-source", type=str, default=None, help="Optional Member 1 processed student_test feature file")
    parser.add_argument("--top-k", type=int, default=6, help="Candidates per GPS observation (for standalone spatial extraction)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold CV splits")
    parser.add_argument("--include-mlp", action="store_true", help="Also include MLPClassifier alongside the 5 Phase 1 models")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = run_all_experiments(
        data_root=args.data_root,
        artifacts_dir=args.artifacts_dir,
        train_source=args.train_source,
        test_source=args.test_source,
        top_k=args.top_k,
        n_splits=args.n_splits,
        include_mlp=args.include_mlp,
    )
    print("\n=== Section 3.1 Consolidated Comparison Table ===")
    print(out["comparison_table"].to_string(index=False))
    print("\n=== Final Model Selection Ranking ===")
    print(json.dumps(out["selection"], indent=2))
