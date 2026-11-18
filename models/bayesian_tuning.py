"""Gaussian-Process Bayesian Hyperparameter Optimisation for the selected model.

Provides a strictly Bayesian (GP-Matern surrogate + Expected Improvement acquisition)
optimisation script over the Member 2 selected HistGradientBoosting model using
trip-grouped GroupKFold cross-validation on the 80 training trips only (zero grid
search or random search, zero evaluator-test leakage).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from models.gradient_boosting import DEFAULT_CONFIG, create_gradient_boosting_pipeline
from models.model_selection import _predict_positive_proba, evaluate_model, run_grouped_cross_validation
from models.schema_adapter import build_partition_dataset
from models.train import _select_trip_stratified_train_diag_slice, save_final_model

# Justified 5D continuous/integer search domain for HistGradientBoosting:
SEARCH_BOUNDS: list[dict[str, Any]] = [
    {"name": "log10_learning_rate", "low": np.log10(0.03), "high": np.log10(0.18), "kind": "log10"},
    {"name": "max_leaf_nodes", "low": 31.0, "high": 127.0, "kind": "int"},
    {"name": "max_depth", "low": 8.0, "high": 18.0, "kind": "int"},
    {"name": "min_samples_leaf", "low": 15.0, "high": 80.0, "kind": "int"},
    {"name": "log10_l2_reg", "low": np.log10(0.1), "high": np.log10(10.0), "kind": "log10"},
]


def _unit_to_params(u: np.ndarray) -> dict[str, Any]:
    """Map normalized [0, 1]^5 vector into HistGradientBoosting hyperparameters."""
    vals: dict[str, Any] = {}
    for idx, spec in enumerate(SEARCH_BOUNDS):
        raw = spec["low"] + float(np.clip(u[idx], 0.0, 1.0)) * (spec["high"] - spec["low"])
        if spec["kind"] == "log10":
            param_name = "learning_rate" if "learning_rate" in spec["name"] else "l2_regularization"
            vals[param_name] = round(float(10.0**raw), 5)
        elif spec["kind"] == "int":
            vals[spec["name"]] = int(round(raw))
    vals["max_iter"] = 250
    vals["class_weight"] = "balanced"
    vals["early_stopping"] = True
    vals["validation_fraction"] = 0.1
    vals["n_iter_no_change"] = 15
    vals["random_state"] = 42
    return vals


def _params_to_unit(params: dict[str, Any]) -> np.ndarray:
    """Map HistGradientBoosting hyperparameter dict back into [0, 1]^5 unit cube."""
    u = np.zeros(len(SEARCH_BOUNDS), dtype=np.float64)
    for idx, spec in enumerate(SEARCH_BOUNDS):
        if spec["kind"] == "log10":
            pname = "learning_rate" if "learning_rate" in spec["name"] else "l2_regularization"
            val = np.log10(float(params[pname]))
        else:
            val = float(params[spec["name"]])
        u[idx] = np.clip((val - spec["low"]) / (spec["high"] - spec["low"]), 0.0, 1.0)
    return u


def _expected_improvement(
    candidates_u: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.002,
) -> np.ndarray:
    """Compute Gaussian-Process Expected Improvement acquisition score."""
    pred = gp.predict(candidates_u, return_std=True)
    mu = np.asarray(pred[0], dtype=np.float64)
    sigma = np.maximum(np.asarray(pred[1], dtype=np.float64), 1e-9)
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-8] = 0.0
    return ei


def run_bayesian_optimisation(
    n_init: int = 3,
    n_bayes_iters: int = 9,
    n_splits: int = 5,
    artifacts_dir: Path | str = Path("models") / "artifacts",
) -> dict[str, Any]:
    """Run GP-EI Bayesian optimisation over 80 training trips and compare Tuned vs Untuned."""
    t_total0 = time.perf_counter()
    art_root = Path(artifacts_dir)
    results_dir = art_root / "results"
    models_dir = art_root / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, meta_tr, graphs_tr = build_partition_dataset("train", top_k=6, use_cache=True)
    X_te, y_te, meta_te, graphs_te = build_partition_dataset("student_test", top_k=6, use_cache=True)

    # Stratified 80-trip representative slice (~113.9k rows) for rapid 5-fold Bayesian trials.
    diag_idx = _select_trip_stratified_train_diag_slice(meta_tr, max_rows=125_000)
    X_opt = X_tr.iloc[diag_idx].reset_index(drop=True)
    y_opt = y_tr[diag_idx]
    groups_opt = meta_tr.iloc[diag_idx]["trip_id"].to_numpy()
    splits = list(GroupKFold(n_splits=n_splits).split(X_opt, y_opt, groups_opt))

    def eval_trial(params: dict[str, Any]) -> tuple[float, float]:
        t0 = time.perf_counter()
        fold_f1: list[float] = []
        for tr_i, val_i in splits:
            est = create_gradient_boosting_pipeline(**params)
            est.fit(X_opt.iloc[tr_i], y_opt[tr_i])
            y_pred, _ = _predict_positive_proba(est, X_opt.iloc[val_i])
            fold_f1.append(float(f1_score(y_opt[val_i], y_pred, average="macro", zero_division=0)))
        return float(np.mean(fold_f1)), float(time.perf_counter() - t0)

    rng = np.random.default_rng(42)
    observed_u: list[np.ndarray] = []
    observed_y: list[float] = []
    trial_records: list[dict[str, Any]] = []

    # Initial design: include untuned default as Trial 1 + space-filling anchor points.
    init_vectors: list[np.ndarray] = [_params_to_unit(DEFAULT_CONFIG)]
    for _ in range(max(0, n_init - 1)):
        init_vectors.append(rng.uniform(0.05, 0.95, size=len(SEARCH_BOUNDS)))

    for trial_idx, u in enumerate(init_vectors, start=1):
        params = _unit_to_params(u) if trial_idx > 1 else dict(DEFAULT_CONFIG)
        score, elapsed = eval_trial(params)
        observed_u.append(u)
        observed_y.append(score)
        trial_records.append(
            {
                "trial": trial_idx,
                "phase": "initial_anchor",
                "cv_macro_f1_surrogate": round(score, 5),
                "trial_time_s": round(elapsed, 2),
                **{k: params[k] for k in ("learning_rate", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization", "max_iter")},
            }
        )
        print(f"Trial {trial_idx:02d} [Init ] | Surrogate CV Macro-F1={score:.5f} | lr={params['learning_rate']} leaves={params['max_leaf_nodes']} depth={params['max_depth']} min_leaf={params['min_samples_leaf']} l2={params['l2_regularization']}")

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(len(SEARCH_BOUNDS)), nu=2.5) + WhiteKernel(noise_level=1e-5)
    for step in range(1, n_bayes_iters + 1):
        trial_idx = len(init_vectors) + step
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=42 + step)
        gp.fit(np.vstack(observed_u), np.asarray(observed_y))

        pool_u = rng.uniform(0.0, 1.0, size=(4000, len(SEARCH_BOUNDS)))
        ei_vals = _expected_improvement(pool_u, gp, y_best=max(observed_y))
        next_u = pool_u[int(np.argmax(ei_vals))]
        params = _unit_to_params(next_u)

        score, elapsed = eval_trial(params)
        observed_u.append(next_u)
        observed_y.append(score)
        trial_records.append(
            {
                "trial": trial_idx,
                "phase": "gp_expected_improvement",
                "cv_macro_f1_surrogate": round(score, 5),
                "trial_time_s": round(elapsed, 2),
                **{k: params[k] for k in ("learning_rate", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization", "max_iter")},
            }
        )
        print(f"Trial {trial_idx:02d} [GP-EI] | Surrogate CV Macro-F1={score:.5f} | lr={params['learning_rate']} leaves={params['max_leaf_nodes']} depth={params['max_depth']} min_leaf={params['min_samples_leaf']} l2={params['l2_regularization']}")

    best_trial_idx = int(np.argmax(observed_y))
    best_record = trial_records[best_trial_idx]
    best_params = {
        "loss": "log_loss",
        "learning_rate": float(best_record["learning_rate"]),
        "max_leaf_nodes": int(best_record["max_leaf_nodes"]),
        "max_depth": int(best_record["max_depth"]),
        "min_samples_leaf": int(best_record["min_samples_leaf"]),
        "l2_regularization": float(best_record["l2_regularization"]),
        "max_iter": int(best_record["max_iter"]),
        "class_weight": "balanced",
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "scoring": "loss",
        "random_state": 42,
    }

    # Verify best configuration with full 5-fold GroupKFold CV on all 1,250,502 training rows
    # and refit on all 80 training trips for student_test evaluation.
    tuned_cv = run_grouped_cross_validation(
        model_factory=lambda: create_gradient_boosting_pipeline(**best_params),
        X=X_tr,
        y=y_tr,
        groups=meta_tr["trip_id"].to_numpy(),
        meta_df=meta_tr,
        road_graphs=graphs_tr,
        n_splits=n_splits,
    )
    tuned_cv.pop("_oof_proba", None)

    tuned_model = create_gradient_boosting_pipeline(**best_params)
    t_fit0 = time.perf_counter()
    tuned_model.fit(X_tr, y_tr)
    tuned_fit_s = float(time.perf_counter() - t_fit0)

    tuned_test = evaluate_model(
        tuned_model,
        X_te,
        y_te,
        meta_df=meta_te,
        road_graphs=graphs_te,
        train_time_s=tuned_fit_s,
    )

    tuned_model_path = models_dir / "selected_best_model_tuned.joblib"
    save_final_model(tuned_model, tuned_model_path)

    # Load untuned baseline metrics from detailed_experiment_results.json for exact apple-to-apple comparison.
    det_path = results_dir / "detailed_experiment_results.json"
    det = json.loads(det_path.read_text(encoding="utf-8"))
    untuned = det["models"]["hist_gradient_boosting"]

    comp_rows = [
        {
            "Variant": "Untuned HistGradientBoosting (Baseline)",
            "CV Macro F1-score": round(untuned["cv"]["f1_macro_mean"], 4),
            "CV Macro F1 Std": round(untuned["cv"]["f1_macro_std"], 4),
            "Student-Test Top-1 Accuracy": round(untuned["student_test"]["top1_accuracy"], 4),
            "Student-Test Macro F1-score": round(untuned["student_test"]["f1_macro"], 4),
            "Log Loss": round(untuned["student_test"]["log_loss"], 4),
            "Route Continuity": round(untuned["student_test"]["route_continuity"], 4),
            "Sequence F1": round(untuned["student_test"]["sequence_f1_mean"], 4),
            "Inference Time (s)": round(untuned["student_test"]["inference_time_s"], 4),
            "Train Time (s)": round(untuned["train_fit"]["train_time_s"], 2),
        },
        {
            "Variant": "Bayesian-Tuned HistGradientBoosting (GP-EI)",
            "CV Macro F1-score": round(tuned_cv["f1_macro_mean"], 4),
            "CV Macro F1 Std": round(tuned_cv["f1_macro_std"], 4),
            "Student-Test Top-1 Accuracy": round(tuned_test["top1_accuracy"], 4),
            "Student-Test Macro F1-score": round(tuned_test["f1_macro"], 4),
            "Log Loss": round(tuned_test["log_loss"], 4),
            "Route Continuity": round(tuned_test["route_continuity"], 4),
            "Sequence F1": round(tuned_test["sequence_f1_mean"], 4),
            "Inference Time (s)": round(tuned_test["inference_time_s"], 4),
            "Train Time (s)": round(tuned_fit_s, 2),
        },
    ]

    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv(results_dir / "bayesian_optimisation_trials.csv", index=False)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(results_dir / "tuned_vs_untuned_comparison.csv", index=False)

    summary_payload = {
        "strategy": "Gaussian Process Surrogate (Matern nu=2.5) + Expected Improvement (Strict Bayesian Optimisation, zero Grid/Random search)",
        "n_trials": len(trial_records),
        "best_trial": int(best_record["trial"]),
        "best_params": best_params,
        "total_optimisation_wall_time_s": round(time.perf_counter() - t_total0, 2),
        "tuned_cv_summary": {
            "f1_macro_mean": tuned_cv["f1_macro_mean"],
            "f1_macro_std": tuned_cv["f1_macro_std"],
            "log_loss_mean": tuned_cv["log_loss_mean"],
            "top1_accuracy_mean": tuned_cv["top1_accuracy_mean"],
        },
        "tuned_student_test": tuned_test,
        "comparison": comp_rows,
    }
    (results_dir / "tuned_vs_untuned_comparison.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print("\n=== Bayesian Tuned vs Untuned Comparison ===")
    print(comp_df.to_string(index=False))
    return summary_payload


if __name__ == "__main__":
    run_bayesian_optimisation()

