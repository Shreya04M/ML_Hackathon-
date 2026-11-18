"""Publication-quality Section 3.3 diagnostic and comparison visualisations.

Generates plots exclusively from measured cross-validation, student-test,
confusion-matrix, latency, and genuine iterative training history records.
Never fabricates training curves for non-iterative algorithms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="muted")

MODEL_COLORS = {
    "logistic_regression": "#4C72B0",
    "knn": "#DD8452",
    "decision_tree": "#C44E52",
    "random_forest": "#55A868",
    "hist_gradient_boosting": "#8172B3",
    "mlp": "#937860",
}


def _ordered_entries(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results.values(),
        key=lambda r: (r["student_test"]["f1_macro"], -r["student_test"]["log_loss"]),
        reverse=True,
    )


def plot_cv_macro_f1_comparison(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Plot 5-fold GroupKFold CV Macro F1 mean ± std with individual fold markers."""
    entries = _ordered_entries(results)
    names = [e["name"] for e in entries]
    means = [e["cv"]["f1_macro_mean"] for e in entries]
    stds = [e["cv"]["f1_macro_std"] for e in entries]
    colors = [MODEL_COLORS.get(e["key"], "#4C72B0") for e in entries]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(entries))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.8)

    for idx, entry in enumerate(entries):
        fold_vals = [f["f1_macro"] for f in entry["cv"]["folds"]]
        jitter = np.linspace(-0.12, 0.12, len(fold_vals))
        ax.scatter(np.full(len(fold_vals), idx) + jitter, fold_vals, color="#111111", s=26, zorder=4, alpha=0.8)
        ax.text(
            idx,
            means[idx] + stds[idx] + 0.012,
            f"{means[idx]:.4f}\n±{stds[idx]:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Macro F1-score")
    ax.set_title("5-Fold Trip-Grouped Cross-Validation Macro F1 Comparison (80 Training Trips)")
    ax.set_ylim(0.0, min(1.08, max(means) + 0.14))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_student_test_metrics(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Plot Student-Test Top-1 Accuracy, Macro F1-score, and Route Continuity."""
    entries = _ordered_entries(results)
    names = [e["name"] for e in entries]
    top1 = [e["student_test"].get("top1_accuracy") or 0.0 for e in entries]
    f1 = [e["student_test"]["f1_macro"] for e in entries]
    cont = [e["student_test"].get("route_continuity") or 0.0 for e in entries]

    x = np.arange(len(entries))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - width, top1, width, label="Observation Top-1 Accuracy", color="#2B6CB0", edgecolor="#1A202C", linewidth=0.6)
    b2 = ax.bar(x, f1, width, label="Candidate Macro F1-score", color="#2F855A", edgecolor="#1A202C", linewidth=0.6)
    b3 = ax.bar(x + width, cont, width, label="Route Continuity (Classifier-Only)", color="#DD6B20", edgecolor="#1A202C", linewidth=0.6)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.set_title("Student-Testing Performance Comparison (10 Held-Out Trips, 19,160 GPS Observations)")
    ax.set_ylim(0.0, 1.09)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_log_loss_comparison(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Plot CV vs Student-Test Log Loss (lower indicates better calibrated emissions)."""
    entries = sorted(results.values(), key=lambda r: r["student_test"]["log_loss"])
    names = [e["name"] for e in entries]
    cv_ll = [e["cv"]["log_loss_mean"] for e in entries]
    st_ll = [e["student_test"]["log_loss"] for e in entries]

    x = np.arange(len(entries))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.0))
    b1 = ax.bar(x - width / 2, cv_ll, width, label="5-Fold Grouped CV Log Loss", color="#718096", edgecolor="#1A202C")
    b2 = ax.bar(x + width / 2, st_ll, width, label="Student-Test Log Loss", color="#805AD5", edgecolor="#1A202C")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Binary Log Loss (Lower is Better)")
    ax.set_title("Candidate Probability Calibration Comparison (Log Loss for HMM Emission Scoring)")
    ax.set_ylim(0.0, max(max(cv_ll), max(st_ll)) * 1.18)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_confusion_matrices(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Plot grid of Student-Test confusion matrices with raw counts and row percentages."""
    entries = _ordered_entries(results)
    n_models = len(entries)
    ncols = 3 if n_models > 4 else 2
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, entry in zip(axes_flat, entries):
        cm_dict = entry["student_test"]["confusion_matrix"]
        mat = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]], dtype=np.int64)
        row_sums = np.maximum(mat.sum(axis=1, keepdims=True), 1)
        pct = 100.0 * mat / row_sums
        labels = np.array(
            [
                [f"{mat[r, c]:,}\n({pct[r, c]:.1f}%)" for c in range(2)]
                for r in range(2)
            ]
        )
        sns.heatmap(
            pct,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            vmin=0,
            vmax=100,
            xticklabels=["Pred 0 (Non-Match)", "Pred 1 (Match)"],
            yticklabels=["True 0", "True 1"],
            ax=ax,
        )
        ax.set_title(f"{entry['name']}\nF1={entry['student_test']['f1_macro']:.4f}", fontsize=10, fontweight="bold")

    for ax in axes_flat[n_models:]:
        ax.axis("off")

    fig.suptitle("Student-Test Confusion Matrices (Candidate-Level Predictions)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overfitting_diagnostics(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Compare Train-Fit vs Grouped-CV vs Student-Test Macro F1 to diagnose generalisation."""
    entries = _ordered_entries(results)
    names = [e["name"] for e in entries]
    tr_f1 = [e["train_fit"]["f1_macro"] for e in entries]
    cv_f1 = [e["cv"]["f1_macro_mean"] for e in entries]
    st_f1 = [e["student_test"]["f1_macro"] for e in entries]

    x = np.arange(len(entries))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.3))
    b1 = ax.bar(x - width, tr_f1, width, label="Train-Fit Macro F1 (80 Trips)", color="#A0AEC0", edgecolor="#1A202C")
    b2 = ax.bar(x, cv_f1, width, label="5-Fold GroupKFold CV Macro F1", color="#3182CE", edgecolor="#1A202C")
    b3 = ax.bar(x + width, st_f1, width, label="Student-Test Macro F1 (10 Trips)", color="#38A169", edgecolor="#1A202C")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}", ha="center", va="bottom", fontsize=7.6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Generalisation & Overfitting Diagnostics: Train vs. Grouped CV vs. Student-Test")
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_computational_efficiency(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Plot Training Time and Inference Latency vs Student-Test Top-1 Accuracy."""
    entries = _ordered_entries(results)
    names = [e["name"] for e in entries]
    train_s = [e["train_fit"]["train_time_s"] for e in entries]
    inf_ms_trip = [e["student_test"]["inference_ms_per_trip"] or (1000.0 * e["student_test"]["inference_time_s"] / 10.0) for e in entries]
    top1 = [e["student_test"].get("top1_accuracy") or e["student_test"]["f1_macro"] for e in entries]
    colors = [MODEL_COLORS.get(e["key"], "#4C72B0") for e in entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    y_pos = np.arange(len(entries))
    ax1.barh(y_pos, train_s, color=colors, edgecolor="#222222", alpha=0.88)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Full 80-Trip Training Time (seconds)")
    ax1.set_title("Training Time Comparison")
    for idx, val in enumerate(train_s):
        ax1.text(val + max(train_s) * 0.02, idx, f"{val:.2f}s", va="center", fontsize=8.5)
    ax1.set_xlim(0.0, max(train_s) * 1.2)

    for entry, ms, t1 in zip(entries, inf_ms_trip, top1):
        ax2.scatter(ms, t1, s=160, color=MODEL_COLORS.get(entry["key"], "#4C72B0"), edgecolors="#111111", zorder=4)
        ax2.annotate(
            f"{entry['name']}\n({ms:.1f} ms/trip, {t1:.3f})",
            (ms, t1),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8.2,
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("Inference Latency per Trip (ms, log scale)")
    ax2.set_ylabel("Student-Test Top-1 Accuracy")
    ax2.set_title("Accuracy vs. Real-Time FastAPI Latency Trade-off")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_iterative_training_history(histories: dict[str, dict[str, list[float]]], out_path: Path) -> bool:
    """Plot genuine training/validation loss histories ONLY for iterative models."""
    valid = {k: v for k, v in histories.items() if v and "train_loss" in v}
    if not valid:
        return False

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
    for ax, (name, hist) in zip(axes[0], valid.items()):
        iters = np.arange(1, len(hist["train_loss"]) + 1)
        ax.plot(iters, hist["train_loss"], label="Training Loss", color="#2B6CB0", linewidth=2.0)
        if "val_loss" in hist:
            ax.plot(iters, hist["val_loss"], label="Validation Loss (10% Early-Stopping Split)", color="#E53E3E", linewidth=2.0, linestyle="--")
            ax.set_ylabel("Binary Log Loss")
        elif "val_accuracy" in hist:
            ax2 = ax.twinx()
            ax2.plot(iters, hist["val_accuracy"], label="Validation Accuracy", color="#38A169", linewidth=1.8, linestyle="--")
            ax2.set_ylabel("Validation Accuracy", color="#38A169")
            ax.set_ylabel("Training Cross-Entropy Loss")
        ax.set_xlabel("Iteration / Epoch")
        ax.set_title(f"{name} — Observed Training History")
        ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Genuine Iteration-Wise Loss Curves (Non-Iterative Models Excluded by Policy)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def plot_feature_importances(
    feature_names: list[str],
    importances_by_model: dict[str, np.ndarray],
    out_path: Path,
) -> bool:
    """Plot top physical & topological feature importances across fitted models."""
    if not importances_by_model:
        return False

    df = pd.DataFrame(importances_by_model, index=feature_names)
    df_norm = df.div(df.sum(axis=0).replace(0, 1.0), axis=1)
    df_norm["mean_importance"] = df_norm.mean(axis=1)
    df_norm = df_norm.sort_values("mean_importance", ascending=True).drop(columns=["mean_importance"])

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    df_norm.plot(kind="barh", ax=ax, width=0.78, edgecolor="#222222", linewidth=0.4)
    ax.set_xlabel("Normalized Feature Importance / Absolute Standardized Coefficient Share")
    ax.set_title("Physical GPS & Topological Feature Contributions Across Model Families")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def generate_all_plots(
    results: dict[str, dict[str, Any]],
    histories: dict[str, dict[str, list[float]]],
    feature_names: list[str],
    importances_by_model: dict[str, np.ndarray],
    plots_dir: Path | str,
) -> list[str]:
    """Generate and save all Section 3.3 visualisations, returning saved file paths."""
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    p1 = out_dir / "cv_macro_f1_comparison.png"
    plot_cv_macro_f1_comparison(results, p1)
    saved.append(str(p1))

    p2 = out_dir / "student_test_metrics_comparison.png"
    plot_student_test_metrics(results, p2)
    saved.append(str(p2))

    p3 = out_dir / "log_loss_comparison.png"
    plot_log_loss_comparison(results, p3)
    saved.append(str(p3))

    p4 = out_dir / "confusion_matrices_grid.png"
    plot_confusion_matrices(results, p4)
    saved.append(str(p4))

    p5 = out_dir / "overfitting_diagnostics.png"
    plot_overfitting_diagnostics(results, p5)
    saved.append(str(p5))

    p6 = out_dir / "computational_efficiency.png"
    plot_computational_efficiency(results, p6)
    saved.append(str(p6))

    p7 = out_dir / "iterative_training_validation_loss.png"
    if plot_iterative_training_history(histories, p7):
        saved.append(str(p7))

    p8 = out_dir / "feature_importance_analysis.png"
    if plot_feature_importances(feature_names, importances_by_model, p8):
        saved.append(str(p8))

    return saved
