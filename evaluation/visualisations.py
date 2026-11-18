
"""Performance visualisations for PathSnap."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OUTPUT_DIR = Path("experiments/results/graphs")


def plot_model_comparison(comparison: pd.DataFrame):
    """Plot cross-validation performance of all models."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = comparison.copy()
    df["label"] = df["model"] + " (" + df["variant"] + ")"

    ax = df.set_index("label")[
        ["mean_accuracy", "mean_macro_f1",
         "mean_precision", "mean_recall"]
    ].plot(kind="bar", figsize=(12, 6))

    ax.set_title("Cross-Validation Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=25, ha="right")
    plt.legend(["Accuracy", "Macro F1", "Precision", "Recall"])
    plt.tight_layout()

    path = OUTPUT_DIR / "model_comparison.png"

    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_execution_time(comparison: pd.DataFrame):
    """Compare training and inference times."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = comparison.copy()
    df["label"] = df["model"] + " (" + df["variant"] + ")"

    paths = []

    for metric, title, filename in [
        ("mean_training_time", "Mean Training Time",
         "training_time.png"),
        ("mean_inference_time", "Mean Inference Time",
         "inference_time.png"),
    ]:
        plt.figure(figsize=(10, 5))

        plt.bar(df["label"], df[metric])

        plt.title(title)
        plt.ylabel("Time (seconds per fold)")
        plt.xlabel("Model")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        path = OUTPUT_DIR / filename

        plt.savefig(path, dpi=300)
        plt.close()

        paths.append(path)

    return paths


def plot_confusion_matrix(matrix, model_name):
    """Generate a confusion matrix heatmap."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Incorrect", "Correct"],
        yticklabels=["Incorrect", "Correct"],
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()

    safe_name = "".join(
        c if c.isalnum() else "_" for c in model_name
    )

    path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"

    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_tuned_vs_untuned(comparison: pd.DataFrame, model_name):
    """Compare the selected model before and after tuning."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = comparison[
        comparison["model"] == model_name
    ].copy()

    if not {"tuned", "untuned"}.issubset(set(df["variant"])):
        raise ValueError("Both tuned and untuned results are required.")

    ax = df.set_index("variant")[
        ["mean_accuracy", "mean_macro_f1",
         "mean_precision", "mean_recall"]
    ].plot(kind="bar", figsize=(9, 5))

    ax.set_title(f"Tuned vs Untuned - {model_name}")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model variant")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)
    plt.legend(["Accuracy", "Macro F1", "Precision", "Recall"])
    plt.tight_layout()

    path = OUTPUT_DIR / "tuned_vs_untuned.png"

    plt.savefig(path, dpi=300)
    plt.close()

    return path