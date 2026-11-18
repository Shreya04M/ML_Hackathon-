
"""Reusable classification metrics for PathSnap."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
)


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None
) -> dict[str, Any]:
    """Calculate binary candidate classification metrics."""

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),

        "precision": float(
            precision_score(y_true, y_pred, average="macro",
                            labels=[0, 1], zero_division=0)
        ),

        "recall": float(
            recall_score(y_true, y_pred, average="macro",
                         labels=[0, 1], zero_division=0)
        ),

        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro",
                     labels=[0, 1], zero_division=0)
        ),

        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),

        "log_loss": None,
    }

    if y_proba is not None:
        results["log_loss"] = float(
            log_loss(y_true, y_proba, labels=[0, 1])
        )

    return results