
"""Group-aware cross-validation for PathSnap."""

from typing import Any
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from evaluation.metrics import calculate_metrics


def run_group_aware_cv(
    estimator: Any,
    X: Any,
    y: Any,
    groups: Any,
    **kwargs: Any
) -> dict[str, Any]:
    """Evaluate a model without leakage between trips."""

    n_splits = kwargs.get("n_splits", 5)

    if not hasattr(X, "iloc"):
        X = np.asarray(X)

    y = np.asarray(y)
    groups = np.asarray(groups)

    cv = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        # Ensure that no trip appears in both sets.
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        assert train_groups.isdisjoint(val_groups), \
            "Data leakage detected!"

        if hasattr(X, "iloc"):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Create a fresh model for each fold.
        model = clone(estimator)

        # Measure training time.
        start = perf_counter()
        model.fit(X_train, y_train)
        training_time = perf_counter() - start

        # Measure prediction time.
        start = perf_counter()
        y_pred = model.predict(X_val)
        inference_time = perf_counter() - start

        # Obtain probabilities if supported.
        y_proba = None

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)

        # Calculate common evaluation metrics.
        metrics = calculate_metrics(
            y_val, y_pred, y_proba
        )

        metrics.update({
            "fold": fold,
            "training_time": training_time,
            "inference_time": inference_time,
            "training_trips": len(train_groups),
            "validation_trips": len(val_groups),
            "training_samples": len(train_idx),
            "validation_samples": len(val_idx),
        })

        fold_results.append(metrics)

    # Calculate mean and standard deviation.
    summary = {}

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "log_loss",
        "training_time",
        "inference_time",
    ]

    for name in metric_names:
        values = [
            result[name]
            for result in fold_results
            if result[name] is not None
        ]

        summary[f"mean_{name}"] = (
            float(np.mean(values)) if values else None
        )

        summary[f"std_{name}"] = (
            float(np.std(values)) if values else None
        )

    return {
        "model": type(estimator).__name__,
        "n_splits": n_splits,
        "fold_results": fold_results,
        "summary": summary,
    }