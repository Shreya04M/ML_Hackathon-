"""HistGradientBoosting model module for PathSnap candidate classification."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DEFAULT_CONFIG: dict[str, Any] = {
    "loss": "log_loss",
    "learning_rate": 0.08,
    "max_iter": 200,
    "max_leaf_nodes": 63,
    "max_depth": 12,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 15,
    "scoring": "loss",
    "random_state": 42,
}


def create_gradient_boosting_pipeline(**kwargs: Any) -> Pipeline:
    """Build a fresh, leakage-free HistGradientBoosting pipeline."""
    cfg = {**DEFAULT_CONFIG, **kwargs}
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(**cfg)),
        ]
    )


def get_gradient_boosting_spec() -> dict[str, Any]:
    """Return algorithm metadata, initial configuration, and pipeline factory."""
    return {
        "key": "hist_gradient_boosting",
        "name": "HistGradientBoosting",
        "algorithm_family": "Boosting",
        "key_configuration": (
            f"lr={DEFAULT_CONFIG['learning_rate']}, max_iter={DEFAULT_CONFIG['max_iter']}, "
            f"max_leaf_nodes={DEFAULT_CONFIG['max_leaf_nodes']}, l2={DEFAULT_CONFIG['l2_regularization']}, "
            f"class_weight={DEFAULT_CONFIG['class_weight']}, early_stopping=True"
        ),
        "default_params": dict(DEFAULT_CONFIG),
        "loss_objective": "L2-Regularised Binary Log Loss via Histogram Gradient Boosting",
        "produces_training_history": True,
        "factory": create_gradient_boosting_pipeline,
    }


def extract_training_history(pipeline: Pipeline) -> dict[str, list[float]] | None:
    """Extract iteration-wise training/validation log loss if recorded by the estimator."""
    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else pipeline
    train_scores = getattr(clf, "train_score_", None)
    val_scores = getattr(clf, "validation_score_", None)
    if train_scores is None or len(train_scores) == 0:
        return None
    # Note: HistGradientBoostingClassifier stores negated loss in *_score_ when scoring='loss'.
    train_loss = [float(-v) for v in train_scores]
    history: dict[str, list[float]] = {"train_loss": train_loss}
    if val_scores is not None and len(val_scores) == len(train_scores):
        history["val_loss"] = [float(-v) for v in val_scores]
    return history


def train_gradient_boosting(X: Any, y: Any, **kwargs: Any) -> Pipeline:
    """Fit and return the HistGradientBoosting pipeline."""
    pipeline = create_gradient_boosting_pipeline(**kwargs)
    pipeline.fit(X, np.asarray(y, dtype=np.int64))
    return pipeline
