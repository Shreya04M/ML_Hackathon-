"""Multi-Layer Perceptron (MLP) model module for PathSnap candidate classification.

Provided to satisfy full compatibility with the repository skeleton while keeping
the primary Phase 1 comparison aligned with the five core model families.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG: dict[str, Any] = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-3,
    "batch_size": 512,
    "learning_rate_init": 1e-3,
    "max_iter": 40,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 5,
    "random_state": 42,
}


def create_mlp_pipeline(**kwargs: Any) -> Pipeline:
    """Build a fresh, leakage-free MLP neural-network pipeline."""
    cfg = {**DEFAULT_CONFIG, **kwargs}
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(**cfg)),
        ]
    )


def get_mlp_spec() -> dict[str, Any]:
    """Return algorithm metadata, initial configuration, and pipeline factory."""
    return {
        "key": "mlp",
        "name": "MLP Neural Network",
        "algorithm_family": "Neural networks",
        "key_configuration": (
            f"hidden={DEFAULT_CONFIG['hidden_layer_sizes']}, act={DEFAULT_CONFIG['activation']}, "
            f"alpha={DEFAULT_CONFIG['alpha']}, batch={DEFAULT_CONFIG['batch_size']}, "
            f"lr={DEFAULT_CONFIG['learning_rate_init']}, early_stopping=True"
        ),
        "default_params": dict(DEFAULT_CONFIG),
        "loss_objective": "L2-Regularised Binary Cross-Entropy via Adam Optimisation",
        "produces_training_history": True,
        "factory": create_mlp_pipeline,
    }


def extract_training_history(pipeline: Pipeline) -> dict[str, list[float]] | None:
    """Extract epoch-wise training loss curve (and validation accuracy) if available."""
    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else pipeline
    loss_curve = getattr(clf, "loss_curve_", None)
    if loss_curve is None or len(loss_curve) == 0:
        return None
    history: dict[str, list[float]] = {"train_loss": [float(v) for v in loss_curve]}
    val_scores = getattr(clf, "validation_scores_", None)
    if val_scores is not None and len(val_scores) == len(loss_curve):
        history["val_accuracy"] = [float(v) for v in val_scores]
    return history


def train_mlp(X: Any, y: Any, **kwargs: Any) -> Pipeline:
    """Fit and return the MLP classifier pipeline."""
    pipeline = create_mlp_pipeline(**kwargs)
    pipeline.fit(X, np.asarray(y, dtype=np.int64))
    return pipeline
