"""Logistic Regression model module for PathSnap candidate classification."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG: dict[str, Any] = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": "balanced",
    "max_iter": 500,
    "random_state": 42,
}


def create_logistic_regression_pipeline(**kwargs: Any) -> Pipeline:
    """Build a fresh, leakage-free Logistic Regression pipeline."""
    cfg = {**DEFAULT_CONFIG, **kwargs}
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**cfg)),
        ]
    )


def get_logistic_regression_spec() -> dict[str, Any]:
    """Return algorithm metadata, initial configuration, and pipeline factory."""
    return {
        "key": "logistic_regression",
        "name": "Logistic Regression",
        "algorithm_family": "Linear and probabilistic",
        "key_configuration": (
            f"C={DEFAULT_CONFIG['C']}, L2 penalty, solver={DEFAULT_CONFIG['solver']}, "
            f"class_weight={DEFAULT_CONFIG['class_weight']}, StandardScaler"
        ),
        "default_params": dict(DEFAULT_CONFIG),
        "loss_objective": "Regularised Binary Cross-Entropy (L2 Log Loss)",
        "produces_training_history": False,
        "factory": create_logistic_regression_pipeline,
    }


def train_logistic_regression(X: Any, y: Any, **kwargs: Any) -> Pipeline:
    """Fit and return the Logistic Regression baseline pipeline."""
    pipeline = create_logistic_regression_pipeline(**kwargs)
    pipeline.fit(X, np.asarray(y, dtype=np.int64))
    return pipeline
