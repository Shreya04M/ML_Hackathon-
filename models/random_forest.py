"""Random Forest bagging ensemble module for PathSnap candidate classification."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DEFAULT_CONFIG: dict[str, Any] = {
    "n_estimators": 120,
    "criterion": "gini",
    "max_depth": 16,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "max_samples": 0.5,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
}


def create_random_forest_pipeline(**kwargs: Any) -> Pipeline:
    """Build a fresh, leakage-free Random Forest pipeline."""
    cfg = {**DEFAULT_CONFIG, **kwargs}
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**cfg)),
        ]
    )


def get_random_forest_spec() -> dict[str, Any]:
    """Return algorithm metadata, initial configuration, and pipeline factory."""
    return {
        "key": "random_forest",
        "name": "Random Forest",
        "algorithm_family": "Ensemble (Bagging)",
        "key_configuration": (
            f"n_estimators={DEFAULT_CONFIG['n_estimators']}, max_depth={DEFAULT_CONFIG['max_depth']}, "
            f"min_samples_leaf={DEFAULT_CONFIG['min_samples_leaf']}, max_samples={DEFAULT_CONFIG['max_samples']}, "
            f"class_weight={DEFAULT_CONFIG['class_weight']}"
        ),
        "default_params": dict(DEFAULT_CONFIG),
        "loss_objective": "Bagged Ensemble Gini Impurity Reduction across Feature Subspaces",
        "produces_training_history": False,
        "factory": create_random_forest_pipeline,
    }


def train_random_forest(X: Any, y: Any, **kwargs: Any) -> Pipeline:
    """Fit and return the Random Forest pipeline."""
    pipeline = create_random_forest_pipeline(**kwargs)
    pipeline.fit(X, np.asarray(y, dtype=np.int64))
    return pipeline
