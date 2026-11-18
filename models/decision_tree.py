"""Regularised Decision Tree model module for PathSnap candidate classification."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

DEFAULT_CONFIG: dict[str, Any] = {
    "criterion": "gini",
    "max_depth": 12,
    "min_samples_split": 40,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": 42,
}


def create_decision_tree_pipeline(**kwargs: Any) -> Pipeline:
    """Build a fresh, leakage-free Decision Tree pipeline."""
    cfg = {**DEFAULT_CONFIG, **kwargs}
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(**cfg)),
        ]
    )


def get_decision_tree_spec() -> dict[str, Any]:
    """Return algorithm metadata, initial configuration, and pipeline factory."""
    return {
        "key": "decision_tree",
        "name": "Decision Tree",
        "algorithm_family": "Tree-based",
        "key_configuration": (
            f"criterion={DEFAULT_CONFIG['criterion']}, max_depth={DEFAULT_CONFIG['max_depth']}, "
            f"min_samples_leaf={DEFAULT_CONFIG['min_samples_leaf']}, class_weight={DEFAULT_CONFIG['class_weight']}"
        ),
        "default_params": dict(DEFAULT_CONFIG),
        "loss_objective": "Weighted Gini Impurity Recursive Binary Partitioning",
        "produces_training_history": False,
        "factory": create_decision_tree_pipeline,
    }


def train_decision_tree(X: Any, y: Any, **kwargs: Any) -> Pipeline:
    """Fit and return the Decision Tree pipeline."""
    pipeline = create_decision_tree_pipeline(**kwargs)
    pipeline.fit(X, np.asarray(y, dtype=np.int64))
    return pipeline
