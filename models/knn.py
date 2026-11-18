"""Distance-Weighted K-Nearest Neighbors (KNN) model module for PathSnap."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG: dict[str, Any] = {
    "n_neighbors": 15,
    "weights": "distance",
    "algorithm": "kd_tree",
    "leaf_size": 40,
    "metric": "minkowski",
    "p": 2,
    "n_jobs": -1,
    "max_train_samples": 120_000,
    "random_state": 42,
}


class StratifiedSubsampledKNN(BaseEstimator, ClassifierMixin):
    """Wrapper around KNeighborsClassifier supporting deterministic stratified bounding.

    Instance-based search scales superlinearly in training set size during inference.
    When ``max_train_samples`` is set and exceeded, fitting draws a deterministic
    stratified subsample of the training fold while preserving class ratios and zero
    validation leakage.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        weights: str = "distance",
        algorithm: str = "kd_tree",
        leaf_size: int = 40,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int = -1,
        max_train_samples: int | None = 120_000,
        random_state: int = 42,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.max_train_samples = max_train_samples
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StratifiedSubsampledKNN":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y_arr)
        self.effective_train_samples_ = len(X_arr)

        if self.max_train_samples is not None and len(X_arr) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            chosen_indices: list[np.ndarray] = []
            for cls in self.classes_:
                cls_idx = np.flatnonzero(y_arr == cls)
                n_take = max(1, int(round(self.max_train_samples * (len(cls_idx) / len(y_arr)))))
                n_take = min(n_take, len(cls_idx))
                chosen_indices.append(rng.choice(cls_idx, size=n_take, replace=False))
            idx = np.sort(np.concatenate(chosen_indices))
            X_fit, y_fit = X_arr[idx], y_arr[idx]
            self.effective_train_samples_ = len(X_fit)
        else:
            X_fit, y_fit = X_arr, y_arr

        self.estimator_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(X_fit, y_fit)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(np.asarray(X, dtype=np.float32))


def create_knn_pipeline(**kwargs: Any) -> Pipeline:
    """Build a fresh, leakage-free Distance-Weighted KNN pipeline."""
    cfg = {**DEFAULT_CONFIG, **kwargs}
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", StratifiedSubsampledKNN(**cfg)),
        ]
    )


def get_knn_spec() -> dict[str, Any]:
    """Return algorithm metadata, initial configuration, and pipeline factory."""
    return {
        "key": "knn",
        "name": "KNN (Distance-Weighted)",
        "algorithm_family": "Instance-based",
        "key_configuration": (
            f"k={DEFAULT_CONFIG['n_neighbors']}, weights={DEFAULT_CONFIG['weights']}, "
            f"algorithm={DEFAULT_CONFIG['algorithm']}, p={DEFAULT_CONFIG['p']}, "
            f"StandardScaler, max_index={DEFAULT_CONFIG['max_train_samples']}"
        ),
        "default_params": dict(DEFAULT_CONFIG),
        "loss_objective": "Non-parametric local Euclidean distance-weighted vote (no parametric training loss)",
        "produces_training_history": False,
        "factory": create_knn_pipeline,
    }


def train_knn(X: Any, y: Any, **kwargs: Any) -> Pipeline:
    """Fit and return the Distance-Weighted KNN pipeline."""
    pipeline = create_knn_pipeline(**kwargs)
    pipeline.fit(X, np.asarray(y, dtype=np.int64))
    return pipeline
