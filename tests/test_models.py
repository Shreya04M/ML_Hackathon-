"""Unit tests for PathSnap ML models, metrics, GroupKFold CV, and serving artifacts."""
from pathlib import Path
import numpy as np
import pandas as pd

from models import (
    FEATURE_COLUMNS,
    evaluate_model,
    load_final_model,
    predict_candidate_probabilities,
    run_grouped_cross_validation,
    train_decision_tree,
    train_gradient_boosting,
    train_knn,
    train_logistic_regression,
    train_mlp,
    train_random_forest,
)
from preprocessing import extract_trip_candidates_and_features
from preprocessing.data_loader import load_trip


def test_all_model_trainers_and_grouped_cv_synthetic() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(180, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X["ortho_dist_m"].to_numpy() < 0.0).astype(np.int64)
    groups = np.repeat(["t1", "t2", "t3", "t4"], 45)

    for fn in (
        lambda: train_logistic_regression(X, y),
        lambda: train_knn(X, y, n_neighbors=5),
        lambda: train_decision_tree(X, y, max_depth=4),
        lambda: train_random_forest(X, y, n_estimators=10, max_depth=5),
        lambda: train_gradient_boosting(X, y, max_iter=15, early_stopping=False),
        lambda: train_mlp(X, y, max_iter=5, early_stopping=False),
    ):
        model = fn()
        probs = predict_candidate_probabilities(model, X.iloc[:12])
        assert probs.shape == (12,)
        assert np.all((probs > 0.0) & (probs < 1.0))

    cv = run_grouped_cross_validation(
        model_factory=lambda: train_logistic_regression(X, y),
        X=X,
        y=y,
        groups=groups,
        n_splits=4,
    )
    assert 0.0 <= cv["f1_macro_mean"] <= 1.0


def test_persisted_tuned_model_inference_on_trip_00000052() -> None:
    model_path = Path("artifacts/models/selected_best_model_tuned.joblib")
    assert model_path.is_file()
    model = load_final_model(model_path)

    trip_dir = Path("Student folder/student_test/00000052")
    raw = load_trip(trip_dir)
    X_df, y, meta_df = extract_trip_candidates_and_features(
        track=raw["track"].to_numpy(dtype=np.float64),
        nodes=raw["nodes"][["longitude", "latitude"]].to_numpy(dtype=np.float64),
        arcs=raw["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64),
        route=raw["route"]["arc_id"].to_numpy(dtype=np.int64),
        trip_id="00000052",
    )
    metrics = evaluate_model(model, X_df[FEATURE_COLUMNS], y, meta_df=meta_df)
    assert metrics["top1_accuracy"] > 0.90
    assert metrics["log_loss"] < 0.25
