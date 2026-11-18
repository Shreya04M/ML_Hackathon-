"""
PathSnap - Member 3
Compatibility testing for all five ML models.
"""

import numpy as np
import optuna

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from optimisation.search_space import define_search_space
from optimisation.bayesian_optimization import (
    optimise_hyperparameters,
)


def test_all_models():

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        weights=[0.75, 0.25],
        random_state=42,
    )

    # Simulate 50 different trips
    groups = np.repeat(np.arange(50), 20)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),

        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(),
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        ),

        "hist_gradient_boosting":
            HistGradientBoostingClassifier(
                random_state=42,
                early_stopping=False,
            ),
    }

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = {}

    for model_name, model in models.items():

        print(f"\nTesting: {model_name}")

        try:

            search_space = define_search_space(
                model_name
            )

            study = optimise_hyperparameters(
                model=model,
                X=X,
                y=y,
                search_space=search_space,
                groups=groups,
                n_trials=3,
                n_splits=3,
            )

            results[model_name] = "PASSED"

            print("Status: PASSED")
            print(f"Best Macro F1: {study.best_value:.4f}")
            print("Best parameters:", study.best_params)

        except Exception as error:

            results[model_name] = "FAILED"

            print("Status: FAILED")
            print("Error:", error)

    print("\n========== FINAL TEST RESULTS ==========")

    for model_name, status in results.items():
        print(f"{model_name:25} : {status}")

    passed = sum(
        status == "PASSED"
        for status in results.values()
    )

    print(f"\nModels passed: {passed}/{len(models)}")

    assert passed == len(models), "Model compatibility test failed"


if __name__ == "__main__":
    test_all_models()
