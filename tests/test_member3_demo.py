"""
PathSnap - Member 3 Demonstration
Test Bayesian optimisation using synthetic data.
"""

import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from optimisation.bayesian_optimization import run_optimisation


def main():

    # Generate temporary sample data
    X, y = make_classification(
        n_samples=1800,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        weights=[0.75, 0.25],
        random_state=42,
    )

    # Simulate 90 different trips
    groups = np.repeat(np.arange(90), 20)

    # Split into 80 training trips and 10 testing trips
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=10,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    train_groups = groups[train_idx]

    print("Sample dataset created successfully")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training trips:", len(np.unique(train_groups)))
    print("Testing trips:", len(np.unique(groups[test_idx])))

    # Temporary baseline model
    baseline_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    # Execute Bayesian optimisation
    results = run_optimisation(
        model_name="random_forest",
        baseline_model=baseline_model,
        X_train=X_train,
        y_train=y_train,
        train_groups=train_groups,
        X_test=X_test,
        y_test=y_test,
        n_trials=12,
        n_splits=5,
        output_dir="experiments/demo_member3",
        artifacts_dir="artifacts/demo_member3",
    )

    print("\nMember 3 demonstration completed!")


if __name__ == "__main__":
    main()
