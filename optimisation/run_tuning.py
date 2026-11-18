"""
PathSnap - Model Tuning
Bayesian Hyperparameter Optimisation
"""

import argparse
from pathlib import Path

import pandas as pd
import joblib

from optimisation.search_space import SUPPORTED_MODELS
from optimisation.bayesian_optimization import run_optimisation


def main():

    parser = argparse.ArgumentParser(
        description="PathSnap Bayesian Model Optimisation"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
    )

    parser.add_argument(
        "--model-path",
        required=True,
    )

    parser.add_argument(
        "--train",
        default="data/processed/train_features.csv",
    )

    parser.add_argument(
        "--test",
        default="data/processed/student_test_features.csv",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    # Step 1: Verify required files

    required_files = [
        args.train,
        args.test,
        args.model_path,
    ]

    for file_path in required_files:

        if not Path(file_path).is_file():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Step 2: Load datasets

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    forbidden = {
        "trip_id", "gps_id", "candidate_id", "label"
    }

    if forbidden.intersection(args.features):
        raise ValueError(
            "Identifiers and labels cannot be model features"
        )

    if len(set(args.features)) != len(args.features):
        raise ValueError("Duplicate feature names")

    required_columns = set(
        args.features + ["trip_id", "label"]
    )

    for name, df in [
        ("Training", train_df),
        ("Testing", test_df),
    ]:

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

        if df[["trip_id", "label"]].isna().any().any():
            raise ValueError(
                f"{name} dataset has missing trip IDs or labels"
            )

    # Step 3: Check for trip leakage

    train_trips = set(train_df["trip_id"])
    test_trips = set(test_df["trip_id"])

    if train_trips.intersection(test_trips):
        raise ValueError(
            "Data leakage: training and testing trips overlap"
        )

    # Step 4: Prepare model inputs

    X_train = train_df[args.features]
    y_train = train_df["label"]

    train_groups = train_df["trip_id"]

    X_test = test_df[args.features]
    y_test = test_df["label"]

    # Step 5: Load trusted baseline model

    baseline_model = joblib.load(args.model_path)

    print("\n========== DATASET SUMMARY ==========")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training trips:", len(train_trips))
    print("Testing trips:", len(test_trips))
    print("Features:", args.features)
    print("Selected model:", args.model)

    # Step 6: Run Bayesian optimisation

    results = run_optimisation(
        model_name=args.model,
        baseline_model=baseline_model,
        X_train=X_train,
        y_train=y_train,
        train_groups=train_groups,
        X_test=X_test,
        y_test=y_test,
        n_trials=args.trials,
        n_splits=args.folds,
        output_dir="experiments",
        artifacts_dir="artifacts",
    )

    print("\nFinal selected version:",
          results["selected_version"])

    print("\nPathSnap model tuning completed!")


if __name__ == "__main__":
    main()
