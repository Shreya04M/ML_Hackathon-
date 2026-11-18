"""
PathSnap - Out-of-Fold Validation Probabilities
Member 3 - Phase 2
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from models.gradient_boosting import (
    create_gradient_boosting_pipeline,
)

from models.schema_adapter import (
    build_partition_dataset,
    FEATURE_COLUMNS,
)


OUTPUT_DIR = Path(
    "experiments/results/hmm_validation"
)

DATA_ROOT = Path("Student folder")

N_SPLITS = 5


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    print("Loading training dataset...", flush=True)

    X, y, metadata, road_graphs = (
        build_partition_dataset(
            partition="train",
            data_root=DATA_ROOT,
            top_k=6,
            cache_dir=None,
            use_cache=False,
        )
    )

    X = X[FEATURE_COLUMNS]

    y = np.asarray(y, dtype=int)

    metadata = metadata.reset_index(drop=True)

    X = X.reset_index(drop=True)

    groups = metadata["trip_id"].to_numpy()

    if not len(X) == len(y) == len(metadata):
        raise ValueError("Dataset alignment error")

    print("Candidate records:", len(X))
    print("Training trips:", len(np.unique(groups)))
    print("Features:", X.shape[1])

    probabilities = np.full(
        len(X),
        np.nan,
        dtype=float,
    )

    fold_ids = np.full(
        len(X),
        -1,
        dtype=int,
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS,
    )

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):

        print(
            f"\nProcessing Fold {fold}/{N_SPLITS}",
            flush=True,
        )

        fold_start = time.perf_counter()

        train_trips = set(groups[train_idx])
        valid_trips = set(groups[valid_idx])

        # Prevent trip-level data leakage

        if train_trips.intersection(valid_trips):
            raise ValueError("Trip leakage detected")

        print("Training trips:", len(train_trips))
        print("Validation trips:", len(valid_trips))

        model = create_gradient_boosting_pipeline()

        model.fit(
            X.iloc[train_idx],
            y[train_idx],
        )

        positive_index = list(
            model.classes_
        ).index(1)

        predicted = model.predict_proba(
            X.iloc[valid_idx]
        )[:, positive_index]

        probabilities[valid_idx] = predicted

        fold_ids[valid_idx] = fold

        elapsed = time.perf_counter() - fold_start

        print(
            f"Fold completed in {elapsed:.2f} seconds",
            flush=True,
        )

    # Verify complete prediction coverage

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Missing or invalid validation probabilities"
        )

    if (fold_ids < 1).any():
        raise ValueError(
            "Some records were not assigned a fold"
        )

    result = metadata.copy()

    result["label"] = y

    result["pred_prob"] = probabilities

    result["pred_log_prob"] = np.log(
        np.clip(probabilities, 1e-12, 1.0)
    )

    result["fold_id"] = fold_ids

    output_path = (
        OUTPUT_DIR
        / "training_oof_probabilities.csv.gz"
    )

    result.to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    print("\n===== VALIDATION DATA GENERATED =====")

    print("Candidate records:", len(result))

    print(
        "Trips:",
        result["trip_id"].nunique(),
    )

    print(
        "GPS observations:",
        result[["trip_id", "obs_id"]]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "Missing probabilities:",
        result["pred_prob"].isna().sum(),
    )

    print(
        "Total runtime:",
        round(time.perf_counter() - start_time, 2),
        "seconds",
    )

    print("Results saved in:", output_path)


if __name__ == "__main__":
    main()
