"""
PathSnap - HMM Emission Probabilities
Route Reconstruction
"""

import numpy as np
import pandas as pd


EPSILON = 1e-12


def calculate_emission_probabilities(
    observations=None,
    candidates=None,
    model=None,
    **kwargs,
):
    """
    Calculate emission probabilities for candidate road segments.

    Supports precomputed ML probabilities or model predictions.
    """

    if candidates is None:
        raise ValueError("Candidate data is required")

    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("Candidates must be a pandas DataFrame")

    df = candidates.copy()

    required_columns = {
        "trip_id",
        "obs_id",
        "candidate_arc_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Candidate dataset is empty")

    if df[list(required_columns)].isna().any().any():
        raise ValueError("Missing trip, observation or candidate IDs")

    # Option 1: Use existing ML probabilities

    if "pred_prob" in df.columns:

        probabilities = df["pred_prob"].to_numpy(dtype=float)

    # Option 2: Generate probabilities using the trained model

    elif model is not None:

        feature_columns = kwargs.get("feature_columns")

        if not feature_columns:
            raise ValueError("Feature columns are required")

        forbidden = {
            "label",
            "ref_arc_id",
            "is_ref_in_candidates",
            "pred_log_prob",
        }

        if forbidden.intersection(feature_columns):
            raise ValueError("Reference information cannot be used")

        positive_index = list(model.classes_).index(1)

        probabilities = model.predict_proba(
            df[feature_columns]
        )[:, positive_index]

    else:

        raise ValueError(
            "Provide pred_prob or a trained model"
        )

    # Validate probabilities

    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability values")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    # Avoid log(0)

    df["emission_raw"] = np.clip(
        probabilities,
        EPSILON,
        1.0,
    )

    # Normalise candidates within each GPS observation

    group_sum = df.groupby(
        ["trip_id", "obs_id"]
    )["emission_raw"].transform("sum")

    df["emission_prob"] = (
        df["emission_raw"] / group_sum
    )

    # Convert probabilities into log-space

    df["emission_log_prob"] = np.log(
        df["emission_prob"]
    )

    return df
