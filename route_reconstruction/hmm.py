"""
PathSnap - HMM Route Reconstruction
Member 3 - Phase 2
"""

import numpy as np
import pandas as pd

from route_reconstruction.emission_probability import (
    calculate_emission_probabilities,
)

from route_reconstruction.transition_probability import (
    calculate_transition_probabilities,
)

from route_reconstruction.viterbi import viterbi_decode


def reconstruct_route(
    observations=None,
    candidates=None,
    road_network=None,
    model=None,
    **kwargs,
):
    """
    Reconstruct candidate road sequences using HMM
    emission scores and Viterbi decoding.
    """

    if candidates is None:
        raise ValueError("Candidate data is required")

    # Handle duplicate road candidates

    candidates = candidates.copy()

    keys = [
        "trip_id",
        "obs_id",
        "candidate_arc_id",
    ]

    duplicate_count = candidates.duplicated(keys).sum()

    if duplicate_count > 0:

        # Verify that repeated roads have consistent nodes

        node_counts = candidates.groupby(keys)[
            ["from_node_id", "to_node_id"]
        ].nunique(dropna=False)

        if (node_counts > 1).any().any():
            raise ValueError(
                "Duplicate candidates have conflicting road nodes"
            )

        if "pred_prob" not in candidates.columns:
            raise ValueError(
                "Duplicate handling requires predicted probabilities"
            )

        probabilities = candidates["pred_prob"].to_numpy(
            dtype=float
        )

        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Invalid predicted probabilities")

        # Retain the highest probability for each road

        candidates = (
            candidates.sort_values(
                "pred_prob",
                ascending=False,
                kind="stable",
            )
            .drop_duplicates(keys, keep="first")
            .reset_index(drop=True)
        )

        print(
            f"Removed {duplicate_count} duplicate candidates"
        )

    

    # Step 1: Calculate emission probabilities

    df = calculate_emission_probabilities(
        observations=observations,
        candidates=candidates,
        model=model,
        **kwargs,
    )

    # Keep candidate ordering consistent

    df = df.sort_values(
        ["trip_id", "obs_id", "candidate_arc_id"]
    ).reset_index(drop=True)

    if df.duplicated(
        ["trip_id", "obs_id", "candidate_arc_id"]
    ).any():
        raise ValueError("Duplicate candidate records")

    # Step 2: Calculate transition probabilities

    transitions = calculate_transition_probabilities(
        candidates=df,
        road_network=road_network,
        **kwargs,
    )

    reconstructed = []

    # Step 3: Process each trip independently

    for trip_id, trip_df in df.groupby("trip_id"):

        observation_groups = list(
            trip_df.groupby("obs_id", sort=True)
        )

        emission_arrays = [
            group["emission_log_prob"].to_numpy()
            for _, group in observation_groups
        ]

        transition_arrays = [
            item["transition_log_prob"]
            for item in transitions[trip_id]
        ]

        # Step 4: Viterbi decoding

        decoded = viterbi_decode(
            emissions=emission_arrays,
            transitions=transition_arrays,
            **kwargs,
        )

        best_path = decoded["best_path"]

        # Step 5: Extract selected candidate roads

        for position, (obs_id, group) in enumerate(
            observation_groups
        ):

            selected = group.iloc[best_path[position]]

            baseline = group.iloc[
                np.argmax(group["emission_prob"].to_numpy())
            ]

            record = {
                "trip_id": trip_id,
                "obs_id": obs_id,
                "classifier_arc_id": baseline[
                    "candidate_arc_id"
                ],
                "hmm_arc_id": selected[
                    "candidate_arc_id"
                ],
                "classifier_from_node": baseline[
                    "from_node_id"
                ],
                "classifier_to_node": baseline[
                    "to_node_id"
                ],
                "hmm_from_node": selected[
                    "from_node_id"
                ],
                "hmm_to_node": selected[
                    "to_node_id"
                ],
                "hmm_emission_prob": selected[
                    "emission_prob"
                ],
            }

            if "timestamp" in group.columns:
                record["timestamp"] = selected["timestamp"]

            # Reference information is copied only
            # after decoding, for evaluation.

            if "ref_arc_id" in group.columns:
                record["ref_arc_id"] = selected[
                    "ref_arc_id"
                ]

            if "is_ref_in_candidates" in group.columns:
                record["is_ref_in_candidates"] = selected[
                    "is_ref_in_candidates"
                ]

            reconstructed.append(record)

    return pd.DataFrame(reconstructed)


def evaluate_route_continuity(
    predicted_route,
    reference_route=None,
):
    """
    Compare direct road connectivity and,
    when available, reference-arc accuracy.
    """

    df = predicted_route.copy()

    results = {}

    for method in ["classifier", "hmm"]:

        connected = 0
        total = 0

        for _, trip in df.groupby("trip_id"):

            trip = trip.sort_values("obs_id")

            arcs = trip[f"{method}_arc_id"].to_numpy()

            from_nodes = trip[
                f"{method}_from_node"
            ].to_numpy()

            to_nodes = trip[
                f"{method}_to_node"
            ].to_numpy()

            if len(trip) < 2:
                continue

            same_arc = arcs[:-1] == arcs[1:]

            forward_connected = (
                to_nodes[:-1] == from_nodes[1:]
            )

            valid = same_arc | forward_connected

            connected += int(valid.sum())
            total += len(valid)

        results[f"{method}_continuity"] = (
            connected / total if total else None
        )

        results[f"{method}_connected_pairs"] = connected

        results[f"{method}_total_pairs"] = total

        if "ref_arc_id" in df.columns:

            valid_reference = df["ref_arc_id"].notna()

            actual = df.loc[
                valid_reference, "ref_arc_id"
            ]

            predicted = df.loc[
                valid_reference, f"{method}_arc_id"
            ]

            results[f"{method}_accuracy"] = (
                float((predicted == actual).mean())
                if len(actual) else None
            )

    results["total_observations"] = len(df)

    results["total_trips"] = df["trip_id"].nunique()

    results["changed_predictions"] = int(
        (
            df["classifier_arc_id"] != df["hmm_arc_id"]
        ).sum()
    )

    return results
