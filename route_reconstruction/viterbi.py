"""
PathSnap - Viterbi Decoding Algorithm
Member 3 - Route Reconstruction
"""

import numpy as np


def viterbi_decode(
    emissions,
    transitions,
    **kwargs,
):
    """
    Find the most probable candidate sequence
    using log-space Viterbi decoding.

    Parameters
    ----------
    emissions:
        List of emission log-probability arrays.

    transitions:
        List of transition log-probability matrices.

    Returns
    -------
    Dictionary containing the best candidate
    indices and final sequence score.
    """

    transition_weight = float(
        kwargs.get("transition_weight", 1.0)
    )

    if (
        not np.isfinite(transition_weight)
        or transition_weight < 0
    ):
        raise ValueError("Invalid transition weight")

    emissions = [
        np.asarray(e, dtype=float)
        for e in emissions
    ]

    transitions = [
        np.asarray(t, dtype=float)
        for t in transitions
    ]

    n_observations = len(emissions)

    if n_observations == 0:
        raise ValueError("No emissions provided")

    if len(transitions) != n_observations - 1:
        raise ValueError(
            "Incorrect number of transition matrices"
        )

    # Validate emission arrays

    for emission in emissions:

        if emission.ndim != 1 or emission.size == 0:
            raise ValueError("Invalid emission array")

        if not np.isfinite(emission).all():
            raise ValueError("Invalid emission values")

    # Dynamic programming storage

    scores = []
    backpointers = []

    # Initialise the first observation

    scores.append(emissions[0].copy())

    backpointers.append(
        np.full(len(emissions[0]), -1, dtype=int)
    )

    # Forward pass

    for t in range(1, n_observations):

        previous_scores = scores[t - 1]
        transition_matrix = transitions[t - 1]
        current_emissions = emissions[t]

        expected_shape = (
            len(previous_scores),
            len(current_emissions),
        )

        if transition_matrix.shape != expected_shape:
            raise ValueError(
                f"Invalid transition shape at step {t}"
            )

        if not np.isfinite(transition_matrix).all():
            raise ValueError("Invalid transition values")

        # Combine previous path and transition scores

        combined_scores = (
            previous_scores[:, None]
            + transition_weight * transition_matrix
        )

        # Find best previous candidate for each state

        best_previous = np.argmax(
            combined_scores,
            axis=0,
        )

        best_scores = np.max(
            combined_scores,
            axis=0,
        )

        # Add current emission probability

        current_scores = (
            best_scores + current_emissions
        )

        scores.append(current_scores)

        backpointers.append(best_previous)

    # Identify best final candidate

    best_last_candidate = int(
        np.argmax(scores[-1])
    )

    best_score = float(
        scores[-1][best_last_candidate]
    )

    # Backtracking

    best_path = [best_last_candidate]

    for t in range(n_observations - 1, 0, -1):

        previous_candidate = int(
            backpointers[t][best_path[-1]]
        )

        best_path.append(previous_candidate)

    best_path.reverse()

    return {
        "best_path": best_path,
        "log_score": best_score,
        "n_observations": n_observations,
    }
