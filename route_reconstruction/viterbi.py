"""
PathSnap - Viterbi Decoding Algorithm
Route Reconstruction
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

        if np.isnan(transition_matrix).any() or np.isposinf(
            transition_matrix
        ).any():
            raise ValueError("Invalid transition values")

        # Combine previous path and transition scores

        if transition_weight == 0:
            combined_scores = np.broadcast_to(
                previous_scores[:, None],
                expected_shape,
            )
        else:
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

        # An all-unreachable transition starts a new independent section.
        if not np.isfinite(best_scores).any():
            scores.append(current_emissions.copy())
            backpointers.append(
                np.full(len(current_emissions), -1, dtype=int)
            )
            continue

        # Add current emission probability
        current_scores = (
            best_scores + current_emissions
        )

        scores.append(current_scores)

        backpointers.append(best_previous)

    segment_starts = [0]
    segment_starts.extend(
        t
        for t in range(1, n_observations)
        if (backpointers[t] == -1).all()
    )
    segment_starts.append(n_observations)

    best_path = [0] * n_observations
    best_score = 0.0
    decoded_segments = []

    for segment_index in range(len(segment_starts) - 1):
        start = segment_starts[segment_index]
        end = segment_starts[segment_index + 1]

        best_state = int(np.argmax(scores[end - 1]))
        best_score += float(scores[end - 1][best_state])
        segment_path = [best_state]

        for t in range(end - 1, start, -1):
            previous_candidate = int(
                backpointers[t][segment_path[-1]]
            )

            if previous_candidate < 0:
                raise ValueError("Invalid Viterbi backpointer")

            segment_path.append(previous_candidate)

        segment_path.reverse()
        best_path[start:end] = segment_path
        decoded_segments.append((start, end))

    return {
        "best_path": best_path,
        "log_score": best_score,
        "n_observations": n_observations,
        "segments": decoded_segments,
    }
