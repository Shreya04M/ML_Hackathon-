"""
PathSnap - Bayesian HMM Optimisation
Distance-Based Transition Tuning
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna


CACHE_DIR = Path("data/processed/hmm_validation_cache")

OUTPUT_DIR = Path("artifacts/hmm")

N_TRIALS = 25

BETA = 30.0
BACKWARD_TOLERANCE_M = 30.0

CONTINUITY_WEIGHT = 0.25

def load_validation_cache(cache_dir=CACHE_DIR):

    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("trip_*.npz"))

    if len(files) != 80:
        raise ValueError(
            f"Expected 80 cached trips, found {len(files)}"
        )

    trips = []

    for path in files:

        with np.load(path, allow_pickle=False) as data:

            trip = {
                key: data[key]
                for key in data.files
            }

        # Precompute transition log scores

        if np.isnan(trip["distance_errors"]).any():
            raise ValueError("Invalid cached distance errors")

        trip["transition_logs"] = (
            -trip["distance_errors"] / BETA
        )

        trips.append(trip)

    return trips


def decode_cached_trip(trip, transition_weight):

    emissions = trip["emissions"]

    counts = trip["candidate_counts"]

    transitions = trip["transition_logs"]

    n, k = emissions.shape

    # A zero transition weight gives
    # classifier-only predictions.

    if transition_weight == 0:

        return np.argmax(
            emissions,
            axis=1,
        )

    scores = emissions[0].copy()

    backpointers = np.zeros(
        (n, k),
        dtype=np.int64,
    )

    segment_starts = [0]
    segment_scores = []

    # Forward Viterbi pass

    for t in range(1, n):

        combined = (
            scores[:, None]
            + transition_weight
            * transitions[t - 1]
        )

        # Mask padded candidate slots

        combined[counts[t - 1]:, :] = -np.inf

        combined[:, counts[t]:] = -np.inf

        best_scores = np.max(combined, axis=0)

        if not np.isfinite(best_scores).any():
            segment_scores.append(scores.copy())
            scores = emissions[t].copy()
            backpointers[t] = -1
            segment_starts.append(t)
            continue

        backpointers[t] = np.argmax(combined, axis=0)
        scores = (
            best_scores + emissions[t]
        )

    segment_starts.append(n)
    segment_scores.append(scores.copy())
    path = np.zeros(n, dtype=np.int64)

    for segment_index in range(len(segment_starts) - 1):
        start = segment_starts[segment_index]
        end = segment_starts[segment_index + 1]
        path[end - 1] = np.argmax(segment_scores[segment_index])

        for t in range(end - 1, start, -1):
            path[t - 1] = backpointers[t, path[t]]

    return path


def evaluate_parameters(trips, transition_weight):

    correct = 0
    total = 0

    connected = 0
    total_pairs = 0

    for trip in trips:

        path = decode_cached_trip(
            trip,
            transition_weight,
        )

        rows = np.arange(len(path))

        arcs = trip["arc_ids"][rows, path]

        from_nodes = trip["from_nodes"][rows, path]

        to_nodes = trip["to_nodes"][rows, path]

        references = trip["references"]

        if (arcs < 0).any():
            raise ValueError(
                "Invalid candidate selected"
            )

        correct += int(
            (arcs == references).sum()
        )

        total += len(arcs)

        same_arc = arcs[:-1] == arcs[1:]

        forward_connected = (
            to_nodes[:-1] == from_nodes[1:]
        )

        connected += int(
            (same_arc | forward_connected).sum()
        )

        total_pairs += max(len(arcs) - 1, 0)

    accuracy = correct / total

    continuity = (
        connected / total_pairs
        if total_pairs else 0.0
    )

    score = (
        accuracy
        + CONTINUITY_WEIGHT * continuity
    )

    return {
        "accuracy": accuracy,
        "continuity": continuity,
        "score": score,
        "observations": total,
    }


def main(cache_dir=CACHE_DIR, output_dir=OUTPUT_DIR):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    print("Loading cached training trips...")

    trips = load_validation_cache(cache_dir)

    observations = sum(
        len(trip["emissions"])
        for trip in trips
    )

    if observations != 208417:
        raise ValueError(
            "Unexpected observation count"
        )

    print("Training trips:", len(trips))

    print("GPS observations:", observations)

    def objective(trial):

        weight = trial.suggest_float(
            "transition_weight",
            0.0,
            2.0,
        )

        metrics = evaluate_parameters(
            trips,
            weight,
        )

        trial.set_user_attr(
            "accuracy",
            metrics["accuracy"],
        )

        trial.set_user_attr(
            "continuity",
            metrics["continuity"],
        )

        print(
            f"Trial {trial.number}: "
            f"weight={weight:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"continuity={metrics['continuity']:.4f}, "
            f"score={metrics['score']:.4f}",
            flush=True,
        )

        return metrics["score"]

    sampler = optuna.samplers.TPESampler(
        seed=42,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    # Include classifier-only and existing
    # HMM settings as reference trials.

    study.enqueue_trial({
        "transition_weight": 0.0
    })

    study.enqueue_trial({
        "transition_weight": 1.0
    })

    study.enqueue_trial({
        "transition_weight": 0.5
    })

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    best = study.best_trial

    results = {
        "beta": BETA,
        "backward_tolerance_m": BACKWARD_TOLERANCE_M,
        "transition_weight":
            best.params["transition_weight"],
        "validation_accuracy":
            best.user_attrs["accuracy"],
        "validation_continuity":
            best.user_attrs["continuity"],
        "objective_score": best.value,
        "continuity_objective_weight":
            CONTINUITY_WEIGHT,
        "training_trips": len(trips),
        "observations": observations,
        "best_trial": best.number,
        "total_trials": N_TRIALS,
        "runtime_seconds":
            time.perf_counter() - start_time,
    }

    study.trials_dataframe().to_csv(
        output_dir / "optimisation_trials.csv",
        index=False,
    )

    with open(
        output_dir / "best_hmm_parameters.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n===== BEST HMM PARAMETERS =====")

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nResults saved in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    arguments = parser.parse_args()
    main(
        cache_dir=arguments.cache_dir,
        output_dir=arguments.output_dir,
    )
