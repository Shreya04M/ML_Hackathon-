"""Evaluate optimized HMM recovery against all student-test route references."""

import json
import time
from pathlib import Path

import pandas as pd

from route_reconstruction.distance_transition import load_gps_track
from route_reconstruction.hmm import (
    evaluate_route_continuity,
    reconstruct_route,
)
from route_reconstruction.road_graph import load_road_graph
from route_reconstruction.route_metrics import (
    evaluate_route_sequence,
    load_reference_route,
)
from route_reconstruction.route_recovery import recover_continuous_route


INPUT_PATH = Path(
    "models/artifacts/results/"
    "m3_handoff_student_test_probabilities.csv.gz"
)
PARAMETER_PATH = Path(
    "route_reconstruction/experiments/phase2_improved/"
    "hmm_optimisation/best_hmm_parameters.json"
)
OUTPUT_DIR = Path(
    "route_reconstruction/experiments/phase2_improved"
)
BASELINE_DIR = Path("experiments/results")
MAX_SPEED_MPS = 60.0
GPS_TOLERANCE_M = 30.0
BACKWARD_TOLERANCE_M = 30.0


def _route_path(trip_id):
    trip_name = f"{int(trip_id):08d}"
    return (
        Path("Student folder")
        / "student_test"
        / trip_name
        / f"{trip_name}.route"
    )


def _mean_metrics(rows, keys):
    if not rows:
        return {key: None for key in keys}

    return {
        key: float(pd.DataFrame(rows)[key].mean())
        for key in keys
    }


def _load_old_route_metrics(filename, routes_filename):
    summary_path = BASELINE_DIR / filename
    routes_path = BASELINE_DIR / routes_filename

    summaries = pd.read_csv(summary_path)

    with routes_path.open(encoding="utf-8") as file:
        routes = json.load(file)

    return summaries, routes


def main(
    input_path=INPUT_PATH,
    parameter_path=PARAMETER_PATH,
    output_dir=OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(input_path)
    with Path(parameter_path).open(encoding="utf-8") as file:
        parameters = json.load(file)

    if (
        parameters.get("training_trips") != 80
        or parameters.get("observations") != 208417
        or parameters.get("total_trials") != 25
    ):
        raise ValueError(
            "HMM parameters are not from the approved 80-trip training run"
        )

    observation_count = candidates[
        ["trip_id", "obs_id"]
    ].drop_duplicates().shape[0]
    if candidates["trip_id"].nunique() != 10 or observation_count != 19160:
        raise ValueError(
            "Expected all ten student-test trips and 19,160 observations"
        )

    required = {
        "trip_id",
        "obs_id",
        "candidate_arc_id",
        "from_node_id",
        "to_node_id",
        "pred_prob",
        "ref_arc_id",
    }
    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")

    old_constrained, old_constrained_routes = _load_old_route_metrics(
        "constrained_recovery/recovery_summary.csv",
        "constrained_recovery/recovered_routes.json",
    )
    old_unconstrained, old_unconstrained_routes = _load_old_route_metrics(
        "optimised_route_recovery/route_recovery_summary.csv",
        "optimised_route_recovery/recovered_routes.json",
    )
    previous_prediction_metrics = pd.read_csv(
        BASELINE_DIR
        / "optimised_reconstruction"
        / "per_trip_comparison.csv"
    ).set_index("trip_id")

    # Keep evaluation labels out of the HMM call.
    inference_columns = [
        column for column in candidates.columns
        if column not in {
            "ref_arc_id",
            "is_ref_in_candidates",
            "label",
            "pred_log_prob",
        }
    ]

    all_predictions = []
    per_trip_rows = []
    new_routes = {}
    old_constrained_route_metrics = []
    old_unconstrained_route_metrics = []
    legacy_runtime_seconds = 0.0
    recovery_runtime_seconds = 0.0
    hmm_runtime_seconds = 0.0
    hmm_pipeline_runtime_seconds = 0.0

    for trip_id, labeled_trip in candidates.groupby("trip_id", sort=True):
        trip_start = time.perf_counter()
        print(
            f"Processing student-test trip {int(trip_id):08d}",
            flush=True,
        )
        labeled_trip = labeled_trip.sort_values("obs_id").reset_index(drop=True)
        inference_trip = labeled_trip[inference_columns].copy()

        graph = load_road_graph(int(trip_id))
        track = load_gps_track(int(trip_id))

        hmm_start = time.perf_counter()
        route = reconstruct_route(
            candidates=inference_trip,
            road_network=graph,
            gps_track=track,
            transition_mode="distance",
            transition_weight=float(parameters["transition_weight"]),
            beta=float(parameters["beta"]),
            backward_tolerance_m=BACKWARD_TOLERANCE_M,
        )
        hmm_elapsed = time.perf_counter() - hmm_start
        hmm_runtime_seconds += hmm_elapsed

        reference_counts = labeled_trip.groupby(
            ["trip_id", "obs_id"]
        )["ref_arc_id"].nunique(dropna=False)
        if (reference_counts != 1).any():
            raise ValueError(f"Conflicting reference labels for trip {trip_id}")

        reference_by_observation = labeled_trip[
            ["trip_id", "obs_id", "ref_arc_id"]
        ].drop_duplicates(["trip_id", "obs_id"])
        evaluated_route = route.merge(
            reference_by_observation,
            on=["trip_id", "obs_id"],
            how="left",
            validate="one_to_one",
        )
        if evaluated_route["ref_arc_id"].isna().any():
            raise ValueError(f"Missing observation labels for trip {trip_id}")

        prediction_metrics = evaluate_route_continuity(evaluated_route)
        all_predictions.append(evaluated_route)
        hmm_pipeline_runtime_seconds += (
            time.perf_counter() - trip_start
        )

        predicted_arcs = route["hmm_arc_id"].astype(int).tolist()
        observations = route["obs_id"].to_numpy(dtype=int)

        legacy_start = time.perf_counter()
        recover_continuous_route(
            predicted_arcs,
            graph,
        )
        legacy_elapsed = time.perf_counter() - legacy_start
        legacy_runtime_seconds += legacy_elapsed

        recovery_start = time.perf_counter()
        recovery = recover_continuous_route(
            predicted_arcs,
            graph,
            gps_track=track,
            observation_ids=observations,
            max_speed_mps=MAX_SPEED_MPS,
            gps_tolerance_m=GPS_TOLERANCE_M,
            backward_tolerance_m=BACKWARD_TOLERANCE_M,
        )
        recovery_elapsed = time.perf_counter() - recovery_start
        recovery_runtime_seconds += recovery_elapsed
        if len(recovery["route_sections"]) != (
            len(recovery["rejected_connections"]) + 1
        ):
            raise ValueError(
                f"Recovery sections do not match rejections for trip {trip_id}"
            )
        if [
            arc
            for section in recovery["route_sections"]
            for arc in section
        ] != recovery["recovered_arcs"]:
            raise ValueError(f"Recovery route alignment failed for trip {trip_id}")
        for section in recovery["route_sections"]:
            for first, second in zip(section[:-1], section[1:]):
                if graph.to_nodes[first] != graph.from_nodes[second]:
                    raise ValueError(
                        f"Invalid directed connection in trip {trip_id}"
                    )

        reference = load_reference_route(_route_path(trip_id))
        route_quality = evaluate_route_sequence(
            recovery["recovered_arcs"],
            reference,
            graph=graph,
        )

        old_constrained_result = old_constrained_routes[str(int(trip_id))]
        old_unconstrained_result = old_unconstrained_routes[str(int(trip_id))]
        old_constrained_quality = evaluate_route_sequence(
            old_constrained_result["recovered_arcs"],
            reference,
            graph=graph,
        )
        old_unconstrained_quality = evaluate_route_sequence(
            old_unconstrained_result["recovered_arcs"],
            reference,
            graph=graph,
        )
        old_constrained_route_metrics.append(old_constrained_quality)
        old_unconstrained_route_metrics.append(old_unconstrained_quality)
        old_prediction = previous_prediction_metrics.loc[int(trip_id)]
        old_constrained_reasons = {}
        for item in old_constrained_result["rejected_connections"]:
            reason = item["reason"]
            old_constrained_reasons[reason] = (
                old_constrained_reasons.get(reason, 0) + 1
            )

        reason_counts = {}
        for item in recovery["rejected_connections"]:
            reason_counts[item["reason"]] = (
                reason_counts.get(item["reason"], 0) + 1
            )

        summary = {
            "trip_id": int(trip_id),
            "observations": len(route),
            "classifier_accuracy": prediction_metrics["classifier_accuracy"],
            "hmm_accuracy": prediction_metrics["hmm_accuracy"],
            "classifier_continuity": prediction_metrics[
                "classifier_continuity"
            ],
            "hmm_continuity": prediction_metrics["hmm_continuity"],
            "previous_hmm_accuracy": old_prediction["hmm_accuracy"],
            "previous_hmm_continuity": old_prediction["hmm_continuity"],
            "inserted_arcs": recovery["inserted_arcs"],
            "rejected_connections": len(
                recovery["rejected_connections"]
            ),
            "unreachable_transitions": reason_counts.get(
                "unreachable", 0
            ),
            "implausible_transitions": reason_counts.get(
                "implausible_distance", 0
            ),
            "invalid_timestamp_transitions": reason_counts.get(
                "invalid_timestamp", 0
            ),
            "invalid_gps_transitions": reason_counts.get(
                "invalid_gps", 0
            ),
            "route_sections": len(recovery["route_sections"]),
            "is_continuous": recovery["is_continuous"],
            "previous_constrained_inserted_arcs":
                old_constrained_result["inserted_arcs"],
            "previous_constrained_rejected_connections": len(
                old_constrained_result["rejected_connections"]
            ),
            "previous_constrained_implausible_transitions":
                old_constrained_reasons.get("implausible_distance", 0),
            "previous_constrained_route_sections": len(
                old_constrained_result["route_sections"]
            ),
            "previous_unconstrained_inserted_arcs":
                old_unconstrained_result["inserted_arcs"],
            "previous_unconstrained_disconnected_pairs": len(
                old_unconstrained_result["disconnected_pairs"]
            ),
            "previous_unconstrained_route_sections": len(
                old_unconstrained_result["route_sections"]
            ),
            "sequence_edit_distance": route_quality[
                "sequence_edit_distance"
            ],
            "sequence_similarity": route_quality[
                "sequence_similarity"
            ],
            "sequence_lcs_f1": route_quality["sequence_lcs_f1"],
            "road_overlap_precision": route_quality[
                "road_overlap_precision"
            ],
            "road_overlap_recall": route_quality[
                "road_overlap_recall"
            ],
            "road_overlap_f1": route_quality["road_overlap_f1"],
            "previous_constrained_sequence_lcs_f1":
                old_constrained_quality["sequence_lcs_f1"],
            "previous_constrained_road_overlap_f1":
                old_constrained_quality["road_overlap_f1"],
            "previous_unconstrained_sequence_lcs_f1":
                old_unconstrained_quality["sequence_lcs_f1"],
            "previous_unconstrained_road_overlap_f1":
                old_unconstrained_quality["road_overlap_f1"],
            "hmm_runtime_seconds": hmm_elapsed,
            "legacy_recovery_runtime_seconds": legacy_elapsed,
            "constrained_recovery_runtime_seconds": recovery_elapsed,
            "trip_runtime_seconds": time.perf_counter() - trip_start,
        }
        per_trip_rows.append(summary)
        new_routes[str(int(trip_id))] = {
            **recovery,
            "route_reference_metrics": route_quality,
        }
        print(
            "  accuracy={:.4f}, continuity={:.4f}, "
            "inserted={}, rejected={}, sections={}, runtime={:.2f}s".format(
                summary["hmm_accuracy"],
                summary["hmm_continuity"],
                summary["inserted_arcs"],
                summary["rejected_connections"],
                summary["route_sections"],
                summary["trip_runtime_seconds"],
            ),
            flush=True,
        )

    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    observation_metrics = evaluate_route_continuity(combined_predictions)
    trip_metrics = pd.DataFrame(per_trip_rows)
    new_route_quality = _mean_metrics(
        [
            item["route_reference_metrics"]
            for item in new_routes.values()
        ],
        [
            "sequence_similarity",
            "sequence_lcs_f1",
            "road_overlap_precision",
            "road_overlap_recall",
            "road_overlap_f1",
        ],
    )
    old_constrained_quality = _mean_metrics(
        old_constrained_route_metrics,
        [
            "sequence_similarity",
            "sequence_lcs_f1",
            "road_overlap_precision",
            "road_overlap_recall",
            "road_overlap_f1",
        ],
    )
    old_unconstrained_quality = _mean_metrics(
        old_unconstrained_route_metrics,
        [
            "sequence_similarity",
            "sequence_lcs_f1",
            "road_overlap_precision",
            "road_overlap_recall",
            "road_overlap_f1",
        ],
    )

    total_sections = int(trip_metrics["route_sections"].sum())
    total_rejections = int(trip_metrics["rejected_connections"].sum())
    old_constrained_reasons = {}
    for trip_result in old_constrained_routes.values():
        for item in trip_result["rejected_connections"]:
            reason = item["reason"]
            old_constrained_reasons[reason] = (
                old_constrained_reasons.get(reason, 0) + 1
            )

    summary = {
        "total_trips": int(trip_metrics["trip_id"].nunique()),
        "total_observations": int(trip_metrics["observations"].sum()),
        "observation_metrics": observation_metrics,
        "recovery": {
            "inserted_arcs": int(trip_metrics["inserted_arcs"].sum()),
            "rejected_connections": total_rejections,
            "unreachable_transitions": int(
                trip_metrics["unreachable_transitions"].sum()
            ),
            "implausible_transitions": int(
                trip_metrics["implausible_transitions"].sum()
            ),
            "invalid_timestamp_transitions": int(
                trip_metrics["invalid_timestamp_transitions"].sum()
            ),
            "route_sections": total_sections,
            "extra_route_sections": int(
                (trip_metrics["route_sections"] - 1).clip(lower=0).sum()
            ),
            "fragmented_trips": int(
                (trip_metrics["route_sections"] > 1).sum()
            ),
            "fully_connected_trips": int(
                trip_metrics["is_continuous"].sum()
            ),
        },
        "route_quality_macro_average": new_route_quality,
        "previous_constrained_route_quality_macro_average":
            old_constrained_quality,
        "previous_unconstrained_route_quality_macro_average":
            old_unconstrained_quality,
        "runtime_seconds": {
            "hmm_total": hmm_runtime_seconds,
            "hmm_pipeline_total": hmm_pipeline_runtime_seconds,
            "legacy_unconstrained_recovery_total": legacy_runtime_seconds,
            "constrained_recovery_total": recovery_runtime_seconds,
            "overall": float(trip_metrics["trip_runtime_seconds"].sum()),
        },
        "parameters": {
            "beta": float(parameters["beta"]),
            "transition_weight": float(parameters["transition_weight"]),
            "max_speed_mps": MAX_SPEED_MPS,
            "gps_tolerance_m": GPS_TOLERANCE_M,
            "backward_tolerance_m": BACKWARD_TOLERANCE_M,
        },
        "previous_constrained_recovery_counts": {
            "inserted_arcs": int(old_constrained["inserted_arcs"].sum()),
            "rejected_connections": int(
                old_constrained["rejected_connections"].sum()
            ),
            "route_sections": int(old_constrained["route_sections"].sum()),
            "fully_connected_trips": int(
                old_constrained["is_continuous"].sum()
            ),
            "implausible_transitions": old_constrained_reasons.get(
                "implausible_distance", 0
            ),
            "unreachable_transitions": old_constrained_reasons.get(
                "unreachable", 0
            ),
        },
        "previous_unconstrained_recovery_counts": {
            "inserted_arcs": int(old_unconstrained["inserted_arcs"].sum()),
            "disconnected_pairs": int(
                old_unconstrained["disconnected_pairs"].sum()
            ),
            "route_sections": int(old_unconstrained["route_sections"].sum()),
            "fully_connected_trips": int(
                old_unconstrained["is_continuous"].sum()
            ),
        },
    }

    trip_metrics.to_csv(output_dir / "per_trip_comparison.csv", index=False)
    prediction_output = combined_predictions.drop(
        columns=["ref_arc_id"],
        errors="ignore",
    )
    prediction_output.to_csv(
        output_dir / "reconstructed_predictions.csv",
        index=False,
    )
    with (output_dir / "recovered_routes.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(new_routes, file, indent=2)
    with (output_dir / "overall_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nEvaluation outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
