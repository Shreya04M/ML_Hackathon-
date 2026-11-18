"""Sequence and road-overlap metrics for recovered routes."""

from collections import Counter
from pathlib import Path


def load_reference_route(route_path):
    """Load a `.route` file as its newline-delimited arc-ID sequence."""

    text = Path(route_path).read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        return [int(value) for value in text.split()]
    except ValueError as exc:
        raise ValueError(f"Invalid route file: {route_path}") from exc


def _edit_distance(first, second):
    previous = list(range(len(second) + 1))

    for i, first_value in enumerate(first, start=1):
        current = [i]

        for j, second_value in enumerate(second, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (first_value != second_value),
            ))

        previous = current

    return previous[-1]


def _lcs_length(first, second):
    previous = [0] * (len(second) + 1)

    for first_value in first:
        current = [0]

        for j, second_value in enumerate(second, start=1):
            if first_value == second_value:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))

        previous = current

    return previous[-1]


def evaluate_route_sequence(
    predicted_route,
    reference_route,
    graph=None,
):
    """Compare route order and shared road length without using connectivity.

    Arc overlap is a length-weighted multiset measure, while the edit and LCS
    measures preserve sequence order.
    """

    predicted = [int(arc_id) for arc_id in predicted_route]
    reference = [int(arc_id) for arc_id in reference_route]

    if graph is not None:
        for arc_id in predicted + reference:
            if arc_id < 0 or arc_id >= len(graph.arcs):
                raise ValueError(f"Invalid route arc ID: {arc_id}")

        lengths = graph.arc_lengths
    else:
        lengths = None

    def arc_length(arc_id):
        return float(lengths[arc_id]) if lengths is not None else 1.0

    predicted_length = sum(arc_length(arc_id) for arc_id in predicted)
    reference_length = sum(arc_length(arc_id) for arc_id in reference)
    predicted_counts = Counter(predicted)
    reference_counts = Counter(reference)

    shared_length = sum(
        min(predicted_counts[arc_id], reference_counts[arc_id])
        * arc_length(arc_id)
        for arc_id in predicted_counts.keys() & reference_counts.keys()
    )

    overlap_precision = (
        shared_length / predicted_length if predicted_length else 0.0
    )
    overlap_recall = (
        shared_length / reference_length if reference_length else 0.0
    )
    overlap_f1 = (
        2 * overlap_precision * overlap_recall
        / (overlap_precision + overlap_recall)
        if overlap_precision + overlap_recall
        else 0.0
    )

    edit_distance = _edit_distance(predicted, reference)
    max_length = max(len(predicted), len(reference))
    lcs = _lcs_length(predicted, reference)

    return {
        "predicted_arc_count": len(predicted),
        "reference_arc_count": len(reference),
        "sequence_edit_distance": edit_distance,
        "sequence_similarity": (
            1.0 - edit_distance / max_length if max_length else 1.0
        ),
        "sequence_lcs_f1": (
            2 * lcs / (len(predicted) + len(reference))
            if predicted or reference else 1.0
        ),
        "predicted_route_length_m": predicted_length,
        "reference_route_length_m": reference_length,
        "shared_arc_length_m": shared_length,
        "road_overlap_precision": overlap_precision,
        "road_overlap_recall": overlap_recall,
        "road_overlap_f1": overlap_f1,
    }
