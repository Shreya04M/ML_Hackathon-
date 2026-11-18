
"""Route evaluation metrics for PathSnap."""

from typing import Any


def evaluate_route(
    predicted_route: Any,
    reference_route: Any
) -> dict[str, Any]:
    """Compare predicted and reference road arc sequences."""

    predicted = list(predicted_route)
    reference = list(reference_route)

    predicted_set = set(predicted)
    reference_set = set(reference)

    common = predicted_set & reference_set

    precision = (
        len(common) / len(predicted_set)
        if predicted_set else 0.0
    )

    recall = (
        len(common) / len(reference_set)
        if reference_set else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    union = predicted_set | reference_set

    jaccard = (
        len(common) / len(union)
        if union else 0.0
    )

    return {
        "route_precision": precision,
        "route_recall": recall,
        "route_f1": f1,
        "jaccard_similarity": jaccard,
        "exact_route_match": predicted == reference,
        "predicted_arc_count": len(predicted),
        "reference_arc_count": len(reference),
        "matched_unique_arcs": len(common),
        "missing_unique_arcs": len(
            reference_set - predicted_set
        ),
        "extra_unique_arcs": len(
            predicted_set - reference_set
        ),
    }