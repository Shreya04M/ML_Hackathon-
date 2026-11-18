"""
PathSnap - Intermediate Road Segment Recovery
Member 3 - Phase 2
"""

import numpy as np


def recover_continuous_route(
    predicted_arcs,
    graph,
    **kwargs,
):
    """
    Recover intermediate road segments between
    consecutive HMM predictions.

    Returns separate route sections if a directed
    connection cannot be found.
    """

    predicted_arcs = [
        int(arc_id)
        for arc_id in predicted_arcs
    ]

    if not predicted_arcs:
        raise ValueError("Predicted route is empty")

    for arc_id in predicted_arcs:

        if arc_id < 0 or arc_id >= len(graph.arcs):
            raise ValueError(
                f"Invalid arc ID: {arc_id}"
            )

    recovered_arcs = [predicted_arcs[0]]

    route_sections = []

    current_section = [predicted_arcs[0]]

    inserted_arcs = 0

    disconnected_pairs = []

    repeated_arcs = 0

    for index in range(
        1, len(predicted_arcs)
    ):

        previous_arc = predicted_arcs[index - 1]

        current_arc = predicted_arcs[index]

        # Avoid consecutive repetitions

        if previous_arc == current_arc:

            repeated_arcs += 1

            continue

        previous_end = int(
            graph.to_nodes[previous_arc]
        )

        current_start = int(
            graph.from_nodes[current_arc]
        )

        # Directly connected road segments

        if previous_end == current_start:

            recovered_arcs.append(current_arc)

            current_section.append(current_arc)

            continue

        # Recover intermediate roads

        distance, middle_arcs = graph.shortest_path(
            previous_end,
            current_start,
        )

        # Do not invent connections

        if not np.isfinite(distance):

            disconnected_pairs.append({
                "observation_index": index,
                "previous_arc": previous_arc,
                "current_arc": current_arc,
            })

            route_sections.append(current_section)

            current_section = [current_arc]

            recovered_arcs.append(current_arc)

            continue

        # Insert recovered intermediate roads

        recovered_arcs.extend(middle_arcs)

        current_section.extend(middle_arcs)

        inserted_arcs += len(middle_arcs)

        recovered_arcs.append(current_arc)

        current_section.append(current_arc)

    route_sections.append(current_section)

    return {
        "recovered_arcs": recovered_arcs,
        "route_sections": route_sections,
        "original_predictions": len(predicted_arcs),
        "recovered_arc_count": len(recovered_arcs),
        "inserted_arcs": inserted_arcs,
        "repeated_arcs_removed": repeated_arcs,
        "disconnected_pairs": disconnected_pairs,
        "is_continuous": len(disconnected_pairs) == 0,
    }
