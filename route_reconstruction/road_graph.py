"""
PathSnap - Weighted Road Network
Member 3 - Phase 2
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


EARTH_RADIUS = 6371000.0


def haversine_distance(point1, point2):
    """
    Calculate geographical distance in metres.

    Coordinates must be longitude, latitude.
    """

    point1 = np.asarray(point1, dtype=float)
    point2 = np.asarray(point2, dtype=float)

    lon1 = np.radians(point1[..., 0])
    lat1 = np.radians(point1[..., 1])

    lon2 = np.radians(point2[..., 0])
    lat2 = np.radians(point2[..., 1])

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return (
        2 * EARTH_RADIUS
        * np.arcsin(np.sqrt(a))
    )


class RoadGraph:

    def __init__(self, nodes, arcs):

        self.nodes = np.asarray(
            nodes, dtype=float
        )

        self.arcs = np.asarray(
            arcs, dtype=np.int64
        )

        if (
            self.nodes.ndim != 2
            or self.nodes.shape[1] != 2
        ):
            raise ValueError("Invalid node format")

        if (
            self.arcs.ndim != 2
            or self.arcs.shape[1] != 2
        ):
            raise ValueError("Invalid arc format")

        if len(self.nodes) == 0 or len(self.arcs) == 0:
            raise ValueError("Empty road network")

        if not np.isfinite(self.nodes).all():
            raise ValueError("Invalid node coordinates")

        if (
            (self.arcs < 0).any()
            or (self.arcs >= len(self.nodes)).any()
        ):
            raise ValueError("Invalid road node IDs")

        self.from_nodes = self.arcs[:, 0]
        self.to_nodes = self.arcs[:, 1]

        # Calculate physical road lengths

        self.arc_lengths = haversine_distance(
            self.nodes[self.from_nodes],
            self.nodes[self.to_nodes],
        )

        self.arc_lengths = np.maximum(
            self.arc_lengths,
            0.01,
        )

        # Construct directed weighted graph.
        # Keep shortest edge for parallel connections.

        edges = {}

        for arc_id, (u, v) in enumerate(self.arcs):

            key = (int(u), int(v))
            length = float(self.arc_lengths[arc_id])

            if key not in edges or length < edges[key][0]:
                edges[key] = (length, arc_id)

        self.edge_arc_ids = {
            key: value[1]
            for key, value in edges.items()
        }

        rows = [key[0] for key in edges]
        columns = [key[1] for key in edges]

        distances = [
            value[0] for value in edges.values()
        ]

        self.graph = csr_matrix(
            (distances, (rows, columns)),
            shape=(
                len(self.nodes),
                len(self.nodes),
            ),
        )

    @lru_cache(maxsize=128)
    def distances_from(self, source):

        return dijkstra(
            self.graph,
            directed=True,
            indices=int(source),
        )

    def shortest_distance(self, source, target):

        source = int(source)
        target = int(target)

        return float(
            self.distances_from(source)[target]
        )

    def shortest_path(self, source, target):
        """
        Return distance and intermediate arc IDs.
        """

        source = int(source)
        target = int(target)

        if source == target:
            return 0.0, []

        distances, predecessors = dijkstra(
            self.graph,
            directed=True,
            indices=source,
            return_predecessors=True,
        )

        if not np.isfinite(distances[target]):
            return float("inf"), []

        path = []
        current = target

        while current != source:

            previous = int(predecessors[current])

            if previous < 0:
                return float("inf"), []

            path.append(
                int(self.edge_arc_ids[(previous, current)])
            )

            current = previous

        path.reverse()

        return float(distances[target]), path


def load_road_graph(
    trip_id,
    partition="student_test",
    data_root="Student folder",
):

    trip_name = f"{int(trip_id):08d}"

    trip_dir = (
        Path(data_root)
        / partition
        / trip_name
    )

    nodes_path = trip_dir / f"{trip_name}.nodes"
    arcs_path = trip_dir / f"{trip_name}.arcs"

    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)

    if not arcs_path.exists():
        raise FileNotFoundError(arcs_path)

    nodes = np.loadtxt(
        nodes_path,
        dtype=float,
        ndmin=2,
    )

    arcs = np.loadtxt(
        arcs_path,
        dtype=np.int64,
        ndmin=2,
    )

    return RoadGraph(nodes, arcs)
