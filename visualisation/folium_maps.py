"""Folium map composition helpers."""
from __future__ import annotations

from typing import Any, Sequence

import folium


def create_folium_map(*layers: Any, **kwargs: Any) -> folium.Map:
    """Create a map and attach supplied Folium layer objects."""
    center = kwargs.pop("center", [0.0, 0.0])
    zoom_start = kwargs.pop("zoom_start", 2)
    fmap = folium.Map(location=center, zoom_start=zoom_start, **kwargs)
    for layer in layers:
        if layer is not None:
            fmap.add_child(layer)
    if layers:
        folium.LayerControl().add_to(fmap)
    return fmap


def _arc_layer(
    segments_df: Any,
    arc_ids: Sequence[int],
    name: str,
    color: str,
    weight: int = 5,
    opacity: float = 0.85,
) -> folium.FeatureGroup:
    """Build a Folium FeatureGroup for a sequence of road arc IDs."""
    layer = folium.FeatureGroup(name=name, show=True)
    seg_lookup = segments_df.set_index("arc_id")
    for aid in arc_ids:
        if int(aid) not in seg_lookup.index:
            continue
        row = seg_lookup.loc[int(aid)]
        folium.PolyLine(
            locations=[(float(row["start_lat"]), float(row["start_lon"])), (float(row["end_lat"]), float(row["end_lon"]))],
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=f"{name}: Arc {int(aid)}",
        ).add_to(layer)
    return layer


def plot_route_comparison_map(
    observations: Any,
    road_network: dict[str, Any],
    predicted_arc_ids: Sequence[int] | None = None,
    reference_arc_ids: Sequence[int] | None = None,
    zoom_start: int = 14,
) -> folium.Map:
    """Create a 4-layer Folium map (Road Network, Raw GPS, Reference Route, Predicted Route)."""
    from .road_network_visualisation import plot_road_network
    from .trajectory_visualisation import plot_trajectory

    pts = list(zip(observations["latitude"].astype(float), observations["longitude"].astype(float)))
    center = [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)] if pts else [0.0, 0.0]

    layers: list[Any] = [
        plot_road_network(road_network, name="Road Network", color="#6c757d", weight=1.5, opacity=0.45),
        plot_trajectory(observations, name="Raw GPS Observations", color="#1976d2", weight=3, opacity=0.8),
    ]
    if reference_arc_ids is not None:
        layers.append(_arc_layer(road_network["segments"], reference_arc_ids, "Reference Route (Ground Truth)", "#2e7d32", weight=6, opacity=0.8))
    if predicted_arc_ids is not None:
        layers.append(_arc_layer(road_network["segments"], predicted_arc_ids, "Predicted Route", "#dc2626", weight=4, opacity=0.9))

    fmap = create_folium_map(*layers, center=center, zoom_start=zoom_start)
    if pts:
        fmap.fit_bounds(pts)
    return fmap
