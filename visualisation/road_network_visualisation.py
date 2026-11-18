"""Road-network plotting helpers."""
from typing import Any

import folium


def plot_road_network(road_network: Any, **kwargs: Any) -> folium.FeatureGroup:
    """Return a Folium layer containing directed road segments.

    ``road_network`` is the dictionary returned by
    ``preprocessing.road_network.load_road_network``.
    """
    segments = road_network["segments"]
    layer = folium.FeatureGroup(name=kwargs.pop("name", "Road network"), show=kwargs.pop("show", True))
    color = kwargs.pop("color", "#6c757d")
    weight = kwargs.pop("weight", 2)
    opacity = kwargs.pop("opacity", 0.65)
    for row in segments.itertuples(index=False):
        folium.PolyLine(
            locations=[(row.start_lat, row.start_lon), (row.end_lat, row.end_lon)],
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=f"Arc {row.arc_id}",
        ).add_to(layer)
    return layer
