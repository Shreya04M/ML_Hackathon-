"""GPS trajectory plotting helpers."""
from typing import Any

import folium


def plot_trajectory(observations: Any, **kwargs: Any) -> folium.FeatureGroup:
    """Return a Folium layer for an ordered GPS trajectory DataFrame."""
    layer = folium.FeatureGroup(name=kwargs.pop("name", "GPS trajectory"), show=kwargs.pop("show", True))
    longitude = kwargs.pop("longitude", "longitude")
    latitude = kwargs.pop("latitude", "latitude")
    points = [(row[latitude], row[longitude]) for _, row in observations.iterrows()]
    if not points:
        return layer
    folium.PolyLine(points, color=kwargs.pop("color", "#1976d2"), weight=kwargs.pop("weight", 4), opacity=kwargs.pop("opacity", 0.9)).add_to(layer)
    folium.CircleMarker(points[0], radius=5, color="#2e7d32", fill=True, tooltip="Start").add_to(layer)
    folium.CircleMarker(points[-1], radius=5, color="#c62828", fill=True, tooltip="End").add_to(layer)
    return layer
