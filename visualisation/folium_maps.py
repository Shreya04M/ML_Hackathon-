"""Folium map composition helpers."""
from typing import Any

import folium


def create_folium_map(*layers: Any, **kwargs: Any) -> folium.Map:
    """Create a map and attach supplied Folium layer objects.

    ``center`` may be a ``(latitude, longitude)`` pair. If omitted, Folium's
    default world view is used. Layers can be FeatureGroups, markers, paths,
    or any other Folium MacroElement.
    """
    center = kwargs.pop("center", [0.0, 0.0])
    zoom_start = kwargs.pop("zoom_start", 2)
    fmap = folium.Map(location=center, zoom_start=zoom_start, **kwargs)
    for layer in layers:
        if layer is not None:
            fmap.add_child(layer)
    if layers:
        folium.LayerControl().add_to(fmap)
    return fmap
