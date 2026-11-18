# Visualisation

Interactive Folium geographical visualisation package supporting all layers specified in `Deliverables.pdf` Section 6:

- [road_network_visualisation.py](road_network_visualisation.py): `plot_road_network()` renders directed `.arcs` geometry.
- [trajectory_visualisation.py](trajectory_visualisation.py): `plot_trajectory()` plots noisy raw GPS points with Start/End markers.
- [folium_maps.py](folium_maps.py): `create_folium_map()` and `plot_route_comparison_map()` compose multi-layer interactive Folium maps with `LayerControl` (`Road Network`, `Raw GPS Observations`, `Reference Route (Green)`, `Predicted Route (Red)`).
- [maps/](maps/): Exported interactive HTML comparison maps ([trip_00000000_comparison.html](maps/trip_00000000_comparison.html), [trip_00000052_comparison.html](maps/trip_00000052_comparison.html)).
