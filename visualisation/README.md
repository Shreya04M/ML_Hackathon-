# Visualisation

Interactive Folium geographical visualisation package (`visualisation/`) supporting all four mandatory layers specified in `Deliverables.pdf` Section 6:

- `road_network_visualisation.py`: `plot_road_network()` renders directed `.arcs` road-network geometry.
- `trajectory_visualisation.py`: `plot_trajectory()` renders noisy raw GPS observations with Start/End markers.
- `folium_maps.py`: `create_folium_map()` and `plot_route_comparison_map()` compose multi-layer interactive maps with `LayerControl` (`Road Network`, `Raw GPS Observations`, `Reference Route (Ground Truth)` in green, and `Predicted Route` in red).
- `maps/`: Exported interactive HTML maps for representative student-testing trips (`trip_00000000_comparison.html`, `trip_00000052_comparison.html`).
