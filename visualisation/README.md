# Visualisation

Folium visualization helpers are implemented for composing maps, displaying
road-network segments, and plotting GPS trajectories.

```python
import folium
from visualisation.folium_maps import create_folium_map
from visualisation.road_network_visualisation import plot_road_network
from visualisation.trajectory_visualisation import plot_trajectory

network_layer = plot_road_network(road_network)
trajectory_layer = plot_trajectory(observations)
map_view = create_folium_map(
    network_layer,
    trajectory_layer,
    center=[observations.latitude.mean(), observations.longitude.mean()],
    zoom_start=13,
)
map_view.save("trajectory_map.html")
```
