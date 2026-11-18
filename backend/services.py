
"""Service-layer integration for PathSnap."""

import os
from pathlib import Path
import folium

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.environ.get(
        "PATHSNAP_TRIP_DIR",
        str(PROJECT_ROOT / "data" / "Student folder" / "student_test")
    )
)


def list_available_trips() -> list[str]:
    """Discover valid trips from the approved dataset."""

    if not DATA_DIR.is_dir():
        raise FileNotFoundError("Trip dataset not available")

    trip_ids = []

    for folder in DATA_DIR.iterdir():

        if not folder.is_dir():
            continue

        trip_id = folder.name

        required_files = [
            folder / f"{trip_id}.track",
            folder / f"{trip_id}.nodes",
            folder / f"{trip_id}.arcs",
        ]

        if all(file.is_file() for file in required_files):
            trip_ids.append(trip_id)

    return sorted(trip_ids)


def predict_trip(trip_id: str) -> dict:
    """Run prediction for an approved trip."""

    if trip_id not in list_available_trips():
        raise ValueError("Invalid trip ID")

    raise NotImplementedError(
        "Model and HMM integration pending"
    )

def generate_trip_map(
    trip_id: str,
    predicted_route: list[int] | None = None
) -> str:
    """Generate a map with GPS, road network and optional prediction."""

    if trip_id not in list_available_trips():
        raise ValueError("Invalid trip ID")

    track_file = DATA_DIR / trip_id / f"{trip_id}.track"

    coordinates = []

    with open(track_file, "r") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            lon = float(parts[0])
            lat = float(parts[1])

            coordinates.append([lat, lon])

    if not coordinates:
        raise ValueError("No GPS coordinates found")

    route_map = folium.Map(
        location=coordinates[0],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # Load road network nodes
    nodes_file = DATA_DIR / trip_id / f"{trip_id}.nodes"
    arcs_file = DATA_DIR / trip_id / f"{trip_id}.arcs"

    nodes = []

    with open(nodes_file, "r") as file:
        for line in file:
            lon, lat = map(float, line.split()[:2])
            nodes.append([lon, lat])

    # Build road network geometry
    road_segments = []

    with open(arcs_file, "r") as file:
        for line in file:
            start, end = map(int, line.split()[:2])

            road_segments.append([
                nodes[start],
                nodes[end]
            ])

    # Display the road network
    folium.GeoJson(
        {
            "type": "MultiLineString",
            "coordinates": road_segments
        },
        name="Road Network",
        style_function=lambda feature: {
            "color": "#808080",
            "weight": 1.5,
            "opacity": 0.5
        }
    ).add_to(route_map)



    folium.PolyLine(
        coordinates,
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip="Raw GPS trajectory"
    ).add_to(route_map)

    folium.Marker(
        coordinates[0],
        tooltip="Start",
        icon=folium.Icon(color="green")
    ).add_to(route_map)

    folium.Marker(
        coordinates[-1],
        tooltip="End",
        icon=folium.Icon(color="red")
    ).add_to(route_map)

    # Display the predicted reconstructed route
    if predicted_route is not None:
        predicted_segments = []

        for arc_id in predicted_route:
            if not 0 <= arc_id < len(road_segments):
                raise ValueError(
                    f"Invalid predicted arc ID: {arc_id}"
                )

            predicted_segments.append(
                road_segments[arc_id]
            )

        if predicted_segments:
            folium.GeoJson(
                {
                    "type": "MultiLineString",
                    "coordinates": predicted_segments
                },
                name="Predicted Route",
                style_function=lambda feature: {
                    "color": "#dc2626",
                    "weight": 5,
                    "opacity": 0.9
                },
                tooltip="Predicted reconstructed route"
            ).add_to(route_map)



    route_map.fit_bounds(coordinates)
    folium.LayerControl().add_to(route_map)

    return route_map.get_root().render()
