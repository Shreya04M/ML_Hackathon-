"""Unit tests for raw trip file loading and schema validation."""
from pathlib import Path
from preprocessing.data_loader import discover_trips, load_trip


def test_load_student_test_trip_00000000() -> None:
    trip_dir = Path("Student folder/student_test/00000000")
    assert trip_dir.is_dir()
    data = load_trip(trip_dir)
    assert data["track"] is not None and list(data["track"].columns) == ["longitude", "latitude", "timestamp"]
    assert data["nodes"] is not None and list(data["nodes"].columns) == ["node_id", "longitude", "latitude"]
    assert data["arcs"] is not None and list(data["arcs"].columns) == ["arc_id", "from_node_id", "to_node_id"]
    assert data["route"] is not None and list(data["route"].columns) == ["route_position", "arc_id"]
    assert len(data["track"]) == 2503
    assert len(data["nodes"]) == 8542
    assert len(data["arcs"]) == 18196
    assert len(data["route"]) == 179


def test_student_test_has_ten_trips() -> None:
    trips = discover_trips(Path("Student folder/student_test"))
    assert len(trips) == 10
