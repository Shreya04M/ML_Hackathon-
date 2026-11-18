"""Integration tests for FastAPI backend endpoints and live pipeline serving."""
import os
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app


def test_health_trips_predict_and_map(monkeypatch) -> None:
    monkeypatch.setenv("PATHSNAP_TRIP_DIR", str(Path("Student folder/student_test").resolve()))
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    trips_resp = client.get("/trips")
    assert trips_resp.status_code == 200
    trips = trips_resp.json()
    assert len(trips) == 10
    assert "00000052" in trips

    predict_resp = client.post("/predict", json={"trip_id": "00000052"})
    assert predict_resp.status_code == 200
    payload = predict_resp.json()
    assert payload["trip_id"] == "00000052"
    assert payload["status"] == "completed"
    assert len(payload["predicted_route"]) > 0
    assert payload["metrics"]["accuracy"] > 0.90
    assert payload["metrics"]["route_f1"] > 0.90
    assert "route_map" in (payload.get("map_html") or "") or "leaflet" in (payload.get("map_html") or "").lower()

    map_resp = client.get("/map/00000052")
    assert map_resp.status_code == 200
    assert "text/html" in map_resp.headers["content-type"]

    bad_resp = client.post("/predict", json={"trip_id": "99999999"})
    assert bad_resp.status_code == 404
