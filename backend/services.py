"""Service-layer integration interfaces."""

def list_available_trips() -> list[str]:
    """List prediction trips. TODO: implement."""
    raise NotImplementedError

def predict_trip(trip_id: str) -> dict[str, str]:
    """Run prediction. TODO: implement."""
    raise NotImplementedError
