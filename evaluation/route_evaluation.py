"""Route evaluation interface."""
from typing import Any

def evaluate_route(predicted_route: Any, reference_route: Any) -> dict[str, Any]:
    """Measure route quality. TODO: define metrics."""
    raise NotImplementedError
