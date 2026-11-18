"""High-level HMM reconstruction interfaces."""
from typing import Any

def reconstruct_route(observations: Any, candidates: Any, road_network: Any, model: Any, **kwargs: Any) -> Any:
    """Reconstruct a continuous route. TODO: connect HMM components."""
    raise NotImplementedError

def evaluate_route_continuity(predicted_route: Any, reference_route: Any) -> dict[str, Any]:
    """Evaluate route continuity. TODO: define metrics."""
    raise NotImplementedError
