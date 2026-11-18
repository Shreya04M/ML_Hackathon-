"""Candidate road-segment generation interfaces."""
from typing import Any

def generate_candidates(observations: Any, road_network: Any, **kwargs: Any) -> Any:
    """Generate candidate segments per GPS observation."""
    raise NotImplementedError
