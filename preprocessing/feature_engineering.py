"""Candidate feature extraction interfaces."""
from typing import Any

def extract_features(observations: Any, candidates: Any, road_network: Any) -> Any:
    """Extract model-ready features without evaluator-label leakage."""
    raise NotImplementedError
