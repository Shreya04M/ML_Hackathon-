"""HMM emission interfaces."""
from typing import Any

def calculate_emission_probabilities(observations: Any, candidates: Any, model: Any, **kwargs: Any) -> Any:
    """Calculate emissions. TODO: define calibration contract."""
    raise NotImplementedError
