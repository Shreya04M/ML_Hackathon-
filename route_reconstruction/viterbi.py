"""Viterbi decoding interface."""
from typing import Any

def viterbi_decode(emissions: Any, transitions: Any, **kwargs: Any) -> Any:
    """Decode best candidate sequence. TODO: define tie-breaking."""
    raise NotImplementedError
