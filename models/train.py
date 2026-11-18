"""Training and artifact interfaces."""
from typing import Any

def save_final_model(model: Any, path: str) -> None:
    """Persist an approved model. TODO: define artifact contract."""
    raise NotImplementedError
