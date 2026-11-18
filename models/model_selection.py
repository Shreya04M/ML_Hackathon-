"""Model evaluation interfaces."""
from typing import Any

def evaluate_model(model: Any, X: Any, y: Any, **kwargs: Any) -> dict[str, Any]:
    """Compute agreed metrics and timing. TODO: implement."""
    raise NotImplementedError

def compare_models(results: Any) -> Any:
    """Compare measured results only. TODO: define selection rules."""
    raise NotImplementedError
