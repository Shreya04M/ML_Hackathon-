"""Classification metrics interface."""
from typing import Any

def calculate_metrics(y_true: Any, y_pred: Any, y_proba: Any | None = None) -> dict[str, Any]:
    """Calculate agreed metrics. TODO: implement after label contract."""
    raise NotImplementedError
