"""Bayesian optimisation interfaces."""
from typing import Any

def define_search_space(model_name: str) -> dict[str, Any]:
    """Describe a search space. TODO: agree backend and parameters."""
    raise NotImplementedError

def optimise_hyperparameters(model: Any, X: Any, y: Any, search_space: Any, **kwargs: Any) -> Any:
    """Optimise parameters and record the search. TODO: implement."""
    raise NotImplementedError
