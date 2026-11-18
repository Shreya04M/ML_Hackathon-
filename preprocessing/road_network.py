"""Road-network loading interfaces."""
from pathlib import Path
from typing import Any

def load_road_network(nodes_path: Path | str, arcs_path: Path | str) -> dict[str, Any]:
    """Load nodes and arcs without assuming an unverified schema."""
    raise NotImplementedError
