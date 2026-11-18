"""Read and validate PathSnap trip files."""
from pathlib import Path
from typing import Any

def load_trip(trip_id: str, partition: str, data_root: Path | str = "data/raw") -> dict[str, Any]:
    """Load one trip's four raw file types. TODO: implement after schema inspection."""
    raise NotImplementedError
