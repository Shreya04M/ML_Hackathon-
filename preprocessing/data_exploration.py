"""Read-only dataset inspection interfaces."""
from pathlib import Path
from typing import Any

def inspect_dataset(data_root: Path | str = "data/raw") -> dict[str, Any]:
    """Summarise available files and schemas without modifying data."""
    raise NotImplementedError
