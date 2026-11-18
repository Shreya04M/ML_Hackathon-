"""Project configuration loaded from environment variables.

This module only resolves paths and does not read or modify dataset contents.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_dataset_root() -> Path:
    """Return the configured dataset directory as an absolute ``Path``.

    ``PATHSNAP_DATA_ROOT`` is required in ``.env``. Relative values are resolved
    relative to the repository root, not the caller's current directory.
    """
    configured = os.getenv("PATHSNAP_DATA_ROOT")
    if not configured:
        raise RuntimeError("PATHSNAP_DATA_ROOT is missing from .env")
    path = Path(configured).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
