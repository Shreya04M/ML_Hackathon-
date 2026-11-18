"""Typed, ID-preserving readers for the PathSnap dataset."""

from pathlib import Path
import pandas as pd


FILE_TYPES = ("track", "nodes", "arcs", "route")


def discover_partitions(dataset_root: Path) -> list[Path]:
    """Return immediate directory partitions discovered under ``dataset_root``."""
    return sorted((p for p in dataset_root.iterdir() if p.is_dir()), key=lambda p: p.name)


def discover_trips(partition_root: Path) -> list[Path]:
    """Return immediate trip directories without assuming their names."""
    return sorted((p for p in partition_root.iterdir() if p.is_dir()), key=lambda p: p.name)


def files_for_trip(trip_root: Path) -> dict[str, list[Path]]:
    """Group files by extension while preserving original paths and IDs."""
    grouped = {kind: [] for kind in FILE_TYPES}
    for path in sorted(trip_root.iterdir(), key=lambda p: p.name):
        extension = path.suffix.lower().lstrip(".")
        if extension in grouped and path.is_file():
            grouped[extension].append(path)
    return grouped

def _read(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="python")

def load_track(path: Path | str) -> pd.DataFrame:
    df = _read(Path(path), ["longitude", "latitude", "timestamp"])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_nodes(path: Path | str) -> pd.DataFrame:
    df = _read(Path(path), ["longitude", "latitude"])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "node_id", range(len(df))); return df

def load_arcs(path: Path | str) -> pd.DataFrame:
    df = _read(Path(path), ["from_node_id", "to_node_id"])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df.insert(0, "arc_id", range(len(df))); return df

def load_route(path: Path | str) -> pd.DataFrame:
    df = _read(Path(path), ["arc_id"])
    df["arc_id"] = pd.to_numeric(df["arc_id"], errors="coerce").astype("Int64")
    df.insert(0, "route_position", range(len(df))); return df

def load_trip(trip_root: Path | str) -> dict:
    grouped = files_for_trip(Path(trip_root)); result = {k: None for k in FILE_TYPES}
    if grouped["track"]: result["track"] = load_track(grouped["track"][0])
    if grouped["nodes"]: result["nodes"] = load_nodes(grouped["nodes"][0])
    if grouped["arcs"]: result["arcs"] = load_arcs(grouped["arcs"][0])
    if grouped["route"]: result["route"] = load_route(grouped["route"][0])
    return result
