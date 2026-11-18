"""Schema adapter bridging Member 1's ``preprocessing/`` pipeline with Member 2's ``models/``.

All spatial candidate generation, metric coordinate projection, and 20-feature
extraction live cleanly in ``preprocessing.spatial_pipeline`` (Member 1 ownership)
to eliminate duplication. This module re-exports the canonical pipeline symbols for
backward compatibility and provides ``load_feature_dataset()`` to normalize external
CSV/Parquet/Joblib feature tables into Member 2's ``(X_df, y, meta_df)`` contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.spatial_pipeline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RAW_ROOT,
    FEATURE_COLUMNS,
    META_COLUMNS,
    build_partition_dataset,
    extract_trip_candidates_and_features,
    load_trip_road_graphs,
)


def load_feature_dataset(
    source: pd.DataFrame | Path | str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Normalize an external Member 1 feature dataset (CSV, Parquet, Joblib, or DataFrame)."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Feature dataset file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".joblib", ".pkl", ".pickle"}:
            import joblib

            loaded = joblib.load(path)
            if isinstance(loaded, pd.DataFrame):
                df = loaded.copy()
            else:
                raise TypeError(f"Unsupported object loaded from {path}: {type(loaded)}")
        else:
            raise ValueError(f"Unsupported feature dataset format: {suffix}")

    alias_map = {
        "trip": "trip_id",
        "obs_idx": "obs_id",
        "observation_id": "obs_id",
        "step_idx": "obs_id",
        "arc_id": "candidate_arc_id",
        "target": "label",
        "is_match": "label",
        "y": "label",
        "true_arc_id": "ref_arc_id",
    }
    rename_dict = {
        col: alias_map[col]
        for col in df.columns
        if col in alias_map and alias_map[col] not in df.columns
    }
    if rename_dict:
        df = df.rename(columns=rename_dict)

    if "label" not in df.columns:
        raise KeyError("Feature dataset must include a 'label' (or 'target'/'is_match') column.")

    y = df["label"].to_numpy(dtype=np.int64)
    present_meta = [c for c in META_COLUMNS if c in df.columns]
    meta_df = df[present_meta].copy() if present_meta else pd.DataFrame(index=df.index)
    if "trip_id" not in meta_df.columns:
        meta_df["trip_id"] = "default_trip"

    if feature_cols is None:
        excluded = set(META_COLUMNS) | {
            "label",
            "trip",
            "obs_idx",
            "observation_id",
            "arc_id",
            "target",
            "is_match",
            "y",
        }
        feature_cols = [
            c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

    X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return X_df, y, meta_df


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_RAW_ROOT",
    "FEATURE_COLUMNS",
    "META_COLUMNS",
    "build_partition_dataset",
    "extract_trip_candidates_and_features",
    "load_feature_dataset",
    "load_trip_road_graphs",
]
