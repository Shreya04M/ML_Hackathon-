"""Data loading, cleaning, road-network, candidate, and feature interfaces."""

from preprocessing.spatial_pipeline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RAW_ROOT,
    FEATURE_COLUMNS,
    META_COLUMNS,
    build_partition_dataset,
    extract_trip_candidates_and_features,
    load_trip_road_graphs,
    project_lonlat_to_meters,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_RAW_ROOT",
    "FEATURE_COLUMNS",
    "META_COLUMNS",
    "build_partition_dataset",
    "extract_trip_candidates_and_features",
    "load_trip_road_graphs",
    "project_lonlat_to_meters",
]
