"""Schema adapter and leakage-free spatial candidate feature pipeline for Member 2.

This module bridges Member 1's preprocessed feature tables (DataFrame, CSV, Parquet,
or Joblib) with Member 2's model training and evaluation suite. When Member 1's
processed artifact is not yet present on disk, it also provides a standalone,
read-only spatial candidate generator and feature extractor directly over the raw
PathSnap dataset (`Student folder/Student folder/{train,student_test}`), strictly
respecting partition isolation and never leaking reference labels into features.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_RAW_ROOT = Path("Student folder") / "Student folder"
DEFAULT_CACHE_DIR = Path("models") / "artifacts" / "cache"

FEATURE_COLUMNS: list[str] = [
    "ortho_dist_m",
    "dist_to_start_m",
    "dist_to_end_m",
    "proj_fraction",
    "is_endpoint_proj",
    "segment_length_m",
    "log_segment_length_m",
    "dist_rank",
    "dist_diff_from_min_m",
    "dist_ratio_to_min",
    "dist_zscore_in_obs",
    "gps_speed_mps",
    "dt_prev_s",
    "heading_diff_deg",
    "heading_cos",
    "heading_sin",
    "speed_weighted_heading_cos",
    "prev_cand_connected",
    "min_dist_from_prev_cand_m",
    "node_in_out_degree_sum",
]

META_COLUMNS: list[str] = [
    "trip_id",
    "obs_id",
    "timestamp",
    "candidate_arc_id",
    "from_node_id",
    "to_node_id",
    "ref_arc_id",
    "is_ref_in_candidates",
]


def _project_lonlat_to_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    ref_lat_deg: float,
) -> np.ndarray:
    """Project WGS84 longitude/latitude degrees into local tangent-plane meters.

    Uses an equirectangular projection centered on ``ref_lat_deg`` so Euclidean
    distances and bearings in local meters avoid angular degree distortion.
    """
    cos_lat0 = float(np.cos(np.radians(ref_lat_deg)))
    x = np.radians(lon) * EARTH_RADIUS_M * cos_lat0
    y = np.radians(lat) * EARTH_RADIUS_M
    return np.column_stack([x, y]).astype(np.float64)


def _compute_gps_kinematics(
    track_xy: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-observation step dt, speed (m/s), heading (rad), and validity."""
    n_obs = len(track_xy)
    dt_prev = np.ones(n_obs, dtype=np.float64)
    speed = np.zeros(n_obs, dtype=np.float64)
    heading = np.zeros(n_obs, dtype=np.float64)
    heading_valid = np.zeros(n_obs, dtype=bool)

    if n_obs < 2:
        return dt_prev, speed, heading, heading_valid

    diffs = np.diff(track_xy, axis=0)
    dts = np.maximum(np.diff(timestamps), 1e-3)
    dists = np.linalg.norm(diffs, axis=1)
    step_speed = np.clip(dists / dts, 0.0, 60.0)

    dt_prev[1:] = dts
    dt_prev[0] = dts[0]
    speed[1:] = step_speed
    speed[0] = step_speed[0]

    # Use central difference where consecutive steps are compact for smoother bearings;
    # fall back to forward/backward difference at endpoints or large GPS gaps (>10s).
    for i in range(n_obs):
        if i == 0:
            vec = track_xy[1] - track_xy[0]
            dt_span = max(timestamps[1] - timestamps[0], 1e-3)
        elif i == n_obs - 1:
            vec = track_xy[-1] - track_xy[-2]
            dt_span = max(timestamps[-1] - timestamps[-2], 1e-3)
        elif dt_prev[i] <= 10.0 and dt_prev[i + 1] <= 10.0:
            vec = track_xy[i + 1] - track_xy[i - 1]
            dt_span = max(timestamps[i + 1] - timestamps[i - 1], 1e-3)
        elif dt_prev[i] <= dt_prev[i + 1]:
            vec = track_xy[i] - track_xy[i - 1]
            dt_span = dt_prev[i]
        else:
            vec = track_xy[i + 1] - track_xy[i]
            dt_span = dt_prev[i + 1]

        norm = float(np.hypot(vec[0], vec[1]))
        # Guard against stationary GPS jitter (< 0.5 m displacement over step).
        if norm >= 0.5 and (norm / dt_span) >= 0.25:
            heading[i] = float(np.arctan2(vec[1], vec[0]))
            heading_valid[i] = True

    return dt_prev, speed, heading, heading_valid


def _build_adaptive_arc_points(
    p1: np.ndarray,
    p2: np.ndarray,
    arc_lengths: np.ndarray,
    step_m: float = 60.0,
    max_samples_per_arc: int = 35,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points along road arcs adaptively so long highway arcs stay indexed."""
    n_arcs = len(p1)
    base_alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)[:, None, None]
    base_pts = (p1[None, :, :] * (1.0 - base_alphas) + p2[None, :, :] * base_alphas).reshape(-1, 2)
    base_ids = np.tile(np.arange(n_arcs, dtype=np.int64), 5)

    long_mask = arc_lengths > (step_m * 4.0)
    if not np.any(long_mask):
        return base_pts, base_ids

    extra_pts: list[np.ndarray] = [base_pts]
    extra_ids: list[np.ndarray] = [base_ids]
    for aid in np.flatnonzero(long_mask):
        n_extra = int(min(max_samples_per_arc, max(6, np.ceil(arc_lengths[aid] / step_m))))
        ts = np.linspace(0.05, 0.95, n_extra)[:, None]
        pts = p1[aid][None, :] * (1.0 - ts) + p2[aid][None, :] * ts
        extra_pts.append(pts)
        extra_ids.append(np.full(n_extra, aid, dtype=np.int64))

    return np.vstack(extra_pts), np.concatenate(extra_ids)


def _associate_reference_arcs(
    track_xy: np.ndarray,
    heading: np.ndarray,
    heading_valid: np.ndarray,
    p1: np.ndarray,
    v: np.ndarray,
    v_len2_safe: np.ndarray,
    arc_heading: np.ndarray,
    arc_valid_dir: np.ndarray,
    route_arcs: np.ndarray,
) -> np.ndarray:
    """Associate each GPS observation with its closest compatible segment in ``route_arcs``.

    Combines orthogonal distance to `.route` segments with a mild directional penalty
    for antiparallel carriageways on bidirectional roads, strictly for training and
    student-test evaluation labeling (never passed as an input feature).
    """
    n_obs = len(track_xy)
    if len(route_arcs) == 0:
        return np.full(n_obs, -1, dtype=np.int64)

    rt_p1 = p1[route_arcs]
    rt_v = v[route_arcs]
    rt_vl2 = v_len2_safe[route_arcs]
    rt_hdg = arc_heading[route_arcs]
    rt_vdir = arc_valid_dir[route_arcs]

    ref_arcs = np.empty(n_obs, dtype=np.int64)
    chunk = 500
    for start in range(0, n_obs, chunk):
        end = min(start + chunk, n_obs)
        diff = track_xy[start:end, None, :] - rt_p1[None, :, :]
        t_proj = np.clip(np.sum(diff * rt_v[None, :, :], axis=2) / rt_vl2[None, :], 0.0, 1.0)
        proj = rt_p1[None, :, :] + t_proj[:, :, None] * rt_v[None, :, :]
        dists = np.linalg.norm(track_xy[start:end, None, :] - proj, axis=2)

        # Tie-break antiparallel arcs along two-way roads using GPS direction when moving.
        ang_diff = np.abs(heading[start:end, None] - rt_hdg[None, :])
        cos_align = np.cos(ang_diff)
        dir_mask = heading_valid[start:end, None] & rt_vdir[None, :]
        dir_penalty = np.where(dir_mask, (1.0 - cos_align) * 4.0, 0.0)
        best_rt_idx = np.argmin(dists + dir_penalty, axis=1)
        ref_arcs[start:end] = route_arcs[best_rt_idx]

    return ref_arcs


def extract_trip_candidates_and_features(
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray | None = None,
    trip_id: str = "unknown",
    top_k: int = 6,
    pool_k: int = 60,
) -> tuple[pd.DataFrame, np.ndarray | None, pd.DataFrame]:
    """Generate candidate road segments and 20 leakage-free features for a single trip.

    Parameters
    ----------
    track : ndarray of shape (T, 3)
        Columns ``[longitude, latitude, timestamp]``.
    nodes : ndarray of shape (N_nodes, 2)
        Columns ``[longitude, latitude]`` indexed by 0-based ``node_id``.
    arcs : ndarray of shape (N_arcs, 2)
        Integer columns ``[from_node_id, to_node_id]`` indexed by 0-based ``arc_id``.
    route : ndarray of shape (L,), optional
        Reference sequence of ``arc_id`` values. Must be ``None`` during unseen
        evaluation inference so ground truth is never touched.
    trip_id : str
        Trip identifier string.
    top_k : int
        Number of unique candidate road segments retained per GPS observation.
    pool_k : int
        Initial sample-point neighbors queried from ``cKDTree`` before exact
        point-to-segment ranking.
    """
    n_obs = len(track)
    n_nodes = len(nodes)
    n_arcs = len(arcs)

    ref_lat = float(np.mean(track[:, 1]))
    track_xy = _project_lonlat_to_meters(track[:, 0], track[:, 1], ref_lat)
    nodes_xy = _project_lonlat_to_meters(nodes[:, 0], nodes[:, 1], ref_lat)

    from_ids = arcs[:, 0].astype(np.int64)
    to_ids = arcs[:, 1].astype(np.int64)
    p1 = nodes_xy[from_ids]
    p2 = nodes_xy[to_ids]
    vec = p2 - p1
    v_len2 = np.sum(vec**2, axis=1)
    arc_lengths = np.sqrt(v_len2)
    v_len2_safe = np.where(v_len2 > 1e-12, v_len2, 1.0)
    arc_valid_dir = arc_lengths > 1e-3
    arc_heading = np.where(arc_valid_dir, np.arctan2(vec[:, 1], vec[:, 0]), 0.0)

    # Precompute node degrees in road network.
    out_deg = np.bincount(from_ids, minlength=n_nodes)
    in_deg = np.bincount(to_ids, minlength=n_nodes)
    arc_deg_sum = (out_deg[to_ids] + in_deg[from_ids]).astype(np.float64)

    # Spatial index on road arcs.
    sample_pts, sample_arc_ids = _build_adaptive_arc_points(p1, p2, arc_lengths)
    tree = cKDTree(sample_pts)
    query_k = min(pool_k, len(sample_pts))
    _, nn_idx = tree.query(track_xy, k=query_k, workers=-1)
    if query_k == 1:
        nn_idx = nn_idx[:, None]
    cand_pool = sample_arc_ids[nn_idx]  # (T, query_k)

    # Exact orthogonal distances from track points to pooled segments.
    cp1 = p1[cand_pool]
    cv = vec[cand_pool]
    cvl2 = v_len2_safe[cand_pool]
    ct = np.clip(np.sum((track_xy[:, None, :] - cp1) * cv, axis=2) / cvl2, 0.0, 1.0)
    cproj = cp1 + ct[:, :, None] * cv
    cdist = np.linalg.norm(track_xy[:, None, :] - cproj, axis=2)

    order = np.argsort(cdist, axis=1)
    sorted_pool = np.take_along_axis(cand_pool, order, axis=1)

    actual_k = min(top_k, n_arcs)
    top_arcs = np.empty((n_obs, actual_k), dtype=np.int64)
    for i in range(n_obs):
        row = sorted_pool[i]
        _, first_pos = np.unique(row, return_index=True)
        uniq_ordered = row[np.sort(first_pos)]
        if len(uniq_ordered) >= actual_k:
            top_arcs[i] = uniq_ordered[:actual_k]
        else:
            pad_len = actual_k - len(uniq_ordered)
            top_arcs[i, : len(uniq_ordered)] = uniq_ordered
            top_arcs[i, len(uniq_ordered) :] = uniq_ordered[-1]

    # Exact geometry for retained top_k candidates.
    tp1 = p1[top_arcs]
    tp2 = p2[top_arcs]
    tvec = vec[top_arcs]
    tvl2 = v_len2_safe[top_arcs]
    t_proj = np.clip(np.sum((track_xy[:, None, :] - tp1) * tvec, axis=2) / tvl2, 0.0, 1.0)
    t_proj_pt = tp1 + t_proj[:, :, None] * tvec

    ortho_dist = np.linalg.norm(track_xy[:, None, :] - t_proj_pt, axis=2)
    dist_to_start = np.linalg.norm(track_xy[:, None, :] - tp1, axis=2)
    dist_to_end = np.linalg.norm(track_xy[:, None, :] - tp2, axis=2)
    seg_len = arc_lengths[top_arcs]
    log_seg_len = np.log1p(seg_len)
    is_endpoint = ((t_proj <= 1e-6) | (t_proj >= 1.0 - 1e-6)).astype(np.float64)

    # Ensure ascending rank ordering within observation.
    obs_sort = np.argsort(ortho_dist, axis=1)
    top_arcs = np.take_along_axis(top_arcs, obs_sort, axis=1)
    ortho_dist = np.take_along_axis(ortho_dist, obs_sort, axis=1)
    dist_to_start = np.take_along_axis(dist_to_start, obs_sort, axis=1)
    dist_to_end = np.take_along_axis(dist_to_end, obs_sort, axis=1)
    t_proj = np.take_along_axis(t_proj, obs_sort, axis=1)
    seg_len = np.take_along_axis(seg_len, obs_sort, axis=1)
    log_seg_len = np.take_along_axis(log_seg_len, obs_sort, axis=1)
    is_endpoint = np.take_along_axis(is_endpoint, obs_sort, axis=1)

    dist_rank = np.tile(np.arange(actual_k, dtype=np.float64), (n_obs, 1))
    min_d = ortho_dist[:, [0]]
    dist_diff_min = ortho_dist - min_d
    dist_ratio_min = (ortho_dist + 0.5) / (min_d + 0.5)
    std_d = np.maximum(np.std(ortho_dist, axis=1, keepdims=True), 1.0)
    dist_z = (ortho_dist - np.mean(ortho_dist, axis=1, keepdims=True)) / std_d

    dt_prev, gps_speed, gps_heading, gps_hdg_valid = _compute_gps_kinematics(track_xy, track[:, 2])
    cand_hdg = arc_heading[top_arcs]
    cand_vdir = arc_valid_dir[top_arcs]
    valid_pair = gps_hdg_valid[:, None] & cand_vdir

    raw_ang_diff = np.abs(gps_heading[:, None] - cand_hdg)
    raw_ang_diff = np.where(raw_ang_diff > np.pi, 2.0 * np.pi - raw_ang_diff, raw_ang_diff)
    heading_diff_deg = np.where(valid_pair, np.degrees(raw_ang_diff), 90.0)
    heading_cos = np.where(valid_pair, np.cos(raw_ang_diff), 0.0)
    heading_sin = np.where(valid_pair, np.sin(raw_ang_diff), 1.0)
    speed_weight = np.minimum(gps_speed[:, None] / 5.0, 1.0)
    speed_weighted_cos = speed_weight * heading_cos

    # Unsupervised previous-observation topological continuity cue (zero label usage).
    cand_from = from_ids[top_arcs]
    cand_to = to_ids[top_arcs]
    prev_connected = np.zeros((n_obs, actual_k), dtype=np.float64)
    min_prev_node_dist = np.zeros((n_obs, actual_k), dtype=np.float64)
    if n_obs > 1:
        prev_arcs = top_arcs[:-1]  # (T-1, K)
        prev_to = cand_to[:-1]     # (T-1, K)
        curr_arcs = top_arcs[1:]   # (T-1, K)
        curr_from = cand_from[1:]  # (T-1, K)

        same_arc = curr_arcs[:, :, None] == prev_arcs[:, None, :]
        edge_conn = curr_from[:, :, None] == prev_to[:, None, :]
        prev_connected[1:] = np.any(same_arc | edge_conn, axis=2).astype(np.float64)

        prev_to_xy = nodes_xy[prev_to]      # (T-1, K, 2)
        curr_from_xy = nodes_xy[curr_from]  # (T-1, K, 2)
        node_dists = np.linalg.norm(curr_from_xy[:, :, None, :] - prev_to_xy[:, None, :, :], axis=3)
        node_dists = np.where(same_arc, 0.0, node_dists)
        min_prev_node_dist[1:] = np.min(node_dists, axis=2)

    deg_sum = arc_deg_sum[top_arcs]

    # Flatten (T, K) -> (T * K,)
    feature_dict = {
        "ortho_dist_m": ortho_dist.ravel(),
        "dist_to_start_m": dist_to_start.ravel(),
        "dist_to_end_m": dist_to_end.ravel(),
        "proj_fraction": t_proj.ravel(),
        "is_endpoint_proj": is_endpoint.ravel(),
        "segment_length_m": seg_len.ravel(),
        "log_segment_length_m": log_seg_len.ravel(),
        "dist_rank": dist_rank.ravel(),
        "dist_diff_from_min_m": dist_diff_min.ravel(),
        "dist_ratio_to_min": dist_ratio_min.ravel(),
        "dist_zscore_in_obs": dist_z.ravel(),
        "gps_speed_mps": np.repeat(gps_speed, actual_k),
        "dt_prev_s": np.repeat(dt_prev, actual_k),
        "heading_diff_deg": heading_diff_deg.ravel(),
        "heading_cos": heading_cos.ravel(),
        "heading_sin": heading_sin.ravel(),
        "speed_weighted_heading_cos": speed_weighted_cos.ravel(),
        "prev_cand_connected": prev_connected.ravel(),
        "min_dist_from_prev_cand_m": min_prev_node_dist.ravel(),
        "node_in_out_degree_sum": deg_sum.ravel(),
    }
    X_df = pd.DataFrame(feature_dict, columns=FEATURE_COLUMNS).astype(np.float32)

    if route is not None:
        ref_arcs = _associate_reference_arcs(
            track_xy, gps_heading, gps_hdg_valid, p1, vec, v_len2_safe, arc_heading, arc_valid_dir, route
        )
        in_cands = np.any(top_arcs == ref_arcs[:, None], axis=1)
        y = (top_arcs == ref_arcs[:, None]).astype(np.int64).ravel()
        ref_rep = np.repeat(ref_arcs, actual_k)
        hit_rep = np.repeat(in_cands, actual_k)
    else:
        y = None
        ref_rep = np.full(n_obs * actual_k, -1, dtype=np.int64)
        hit_rep = np.zeros(n_obs * actual_k, dtype=bool)

    meta_df = pd.DataFrame(
        {
            "trip_id": np.repeat(trip_id, n_obs * actual_k),
            "obs_id": np.repeat(np.arange(n_obs, dtype=np.int64), actual_k),
            "timestamp": np.repeat(track[:, 2].astype(np.float64), actual_k),
            "candidate_arc_id": top_arcs.ravel(),
            "from_node_id": cand_from.ravel(),
            "to_node_id": cand_to.ravel(),
            "ref_arc_id": ref_rep,
            "is_ref_in_candidates": hit_rep,
        }
    )
    return X_df, y, meta_df


def load_trip_road_graphs(
    partition: str,
    data_root: Path | str = DEFAULT_RAW_ROOT,
) -> dict[str, dict[str, np.ndarray]]:
    """Load directed arc endpoints (``from_node``, ``to_node``) per trip for continuity metrics."""
    part_dir = Path(data_root) / partition
    if not part_dir.exists():
        return {}
    graphs: dict[str, dict[str, np.ndarray]] = {}
    for trip_dir in sorted(part_dir.iterdir()):
        if not trip_dir.is_dir():
            continue
        trip_id = trip_dir.name
        arcs_path = trip_dir / f"{trip_id}.arcs"
        if not arcs_path.exists():
            continue
        arcs = np.loadtxt(arcs_path, dtype=np.int64)
        if arcs.ndim == 1:
            arcs = arcs[None, :]
        graphs[trip_id] = {"from_node": arcs[:, 0], "to_node": arcs[:, 1]}
    return graphs


def build_partition_dataset(
    partition: str = "train",
    data_root: Path | str = DEFAULT_RAW_ROOT,
    top_k: int = 6,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Build or load cached feature matrix, labels, metadata, and road graphs for a partition."""
    data_root = Path(data_root)
    part_dir = data_root / partition
    if not part_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {part_dir}")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{partition}_k{top_k}_features.parquet"

    road_graphs = load_trip_road_graphs(partition, data_root)

    if use_cache and cache_path is not None and cache_path.exists():
        logger.info("Loading cached %s features from %s", partition, cache_path)
        combined = pd.read_parquet(cache_path)
        y = combined["label"].to_numpy(dtype=np.int64)
        meta_df = combined[META_COLUMNS].copy()
        X_df = combined[FEATURE_COLUMNS].astype(np.float32)
        return X_df, y, meta_df, road_graphs

    trip_dirs = sorted(p for p in part_dir.iterdir() if p.is_dir())
    x_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name
        track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64)
        nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64)
        arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)
        if arcs.ndim == 1:
            arcs = arcs[None, :]
        route_file = trip_dir / f"{trip_id}.route"
        route = np.loadtxt(route_file, dtype=np.int64) if route_file.exists() else None
        if route is not None and route.ndim == 0:
            route = np.array([int(route)], dtype=np.int64)

        x_trip, y_trip, m_trip = extract_trip_candidates_and_features(
            track=track,
            nodes=nodes,
            arcs=arcs,
            route=route,
            trip_id=trip_id,
            top_k=top_k,
        )
        x_parts.append(x_trip)
        if y_trip is not None:
            y_parts.append(y_trip)
        meta_parts.append(m_trip)

    X_df = pd.concat(x_parts, ignore_index=True)
    y = np.concatenate(y_parts) if y_parts else np.empty(0, dtype=np.int64)
    meta_df = pd.concat(meta_parts, ignore_index=True)

    if cache_path is not None:
        combined = pd.concat([meta_df, X_df], axis=1)
        combined["label"] = y
        combined.to_parquet(cache_path, index=False)
        logger.info("Cached %s dataset (%d rows) to %s", partition, len(combined), cache_path)

    return X_df, y, meta_df, road_graphs


def load_feature_dataset(
    source: pd.DataFrame | Path | str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Normalize an external Member 1 feature dataset (CSV, Parquet, Joblib, or DataFrame).

    Accepts flexible column names from Member 1's pipeline and maps them into the
    standardized ``(X_df, y, meta_df)`` contract expected by ``models/``.
    """
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
    rename_dict = {col: alias_map[col] for col in df.columns if col in alias_map and alias_map[col] not in df.columns}
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
        excluded = set(META_COLUMNS) | {"label", "trip", "obs_idx", "observation_id", "arc_id", "target", "is_match", "y"}
        feature_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return X_df, y, meta_df
