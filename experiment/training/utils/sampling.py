from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment.training.data.features import FeatureStore
from experiment.training.utils.common import ensure_dir


RAW_CONSISTENCY_PROFILE_VERSION = "raw_consistency_v1"


def _profile_paths(feature_store: FeatureStore) -> tuple[Path, Path]:
    phase_dir = ensure_dir(feature_store.phase_dir / "sampling_profiles")
    stem = f"{RAW_CONSISTENCY_PROFILE_VERSION}_f16"
    return (
        phase_dir / f"{stem}.npy",
        phase_dir / f"{stem}.json",
    )


def _raw_group_slices(feature_store: FeatureStore) -> tuple[slice, slice]:
    core_groups = feature_store.manifest["core_groups"]
    raw_spec = core_groups["raw_x"]
    missing_spec = core_groups["missing_mask"]
    return (
        slice(int(raw_spec["start"]), int(raw_spec["end"])),
        slice(int(missing_spec["start"]), int(missing_spec["end"])),
    )


def _compute_raw_feature_stats(
    core: np.ndarray,
    raw_slice: slice,
    missing_slice: slice,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = int(raw_slice.stop - raw_slice.start)
    count = np.zeros(feature_dim, dtype=np.float64)
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)

    num_nodes = int(core.shape[0])
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        raw_chunk = np.asarray(core[start:end, raw_slice], dtype=np.float32)
        missing_chunk = np.asarray(core[start:end, missing_slice], dtype=np.float32)
        observed = missing_chunk < 0.5
        total += np.sum(np.where(observed, raw_chunk, 0.0), axis=0, dtype=np.float64)
        total_sq += np.sum(np.where(observed, raw_chunk * raw_chunk, 0.0), axis=0, dtype=np.float64)
        count += np.sum(observed, axis=0, dtype=np.float64)

    count = np.maximum(count, 1.0)
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1e-6)
    std = np.sqrt(variance, dtype=np.float64)
    return mean.astype(np.float32), std.astype(np.float32)


def load_or_build_raw_consistency_profile(
    feature_store: FeatureStore,
    *,
    chunk_size: int = 250000,
) -> np.ndarray:
    profile_path, meta_path = _profile_paths(feature_store)
    if profile_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") == RAW_CONSISTENCY_PROFILE_VERSION:
            return np.load(profile_path, mmap_mode="r")

    raw_slice, missing_slice = _raw_group_slices(feature_store)
    mean, std = _compute_raw_feature_stats(
        feature_store.core,
        raw_slice,
        missing_slice,
        chunk_size=chunk_size,
    )

    num_nodes = int(feature_store.core.shape[0])
    profile_dim = int((raw_slice.stop - raw_slice.start) + (missing_slice.stop - missing_slice.start))
    profile = np.lib.format.open_memmap(
        profile_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_nodes, profile_dim),
    )

    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        raw_chunk = np.asarray(feature_store.core[start:end, raw_slice], dtype=np.float32)
        missing_chunk = np.asarray(feature_store.core[start:end, missing_slice], dtype=np.float32)
        filled = np.where(missing_chunk > 0.5, mean.reshape(1, -1), raw_chunk)
        zscore = (filled - mean.reshape(1, -1)) / std.reshape(1, -1)
        chunk_profile = np.concatenate([zscore, missing_chunk], axis=1).astype(np.float32, copy=False)
        norm = np.linalg.norm(chunk_profile, axis=1, keepdims=True)
        chunk_profile = chunk_profile / np.clip(norm, 1e-6, None)
        profile[start:end] = chunk_profile.astype(np.float16)

    meta = {
        "version": RAW_CONSISTENCY_PROFILE_VERSION,
        "num_nodes": num_nodes,
        "profile_dim": profile_dim,
        "raw_slice": [int(raw_slice.start), int(raw_slice.stop)],
        "missing_slice": [int(missing_slice.start), int(missing_slice.stop)],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return np.load(profile_path, mmap_mode="r")
