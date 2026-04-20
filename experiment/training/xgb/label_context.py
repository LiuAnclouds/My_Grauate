from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment.training.common import ensure_dir
from experiment.training.features import load_graph_cache

KNOWN_LABEL_ANCHOR_NAMES = [
    "known_normal",
    "known_fraud",
    "known_bg2",
    "known_bg3",
    "known_foreground",
    "known_background",
]
KNOWN_LABEL_BASE_NAMES = (
    "known_normal",
    "known_fraud",
    "known_bg2",
    "known_bg3",
)
BEST_LABEL_CONTEXT_PRESET_VERSION = "best_label_context_v1"
BEST_LABEL_CONTEXT_ANCHORS = (
    "known_fraud",
    "known_bg2",
    "known_bg3",
    "known_background",
)
BEST_LABEL_CONTEXT_RELATION_ANCHORS = (
    "known_fraud",
    "known_background",
)
BEST_LABEL_CONTEXT_RECENT_WINDOWS = (14, 60)
BEST_LABEL_CONTEXT_RELATION_WINDOW = 30


def build_known_label_code(
    phase: str,
    labels: np.ndarray,
    split,
    num_nodes: int,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int8)
    codes = np.full(num_nodes, -1, dtype=np.int8)

    bg2_mask = labels_arr == 2
    bg3_mask = labels_arr == 3
    codes[bg2_mask] = 2
    codes[bg3_mask] = 3

    if phase == "phase1":
        train_ids = np.asarray(split.train_ids, dtype=np.int32)
        train_labels = labels_arr[train_ids]
        codes[train_ids[train_labels == 0]] = 0
        codes[train_ids[train_labels == 1]] = 1

    return codes


def build_known_label_anchor_matrix(
    phase: str,
    labels: np.ndarray,
    split,
    num_nodes: int,
) -> np.ndarray:
    codes = build_known_label_code(
        phase=phase,
        labels=labels,
        split=split,
        num_nodes=num_nodes,
    )
    anchors = np.zeros((num_nodes, len(KNOWN_LABEL_ANCHOR_NAMES)), dtype=np.float32)

    normal_mask = codes == 0
    fraud_mask = codes == 1
    bg2_mask = codes == 2
    bg3_mask = codes == 3
    foreground_mask = normal_mask | fraud_mask
    background_mask = bg2_mask | bg3_mask

    anchors[normal_mask, 0] = 1.0
    anchors[fraud_mask, 1] = 1.0
    anchors[bg2_mask, 2] = 1.0
    anchors[bg3_mask, 3] = 1.0
    anchors[foreground_mask, 4] = 1.0
    anchors[background_mask, 5] = 1.0
    return anchors


def _edge_rows_from_ptr(ptr: np.ndarray, num_nodes: int) -> np.ndarray:
    degree = np.diff(np.asarray(ptr, dtype=np.int64)).astype(np.int32, copy=False)
    return np.repeat(np.arange(num_nodes, dtype=np.int32), degree)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator_f = np.asarray(numerator, dtype=np.float32)
    denominator_f = np.asarray(denominator, dtype=np.float32)
    output = np.zeros_like(numerator_f, dtype=np.float32)
    valid = denominator_f > 0.0
    output[valid] = numerator_f[valid] / denominator_f[valid]
    return output


def _count_known_labels(
    rows: np.ndarray,
    neighbor_codes: np.ndarray,
    mask: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    if not np.any(mask):
        return np.zeros((num_nodes, len(KNOWN_LABEL_BASE_NAMES)), dtype=np.float32)
    masked_rows = np.asarray(rows[mask], dtype=np.int64)
    masked_codes = np.asarray(neighbor_codes[mask], dtype=np.int64)
    valid = masked_codes >= 0
    if not np.any(valid):
        return np.zeros((num_nodes, len(KNOWN_LABEL_BASE_NAMES)), dtype=np.float32)
    flat_index = masked_rows[valid] * len(KNOWN_LABEL_BASE_NAMES) + masked_codes[valid]
    counts = np.bincount(
        flat_index,
        minlength=num_nodes * len(KNOWN_LABEL_BASE_NAMES),
    )
    return counts.reshape(num_nodes, len(KNOWN_LABEL_BASE_NAMES)).astype(np.float32, copy=False)


def _named_count_views(counts: np.ndarray) -> dict[str, np.ndarray]:
    normal = np.asarray(counts[:, 0], dtype=np.float32, copy=False)
    fraud = np.asarray(counts[:, 1], dtype=np.float32, copy=False)
    bg2 = np.asarray(counts[:, 2], dtype=np.float32, copy=False)
    bg3 = np.asarray(counts[:, 3], dtype=np.float32, copy=False)
    return {
        "known_normal": normal,
        "known_fraud": fraud,
        "known_bg2": bg2,
        "known_bg3": bg3,
        "known_foreground": (normal + fraud).astype(np.float32, copy=False),
        "known_background": (bg2 + bg3).astype(np.float32, copy=False),
    }


def _anchor_suffix(anchor_name: str) -> str:
    return anchor_name.replace("known_", "")


def _validate_anchor_names(anchor_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    resolved = tuple(str(name) for name in anchor_names)
    unknown = [name for name in resolved if name not in KNOWN_LABEL_ANCHOR_NAMES]
    if unknown:
        raise KeyError(f"Unknown label-context anchors: {unknown}")
    if not resolved:
        raise ValueError("label-context anchors must not be empty.")
    return resolved


def _feature_names(
    *,
    anchor_names: tuple[str, ...],
    relation_anchor_names: tuple[str, ...],
    recent_windows: tuple[int, ...],
    relation_window: int,
) -> list[str]:
    names: list[str] = []
    for anchor_name in anchor_names:
        suffix = _anchor_suffix(anchor_name)
        names.extend(
            [
                f"labelctx_snap_in_{suffix}_count",
                f"labelctx_snap_out_{suffix}_count",
                f"labelctx_snap_in_{suffix}_ratio",
                f"labelctx_snap_out_{suffix}_ratio",
            ]
        )
    for recent_days in recent_windows:
        for anchor_name in anchor_names:
            suffix = _anchor_suffix(anchor_name)
            names.extend(
                [
                    f"labelctx_recent{recent_days}_in_{suffix}_count",
                    f"labelctx_recent{recent_days}_out_{suffix}_count",
                    f"labelctx_recent{recent_days}_in_{suffix}_ratio",
                    f"labelctx_recent{recent_days}_out_{suffix}_ratio",
                    f"labelctx_recent{recent_days}_in_{suffix}_over_snap",
                    f"labelctx_recent{recent_days}_out_{suffix}_over_snap",
                ]
            )
    for edge_type in range(1, 12):
        for anchor_name in relation_anchor_names:
            suffix = _anchor_suffix(anchor_name)
            names.extend(
                [
                    f"labelctx_rel{relation_window}_in_type_{edge_type}_{suffix}_count",
                    f"labelctx_rel{relation_window}_out_type_{edge_type}_{suffix}_count",
                ]
            )
    return names


def _build_phase_temporal_label_context(
    phase: str,
    feature_dir: Path,
    labels: np.ndarray,
    split,
    split_ids: dict[str, np.ndarray],
    anchor_names: tuple[str, ...],
    relation_anchor_names: tuple[str, ...],
    recent_windows: tuple[int, ...],
    relation_window: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    graph_cache = load_graph_cache(phase, outdir=feature_dir)
    num_nodes = int(graph_cache.num_nodes)
    label_codes = build_known_label_code(
        phase=phase,
        labels=labels,
        split=split,
        num_nodes=num_nodes,
    )
    feature_names = _feature_names(
        anchor_names=anchor_names,
        relation_anchor_names=relation_anchor_names,
        recent_windows=recent_windows,
        relation_window=relation_window,
    )
    output = {
        split_name: np.zeros((node_ids.shape[0], len(feature_names)), dtype=np.float32)
        for split_name, node_ids in split_ids.items()
    }
    split_nodes = {
        split_name: np.asarray(node_ids, dtype=np.int32)
        for split_name, node_ids in split_ids.items()
    }
    bucket_ids = np.asarray(graph_cache.node_time_bucket, dtype=np.int8)
    split_bucket_views = {
        split_name: {
            bucket_idx: np.flatnonzero(bucket_ids[node_ids] == bucket_idx).astype(np.int32, copy=False)
            for bucket_idx in range(len(graph_cache.time_windows))
        }
        for split_name, node_ids in split_nodes.items()
    }

    out_rows = _edge_rows_from_ptr(graph_cache.out_ptr, num_nodes)
    out_neighbors = np.asarray(graph_cache.out_neighbors, dtype=np.int32)
    out_neighbor_codes = label_codes[out_neighbors]
    out_timestamps = np.asarray(graph_cache.out_edge_timestamp, dtype=np.int32)
    out_types = np.asarray(graph_cache.out_edge_type, dtype=np.int16)

    in_rows = _edge_rows_from_ptr(graph_cache.in_ptr, num_nodes)
    in_neighbors = np.asarray(graph_cache.in_neighbors, dtype=np.int32)
    in_neighbor_codes = label_codes[in_neighbors]
    in_timestamps = np.asarray(graph_cache.in_edge_timestamp, dtype=np.int32)
    in_types = np.asarray(graph_cache.in_edge_type, dtype=np.int16)

    for bucket_idx, window in enumerate(graph_cache.time_windows):
        active_splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name, node_ids in split_nodes.items():
            positions = split_bucket_views[split_name][bucket_idx]
            if positions.size == 0:
                continue
            active_splits[split_name] = (positions, node_ids[positions])
        if not active_splits:
            continue

        snapshot_end = int(window["end_day"])
        out_snapshot_mask = out_timestamps <= snapshot_end
        in_snapshot_mask = in_timestamps <= snapshot_end
        out_snapshot_total = np.bincount(
            out_rows[out_snapshot_mask],
            minlength=num_nodes,
        ).astype(np.float32, copy=False)
        in_snapshot_total = np.bincount(
            in_rows[in_snapshot_mask],
            minlength=num_nodes,
        ).astype(np.float32, copy=False)
        out_snapshot_named = _named_count_views(
            _count_known_labels(
                rows=out_rows,
                neighbor_codes=out_neighbor_codes,
                mask=out_snapshot_mask,
                num_nodes=num_nodes,
            )
        )
        in_snapshot_named = _named_count_views(
            _count_known_labels(
                rows=in_rows,
                neighbor_codes=in_neighbor_codes,
                mask=in_snapshot_mask,
                num_nodes=num_nodes,
            )
        )

        col = 0
        for anchor_name in anchor_names:
            in_count = in_snapshot_named[anchor_name]
            out_count = out_snapshot_named[anchor_name]
            for split_name, (positions, target_nodes) in active_splits.items():
                output[split_name][positions, col] = in_count[target_nodes]
                output[split_name][positions, col + 1] = out_count[target_nodes]
                output[split_name][positions, col + 2] = _safe_ratio(
                    in_count[target_nodes],
                    in_snapshot_total[target_nodes],
                )
                output[split_name][positions, col + 3] = _safe_ratio(
                    out_count[target_nodes],
                    out_snapshot_total[target_nodes],
                )
            col += 4

        for recent_days in recent_windows:
            recent_cutoff = snapshot_end - int(recent_days)
            out_recent_mask = out_snapshot_mask & (out_timestamps > recent_cutoff)
            in_recent_mask = in_snapshot_mask & (in_timestamps > recent_cutoff)
            out_recent_total = np.bincount(
                out_rows[out_recent_mask],
                minlength=num_nodes,
            ).astype(np.float32, copy=False)
            in_recent_total = np.bincount(
                in_rows[in_recent_mask],
                minlength=num_nodes,
            ).astype(np.float32, copy=False)
            out_recent_named = _named_count_views(
                _count_known_labels(
                    rows=out_rows,
                    neighbor_codes=out_neighbor_codes,
                    mask=out_recent_mask,
                    num_nodes=num_nodes,
                )
            )
            in_recent_named = _named_count_views(
                _count_known_labels(
                    rows=in_rows,
                    neighbor_codes=in_neighbor_codes,
                    mask=in_recent_mask,
                    num_nodes=num_nodes,
                )
            )
            for anchor_name in anchor_names:
                in_recent = in_recent_named[anchor_name]
                out_recent = out_recent_named[anchor_name]
                in_snapshot = in_snapshot_named[anchor_name]
                out_snapshot = out_snapshot_named[anchor_name]
                for split_name, (positions, target_nodes) in active_splits.items():
                    output[split_name][positions, col] = in_recent[target_nodes]
                    output[split_name][positions, col + 1] = out_recent[target_nodes]
                    output[split_name][positions, col + 2] = _safe_ratio(
                        in_recent[target_nodes],
                        in_recent_total[target_nodes],
                    )
                    output[split_name][positions, col + 3] = _safe_ratio(
                        out_recent[target_nodes],
                        out_recent_total[target_nodes],
                    )
                    output[split_name][positions, col + 4] = _safe_ratio(
                        in_recent[target_nodes],
                        in_snapshot[target_nodes],
                    )
                    output[split_name][positions, col + 5] = _safe_ratio(
                        out_recent[target_nodes],
                        out_snapshot[target_nodes],
                    )
                col += 6

        relation_cutoff = snapshot_end - int(relation_window)
        out_relation_mask = out_snapshot_mask & (out_timestamps > relation_cutoff)
        in_relation_mask = in_snapshot_mask & (in_timestamps > relation_cutoff)
        for edge_type in range(1, 12):
            out_type_named = _named_count_views(
                _count_known_labels(
                    rows=out_rows,
                    neighbor_codes=out_neighbor_codes,
                    mask=out_relation_mask & (out_types == edge_type),
                    num_nodes=num_nodes,
                )
            )
            in_type_named = _named_count_views(
                _count_known_labels(
                    rows=in_rows,
                    neighbor_codes=in_neighbor_codes,
                    mask=in_relation_mask & (in_types == edge_type),
                    num_nodes=num_nodes,
                )
            )
            for anchor_name in relation_anchor_names:
                in_count = in_type_named[anchor_name]
                out_count = out_type_named[anchor_name]
                for split_name, (positions, target_nodes) in active_splits.items():
                    output[split_name][positions, col] = in_count[target_nodes]
                    output[split_name][positions, col + 1] = out_count[target_nodes]
                col += 2

    return output, feature_names


def load_or_build_temporal_label_context_features(
    *,
    cache_dir: Path,
    feature_dir: Path,
    split,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    phase1_ids: dict[str, np.ndarray],
    phase2_ids: dict[str, np.ndarray],
    anchor_names: tuple[str, ...] = BEST_LABEL_CONTEXT_ANCHORS,
    relation_anchor_names: tuple[str, ...] = BEST_LABEL_CONTEXT_RELATION_ANCHORS,
    recent_windows: tuple[int, ...] = BEST_LABEL_CONTEXT_RECENT_WINDOWS,
    relation_window: int = BEST_LABEL_CONTEXT_RELATION_WINDOW,
    primary_phase: str = "phase1",
    external_phase: str = "phase2",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    anchor_names = _validate_anchor_names(anchor_names)
    relation_anchor_names = _validate_anchor_names(relation_anchor_names)
    feature_names_path = cache_dir / "label_context_feature_names.json"
    phase1_train_path = cache_dir / "phase1_train_label_context.npy"
    phase1_val_path = cache_dir / "phase1_val_label_context.npy"
    phase2_external_path = cache_dir / "phase2_external_label_context.npy"
    if (
        feature_names_path.exists()
        and phase1_train_path.exists()
        and phase1_val_path.exists()
        and phase2_external_path.exists()
    ):
        return (
            {
                "train": np.load(phase1_train_path, mmap_mode="r"),
                "val": np.load(phase1_val_path, mmap_mode="r"),
            },
            {
                "external": np.load(phase2_external_path, mmap_mode="r"),
            },
            list(json.loads(feature_names_path.read_text(encoding="utf-8"))),
        )

    ensure_dir(cache_dir)
    phase1_output, feature_names = _build_phase_temporal_label_context(
        phase=str(primary_phase),
        feature_dir=feature_dir,
        labels=phase1_y,
        split=split,
        split_ids=phase1_ids,
        anchor_names=anchor_names,
        relation_anchor_names=relation_anchor_names,
        recent_windows=tuple(int(window) for window in recent_windows),
        relation_window=int(relation_window),
    )
    phase2_has_rows = any(np.asarray(node_ids, dtype=np.int32).size for node_ids in phase2_ids.values())
    if phase2_has_rows:
        phase2_output, phase2_feature_names = _build_phase_temporal_label_context(
            phase=str(external_phase),
            feature_dir=feature_dir,
            labels=phase2_y,
            split=split,
            split_ids=phase2_ids,
            anchor_names=anchor_names,
            relation_anchor_names=relation_anchor_names,
            recent_windows=tuple(int(window) for window in recent_windows),
            relation_window=int(relation_window),
        )
        if phase2_feature_names != feature_names:
            raise AssertionError("Phase1/phase2 temporal label context feature names are not aligned.")
    else:
        phase2_output = {
            split_name: np.zeros((np.asarray(node_ids, dtype=np.int32).size, len(feature_names)), dtype=np.float32)
            for split_name, node_ids in phase2_ids.items()
        }

    np.save(phase1_train_path, np.asarray(phase1_output["train"], dtype=np.float32))
    np.save(phase1_val_path, np.asarray(phase1_output["val"], dtype=np.float32))
    np.save(phase2_external_path, np.asarray(phase2_output["external"], dtype=np.float32))
    feature_names_path.write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (
        {
            "train": np.load(phase1_train_path, mmap_mode="r"),
            "val": np.load(phase1_val_path, mmap_mode="r"),
        },
        {
            "external": np.load(phase2_external_path, mmap_mode="r"),
        },
        feature_names,
    )
