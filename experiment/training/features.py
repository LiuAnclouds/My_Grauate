from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm

from experiment.datasets.registry import get_active_dataset_spec
from experiment.eda.analysis import compute_degree_arrays, compute_temporal_core
from experiment.eda.data_loader import PhaseData, load_phase
from experiment.training.common import (
    EDA_OUTPUT_ROOT,
    FEATURE_OUTPUT_ROOT,
    REPO_ROOT,
    ensure_dir,
    write_json,
)

# Default schema for the original XinYe dataset. These values are updated at runtime
# when a different dataset is converted into the shared phase contract.
RAW_FEATURE_COUNT = 17
NUM_EDGE_TYPES = 11
# 时间窗口
NUM_TIME_WINDOWS = 4
DEFAULT_STRONG_PAIRS = ((2, 3), (6, 8), (15, 16))
TEMPORAL_RECENT_WINDOWS = (14, 60)
TEMPORAL_RELATION_RECENT_WINDOW = 30
ACTIVATION_EARLY_WINDOWS = (7, 30)
ACTIVATION_RELATION_WINDOW = 30
GRAPH_RECENT_WINDOWS = (3, 7, 14, 30)
TEMPORAL_BUCKET_DAYS = 30
TEMPORAL_MOTIF_RECENT_DAYS = 14
TEMPORAL_MOTIF_BURST_DAYS = 7
TEMPORAL_DUAL_SCALE_PAIRS = ((3, 14), (7, 30), (14, 60))
TEMPORAL_CONCENTRATION_WINDOWS = (7, 30, 90)
TEMPORAL_CONCENTRATION_RECENT_WINDOWS = (7, 30)
TEMPORAL_CONCENTRATION_FOCUS_WINDOWS = (7, 30)
TEMPORAL_ADAPTIVE_WINDOWS = (3, 7, 14, 30, 60, 90)
TEMPORAL_ADAPTIVE_CONTRAST_WINDOWS = (7, 60)
ACTIVE_FEATURE_STRONG_PAIRS = tuple(get_active_dataset_spec().strong_pairs)
BACKGROUND_LABELS = tuple(get_active_dataset_spec().background_labels)

def _env_override_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable `{name}` must be an integer, got `{raw}`.") from exc
    return max(value, 1)


UTPM_ATTR_PROJ_DIM = _env_override_int("GRADPROJ_UTPM_ATTR_PROJ_DIM", 32)
UTPM_ATTR_STATS_DIM = 16
UTPM_GRAPH_CORE_DIM = 16
UTPM_RELATION_CORE_DIM = 16
UTPM_TEMPORAL_CORE_DIM = 24
UTPM_TEMPORAL_RISK_DIM = 16
UTPM_AUX_LABEL_DIM = 8
UTPM_TEMPORAL_SHIFT_DIM = 16


def _set_feature_schema(
    raw_feature_count: int,
    num_edge_types: int,
    strong_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    background_labels: tuple[int, ...] | list[int] | None = None,
) -> None:
    global RAW_FEATURE_COUNT, NUM_EDGE_TYPES, ACTIVE_FEATURE_STRONG_PAIRS, BACKGROUND_LABELS
    RAW_FEATURE_COUNT = max(int(raw_feature_count), 1)
    NUM_EDGE_TYPES = max(int(num_edge_types), 1)
    ACTIVE_FEATURE_STRONG_PAIRS = tuple(
        (int(left), int(right))
        for left, right in (strong_pairs or ())
    )
    BACKGROUND_LABELS = tuple(
        sorted(
            {
                int(label)
                for label in (background_labels or ())
                if int(label) not in {-100, 0, 1}
            }
        )
    )


def _active_background_labels() -> tuple[int, ...]:
    return BACKGROUND_LABELS


def _aux_label_prefix(label: int) -> str:
    return f"aux_label_{int(label)}"


def _feature_schema_payload() -> dict[str, Any]:
    return {
        "raw_feature_count": int(RAW_FEATURE_COUNT),
        "num_edge_types": int(NUM_EDGE_TYPES),
        "background_labels": [int(label) for label in _active_background_labels()],
        "strong_pairs": [[int(left), int(right)] for left, right in ACTIVE_FEATURE_STRONG_PAIRS],
        "num_time_windows": int(NUM_TIME_WINDOWS),
        "utpm_attr_proj_dim": int(UTPM_ATTR_PROJ_DIM),
        "utpm_unified_groups": utpm_unified_feature_groups(),
        "utpm_unified_dim": int(
            UTPM_ATTR_PROJ_DIM
            + UTPM_ATTR_STATS_DIM
            + UTPM_GRAPH_CORE_DIM
            + UTPM_RELATION_CORE_DIM
            + UTPM_TEMPORAL_CORE_DIM
            + UTPM_TEMPORAL_RISK_DIM
            + UTPM_AUX_LABEL_DIM
        ),
        "utpm_shift_compact_groups": utpm_shift_compact_feature_groups(),
        "utpm_shift_compact_dim": int(
            UTPM_ATTR_PROJ_DIM
            + UTPM_ATTR_STATS_DIM
            + UTPM_GRAPH_CORE_DIM
            + UTPM_RELATION_CORE_DIM
            + UTPM_TEMPORAL_CORE_DIM
            + UTPM_TEMPORAL_RISK_DIM
            + UTPM_AUX_LABEL_DIM
            + UTPM_TEMPORAL_SHIFT_DIM
        ),
        "utpm_shift_enhanced_groups": utpm_shift_enhanced_feature_groups(),
    }


def _active_strong_pairs(raw_feature_count: int) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left, right in ACTIVE_FEATURE_STRONG_PAIRS
        if left < int(raw_feature_count) and right < int(raw_feature_count)
    ]


def utpm_unified_feature_groups() -> list[str]:
    return [
        "utpm_attr_proj",
        "utpm_attr_stats",
        "utpm_graph_core",
        "utpm_relation_core",
        "utpm_temporal_core",
        "utpm_temporal_risk",
        "utpm_aux_label",
    ]


def utpm_temporal_shift_feature_names() -> list[str]:
    return [
        "tdd_out_density_delta_7_30",
        "tdd_in_density_delta_7_30",
        "tdd_total_density_delta_7_30",
        "tdd_out_burst_contrast_7_30",
        "tdd_in_burst_contrast_7_30",
        "tdd_total_burst_contrast_7_30",
        "tdd_total_density_delta_14_60",
        "tdd_total_ratio_14_60",
        "tdd_total_burst_contrast_14_60",
        "tccr_total_top_share_7",
        "tccr_total_top_share_30",
        "tccr_total_entropy_7",
        "tccr_total_entropy_30",
        "tcsf_total_top_share_7",
        "tcsf_total_top_share_30",
        "taw_total_density_7_over_60_z",
    ]


def utpm_shift_compact_feature_groups() -> list[str]:
    return [
        *utpm_unified_feature_groups(),
        "utpm_temporal_shift",
    ]


def utpm_shift_enhanced_feature_groups() -> list[str]:
    return [
        *utpm_shift_compact_feature_groups(),
        "graph_time_detrend",
        "temporal_motif_lite",
        "temporal_counterparty_concentration",
        "temporal_adaptive_window",
        "neighbor_similarity",
        "temporal_bucket_norm",
        "activation_early",
    ]


def _clip_nonfinite(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if np.all(np.isfinite(arr)):
        return arr
    fixed = arr.copy()
    fixed[~np.isfinite(fixed)] = np.float32(fill_value)
    return fixed


def _pad_feature_block(block: np.ndarray, width: int) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    if block.ndim != 2:
        raise AssertionError(f"Feature block must be 2D, got {block.shape}")
    if block.shape[1] == width:
        return block.astype(np.float32, copy=False)
    if block.shape[1] > width:
        return block[:, :width].astype(np.float32, copy=False)
    padded = np.zeros((block.shape[0], width), dtype=np.float32)
    padded[:, : block.shape[1]] = block
    return padded


def _stable_row_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    den = np.maximum(np.asarray(denominator, dtype=np.float32), 1.0)
    return np.asarray(numerator, dtype=np.float32) / den


def _load_projection_train_ids(phase: str, num_nodes: int) -> np.ndarray:
    split_path = EDA_OUTPUT_ROOT / "recommended_split.json"
    if not split_path.exists():
        return np.arange(min(num_nodes, 250000), dtype=np.int32)
    split_summary = json.loads(split_path.read_text(encoding="utf-8"))
    train_phase = str(split_summary.get("train_phase", "graph"))
    train_path = None
    train_split = split_summary.get("train_split")
    if isinstance(train_split, dict):
        train_path = train_split.get("id_path")
    if train_phase != phase or not train_path:
        return np.arange(min(num_nodes, 250000), dtype=np.int32)
    full_path = EDA_OUTPUT_ROOT / str(train_path)
    if not full_path.exists():
        return np.arange(min(num_nodes, 250000), dtype=np.int32)
    train_ids = np.asarray(np.load(full_path), dtype=np.int32)
    if train_ids.size == 0:
        return np.arange(min(num_nodes, 250000), dtype=np.int32)
    return train_ids


def _fit_utpm_attr_projection(
    x: np.ndarray,
    missing_mask: np.ndarray,
    train_ids: np.ndarray,
    output_dim: int = UTPM_ATTR_PROJ_DIM,
) -> np.ndarray:
    raw = np.asarray(x, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=np.float32) > 0.5
    train_rows = np.asarray(train_ids, dtype=np.int32)
    if train_rows.size == 0:
        train_rows = np.arange(min(raw.shape[0], 250000), dtype=np.int32)

    medians = np.zeros(raw.shape[1], dtype=np.float32)
    for feature_idx in range(raw.shape[1]):
        observed = raw[train_rows, feature_idx][~missing[train_rows, feature_idx]]
        medians[feature_idx] = float(np.median(observed)) if observed.size else 0.0
    filled = np.where(missing, medians.reshape(1, -1), raw).astype(np.float32, copy=False)
    train_filled = filled[train_rows]
    mean = np.mean(train_filled, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = np.std(train_filled, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    standardized = (filled - mean.reshape(1, -1)) / std.reshape(1, -1)

    n_components = min(int(output_dim), int(standardized.shape[1]), max(int(train_rows.size) - 1, 1))
    if n_components <= 0:
        return np.zeros((raw.shape[0], output_dim), dtype=np.float32)
    projector = TruncatedSVD(n_components=n_components, random_state=42)
    projector.fit(standardized[train_rows])
    projected = projector.transform(standardized).astype(np.float32, copy=False)
    return _pad_feature_block(projected, output_dim)


def _build_utpm_attr_stats_block(
    x: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(x, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=np.float32)
    observed = 1.0 - missing
    valid_count = np.maximum(np.sum(observed, axis=1, dtype=np.float32), 1.0)
    observed_raw = raw * observed
    missing_count = np.sum(missing, axis=1, dtype=np.float32)
    valid_mean = np.sum(observed_raw, axis=1, dtype=np.float32) / valid_count
    valid_sq_mean = np.sum(observed_raw * observed_raw, axis=1, dtype=np.float32) / valid_count
    valid_std = np.sqrt(np.maximum(valid_sq_mean - valid_mean * valid_mean, 0.0), dtype=np.float32)

    valid_min = np.min(np.where(observed > 0.5, raw, np.inf), axis=1)
    valid_min = np.where(np.isfinite(valid_min), valid_min, 0.0).astype(np.float32, copy=False)
    valid_max = np.max(np.where(observed > 0.5, raw, -np.inf), axis=1)
    valid_max = np.where(np.isfinite(valid_max), valid_max, 0.0).astype(np.float32, copy=False)

    zero_ratio = np.sum((raw == 0.0) & (observed > 0.5), axis=1, dtype=np.float32) / valid_count
    pos_ratio = np.sum((raw > 0.0) & (observed > 0.5), axis=1, dtype=np.float32) / valid_count
    neg_ratio = np.sum((raw < 0.0) & (observed > 0.5), axis=1, dtype=np.float32) / valid_count
    abs_raw = np.abs(observed_raw, dtype=np.float32)
    l1_norm = np.sum(abs_raw, axis=1, dtype=np.float32)
    l2_norm = np.sqrt(
        np.sum(observed_raw * observed_raw, axis=1, dtype=np.float32)
    ).astype(np.float32, copy=False)
    mean_abs = l1_norm / valid_count
    max_abs = np.max(abs_raw, axis=1)

    attr_stats = np.column_stack(
        [
            missing_count,
            missing_count / float(raw.shape[1]),
            valid_count,
            valid_mean,
            valid_std,
            valid_min,
            valid_max,
            zero_ratio,
            pos_ratio,
            neg_ratio,
            l1_norm,
            l2_norm,
            mean_abs,
            max_abs.astype(np.float32, copy=False),
            raw[:, 0].astype(np.float32, copy=False),
            raw[:, min(1, raw.shape[1] - 1)].astype(np.float32, copy=False),
        ]
    )
    return _pad_feature_block(_clip_nonfinite(attr_stats), UTPM_ATTR_STATS_DIM)


def _core_name_to_index_map(core_spans: dict[str, dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for spec in core_spans.values():
        start = int(spec["start"])
        for offset, name in enumerate(spec["names"]):
            mapping[str(name)] = start + offset
    return mapping


def _take_named_core_features(
    core: np.ndarray,
    name_to_index: dict[str, int],
    names: list[str],
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    num_nodes = int(core.shape[0])
    for name in names:
        col_idx = name_to_index.get(str(name))
        if col_idx is None:
            blocks.append(np.zeros((num_nodes, 1), dtype=np.float32))
            continue
        blocks.append(np.asarray(core[:, col_idx : col_idx + 1], dtype=np.float32))
    if not blocks:
        return np.zeros((num_nodes, 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def _normalized_entropy(counts: np.ndarray) -> np.ndarray:
    count_arr = np.asarray(counts, dtype=np.float32)
    total = np.sum(count_arr, axis=1, keepdims=True, dtype=np.float32)
    prob = np.divide(count_arr, np.maximum(total, 1.0), out=np.zeros_like(count_arr), where=total > 0)
    entropy = -np.sum(prob * np.log(np.clip(prob, 1e-6, None)), axis=1, dtype=np.float32)
    denom = float(np.log(max(count_arr.shape[1], 2)))
    return (entropy / max(denom, 1e-6)).astype(np.float32, copy=False)


def _topk_share(counts: np.ndarray, k: int) -> np.ndarray:
    count_arr = np.asarray(counts, dtype=np.float32)
    if count_arr.shape[1] == 0:
        return np.zeros(count_arr.shape[0], dtype=np.float32)
    k = max(1, min(int(k), count_arr.shape[1]))
    part = np.partition(count_arr, count_arr.shape[1] - k, axis=1)[:, -k:]
    numerator = np.sum(part, axis=1, dtype=np.float32)
    denominator = np.sum(count_arr, axis=1, dtype=np.float32)
    return _stable_row_divide(numerator, denominator)


def _build_utpm_relation_core_block(
    core: np.ndarray,
    core_spans: dict[str, dict[str, Any]],
) -> np.ndarray:
    edge_type_spec = core_spans.get("edge_type")
    if edge_type_spec is None:
        return np.zeros((core.shape[0], UTPM_RELATION_CORE_DIM), dtype=np.float32)
    edge_type_block = np.asarray(
        core[:, int(edge_type_spec["start"]) : int(edge_type_spec["end"])],
        dtype=np.float32,
    )
    in_counts = edge_type_block[:, 0::2]
    out_counts = edge_type_block[:, 1::2]
    total_counts = in_counts + out_counts

    relation_recent_spec = core_spans.get("temporal_relation_recent")
    if relation_recent_spec is not None:
        recent_block = np.asarray(
            core[:, int(relation_recent_spec["start"]) : int(relation_recent_spec["end"])],
            dtype=np.float32,
        )
        recent_in = recent_block[:, 0::2]
        recent_out = recent_block[:, 1::2]
    else:
        recent_in = np.zeros_like(in_counts)
        recent_out = np.zeros_like(out_counts)

    features = np.column_stack(
        [
            _normalized_entropy(in_counts),
            _normalized_entropy(out_counts),
            _normalized_entropy(total_counts),
            _topk_share(in_counts, 1),
            _topk_share(out_counts, 1),
            _topk_share(total_counts, 1),
            _topk_share(in_counts, 2),
            _topk_share(out_counts, 2),
            _topk_share(total_counts, 2),
            np.mean(in_counts > 0.0, axis=1, dtype=np.float32),
            np.mean(out_counts > 0.0, axis=1, dtype=np.float32),
            np.mean(total_counts > 0.0, axis=1, dtype=np.float32),
            _normalized_entropy(recent_in),
            _normalized_entropy(recent_out),
            _topk_share(recent_in, 1),
            _topk_share(recent_out, 1),
        ]
    )
    return _pad_feature_block(_clip_nonfinite(features), UTPM_RELATION_CORE_DIM)


def _build_utpm_aux_label_block(
    core: np.ndarray,
    core_spans: dict[str, dict[str, Any]],
) -> np.ndarray:
    name_to_index = _core_name_to_index_map(core_spans)
    aux_names = [
        name
        for name in sorted(name_to_index)
        if name.startswith("aux_label_")
    ][:UTPM_AUX_LABEL_DIM]
    if not aux_names:
        return np.zeros((core.shape[0], UTPM_AUX_LABEL_DIM), dtype=np.float32)
    return _pad_feature_block(_take_named_core_features(core, name_to_index, aux_names), UTPM_AUX_LABEL_DIM)


def _build_utpm_blocks(
    *,
    core: np.ndarray,
    core_spans: dict[str, dict[str, Any]],
    x: np.ndarray,
    missing_mask: np.ndarray,
    train_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    name_to_index = _core_name_to_index_map(core_spans)
    graph_feature_names = [
        "indegree",
        "outdegree",
        "total_degree",
        "out_over_in_plus1",
        "in_over_out_plus1",
        "unique_out",
        "unique_in",
        "multi_out",
        "multi_in",
        "unique_out_ratio_v2",
        "unique_in_ratio_v2",
        "days_since_out_mean",
        "days_since_in_mean",
        "days_since_out_last",
        "days_since_in_last",
        "out_last30_ratio_v2",
    ]
    temporal_core_names = [
        "first_active",
        "last_active",
        "active_span",
        "early_late_total_ratio",
        "snapshot_day",
        "snapshot_age",
        "snap_in_count",
        "snap_out_count",
        "snap_total_count",
        "snap_bg_total_ratio",
        "window_1_total_count",
        "window_2_total_count",
        "window_3_total_count",
        "window_4_total_count",
        "recent_14_in_count",
        "recent_14_out_count",
        "recent_14_total_count",
        "recent_60_in_count",
        "recent_60_out_count",
        "recent_60_total_count",
        "recent_14_total_over_snapshot",
        "recent_60_total_over_snapshot",
        "recent_14_bg_total_over_snapshot",
        "recent_60_bg_total_over_snapshot",
    ]
    temporal_risk_names = [
        "bg_in_count",
        "bg_out_count",
        "bg_total_count",
        "bg_in_ratio",
        "bg_out_ratio",
        "bg_total_ratio",
        "touch_background",
        "tcc_out_top_share_7",
        "tcc_in_top_share_7",
        "tcc_total_top_share_7",
        "tcc_out_entropy_30",
        "tcc_in_entropy_30",
        "tcc_total_entropy_30",
        f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_recip_out_ratio",
        f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_recip_in_ratio",
        "taw_total_count_peak_z",
    ]
    return {
        "utpm_attr_proj": _fit_utpm_attr_projection(x=x, missing_mask=missing_mask, train_ids=train_ids),
        "utpm_attr_stats": _build_utpm_attr_stats_block(x=x, missing_mask=missing_mask),
        "utpm_graph_core": _pad_feature_block(
            _take_named_core_features(core, name_to_index, graph_feature_names),
            UTPM_GRAPH_CORE_DIM,
        ),
        "utpm_relation_core": _build_utpm_relation_core_block(core=core, core_spans=core_spans),
        "utpm_temporal_core": _pad_feature_block(
            _take_named_core_features(core, name_to_index, temporal_core_names),
            UTPM_TEMPORAL_CORE_DIM,
        ),
        "utpm_temporal_risk": _pad_feature_block(
            _take_named_core_features(core, name_to_index, temporal_risk_names),
            UTPM_TEMPORAL_RISK_DIM,
        ),
        "utpm_aux_label": _build_utpm_aux_label_block(core=core, core_spans=core_spans),
        "utpm_temporal_shift": _pad_feature_block(
            _take_named_core_features(core, name_to_index, utpm_temporal_shift_feature_names()),
            UTPM_TEMPORAL_SHIFT_DIM,
        ),
    }


@dataclass(frozen=True)
class GraphCache:
    phase: str # 数据阶段
    num_nodes: int # 节点个数
    max_day: int # 最大天数，用于初始化
    num_edge_types: int # 原始边类型总数
    num_relations: int # 关系种类数
    time_windows: list[dict[str, int]] # 时间窗口特征   "window_idx": 1,"start_day": 1,"end_day": 17
    out_ptr: np.ndarray # 节点N的所有出边，扁平话数组，节点 i 的所有出边，位于 out_neighbors[out_ptr[i] : out_ptr[i+1]] 这段切片里。
    out_neighbors: np.ndarray # 节点N的所有出领居节点ID
    out_edge_type: np.ndarray # 所有出边类型，和Neighbor对应
    out_edge_timestamp: np.ndarray # 边的时间戳特征
    in_ptr: np.ndarray # 节点N的所有入边
    in_neighbors: np.ndarray # 入邻居
    in_edge_type: np.ndarray # 入边类型
    in_edge_timestamp: np.ndarray # 入边时间戳
    first_active: np.ndarray # 第一次活动时间，按最小边算
    node_time_bucket: np.ndarray # 时间组编号


@dataclass(frozen=True)
class HybridFeatureNormalizerState:
    mode: str
    feature_names: list[str]
    raw_indices: list[int]
    raw_medians: list[float]
    raw_means: list[float]
    raw_stds: list[float]
    log_indices: list[int]
    log_means: list[float]
    log_stds: list[float]
    z_indices: list[int]
    z_means: list[float]
    z_stds: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feature_names": self.feature_names,
            "raw_indices": self.raw_indices,
            "raw_medians": self.raw_medians,
            "raw_means": self.raw_means,
            "raw_stds": self.raw_stds,
            "log_indices": self.log_indices,
            "log_means": self.log_means,
            "log_stds": self.log_stds,
            "z_indices": self.z_indices,
            "z_means": self.z_means,
            "z_stds": self.z_stds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> HybridFeatureNormalizerState | None:
        if not payload:
            return None
        return cls(
            mode=str(payload["mode"]),
            feature_names=list(payload["feature_names"]),
            raw_indices=[int(idx) for idx in payload["raw_indices"]],
            raw_medians=[float(value) for value in payload["raw_medians"]],
            raw_means=[float(value) for value in payload["raw_means"]],
            raw_stds=[float(value) for value in payload["raw_stds"]],
            log_indices=[int(idx) for idx in payload["log_indices"]],
            log_means=[float(value) for value in payload["log_means"]],
            log_stds=[float(value) for value in payload["log_stds"]],
            z_indices=[int(idx) for idx in payload["z_indices"]],
            z_means=[float(value) for value in payload["z_means"]],
            z_stds=[float(value) for value in payload["z_stds"]],
        )


# 特征读取器
class FeatureStore:
    def __init__(
        self,
        phase: str, # 数据阶段
        selected_groups: list[str],
        outdir: Path = FEATURE_OUTPUT_ROOT,
        normalizer_state: HybridFeatureNormalizerState | None = None,
    ) -> None:
        self.phase = phase
        self.phase_dir = outdir / phase
        self.manifest = load_feature_manifest(phase, outdir=outdir)
        self.selected_groups = selected_groups
        self.normalizer_state = normalizer_state
        self.core = np.load(self.phase_dir / self.manifest["core_file"], mmap_mode="r")
        self.neighbor = None
        neighbor_file = self.manifest.get("neighbor_file")
        if neighbor_file and (self.phase_dir / neighbor_file).exists():
            self.neighbor = np.load(self.phase_dir / neighbor_file, mmap_mode="r")
        self._group_specs = self._resolve_group_specs(selected_groups)
        self.feature_names = [
            name
            for spec in self._group_specs
            for name in spec["names"]
        ]

    def _resolve_group_specs(self, selected_groups: list[str]) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for group_name in selected_groups:
            if group_name in self.manifest["core_groups"]:
                spec = dict(self.manifest["core_groups"][group_name])
                spec["source"] = "core"
                spec["group_name"] = group_name
                specs.append(spec)
                continue
            if group_name in self.manifest["neighbor_groups"]:
                if self.neighbor is None:
                    raise FileNotFoundError(
                        f"{self.phase}: neighbor feature file is missing, run build_features first."
                    )
                spec = dict(self.manifest["neighbor_groups"][group_name])
                spec["source"] = "neighbor"
                spec["group_name"] = group_name
                specs.append(spec)
                continue
            raise KeyError(f"{self.phase}: unknown feature group '{group_name}'")
        return specs

    def take_rows(self, node_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(node_ids, dtype=np.int32)
        blocks: list[np.ndarray] = []
        for spec in self._group_specs:
            matrix = self.core if spec["source"] == "core" else self.neighbor
            blocks.append(
                np.asarray(matrix[rows, spec["start"] : spec["end"]], dtype=np.float32)
            )
        if not blocks:
            raise ValueError("No feature groups selected.")
        features = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        if self.normalizer_state is not None:
            features = apply_hybrid_feature_normalizer(features, self.normalizer_state)
        return features

    @property
    def input_dim(self) -> int:
        return int(sum(spec["end"] - spec["start"] for spec in self._group_specs))


def _build_edge_time_windows(
    timestamps: np.ndarray,
    n_windows: int = NUM_TIME_WINDOWS,
) -> list[dict[str, int]]:
    quantiles = np.quantile(timestamps, np.linspace(0.0, 1.0, num=n_windows + 1))
    windows: list[dict[str, int]] = []
    for idx in range(n_windows):
        start_day = int(np.floor(quantiles[idx]))
        end_day = int(np.ceil(quantiles[idx + 1]))
        if idx > 0 and start_day <= windows[-1]["end_day"]:
            start_day = windows[-1]["end_day"] + 1
        if end_day < start_day:
            end_day = start_day
        windows.append(
            {
                "window_idx": idx + 1,
                "start_day": start_day,
                "end_day": end_day,
            }
        )
    windows[-1]["end_day"] = int(np.max(timestamps))
    return windows


def _assign_node_time_bucket(
    first_active: np.ndarray,
    time_windows: list[dict[str, int]],
) -> np.ndarray:
    buckets = np.zeros(first_active.shape[0], dtype=np.int8)
    for window in time_windows:
        mask = (first_active >= window["start_day"]) & (first_active <= window["end_day"])
        buckets[mask] = int(window["window_idx"] - 1)
    return buckets


def _group_definition() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    active_pairs = _active_strong_pairs(RAW_FEATURE_COUNT)
    core_groups: dict[str, list[str]] = {
        "raw_x": [f"x{i}" for i in range(RAW_FEATURE_COUNT)],
        "missing_mask": [f"x{i}_is_neg1" for i in range(RAW_FEATURE_COUNT)],
        "missing_summary": ["missing_count"],
        "strong_combo": [],
        "graph_direction": [
            "indegree",
            "outdegree",
            "total_degree",
            "out_over_in_plus1",
            "in_over_out_plus1",
        ],
        "edge_type": [],
        "background": [
            "bg_in_count",
            "bg_out_count",
            "bg_total_count",
            "bg_in_ratio",
            "bg_out_ratio",
            "bg_total_ratio",
            "touch_background",
        ],
        "time": [
            "first_active",
            "last_active",
            "active_span",
        ],
        "graph_stats": [
            "unique_out",
            "unique_in",
            "multi_out",
            "multi_in",
            "unique_out_ratio_v2",
            "unique_in_ratio_v2",
            "days_since_out_mean",
            "days_since_in_mean",
            "days_since_out_last",
            "days_since_in_last",
        ],
        "graph_time_detrend": [
            "gtd_days_since_out_mean_z",
            "gtd_days_since_in_mean_z",
            "gtd_days_since_out_last_z",
            "gtd_days_since_in_last_z",
            "gtd_out_last30_ratio_z",
            "gtd_in_last30_ratio_z",
        ],
        "temporal_motif_lite": [
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_recip_out_unique",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_recip_in_unique",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_recip_out_ratio",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_recip_in_ratio",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_repeat_out_ratio",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_repeat_in_ratio",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_neighbor_totaldeg_out_mean",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_neighbor_totaldeg_in_mean",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_neighbor_totaldeg_out_logsum",
            f"tm{TEMPORAL_MOTIF_RECENT_DAYS}_neighbor_totaldeg_in_logsum",
            f"tm{TEMPORAL_MOTIF_BURST_DAYS}_out_burst_maxday",
            f"tm{TEMPORAL_MOTIF_BURST_DAYS}_in_burst_maxday",
        ],
        "temporal_counterparty_concentration": [],
        "temporal_counterparty_concentration_recent": [],
        "temporal_counterparty_share_focus": [],
        "temporal_dual_scale_delta": [],
        "temporal_adaptive_window": [
            "taw_out_count_peak_z",
            "taw_in_count_peak_z",
            "taw_total_count_peak_z",
            "taw_out_ratio_peak_z",
            "taw_in_ratio_peak_z",
            "taw_total_ratio_peak_z",
            f"taw_out_density_{TEMPORAL_ADAPTIVE_CONTRAST_WINDOWS[0]}_over_{TEMPORAL_ADAPTIVE_CONTRAST_WINDOWS[1]}_z",
            f"taw_in_density_{TEMPORAL_ADAPTIVE_CONTRAST_WINDOWS[0]}_over_{TEMPORAL_ADAPTIVE_CONTRAST_WINDOWS[1]}_z",
            f"taw_total_density_{TEMPORAL_ADAPTIVE_CONTRAST_WINDOWS[0]}_over_{TEMPORAL_ADAPTIVE_CONTRAST_WINDOWS[1]}_z",
            "taw_out_peak_window",
            "taw_in_peak_window",
            "taw_total_peak_window",
        ],
        "neighbor_similarity": [
            "out_cosine_sum",
            "in_cosine_sum",
            "out_cosine_mean",
            "in_cosine_mean",
            "out_cosine_max",
            "in_cosine_max",
        ],
        "temporal_bucket_norm": [],
        "temporal_snapshot": [
            "snap_in_count",
            "snap_out_count",
            "snap_total_count",
            "snap_out_over_in_plus1",
            "snap_in_over_out_plus1",
            "snap_bg_in_count",
            "snap_bg_out_count",
            "snap_bg_total_count",
            "snap_bg_in_ratio",
            "snap_bg_out_ratio",
            "snap_bg_total_ratio",
            "snapshot_day",
            "snapshot_age",
        ],
        "temporal_recent": [],
        "temporal_relation_recent": [],
        "temporal_neighbor": [],
        "activation_early": [],
        "utpm_attr_proj": [f"utpm_attr_proj_{i}" for i in range(UTPM_ATTR_PROJ_DIM)],
        "utpm_attr_stats": [
            "utpm_missing_count",
            "utpm_missing_ratio",
            "utpm_valid_count",
            "utpm_valid_mean",
            "utpm_valid_std",
            "utpm_valid_min",
            "utpm_valid_max",
            "utpm_zero_ratio",
            "utpm_pos_ratio",
            "utpm_neg_ratio",
            "utpm_l1_norm",
            "utpm_l2_norm",
            "utpm_mean_abs",
            "utpm_max_abs",
            "utpm_raw_x0",
            "utpm_raw_x1",
        ],
        "utpm_graph_core": [f"utpm_graph_core_{i}" for i in range(UTPM_GRAPH_CORE_DIM)],
        "utpm_relation_core": [f"utpm_relation_core_{i}" for i in range(UTPM_RELATION_CORE_DIM)],
        "utpm_temporal_core": [f"utpm_temporal_core_{i}" for i in range(UTPM_TEMPORAL_CORE_DIM)],
        "utpm_temporal_risk": [f"utpm_temporal_risk_{i}" for i in range(UTPM_TEMPORAL_RISK_DIM)],
        "utpm_aux_label": [f"utpm_aux_label_{i}" for i in range(UTPM_AUX_LABEL_DIM)],
        "utpm_temporal_shift": [f"utpm_temporal_shift_{i}" for i in range(UTPM_TEMPORAL_SHIFT_DIM)],
    }
    for left, right in active_pairs:
        core_groups["strong_combo"].extend(
            [
                f"x{left}_x{right}_mean",
                f"x{left}_x{right}_absdiff",
            ]
        )
    for edge_type in range(1, NUM_EDGE_TYPES + 1):
        core_groups["edge_type"].extend(
            [
                f"in_type_{edge_type}_count",
                f"out_type_{edge_type}_count",
            ]
        )
        core_groups["background"].extend(
            [
                f"bg_in_type_{edge_type}_count",
                f"bg_out_type_{edge_type}_count",
            ]
        )
    for window_idx in range(1, NUM_TIME_WINDOWS + 1):
        core_groups["time"].extend(
            [
                f"window_{window_idx}_in_count",
                f"window_{window_idx}_out_count",
                f"window_{window_idx}_total_count",
            ]
        )
    core_groups["time"].append("early_late_total_ratio")
    for recent_days in GRAPH_RECENT_WINDOWS:
        core_groups["graph_stats"].extend(
            [
                f"out_last{recent_days}",
                f"in_last{recent_days}",
                f"out_last{recent_days}_ratio_v2",
                f"in_last{recent_days}_ratio_v2",
            ]
        )
    for bg_label in _active_background_labels():
        prefix = _aux_label_prefix(bg_label)
        core_groups["graph_stats"].extend(
            [
                f"{prefix}_out_count",
                f"{prefix}_in_count",
                f"{prefix}_total_count",
                f"{prefix}_out_ratio",
                f"{prefix}_in_ratio",
                f"{prefix}_out_last14",
                f"{prefix}_in_last14",
            ]
        )
    for feature_idx in range(RAW_FEATURE_COUNT):
        core_groups["temporal_bucket_norm"].extend(
            [
                f"tb{TEMPORAL_BUCKET_DAYS}_x{feature_idx}_z",
                f"tb{TEMPORAL_BUCKET_DAYS}_x{feature_idx}_missing_z",
            ]
        )
    core_groups["temporal_bucket_norm"].extend(
        [
            f"tb{TEMPORAL_BUCKET_DAYS}_missing_count_z",
            f"tb{TEMPORAL_BUCKET_DAYS}_log_indegree_z",
            f"tb{TEMPORAL_BUCKET_DAYS}_log_outdegree_z",
            f"tb{TEMPORAL_BUCKET_DAYS}_log_total_degree_z",
            f"tb{TEMPORAL_BUCKET_DAYS}_bg_in_ratio_z",
            f"tb{TEMPORAL_BUCKET_DAYS}_bg_out_ratio_z",
            f"tb{TEMPORAL_BUCKET_DAYS}_bg_total_ratio_z",
            f"tb{TEMPORAL_BUCKET_DAYS}_log_bg_total_count_z",
        ]
    )
    for recent_days in TEMPORAL_RECENT_WINDOWS:
        core_groups["temporal_recent"].extend(
            [
                f"recent_{recent_days}_in_count",
                f"recent_{recent_days}_out_count",
                f"recent_{recent_days}_total_count",
                f"recent_{recent_days}_bg_in_count",
                f"recent_{recent_days}_bg_out_count",
                f"recent_{recent_days}_bg_total_count",
                f"recent_{recent_days}_total_over_snapshot",
                f"recent_{recent_days}_bg_total_over_snapshot",
            ]
        )
    for recent_days in TEMPORAL_CONCENTRATION_WINDOWS:
        core_groups["temporal_counterparty_concentration"].extend(
            [
                f"tcc_out_top_share_{recent_days}",
                f"tcc_in_top_share_{recent_days}",
                f"tcc_total_top_share_{recent_days}",
                f"tcc_out_entropy_{recent_days}",
                f"tcc_in_entropy_{recent_days}",
                f"tcc_total_entropy_{recent_days}",
            ]
        )
    for recent_days in TEMPORAL_CONCENTRATION_RECENT_WINDOWS:
        core_groups["temporal_counterparty_concentration_recent"].extend(
            [
                f"tccr_out_top_share_{recent_days}",
                f"tccr_in_top_share_{recent_days}",
                f"tccr_total_top_share_{recent_days}",
                f"tccr_out_entropy_{recent_days}",
                f"tccr_in_entropy_{recent_days}",
                f"tccr_total_entropy_{recent_days}",
            ]
        )
    for recent_days in TEMPORAL_CONCENTRATION_FOCUS_WINDOWS:
        core_groups["temporal_counterparty_share_focus"].extend(
            [
                f"tcsf_out_top_share_{recent_days}",
                f"tcsf_in_top_share_{recent_days}",
                f"tcsf_total_top_share_{recent_days}",
            ]
        )
    for short_window, long_window in TEMPORAL_DUAL_SCALE_PAIRS:
        for prefix in ("out", "in", "total"):
            core_groups["temporal_dual_scale_delta"].extend(
                [
                    f"tdd_{prefix}_density_delta_{short_window}_{long_window}",
                    f"tdd_{prefix}_ratio_{short_window}_{long_window}",
                    f"tdd_{prefix}_burst_contrast_{short_window}_{long_window}",
                ]
            )
    for edge_type in range(1, NUM_EDGE_TYPES + 1):
        core_groups["temporal_relation_recent"].extend(
            [
                f"recent_{TEMPORAL_RELATION_RECENT_WINDOW}_in_type_{edge_type}_count",
                f"recent_{TEMPORAL_RELATION_RECENT_WINDOW}_out_type_{edge_type}_count",
            ]
        )
    for reducer in ("mean", "max", "missing_ratio"):
        for prefix in ("in", "out"):
            core_groups["temporal_neighbor"].extend(
                [f"snap_{prefix}_neighbor_{reducer}_x{i}" for i in range(RAW_FEATURE_COUNT)]
            )
    for early_days in ACTIVATION_EARLY_WINDOWS:
        core_groups["activation_early"].extend(
            [
                f"start_{early_days}_in_count",
                f"start_{early_days}_out_count",
                f"start_{early_days}_total_count",
                f"start_{early_days}_bg_in_count",
                f"start_{early_days}_bg_out_count",
                f"start_{early_days}_bg_total_count",
            ]
        )
    for edge_type in range(1, NUM_EDGE_TYPES + 1):
        core_groups["activation_early"].extend(
            [
                f"start_{ACTIVATION_RELATION_WINDOW}_in_type_{edge_type}_count",
                f"start_{ACTIVATION_RELATION_WINDOW}_out_type_{edge_type}_count",
            ]
        )

    neighbor_groups = {"neighbor": []}
    for reducer in ("mean", "max", "missing_ratio"):
        for prefix in ("in", "out"):
            neighbor_groups["neighbor"].extend(
                [f"{prefix}_neighbor_{reducer}_x{i}" for i in range(RAW_FEATURE_COUNT)]
            )
    return core_groups, neighbor_groups


def _allocate_group_spans(groups: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    spans: dict[str, dict[str, Any]] = {}
    col = 0
    for group_name, feature_names in groups.items():
        spans[group_name] = {
            "start": col,
            "end": col + len(feature_names),
            "names": feature_names,
        }
        col += len(feature_names)
    return spans


def _write_graph_arrays(
    phase_dir: Path,
    prefix: str,
    centers: np.ndarray,
    neighbors: np.ndarray,
    edge_type: np.ndarray,
    edge_timestamp: np.ndarray,
    num_nodes: int,
) -> dict[str, str]:
    counts = np.bincount(centers, minlength=num_nodes).astype(np.int64, copy=False)
    ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    order = np.argsort(centers, kind="stable")
    neighbor_ordered = neighbors[order].astype(np.int32, copy=False)
    edge_type_ordered = edge_type[order].astype(np.int16, copy=False)
    edge_timestamp_ordered = edge_timestamp[order].astype(np.int32, copy=False)

    graph_dir = ensure_dir(phase_dir / "graph")
    ptr_path = graph_dir / f"{prefix}_ptr.npy"
    neighbor_path = graph_dir / f"{prefix}_neighbors.npy"
    edge_type_path = graph_dir / f"{prefix}_edge_type.npy"
    edge_time_path = graph_dir / f"{prefix}_edge_timestamp.npy"
    np.save(ptr_path, ptr)
    np.save(neighbor_path, neighbor_ordered)
    np.save(edge_type_path, edge_type_ordered)
    np.save(edge_time_path, edge_timestamp_ordered)
    return {
        f"{prefix}_ptr": str(ptr_path.name),
        f"{prefix}_neighbors": str(neighbor_path.name),
        f"{prefix}_edge_type": str(edge_type_path.name),
        f"{prefix}_edge_timestamp": str(edge_time_path.name),
    }


def _bincount_float(indices: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(indices, weights=weights, minlength=size).astype(np.float32, copy=False)


def _stable_std(value: float) -> float:
    return float(max(value, 1e-6))


def _feature_normalization_type(feature_name: str) -> str:
    if feature_name.endswith("_is_neg1") or feature_name == "touch_background":
        return "identity"
    if feature_name.startswith(f"tb{TEMPORAL_BUCKET_DAYS}_"):
        return "identity"
    if feature_name.startswith("x") and feature_name[1:].isdigit():
        return "raw"
    log_prefixes = (
        "in_type_",
        "out_type_",
        "bg_in_type_",
        "bg_out_type_",
        "window_",
    )
    log_exact = {
        "missing_count",
        "indegree",
        "outdegree",
        "total_degree",
        "bg_in_count",
        "bg_out_count",
        "bg_total_count",
        "first_active",
        "last_active",
        "active_span",
        "snap_in_count",
        "snap_out_count",
        "snap_total_count",
        "snap_bg_in_count",
        "snap_bg_out_count",
        "snap_bg_total_count",
        "snapshot_day",
        "snapshot_age",
        "unique_out",
        "unique_in",
        "multi_out",
        "multi_in",
        "days_since_out_mean",
        "days_since_in_mean",
        "days_since_out_last",
        "days_since_in_last",
    }
    if feature_name.startswith("out_last") or feature_name.startswith("in_last"):
        if "_ratio_" not in feature_name and not feature_name.endswith("_ratio"):
            return "log"
    if feature_name.startswith("aux_label_"):
        if feature_name.endswith("_count") or feature_name.endswith("_last14"):
            return "log"
    if (
        feature_name.startswith("recent_")
        or feature_name.startswith("start_")
    ) and feature_name.endswith("_count"):
        return "log"
    if feature_name in log_exact or feature_name.startswith(log_prefixes):
        return "log"
    return "zscore"


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / (denominator + 1.0)


def _bucketwise_zscore(
    values: np.ndarray,
    bucket_ids: np.ndarray,
    num_buckets: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    values_f = np.asarray(values, dtype=np.float32)
    buckets = np.asarray(bucket_ids, dtype=np.int32)
    output = np.zeros(values_f.shape[0], dtype=np.float32)

    if valid_mask is None:
        global_valid = np.ones(values_f.shape[0], dtype=bool)
    else:
        global_valid = np.asarray(valid_mask, dtype=bool)

    if np.any(global_valid):
        global_mean = float(np.mean(values_f[global_valid], dtype=np.float64))
        global_std = float(np.std(values_f[global_valid], dtype=np.float64))
    else:
        global_mean = 0.0
        global_std = 1.0
    global_std = max(global_std, 1e-6)

    for bucket_idx in range(num_buckets):
        bucket_mask = buckets == bucket_idx
        if not np.any(bucket_mask):
            continue
        if valid_mask is None:
            bucket_valid = bucket_mask
        else:
            bucket_valid = bucket_mask & global_valid
        if np.any(bucket_valid):
            mean = float(np.mean(values_f[bucket_valid], dtype=np.float64))
            std = float(np.std(values_f[bucket_valid], dtype=np.float64))
            std = max(std, 1e-6)
        else:
            mean = global_mean
            std = global_std
        if valid_mask is None:
            output[bucket_mask] = ((values_f[bucket_mask] - mean) / std).astype(np.float32, copy=False)
        else:
            observed_mask = bucket_mask & global_valid
            output[observed_mask] = ((values_f[observed_mask] - mean) / std).astype(
                np.float32,
                copy=False,
            )
    return output


def _build_temporal_bucket_norm_feature_block(
    x: np.ndarray,
    missing_mask: np.ndarray,
    first_active: np.ndarray,
    indegree: np.ndarray,
    outdegree: np.ndarray,
    bg_in_count: np.ndarray,
    bg_out_count: np.ndarray,
) -> np.ndarray:
    bucket_ids = np.maximum((np.asarray(first_active, dtype=np.int32) - 1) // TEMPORAL_BUCKET_DAYS, 0)
    num_buckets = int(bucket_ids.max()) + 1 if bucket_ids.size else 1
    missing_count = np.sum(missing_mask, axis=1, dtype=np.float32)

    indegree_f = indegree.astype(np.float32, copy=False)
    outdegree_f = outdegree.astype(np.float32, copy=False)
    total_degree_f = (indegree_f + outdegree_f).astype(np.float32, copy=False)
    bg_in_ratio = _safe_ratio(bg_in_count.astype(np.float32, copy=False), indegree_f)
    bg_out_ratio = _safe_ratio(bg_out_count.astype(np.float32, copy=False), outdegree_f)
    bg_total_count = (bg_in_count + bg_out_count).astype(np.float32, copy=False)
    bg_total_ratio = _safe_ratio(bg_total_count, total_degree_f)

    blocks: list[np.ndarray] = []
    raw_feature_count = int(x.shape[1])
    for feature_idx in range(raw_feature_count):
        observed_mask = missing_mask[:, feature_idx] < 0.5
        blocks.append(
            _bucketwise_zscore(
                values=x[:, feature_idx],
                bucket_ids=bucket_ids,
                num_buckets=num_buckets,
                valid_mask=observed_mask,
            ).reshape(-1, 1)
        )
        missing_values = missing_mask[:, feature_idx].astype(np.float32, copy=False)
        bucket_missing_z = np.zeros_like(missing_values)
        for bucket_idx in range(num_buckets):
            bucket_mask = bucket_ids == bucket_idx
            if not np.any(bucket_mask):
                continue
            rate = float(np.mean(missing_values[bucket_mask], dtype=np.float64))
            scale = max(np.sqrt(rate * max(1.0 - rate, 0.0)), 1e-3)
            bucket_missing_z[bucket_mask] = ((missing_values[bucket_mask] - rate) / scale).astype(
                np.float32,
                copy=False,
            )
        blocks.append(bucket_missing_z.reshape(-1, 1))

    for values in (
        missing_count,
        np.log1p(indegree_f),
        np.log1p(outdegree_f),
        np.log1p(total_degree_f),
        bg_in_ratio,
        bg_out_ratio,
        bg_total_ratio,
        np.log1p(bg_total_count),
    ):
        blocks.append(
            _bucketwise_zscore(
                values=values,
                bucket_ids=bucket_ids,
                num_buckets=num_buckets,
            ).reshape(-1, 1)
        )

    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def _build_temporal_safe_feature_blocks(
    data: PhaseData,
    time_windows: list[dict[str, int]],
    node_time_bucket: np.ndarray,
    first_active: np.ndarray,
    background_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    num_nodes = data.num_nodes
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_type = np.asarray(data.edge_type, dtype=np.int16)

    snapshot_blocks = np.zeros((num_nodes, 13), dtype=np.float32)
    recent_blocks = np.zeros((num_nodes, len(TEMPORAL_RECENT_WINDOWS) * 8), dtype=np.float32)
    num_edge_types = max(int(np.max(edge_type)), 1)
    relation_blocks = np.zeros((num_nodes, num_edge_types * 2), dtype=np.float32)

    for bucket_idx, window in enumerate(time_windows):
        bucket_mask = node_time_bucket == bucket_idx
        if not np.any(bucket_mask):
            continue

        snapshot_end = int(window["end_day"])
        snapshot_edge_mask = edge_timestamp <= snapshot_end
        snap_src = src[snapshot_edge_mask]
        snap_dst = dst[snapshot_edge_mask]

        snap_in_count = np.bincount(snap_dst, minlength=num_nodes).astype(np.float32, copy=False)
        snap_out_count = np.bincount(snap_src, minlength=num_nodes).astype(np.float32, copy=False)
        snap_total_count = snap_in_count + snap_out_count

        snap_bg_out_mask = background_mask[snap_dst]
        snap_bg_in_mask = background_mask[snap_src]
        snap_bg_out_count = np.bincount(
            snap_src[snap_bg_out_mask],
            minlength=num_nodes,
        ).astype(np.float32, copy=False)
        snap_bg_in_count = np.bincount(
            snap_dst[snap_bg_in_mask],
            minlength=num_nodes,
        ).astype(np.float32, copy=False)
        snap_bg_total_count = snap_bg_in_count + snap_bg_out_count

        snapshot_bucket = np.column_stack(
            [
                snap_in_count,
                snap_out_count,
                snap_total_count,
                _safe_ratio(snap_out_count, snap_in_count),
                _safe_ratio(snap_in_count, snap_out_count),
                snap_bg_in_count,
                snap_bg_out_count,
                snap_bg_total_count,
                _safe_ratio(snap_bg_in_count, snap_in_count),
                _safe_ratio(snap_bg_out_count, snap_out_count),
                _safe_ratio(snap_bg_total_count, snap_total_count),
                np.full(num_nodes, snapshot_end, dtype=np.float32),
                np.clip(snapshot_end - first_active, 0, None).astype(np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)
        snapshot_blocks[bucket_mask] = snapshot_bucket[bucket_mask]

        recent_columns: list[np.ndarray] = []
        for recent_days in TEMPORAL_RECENT_WINDOWS:
            recent_edge_mask = snapshot_edge_mask & (edge_timestamp > snapshot_end - recent_days)
            recent_src = src[recent_edge_mask]
            recent_dst = dst[recent_edge_mask]

            recent_in_count = np.bincount(recent_dst, minlength=num_nodes).astype(np.float32, copy=False)
            recent_out_count = np.bincount(recent_src, minlength=num_nodes).astype(np.float32, copy=False)
            recent_total_count = recent_in_count + recent_out_count

            recent_bg_out_mask = background_mask[recent_dst]
            recent_bg_in_mask = background_mask[recent_src]
            recent_bg_out_count = np.bincount(
                recent_src[recent_bg_out_mask],
                minlength=num_nodes,
            ).astype(np.float32, copy=False)
            recent_bg_in_count = np.bincount(
                recent_dst[recent_bg_in_mask],
                minlength=num_nodes,
            ).astype(np.float32, copy=False)
            recent_bg_total_count = recent_bg_in_count + recent_bg_out_count

            recent_columns.extend(
                [
                    recent_in_count,
                    recent_out_count,
                    recent_total_count,
                    recent_bg_in_count,
                    recent_bg_out_count,
                    recent_bg_total_count,
                    _safe_ratio(recent_total_count, snap_total_count),
                    _safe_ratio(recent_bg_total_count, snap_bg_total_count),
                ]
            )
        recent_bucket = np.column_stack(recent_columns).astype(np.float32, copy=False)
        recent_blocks[bucket_mask] = recent_bucket[bucket_mask]

        relation_edge_mask = snapshot_edge_mask & (
            edge_timestamp > snapshot_end - TEMPORAL_RELATION_RECENT_WINDOW
        )
        relation_src = src[relation_edge_mask]
        relation_dst = dst[relation_edge_mask]
        relation_type = edge_type[relation_edge_mask]
        relation_columns: list[np.ndarray] = []
        for current_type in range(1, num_edge_types + 1):
            type_mask = relation_type == current_type
            relation_columns.extend(
                [
                    np.bincount(
                        relation_dst[type_mask],
                        minlength=num_nodes,
                    ).astype(np.float32, copy=False),
                    np.bincount(
                        relation_src[type_mask],
                        minlength=num_nodes,
                    ).astype(np.float32, copy=False),
                ]
            )
        relation_bucket = np.column_stack(relation_columns).astype(np.float32, copy=False)
        relation_blocks[bucket_mask] = relation_bucket[bucket_mask]

    return {
        "temporal_snapshot": snapshot_blocks,
        "temporal_recent": recent_blocks,
        "temporal_relation_recent": relation_blocks,
    }


def _build_activation_early_feature_block(
    data: PhaseData,
    first_active: np.ndarray,
    background_mask: np.ndarray,
) -> np.ndarray:
    num_nodes = data.num_nodes
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_type = np.asarray(data.edge_type, dtype=np.int16)

    columns: list[np.ndarray] = []
    for early_days in ACTIVATION_EARLY_WINDOWS:
        out_mask = (edge_timestamp >= first_active[src]) & (
            edge_timestamp <= first_active[src] + early_days - 1
        )
        in_mask = (edge_timestamp >= first_active[dst]) & (
            edge_timestamp <= first_active[dst] + early_days - 1
        )

        early_out_count = np.bincount(src[out_mask], minlength=num_nodes).astype(np.float32, copy=False)
        early_in_count = np.bincount(dst[in_mask], minlength=num_nodes).astype(np.float32, copy=False)
        early_total_count = early_in_count + early_out_count

        early_bg_out_mask = out_mask & background_mask[dst]
        early_bg_in_mask = in_mask & background_mask[src]
        early_bg_out_count = np.bincount(
            src[early_bg_out_mask],
            minlength=num_nodes,
        ).astype(np.float32, copy=False)
        early_bg_in_count = np.bincount(
            dst[early_bg_in_mask],
            minlength=num_nodes,
        ).astype(np.float32, copy=False)
        early_bg_total_count = early_bg_in_count + early_bg_out_count

        columns.extend(
            [
                early_in_count,
                early_out_count,
                early_total_count,
                early_bg_in_count,
                early_bg_out_count,
                early_bg_total_count,
            ]
        )

    relation_window = ACTIVATION_RELATION_WINDOW
    out_relation_mask = (edge_timestamp >= first_active[src]) & (
        edge_timestamp <= first_active[src] + relation_window - 1
    )
    in_relation_mask = (edge_timestamp >= first_active[dst]) & (
        edge_timestamp <= first_active[dst] + relation_window - 1
    )
    num_edge_types = max(int(np.max(edge_type)), 1)
    for current_type in range(1, num_edge_types + 1):
        typed_out_mask = out_relation_mask & (edge_type == current_type)
        typed_in_mask = in_relation_mask & (edge_type == current_type)
        columns.extend(
            [
                np.bincount(dst[typed_in_mask], minlength=num_nodes).astype(np.float32, copy=False),
                np.bincount(src[typed_out_mask], minlength=num_nodes).astype(np.float32, copy=False),
            ]
        )

    return np.column_stack(columns).astype(np.float32, copy=False)


def _build_temporal_neighbor_feature_block(
    data: PhaseData,
    time_windows: list[dict[str, int]],
    node_time_bucket: np.ndarray,
    x: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    num_nodes = data.num_nodes
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    raw_feature_count = int(x.shape[1])
    block = np.zeros((num_nodes, raw_feature_count * 2 * 3), dtype=np.float32)

    for bucket_idx, window in enumerate(time_windows):
        bucket_mask = node_time_bucket == bucket_idx
        if not np.any(bucket_mask):
            continue

        snapshot_end = int(window["end_day"])
        snapshot_edge_mask = edge_timestamp <= snapshot_end
        snap_src = src[snapshot_edge_mask]
        snap_dst = dst[snapshot_edge_mask]
        snap_indegree = np.bincount(snap_dst, minlength=num_nodes).astype(np.float32, copy=False)
        snap_outdegree = np.bincount(snap_src, minlength=num_nodes).astype(np.float32, copy=False)
        in_den = np.maximum(snap_indegree, 1.0)
        out_den = np.maximum(snap_outdegree, 1.0)

        col = 0
        for reducer in ("mean", "max", "missing_ratio"):
            for feature_idx in range(raw_feature_count):
                values = x[:, feature_idx] if reducer != "missing_ratio" else missing_mask[:, feature_idx]
                in_center_values = values[snap_src]
                out_center_values = values[snap_dst]
                if reducer in {"mean", "missing_ratio"}:
                    in_result = _bincount_float(snap_dst, in_center_values, num_nodes) / in_den
                    out_result = _bincount_float(snap_src, out_center_values, num_nodes) / out_den
                else:
                    in_result = np.full(num_nodes, -np.inf, dtype=np.float32)
                    out_result = np.full(num_nodes, -np.inf, dtype=np.float32)
                    np.maximum.at(in_result, snap_dst, in_center_values.astype(np.float32, copy=False))
                    np.maximum.at(out_result, snap_src, out_center_values.astype(np.float32, copy=False))
                    in_result[snap_indegree == 0] = 0.0
                    out_result[snap_outdegree == 0] = 0.0
                block[bucket_mask, col] = in_result[bucket_mask]
                block[bucket_mask, col + 1] = out_result[bucket_mask]
                col += 2

    return block


def _build_graph_stats_feature_block(
    data: PhaseData,
    indegree: np.ndarray,
    outdegree: np.ndarray,
) -> np.ndarray:
    num_nodes = data.num_nodes
    max_day = int(np.max(data.edge_timestamp))
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    outdegree_f = np.asarray(outdegree, dtype=np.float32)
    indegree_f = np.asarray(indegree, dtype=np.float32)

    pair_code = src.astype(np.int64, copy=False) * np.int64(num_nodes) + dst.astype(np.int64, copy=False)
    _, unique_idx = np.unique(pair_code, return_index=True)
    unique_out = np.bincount(src[unique_idx], minlength=num_nodes).astype(np.float32, copy=False)
    unique_in = np.bincount(dst[unique_idx], minlength=num_nodes).astype(np.float32, copy=False)
    multi_out = np.clip(outdegree_f - unique_out, 0.0, None).astype(np.float32, copy=False)
    multi_in = np.clip(indegree_f - unique_in, 0.0, None).astype(np.float32, copy=False)

    out_ts_sum = np.bincount(src, weights=edge_timestamp, minlength=num_nodes).astype(np.float32, copy=False)
    in_ts_sum = np.bincount(dst, weights=edge_timestamp, minlength=num_nodes).astype(np.float32, copy=False)
    out_mean_ts = out_ts_sum / np.maximum(outdegree_f, 1.0)
    in_mean_ts = in_ts_sum / np.maximum(indegree_f, 1.0)
    days_since_out_mean = np.full(num_nodes, float(max_day + 1), dtype=np.float32)
    days_since_in_mean = np.full(num_nodes, float(max_day + 1), dtype=np.float32)
    out_mask = outdegree_f > 0
    in_mask = indegree_f > 0
    days_since_out_mean[out_mask] = float(max_day) - out_mean_ts[out_mask]
    days_since_in_mean[in_mask] = float(max_day) - in_mean_ts[in_mask]

    out_last_ts = np.full(num_nodes, -1, dtype=np.int32)
    in_last_ts = np.full(num_nodes, -1, dtype=np.int32)
    np.maximum.at(out_last_ts, src, edge_timestamp)
    np.maximum.at(in_last_ts, dst, edge_timestamp)
    days_since_out_last = np.full(num_nodes, float(max_day + 1), dtype=np.float32)
    days_since_in_last = np.full(num_nodes, float(max_day + 1), dtype=np.float32)
    out_last_mask = out_last_ts >= 0
    in_last_mask = in_last_ts >= 0
    days_since_out_last[out_last_mask] = float(max_day) - out_last_ts[out_last_mask].astype(np.float32, copy=False)
    days_since_in_last[in_last_mask] = float(max_day) - in_last_ts[in_last_mask].astype(np.float32, copy=False)

    columns: list[np.ndarray] = [
        unique_out,
        unique_in,
        multi_out,
        multi_in,
        unique_out / (outdegree_f + 1.0),
        unique_in / (indegree_f + 1.0),
        days_since_out_mean,
        days_since_in_mean,
        days_since_out_last,
        days_since_in_last,
    ]

    for recent_days in GRAPH_RECENT_WINDOWS:
        recent_mask = edge_timestamp > (max_day - recent_days)
        recent_out = np.bincount(src[recent_mask], minlength=num_nodes).astype(np.float32, copy=False)
        recent_in = np.bincount(dst[recent_mask], minlength=num_nodes).astype(np.float32, copy=False)
        columns.extend(
            [
                recent_out,
                recent_in,
                recent_out / (outdegree_f + 1.0),
                recent_in / (indegree_f + 1.0),
            ]
        )

    labels = np.asarray(data.y, dtype=np.int32)
    recent14_mask = edge_timestamp > (max_day - 14)
    for bg_label in _active_background_labels():
        bg_out_neighbor = labels[dst] == bg_label
        bg_in_neighbor = labels[src] == bg_label
        bg_out_count = np.bincount(src[bg_out_neighbor], minlength=num_nodes).astype(np.float32, copy=False)
        bg_in_count = np.bincount(dst[bg_in_neighbor], minlength=num_nodes).astype(np.float32, copy=False)
        bg_out_recent = np.bincount(
            src[bg_out_neighbor & recent14_mask],
            minlength=num_nodes,
        ).astype(np.float32, copy=False)
        bg_in_recent = np.bincount(
            dst[bg_in_neighbor & recent14_mask],
            minlength=num_nodes,
        ).astype(np.float32, copy=False)
        bg_total = bg_in_count + bg_out_count
        columns.extend(
            [
                bg_out_count,
                bg_in_count,
                bg_total,
                bg_out_count / (outdegree_f + 1.0),
                bg_in_count / (indegree_f + 1.0),
                bg_out_recent,
                bg_in_recent,
            ]
        )

    return np.column_stack(columns).astype(np.float32, copy=False)


def _build_graph_time_detrend_feature_block(
    data: PhaseData,
    indegree: np.ndarray,
    outdegree: np.ndarray,
    first_active: np.ndarray,
) -> np.ndarray:
    num_nodes = data.num_nodes
    max_day = int(np.max(data.edge_timestamp))
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    outdegree_f = np.asarray(outdegree, dtype=np.float32)
    indegree_f = np.asarray(indegree, dtype=np.float32)

    out_ts_sum = np.bincount(src, weights=edge_timestamp, minlength=num_nodes).astype(np.float32, copy=False)
    in_ts_sum = np.bincount(dst, weights=edge_timestamp, minlength=num_nodes).astype(np.float32, copy=False)
    out_mean_ts = out_ts_sum / np.maximum(outdegree_f, 1.0)
    in_mean_ts = in_ts_sum / np.maximum(indegree_f, 1.0)
    days_since_out_mean = np.full(num_nodes, float(max_day + 1), dtype=np.float32)
    days_since_in_mean = np.full(num_nodes, float(max_day + 1), dtype=np.float32)
    out_mask = outdegree_f > 0
    in_mask = indegree_f > 0
    days_since_out_mean[out_mask] = float(max_day) - out_mean_ts[out_mask]
    days_since_in_mean[in_mask] = float(max_day) - in_mean_ts[in_mask]

    out_last_ts = np.full(num_nodes, -1, dtype=np.int32)
    in_last_ts = np.full(num_nodes, -1, dtype=np.int32)
    np.maximum.at(out_last_ts, src, edge_timestamp)
    np.maximum.at(in_last_ts, dst, edge_timestamp)
    days_since_out_last = np.full(num_nodes, float(max_day + 1), dtype=np.float32)
    days_since_in_last = np.full(num_nodes, float(max_day + 1), dtype=np.float32)
    out_last_mask = out_last_ts >= 0
    in_last_mask = in_last_ts >= 0
    days_since_out_last[out_last_mask] = float(max_day) - out_last_ts[out_last_mask].astype(
        np.float32,
        copy=False,
    )
    days_since_in_last[in_last_mask] = float(max_day) - in_last_ts[in_last_mask].astype(
        np.float32,
        copy=False,
    )

    recent_mask = edge_timestamp > (max_day - 30)
    recent_out = np.bincount(src[recent_mask], minlength=num_nodes).astype(np.float32, copy=False)
    recent_in = np.bincount(dst[recent_mask], minlength=num_nodes).astype(np.float32, copy=False)
    recent_out_ratio = _safe_ratio(recent_out, outdegree_f)
    recent_in_ratio = _safe_ratio(recent_in, indegree_f)

    bucket_ids = np.maximum((np.asarray(first_active, dtype=np.int32) - 1) // TEMPORAL_BUCKET_DAYS, 0)
    num_buckets = int(bucket_ids.max()) + 1 if bucket_ids.size else 1
    columns = [
        _bucketwise_zscore(days_since_out_mean, bucket_ids=bucket_ids, num_buckets=num_buckets),
        _bucketwise_zscore(days_since_in_mean, bucket_ids=bucket_ids, num_buckets=num_buckets),
        _bucketwise_zscore(days_since_out_last, bucket_ids=bucket_ids, num_buckets=num_buckets),
        _bucketwise_zscore(days_since_in_last, bucket_ids=bucket_ids, num_buckets=num_buckets),
        _bucketwise_zscore(recent_out_ratio, bucket_ids=bucket_ids, num_buckets=num_buckets),
        _bucketwise_zscore(recent_in_ratio, bucket_ids=bucket_ids, num_buckets=num_buckets),
    ]
    return np.column_stack(columns).astype(np.float32, copy=False)


def _update_peak_feature(
    peak_values: np.ndarray,
    candidate: np.ndarray,
    window_idx: int,
    peak_window: np.ndarray | None = None,
) -> None:
    better_mask = np.asarray(candidate, dtype=np.float32) > peak_values
    peak_values[better_mask] = candidate[better_mask]
    if peak_window is not None:
        peak_window[better_mask] = float(window_idx)


def _build_temporal_adaptive_window_feature_block(
    data: PhaseData,
    indegree: np.ndarray,
    outdegree: np.ndarray,
    first_active: np.ndarray,
) -> np.ndarray:
    num_nodes = data.num_nodes
    max_day = int(np.max(data.edge_timestamp))
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)

    outdegree_f = np.asarray(outdegree, dtype=np.float32)
    indegree_f = np.asarray(indegree, dtype=np.float32)
    total_degree_f = (outdegree_f + indegree_f).astype(np.float32, copy=False)
    out_valid = outdegree_f > 0
    in_valid = indegree_f > 0
    total_valid = total_degree_f > 0

    bucket_ids = np.maximum((np.asarray(first_active, dtype=np.int32) - 1) // TEMPORAL_BUCKET_DAYS, 0)
    num_buckets = int(bucket_ids.max()) + 1 if bucket_ids.size else 1

    out_count_peak_z = np.full(num_nodes, -np.inf, dtype=np.float32)
    in_count_peak_z = np.full(num_nodes, -np.inf, dtype=np.float32)
    total_count_peak_z = np.full(num_nodes, -np.inf, dtype=np.float32)
    out_ratio_peak_z = np.full(num_nodes, -np.inf, dtype=np.float32)
    in_ratio_peak_z = np.full(num_nodes, -np.inf, dtype=np.float32)
    total_ratio_peak_z = np.full(num_nodes, -np.inf, dtype=np.float32)
    out_peak_window = np.zeros(num_nodes, dtype=np.float32)
    in_peak_window = np.zeros(num_nodes, dtype=np.float32)
    total_peak_window = np.zeros(num_nodes, dtype=np.float32)

    short_window, long_window = TEMPORAL_ADAPTIVE_CONTRAST_WINDOWS
    short_out = short_in = short_total = None
    long_out = long_in = long_total = None

    for window_idx, recent_days in enumerate(TEMPORAL_ADAPTIVE_WINDOWS):
        recent_mask = edge_timestamp > (max_day - int(recent_days))
        out_count = np.bincount(src[recent_mask], minlength=num_nodes).astype(np.float32, copy=False)
        in_count = np.bincount(dst[recent_mask], minlength=num_nodes).astype(np.float32, copy=False)
        total_count = (out_count + in_count).astype(np.float32, copy=False)

        _update_peak_feature(
            out_count_peak_z,
            _bucketwise_zscore(out_count, bucket_ids=bucket_ids, num_buckets=num_buckets),
            window_idx,
            peak_window=out_peak_window,
        )
        _update_peak_feature(
            in_count_peak_z,
            _bucketwise_zscore(in_count, bucket_ids=bucket_ids, num_buckets=num_buckets),
            window_idx,
            peak_window=in_peak_window,
        )
        _update_peak_feature(
            total_count_peak_z,
            _bucketwise_zscore(total_count, bucket_ids=bucket_ids, num_buckets=num_buckets),
            window_idx,
            peak_window=total_peak_window,
        )

        _update_peak_feature(
            out_ratio_peak_z,
            _bucketwise_zscore(
                _safe_ratio(out_count, outdegree_f),
                bucket_ids=bucket_ids,
                num_buckets=num_buckets,
                valid_mask=out_valid,
            ),
            window_idx,
        )
        _update_peak_feature(
            in_ratio_peak_z,
            _bucketwise_zscore(
                _safe_ratio(in_count, indegree_f),
                bucket_ids=bucket_ids,
                num_buckets=num_buckets,
                valid_mask=in_valid,
            ),
            window_idx,
        )
        _update_peak_feature(
            total_ratio_peak_z,
            _bucketwise_zscore(
                _safe_ratio(total_count, total_degree_f),
                bucket_ids=bucket_ids,
                num_buckets=num_buckets,
                valid_mask=total_valid,
            ),
            window_idx,
        )

        if recent_days == short_window:
            short_out = out_count
            short_in = in_count
            short_total = total_count
        if recent_days == long_window:
            long_out = out_count
            long_in = in_count
            long_total = total_count

    if short_out is None or short_in is None or short_total is None:
        raise AssertionError("Missing short-window counts for temporal adaptive features.")
    if long_out is None or long_in is None or long_total is None:
        raise AssertionError("Missing long-window counts for temporal adaptive features.")

    short_out_density = short_out / float(short_window)
    short_in_density = short_in / float(short_window)
    short_total_density = short_total / float(short_window)
    long_out_density = long_out / float(long_window)
    long_in_density = long_in / float(long_window)
    long_total_density = long_total / float(long_window)
    peak_window_scale = float(max(len(TEMPORAL_ADAPTIVE_WINDOWS) - 1, 1))

    return np.column_stack(
        [
            out_count_peak_z,
            in_count_peak_z,
            total_count_peak_z,
            out_ratio_peak_z,
            in_ratio_peak_z,
            total_ratio_peak_z,
            _bucketwise_zscore(
                np.log1p(short_out_density) - np.log1p(long_out_density),
                bucket_ids=bucket_ids,
                num_buckets=num_buckets,
                valid_mask=out_valid,
            ),
            _bucketwise_zscore(
                np.log1p(short_in_density) - np.log1p(long_in_density),
                bucket_ids=bucket_ids,
                num_buckets=num_buckets,
                valid_mask=in_valid,
            ),
            _bucketwise_zscore(
                np.log1p(short_total_density) - np.log1p(long_total_density),
                bucket_ids=bucket_ids,
                num_buckets=num_buckets,
                valid_mask=total_valid,
            ),
            out_peak_window / peak_window_scale,
            in_peak_window / peak_window_scale,
            total_peak_window / peak_window_scale,
        ]
    ).astype(np.float32, copy=False)


def _build_temporal_dual_scale_delta_feature_block(
    data: PhaseData,
    indegree: np.ndarray,
    outdegree: np.ndarray,
    first_active: np.ndarray,
) -> np.ndarray:
    num_nodes = data.num_nodes
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    if edge_timestamp.size == 0:
        return np.zeros((num_nodes, len(TEMPORAL_DUAL_SCALE_PAIRS) * 9), dtype=np.float32)

    max_day = int(np.max(edge_timestamp))
    outdegree_f = np.asarray(outdegree, dtype=np.float32)
    indegree_f = np.asarray(indegree, dtype=np.float32)
    total_degree_f = (outdegree_f + indegree_f).astype(np.float32, copy=False)
    out_valid = outdegree_f > 0
    in_valid = indegree_f > 0
    total_valid = total_degree_f > 0

    bucket_ids = np.maximum((np.asarray(first_active, dtype=np.int32) - 1) // TEMPORAL_BUCKET_DAYS, 0)
    num_buckets = int(bucket_ids.max()) + 1 if bucket_ids.size else 1
    columns: list[np.ndarray] = []

    for short_window, long_window in TEMPORAL_DUAL_SCALE_PAIRS:
        short_mask = edge_timestamp > (max_day - int(short_window))
        long_mask = edge_timestamp > (max_day - int(long_window))

        short_out = np.bincount(src[short_mask], minlength=num_nodes).astype(np.float32, copy=False)
        short_in = np.bincount(dst[short_mask], minlength=num_nodes).astype(np.float32, copy=False)
        short_total = (short_out + short_in).astype(np.float32, copy=False)
        long_out = np.bincount(src[long_mask], minlength=num_nodes).astype(np.float32, copy=False)
        long_in = np.bincount(dst[long_mask], minlength=num_nodes).astype(np.float32, copy=False)
        long_total = (long_out + long_in).astype(np.float32, copy=False)

        short_out_density = short_out / float(short_window)
        short_in_density = short_in / float(short_window)
        short_total_density = short_total / float(short_window)
        long_out_density = long_out / float(long_window)
        long_in_density = long_in / float(long_window)
        long_total_density = long_total / float(long_window)

        short_out_burst = _max_daily_burst_feature(src[short_mask], edge_timestamp[short_mask], num_nodes)
        short_in_burst = _max_daily_burst_feature(dst[short_mask], edge_timestamp[short_mask], num_nodes)
        short_total_burst = _max_daily_burst_feature(
            np.concatenate([src[short_mask], dst[short_mask]]).astype(np.int32, copy=False),
            np.concatenate([edge_timestamp[short_mask], edge_timestamp[short_mask]]).astype(np.int32, copy=False),
            num_nodes,
        )

        columns.extend(
            [
                _bucketwise_zscore(
                    np.log1p(short_out_density) - np.log1p(long_out_density),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=out_valid,
                ),
                _bucketwise_zscore(
                    _safe_ratio(short_out, long_out),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=out_valid,
                ),
                _bucketwise_zscore(
                    np.log1p(short_out_burst) - np.log1p(long_out_density),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=out_valid,
                ),
                _bucketwise_zscore(
                    np.log1p(short_in_density) - np.log1p(long_in_density),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=in_valid,
                ),
                _bucketwise_zscore(
                    _safe_ratio(short_in, long_in),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=in_valid,
                ),
                _bucketwise_zscore(
                    np.log1p(short_in_burst) - np.log1p(long_in_density),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=in_valid,
                ),
                _bucketwise_zscore(
                    np.log1p(short_total_density) - np.log1p(long_total_density),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=total_valid,
                ),
                _bucketwise_zscore(
                    _safe_ratio(short_total, long_total),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=total_valid,
                ),
                _bucketwise_zscore(
                    np.log1p(short_total_burst) - np.log1p(long_total_density),
                    bucket_ids=bucket_ids,
                    num_buckets=num_buckets,
                    valid_mask=total_valid,
                ),
            ]
        )

    return np.column_stack(columns).astype(np.float32, copy=False)


def _partner_concentration_stats(
    centers: np.ndarray,
    partners: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    if centers.size == 0:
        zeros = np.zeros(num_nodes, dtype=np.float32)
        return zeros, zeros

    pair_code = (
        centers.astype(np.int64, copy=False) * np.int64(num_nodes)
        + partners.astype(np.int64, copy=False)
    )
    unique_pair_code, pair_counts = np.unique(pair_code, return_counts=True)
    center_ids = (unique_pair_code // np.int64(num_nodes)).astype(np.int32, copy=False)
    total_counts = np.bincount(centers, minlength=num_nodes).astype(np.float32, copy=False)
    pair_counts_f = pair_counts.astype(np.float32, copy=False)

    max_pair_count = np.zeros(num_nodes, dtype=np.float32)
    np.maximum.at(max_pair_count, center_ids, pair_counts_f)
    top_share = _safe_ratio(max_pair_count, total_counts)

    probs = pair_counts_f / np.maximum(total_counts[center_ids], 1.0)
    entropy_terms = -(probs * np.log(np.clip(probs, 1e-8, None))).astype(np.float32, copy=False)
    entropy = np.bincount(center_ids, weights=entropy_terms, minlength=num_nodes).astype(np.float32, copy=False)
    unique_partner_count = np.bincount(center_ids, minlength=num_nodes).astype(np.float32, copy=False)
    entropy_norm = np.zeros(num_nodes, dtype=np.float32)
    valid = unique_partner_count > 1.0
    entropy_norm[valid] = (
        entropy[valid]
        / np.maximum(np.log(unique_partner_count[valid]), 1e-6)
    ).astype(np.float32, copy=False)
    return top_share.astype(np.float32, copy=False), entropy_norm.astype(np.float32, copy=False)


def _build_temporal_counterparty_concentration_feature_block(
    data: PhaseData,
) -> np.ndarray:
    num_nodes = data.num_nodes
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    if edge_timestamp.size == 0:
        return np.zeros((num_nodes, len(TEMPORAL_CONCENTRATION_WINDOWS) * 6), dtype=np.float32)

    max_day = int(np.max(edge_timestamp))
    columns: list[np.ndarray] = []
    for recent_days in TEMPORAL_CONCENTRATION_WINDOWS:
        recent_mask = edge_timestamp > (max_day - int(recent_days))
        recent_src = src[recent_mask]
        recent_dst = dst[recent_mask]
        out_top_share, out_entropy = _partner_concentration_stats(
            centers=recent_src,
            partners=recent_dst,
            num_nodes=num_nodes,
        )
        in_top_share, in_entropy = _partner_concentration_stats(
            centers=recent_dst,
            partners=recent_src,
            num_nodes=num_nodes,
        )
        total_top_share, total_entropy = _partner_concentration_stats(
            centers=np.concatenate([recent_src, recent_dst]).astype(np.int32, copy=False),
            partners=np.concatenate([recent_dst, recent_src]).astype(np.int32, copy=False),
            num_nodes=num_nodes,
        )
        columns.extend(
            [
                out_top_share,
                in_top_share,
                total_top_share,
                out_entropy,
                in_entropy,
                total_entropy,
            ]
        )
    return np.column_stack(columns).astype(np.float32, copy=False)


def _build_temporal_counterparty_concentration_recent_feature_block(
    data: PhaseData,
) -> np.ndarray:
    num_nodes = data.num_nodes
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    if edge_timestamp.size == 0:
        return np.zeros((num_nodes, len(TEMPORAL_CONCENTRATION_RECENT_WINDOWS) * 6), dtype=np.float32)

    max_day = int(np.max(edge_timestamp))
    columns: list[np.ndarray] = []
    for recent_days in TEMPORAL_CONCENTRATION_RECENT_WINDOWS:
        recent_mask = edge_timestamp > (max_day - int(recent_days))
        recent_src = src[recent_mask]
        recent_dst = dst[recent_mask]
        out_top_share, out_entropy = _partner_concentration_stats(
            centers=recent_src,
            partners=recent_dst,
            num_nodes=num_nodes,
        )
        in_top_share, in_entropy = _partner_concentration_stats(
            centers=recent_dst,
            partners=recent_src,
            num_nodes=num_nodes,
        )
        total_top_share, total_entropy = _partner_concentration_stats(
            centers=np.concatenate([recent_src, recent_dst]).astype(np.int32, copy=False),
            partners=np.concatenate([recent_dst, recent_src]).astype(np.int32, copy=False),
            num_nodes=num_nodes,
        )
        columns.extend(
            [
                out_top_share,
                in_top_share,
                total_top_share,
                out_entropy,
                in_entropy,
                total_entropy,
            ]
        )
    return np.column_stack(columns).astype(np.float32, copy=False)


def _build_temporal_counterparty_share_focus_feature_block(
    data: PhaseData,
) -> np.ndarray:
    num_nodes = data.num_nodes
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    if edge_timestamp.size == 0:
        return np.zeros((num_nodes, len(TEMPORAL_CONCENTRATION_FOCUS_WINDOWS) * 3), dtype=np.float32)

    max_day = int(np.max(edge_timestamp))
    columns: list[np.ndarray] = []
    for recent_days in TEMPORAL_CONCENTRATION_FOCUS_WINDOWS:
        recent_mask = edge_timestamp > (max_day - int(recent_days))
        recent_src = src[recent_mask]
        recent_dst = dst[recent_mask]
        out_top_share, _ = _partner_concentration_stats(
            centers=recent_src,
            partners=recent_dst,
            num_nodes=num_nodes,
        )
        in_top_share, _ = _partner_concentration_stats(
            centers=recent_dst,
            partners=recent_src,
            num_nodes=num_nodes,
        )
        total_top_share, _ = _partner_concentration_stats(
            centers=np.concatenate([recent_src, recent_dst]).astype(np.int32, copy=False),
            partners=np.concatenate([recent_dst, recent_src]).astype(np.int32, copy=False),
            num_nodes=num_nodes,
        )
        columns.extend([out_top_share, in_top_share, total_top_share])
    return np.column_stack(columns).astype(np.float32, copy=False)


def _max_daily_burst_feature(
    centers: np.ndarray,
    timestamps: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    if centers.size == 0:
        return np.zeros(num_nodes, dtype=np.float32)
    max_day = int(np.max(timestamps)) + 1
    key = centers.astype(np.int64, copy=False) * np.int64(max_day) + timestamps.astype(np.int64, copy=False)
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    centers_sorted = centers[order]
    _, first_idx, counts = np.unique(key_sorted, return_index=True, return_counts=True)
    burst = np.zeros(num_nodes, dtype=np.float32)
    np.maximum.at(
        burst,
        centers_sorted[first_idx],
        counts.astype(np.float32, copy=False),
    )
    return burst


def _build_temporal_motif_lite_feature_block(
    data: PhaseData,
    indegree: np.ndarray,
    outdegree: np.ndarray,
) -> np.ndarray:
    num_nodes = data.num_nodes
    max_day = int(np.max(data.edge_timestamp))
    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_timestamp = np.asarray(data.edge_timestamp, dtype=np.int32)
    outdegree_f = np.asarray(outdegree, dtype=np.float32)
    indegree_f = np.asarray(indegree, dtype=np.float32)
    total_degree = (outdegree_f + indegree_f).astype(np.float32, copy=False)

    recent_mask = edge_timestamp > (max_day - TEMPORAL_MOTIF_RECENT_DAYS)
    recent_src = src[recent_mask]
    recent_dst = dst[recent_mask]
    recent_ts = edge_timestamp[recent_mask]
    recent_out = np.bincount(recent_src, minlength=num_nodes).astype(np.float32, copy=False)
    recent_in = np.bincount(recent_dst, minlength=num_nodes).astype(np.float32, copy=False)

    if recent_src.size == 0:
        return np.zeros((num_nodes, 12), dtype=np.float32)

    recent_pair_code = (
        recent_src.astype(np.int64, copy=False) * np.int64(num_nodes)
        + recent_dst.astype(np.int64, copy=False)
    )
    unique_pair_code = np.unique(recent_pair_code)
    pair_src = (unique_pair_code // np.int64(num_nodes)).astype(np.int32, copy=False)
    pair_dst = (unique_pair_code % np.int64(num_nodes)).astype(np.int32, copy=False)
    reverse_code = (
        pair_dst.astype(np.int64, copy=False) * np.int64(num_nodes)
        + pair_src.astype(np.int64, copy=False)
    )
    reciprocal_mask = np.isin(reverse_code, unique_pair_code)
    recip_out_unique = np.bincount(pair_src[reciprocal_mask], minlength=num_nodes).astype(np.float32, copy=False)
    recip_in_unique = np.bincount(pair_dst[reciprocal_mask], minlength=num_nodes).astype(np.float32, copy=False)
    unique_out_recent = np.bincount(pair_src, minlength=num_nodes).astype(np.float32, copy=False)
    unique_in_recent = np.bincount(pair_dst, minlength=num_nodes).astype(np.float32, copy=False)

    repeat_out_ratio = 1.0 - _safe_ratio(unique_out_recent, recent_out)
    repeat_in_ratio = 1.0 - _safe_ratio(unique_in_recent, recent_in)
    recip_out_ratio = _safe_ratio(recip_out_unique, unique_out_recent)
    recip_in_ratio = _safe_ratio(recip_in_unique, unique_in_recent)

    recent_out_neighbor_deg_logsum = np.bincount(
        recent_src,
        weights=np.log1p(total_degree[recent_dst]).astype(np.float32, copy=False),
        minlength=num_nodes,
    ).astype(np.float32, copy=False)
    recent_in_neighbor_deg_logsum = np.bincount(
        recent_dst,
        weights=np.log1p(total_degree[recent_src]).astype(np.float32, copy=False),
        minlength=num_nodes,
    ).astype(np.float32, copy=False)
    recent_out_neighbor_deg_sum = np.bincount(
        recent_src,
        weights=total_degree[recent_dst].astype(np.float32, copy=False),
        minlength=num_nodes,
    ).astype(np.float32, copy=False)
    recent_in_neighbor_deg_sum = np.bincount(
        recent_dst,
        weights=total_degree[recent_src].astype(np.float32, copy=False),
        minlength=num_nodes,
    ).astype(np.float32, copy=False)

    burst_mask = recent_ts > (max_day - TEMPORAL_MOTIF_BURST_DAYS)
    out_burst_maxday = _max_daily_burst_feature(recent_src[burst_mask], recent_ts[burst_mask], num_nodes)
    in_burst_maxday = _max_daily_burst_feature(recent_dst[burst_mask], recent_ts[burst_mask], num_nodes)

    return np.column_stack(
        [
            recip_out_unique,
            recip_in_unique,
            recip_out_ratio,
            recip_in_ratio,
            repeat_out_ratio.astype(np.float32, copy=False),
            repeat_in_ratio.astype(np.float32, copy=False),
            recent_out_neighbor_deg_sum / np.maximum(recent_out, 1.0),
            recent_in_neighbor_deg_sum / np.maximum(recent_in, 1.0),
            recent_out_neighbor_deg_logsum,
            recent_in_neighbor_deg_logsum,
            out_burst_maxday,
            in_burst_maxday,
        ]
    ).astype(np.float32, copy=False)


def _build_neighbor_similarity_feature_block(
    x: np.ndarray,
    missing_mask: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    indegree: np.ndarray,
    outdegree: np.ndarray,
) -> np.ndarray:
    num_nodes = x.shape[0]
    invalid_mask = (missing_mask > 0.0) | (~np.isfinite(x))
    filled = np.where(invalid_mask, 0.0, x).astype(np.float32, copy=False)
    node_norm = np.sqrt(np.sum(filled * filled, axis=1, dtype=np.float32)).astype(np.float32, copy=False)
    dot = np.zeros(src.shape[0], dtype=np.float32)
    for feature_idx in range(filled.shape[1]):
        dot += filled[src, feature_idx] * filled[dst, feature_idx]
    denom = np.maximum(node_norm[src] * node_norm[dst], 1e-6)
    sim = (dot / denom).astype(np.float32, copy=False)

    out_sum = np.bincount(src, weights=sim, minlength=num_nodes).astype(np.float32, copy=False)
    in_sum = np.bincount(dst, weights=sim, minlength=num_nodes).astype(np.float32, copy=False)

    out_max = np.full(num_nodes, -1.0, dtype=np.float32)
    in_max = np.full(num_nodes, -1.0, dtype=np.float32)
    np.maximum.at(out_max, src, sim)
    np.maximum.at(in_max, dst, sim)
    out_max[np.asarray(outdegree) == 0] = 0.0
    in_max[np.asarray(indegree) == 0] = 0.0

    outdegree_f = np.asarray(outdegree, dtype=np.float32)
    indegree_f = np.asarray(indegree, dtype=np.float32)
    return np.column_stack(
        [
            out_sum,
            in_sum,
            out_sum / np.maximum(outdegree_f, 1.0),
            in_sum / np.maximum(indegree_f, 1.0),
            out_max,
            in_max,
        ]
    ).astype(np.float32, copy=False)


def apply_hybrid_feature_normalizer(
    features: np.ndarray,
    state: HybridFeatureNormalizerState,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32).copy()
    if x.shape[1] != len(state.feature_names):
        raise ValueError(
            f"Feature dimension mismatch: expected {len(state.feature_names)}, got {x.shape[1]}"
        )
    if state.raw_indices:
        raw_idx = np.asarray(state.raw_indices, dtype=np.int64)
        raw_values = x[:, raw_idx]
        medians = np.asarray(state.raw_medians, dtype=np.float32)
        means = np.asarray(state.raw_means, dtype=np.float32)
        stds = np.asarray(state.raw_stds, dtype=np.float32)
        missing = raw_values == -1.0
        raw_values = np.where(missing, medians.reshape(1, -1), raw_values)
        x[:, raw_idx] = (raw_values - means.reshape(1, -1)) / stds.reshape(1, -1)
    if state.log_indices:
        log_idx = np.asarray(state.log_indices, dtype=np.int64)
        log_values = np.log1p(np.clip(x[:, log_idx], 0.0, None))
        means = np.asarray(state.log_means, dtype=np.float32)
        stds = np.asarray(state.log_stds, dtype=np.float32)
        x[:, log_idx] = (log_values - means.reshape(1, -1)) / stds.reshape(1, -1)
    if state.z_indices:
        z_idx = np.asarray(state.z_indices, dtype=np.int64)
        z_values = x[:, z_idx]
        means = np.asarray(state.z_means, dtype=np.float32)
        stds = np.asarray(state.z_stds, dtype=np.float32)
        x[:, z_idx] = (z_values - means.reshape(1, -1)) / stds.reshape(1, -1)
    return x.astype(np.float32, copy=False)


def build_hybrid_feature_normalizer(
    phase: str,
    selected_groups: list[str],
    train_ids: np.ndarray,
    outdir: Path = FEATURE_OUTPUT_ROOT,
) -> HybridFeatureNormalizerState:
    train_store = FeatureStore(phase, selected_groups, outdir=outdir)
    x_train = train_store.take_rows(np.asarray(train_ids, dtype=np.int32))
    feature_names = list(train_store.feature_names)

    raw_indices: list[int] = []
    raw_medians: list[float] = []
    raw_means: list[float] = []
    raw_stds: list[float] = []
    log_indices: list[int] = []
    log_means: list[float] = []
    log_stds: list[float] = []
    z_indices: list[int] = []
    z_means: list[float] = []
    z_stds: list[float] = []

    with tqdm(
        total=len(feature_names),
        desc=f"feature_norm:{phase}",
        unit="feat",
        dynamic_ncols=True,
        leave=False,
    ) as normalizer_pbar:
        for feature_idx, feature_name in enumerate(feature_names):
            mode = _feature_normalization_type(feature_name)
            column = x_train[:, feature_idx].astype(np.float32, copy=False)
            if mode == "identity":
                normalizer_pbar.update(1)
                continue
            if mode == "raw":
                valid = column[column != -1.0]
                if valid.size == 0:
                    median = 0.0
                    mean = 0.0
                    std = 1.0
                else:
                    median = float(np.median(valid))
                    filled = np.where(column == -1.0, median, column)
                    mean = float(np.mean(filled))
                    std = _stable_std(float(np.std(filled)))
                raw_indices.append(feature_idx)
                raw_medians.append(median)
                raw_means.append(mean)
                raw_stds.append(std)
                normalizer_pbar.update(1)
                continue
            if mode == "log":
                transformed = np.log1p(np.clip(column, 0.0, None))
                log_indices.append(feature_idx)
                log_means.append(float(np.mean(transformed)))
                log_stds.append(_stable_std(float(np.std(transformed))))
                normalizer_pbar.update(1)
                continue
            z_indices.append(feature_idx)
            z_means.append(float(np.mean(column)))
            z_stds.append(_stable_std(float(np.std(column))))
            normalizer_pbar.update(1)

    return HybridFeatureNormalizerState(
        mode="hybrid",
        feature_names=feature_names,
        raw_indices=raw_indices,
        raw_medians=raw_medians,
        raw_means=raw_means,
        raw_stds=raw_stds,
        log_indices=log_indices,
        log_means=log_means,
        log_stds=log_stds,
        z_indices=z_indices,
        z_means=z_means,
        z_stds=z_stds,
    )


def _build_neighbor_features(
    data: PhaseData,
    phase_dir: Path,
    indegree: np.ndarray,
    outdegree: np.ndarray,
    x: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[str, dict[str, dict[str, Any]]]:
    _, neighbor_groups = _group_definition()
    neighbor_spans = _allocate_group_spans(neighbor_groups)
    shape = (data.num_nodes, neighbor_spans["neighbor"]["end"])
    neighbor_path = phase_dir / "neighbor_features.npy"
    matrix = np.lib.format.open_memmap(
        neighbor_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    src = data.edge_index[:, 0]
    dst = data.edge_index[:, 1]
    in_den = np.maximum(indegree.astype(np.float32, copy=False), 1.0)
    out_den = np.maximum(outdegree.astype(np.float32, copy=False), 1.0)

    col = 0
    with tqdm(
        total=3 * int(x.shape[1]),
        desc=f"neighbor_features:{phase_dir.name}",
        unit="feat",
        dynamic_ncols=True,
        leave=False,
    ) as neighbor_pbar:
        for reducer in ("mean", "max", "missing_ratio"):
            neighbor_pbar.set_postfix(reducer=reducer, refresh=False)
            raw_feature_count = int(x.shape[1])
            for feature_idx in range(raw_feature_count):
                values = x[:, feature_idx] if reducer != "missing_ratio" else missing_mask[:, feature_idx]
                in_center_values = values[src]
                out_center_values = values[dst]
                if reducer == "mean" or reducer == "missing_ratio":
                    in_result = _bincount_float(dst, in_center_values, data.num_nodes) / in_den
                    out_result = _bincount_float(src, out_center_values, data.num_nodes) / out_den
                else:
                    in_result = np.full(data.num_nodes, -np.inf, dtype=np.float32)
                    out_result = np.full(data.num_nodes, -np.inf, dtype=np.float32)
                    np.maximum.at(in_result, dst, in_center_values.astype(np.float32, copy=False))
                    np.maximum.at(out_result, src, out_center_values.astype(np.float32, copy=False))
                    in_result[indegree == 0] = 0.0
                    out_result[outdegree == 0] = 0.0
                matrix[:, col] = in_result
                matrix[:, col + raw_feature_count] = out_result
                col += 1
                neighbor_pbar.update(1)
            col += raw_feature_count
    del matrix
    return neighbor_path.name, neighbor_spans


def build_phase_feature_artifacts(
    phase: str,
    outdir: Path = FEATURE_OUTPUT_ROOT,
    build_neighbor: bool = True,
    ) -> dict[str, Any]:
    with tqdm(
        total=13,
        desc=f"build_features:{phase}",
        unit="step",
        dynamic_ncols=True,
    ) as phase_pbar:
        data = load_phase(phase, repo_root=REPO_ROOT)
        dataset_spec = get_active_dataset_spec()
        raw_feature_count = int(data.x.shape[1])
        num_edge_types = int(np.max(data.edge_type)) if data.edge_type.size else 1
        _set_feature_schema(
            raw_feature_count=raw_feature_count,
            num_edge_types=num_edge_types,
            strong_pairs=dataset_spec.strong_pairs,
            background_labels=dataset_spec.background_labels,
        )
        phase_dir = ensure_dir(outdir / phase)
        core_groups, _ = _group_definition()
        core_spans = _allocate_group_spans(core_groups)
        core_dim = max(spec["end"] for spec in core_spans.values())
        core_path = phase_dir / "core_features.npy"
        core = np.lib.format.open_memmap(
            core_path,
            mode="w+",
            dtype=np.float32,
            shape=(data.num_nodes, core_dim),
        )
        phase_pbar.set_postfix(nodes=data.num_nodes, edges=data.num_edges, refresh=False)
        phase_pbar.update(1)
        existing_manifest_path = phase_dir / "feature_manifest.json"
        preserved_neighbor_file = None
        preserved_neighbor_spans: dict[str, dict[str, Any]] = {}
        if not build_neighbor and existing_manifest_path.exists():
            existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            candidate_neighbor_file = existing_manifest.get("neighbor_file")
            candidate_neighbor_spans = existing_manifest.get("neighbor_groups", {})
            if candidate_neighbor_file and (phase_dir / candidate_neighbor_file).exists():
                # Allow fast core-only rebuilds without invalidating the existing neighbor cache.
                preserved_neighbor_file = str(candidate_neighbor_file)
                preserved_neighbor_spans = dict(candidate_neighbor_spans)

        x = np.asarray(data.x, dtype=np.float32).copy()
        nonfinite_mask = ~np.isfinite(x)
        if np.any(nonfinite_mask):
            x[nonfinite_mask] = -1.0
        missing_mask = (x == -1).astype(np.float32, copy=False)
        indegree, outdegree, total_degree = compute_degree_arrays(data)
        temporal = compute_temporal_core(data)
        first_active = temporal["first_active"].astype(np.int32, copy=False)
        last_active = temporal["last_active"].astype(np.int32, copy=False)
        active_span = temporal["active_span"].astype(np.int32, copy=False)
        time_windows = _build_edge_time_windows(data.edge_timestamp)
        node_time_bucket = _assign_node_time_bucket(first_active, time_windows)
        phase_pbar.update(1)

        src = data.edge_index[:, 0]
        dst = data.edge_index[:, 1]
        background_mask = np.isin(data.y, _active_background_labels())

        col = 0
        core[:, col : col + raw_feature_count] = x
        col += raw_feature_count

        core[:, col : col + raw_feature_count] = missing_mask
        col += raw_feature_count

        core[:, col] = np.sum(missing_mask, axis=1, dtype=np.float32)
        col += 1
        phase_pbar.update(1)

        for left, right in _active_strong_pairs(raw_feature_count):
            core[:, col] = (x[:, left] + x[:, right]) / 2.0
            core[:, col + 1] = np.abs(x[:, left] - x[:, right])
            col += 2

        indegree_f = indegree.astype(np.float32, copy=False)
        outdegree_f = outdegree.astype(np.float32, copy=False)
        total_degree_f = total_degree.astype(np.float32, copy=False)
        core[:, col] = indegree_f
        core[:, col + 1] = outdegree_f
        core[:, col + 2] = total_degree_f
        core[:, col + 3] = outdegree_f / (indegree_f + 1.0)
        core[:, col + 4] = indegree_f / (outdegree_f + 1.0)
        col += 5
        phase_pbar.update(1)

        for edge_type in range(1, num_edge_types + 1):
            mask_t = data.edge_type == edge_type
            core[:, col] = np.bincount(dst[mask_t], minlength=data.num_nodes).astype(np.float32, copy=False)
            core[:, col + 1] = np.bincount(src[mask_t], minlength=data.num_nodes).astype(np.float32, copy=False)
            col += 2
        phase_pbar.update(1)

        bg_out_mask = background_mask[dst]
        bg_in_mask = background_mask[src]
        bg_out_count = np.bincount(src[bg_out_mask], minlength=data.num_nodes).astype(np.float32, copy=False)
        bg_in_count = np.bincount(dst[bg_in_mask], minlength=data.num_nodes).astype(np.float32, copy=False)
        bg_total_count = bg_in_count + bg_out_count
        core[:, col] = bg_in_count
        core[:, col + 1] = bg_out_count
        core[:, col + 2] = bg_total_count
        core[:, col + 3] = bg_in_count / (indegree_f + 1.0)
        core[:, col + 4] = bg_out_count / (outdegree_f + 1.0)
        core[:, col + 5] = bg_total_count / (total_degree_f + 1.0)
        core[:, col + 6] = (bg_total_count > 0).astype(np.float32, copy=False)
        col += 7
        phase_pbar.update(1)

        for edge_type in range(1, num_edge_types + 1):
            mask_t = data.edge_type == edge_type
            mask_bg_in = mask_t & bg_in_mask
            mask_bg_out = mask_t & bg_out_mask
            core[:, col] = np.bincount(dst[mask_bg_in], minlength=data.num_nodes).astype(np.float32, copy=False)
            core[:, col + 1] = np.bincount(src[mask_bg_out], minlength=data.num_nodes).astype(np.float32, copy=False)
            col += 2
        phase_pbar.update(1)

        core[:, col] = first_active.astype(np.float32, copy=False)
        core[:, col + 1] = last_active.astype(np.float32, copy=False)
        core[:, col + 2] = active_span.astype(np.float32, copy=False)
        col += 3

        window_total_counts: list[np.ndarray] = []
        for window in time_windows:
            if window["window_idx"] == NUM_TIME_WINDOWS:
                mask_w = (data.edge_timestamp >= window["start_day"]) & (
                    data.edge_timestamp <= window["end_day"]
                )
            else:
                mask_w = (data.edge_timestamp >= window["start_day"]) & (
                    data.edge_timestamp < window["end_day"] + 1
                )
            in_count = np.bincount(dst[mask_w], minlength=data.num_nodes).astype(np.float32, copy=False)
            out_count = np.bincount(src[mask_w], minlength=data.num_nodes).astype(np.float32, copy=False)
            total_count = in_count + out_count
            window_total_counts.append(total_count)
            core[:, col] = in_count
            core[:, col + 1] = out_count
            core[:, col + 2] = total_count
            col += 3
        split_idx = max(len(window_total_counts) // 2, 1)
        early_total = np.sum(np.stack(window_total_counts[:split_idx], axis=0), axis=0)
        if split_idx < len(window_total_counts):
            late_total = np.sum(np.stack(window_total_counts[split_idx:], axis=0), axis=0)
        else:
            late_total = np.zeros_like(early_total)
        core[:, col] = (early_total + 1.0) / (late_total + 1.0)
        col += 1
        phase_pbar.update(1)

        graph_stats_block = _build_graph_stats_feature_block(
            data=data,
            indegree=indegree,
            outdegree=outdegree,
        )
        core[:, col : col + graph_stats_block.shape[1]] = graph_stats_block
        col += graph_stats_block.shape[1]

        graph_time_detrend_block = _build_graph_time_detrend_feature_block(
            data=data,
            indegree=indegree,
            outdegree=outdegree,
            first_active=first_active,
        )
        core[:, col : col + graph_time_detrend_block.shape[1]] = graph_time_detrend_block
        col += graph_time_detrend_block.shape[1]

        temporal_motif_lite_block = _build_temporal_motif_lite_feature_block(
            data=data,
            indegree=indegree,
            outdegree=outdegree,
        )
        core[:, col : col + temporal_motif_lite_block.shape[1]] = temporal_motif_lite_block
        col += temporal_motif_lite_block.shape[1]

        temporal_counterparty_concentration_block = _build_temporal_counterparty_concentration_feature_block(
            data=data,
        )
        core[:, col : col + temporal_counterparty_concentration_block.shape[1]] = (
            temporal_counterparty_concentration_block
        )
        col += temporal_counterparty_concentration_block.shape[1]

        temporal_counterparty_concentration_recent_block = (
            _build_temporal_counterparty_concentration_recent_feature_block(
                data=data,
            )
        )
        core[:, col : col + temporal_counterparty_concentration_recent_block.shape[1]] = (
            temporal_counterparty_concentration_recent_block
        )
        col += temporal_counterparty_concentration_recent_block.shape[1]

        temporal_counterparty_share_focus_block = _build_temporal_counterparty_share_focus_feature_block(
            data=data,
        )
        core[:, col : col + temporal_counterparty_share_focus_block.shape[1]] = (
            temporal_counterparty_share_focus_block
        )
        col += temporal_counterparty_share_focus_block.shape[1]

        temporal_dual_scale_delta_block = _build_temporal_dual_scale_delta_feature_block(
            data=data,
            indegree=indegree,
            outdegree=outdegree,
            first_active=first_active,
        )
        core[:, col : col + temporal_dual_scale_delta_block.shape[1]] = temporal_dual_scale_delta_block
        col += temporal_dual_scale_delta_block.shape[1]

        temporal_adaptive_window_block = _build_temporal_adaptive_window_feature_block(
            data=data,
            indegree=indegree,
            outdegree=outdegree,
            first_active=first_active,
        )
        core[:, col : col + temporal_adaptive_window_block.shape[1]] = temporal_adaptive_window_block
        col += temporal_adaptive_window_block.shape[1]

        neighbor_similarity_block = _build_neighbor_similarity_feature_block(
            x=x,
            missing_mask=missing_mask,
            src=src,
            dst=dst,
            indegree=indegree,
            outdegree=outdegree,
        )
        core[:, col : col + neighbor_similarity_block.shape[1]] = neighbor_similarity_block
        col += neighbor_similarity_block.shape[1]

        temporal_bucket_norm_block = _build_temporal_bucket_norm_feature_block(
            x=x,
            missing_mask=missing_mask,
            first_active=first_active,
            indegree=indegree,
            outdegree=outdegree,
            bg_in_count=bg_in_count,
            bg_out_count=bg_out_count,
        )
        core[:, col : col + temporal_bucket_norm_block.shape[1]] = temporal_bucket_norm_block
        col += temporal_bucket_norm_block.shape[1]

        temporal_safe_blocks = _build_temporal_safe_feature_blocks(
            data=data,
            time_windows=time_windows,
            node_time_bucket=node_time_bucket,
            first_active=first_active,
            background_mask=background_mask,
        )
        for group_name in (
            "temporal_snapshot",
            "temporal_recent",
            "temporal_relation_recent",
        ):
            block = temporal_safe_blocks[group_name]
            core[:, col : col + block.shape[1]] = block
            col += block.shape[1]
        temporal_neighbor_block = _build_temporal_neighbor_feature_block(
            data=data,
            time_windows=time_windows,
            node_time_bucket=node_time_bucket,
            x=x,
            missing_mask=missing_mask,
        )
        core[:, col : col + temporal_neighbor_block.shape[1]] = temporal_neighbor_block
        col += temporal_neighbor_block.shape[1]
        activation_early_block = _build_activation_early_feature_block(
            data=data,
            first_active=first_active,
            background_mask=background_mask,
        )
        core[:, col : col + activation_early_block.shape[1]] = activation_early_block
        col += activation_early_block.shape[1]

        projection_train_ids = _load_projection_train_ids(phase=phase, num_nodes=data.num_nodes)
        utpm_blocks = _build_utpm_blocks(
            core=core,
            core_spans=core_spans,
            x=x,
            missing_mask=missing_mask,
            train_ids=projection_train_ids,
        )
        for group_name in (*utpm_unified_feature_groups(), "utpm_temporal_shift"):
            group_spec = core_spans[group_name]
            block = utpm_blocks[group_name]
            width = int(group_spec["end"] - group_spec["start"])
            if int(block.shape[1]) != width:
                raise AssertionError(
                    f"{phase}: {group_name} width mismatch, got {block.shape[1]}, expected {width}"
                )
            core[:, col : col + width] = block
            col += width

        expected_col = core_dim
        if col != expected_col:
            raise AssertionError(f"{phase}: core feature width mismatch, got {col}, expected {expected_col}")
        del core
        phase_pbar.update(1)

        neighbor_file = preserved_neighbor_file
        neighbor_spans: dict[str, dict[str, Any]] = preserved_neighbor_spans
        if build_neighbor:
            neighbor_file, neighbor_spans = _build_neighbor_features(
                data=data,
                phase_dir=phase_dir,
                indegree=indegree,
                outdegree=outdegree,
                x=x,
                missing_mask=missing_mask,
            )
        phase_pbar.update(1)

        graph_meta = {}
        graph_meta.update(
            _write_graph_arrays(
                phase_dir=phase_dir,
                prefix="out",
                centers=src,
                neighbors=dst,
                edge_type=data.edge_type,
                edge_timestamp=data.edge_timestamp,
                num_nodes=data.num_nodes,
            )
        )
        phase_pbar.update(1)
        graph_meta.update(
            _write_graph_arrays(
                phase_dir=phase_dir,
                prefix="in",
                centers=dst,
                neighbors=src,
                edge_type=data.edge_type,
                edge_timestamp=data.edge_timestamp,
                num_nodes=data.num_nodes,
            )
        )
        phase_pbar.update(1)
        graph_dir = ensure_dir(phase_dir / "graph")
        first_active_file = graph_dir / "first_active.npy"
        node_time_bucket_file = graph_dir / "node_time_bucket.npy"
        np.save(first_active_file, first_active)
        np.save(node_time_bucket_file, node_time_bucket)
        phase_pbar.update(1)

        manifest = {
            "phase": phase,
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges,
            "core_file": core_path.name,
            "neighbor_file": neighbor_file,
            "core_groups": core_spans,
            "neighbor_groups": neighbor_spans,
            "graph_meta": {
                "dir": "graph",
                "files": graph_meta,
                "first_active_file": first_active_file.name,
                "node_time_bucket_file": node_time_bucket_file.name,
                "num_edge_types": num_edge_types,
                "num_relations": num_edge_types * 2,
                "max_day": int(data.edge_timestamp.max()),
                "time_windows": time_windows,
            },
            "feature_schema": _feature_schema_payload(),
        }
        write_json(phase_dir / "feature_manifest.json", manifest)
        phase_pbar.update(1)
    return manifest


def build_feature_artifacts(
    phases: list[str],
    outdir: Path = FEATURE_OUTPUT_ROOT,
    build_neighbor: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with tqdm(
        phases,
        desc="build_features:phases",
        unit="phase",
        dynamic_ncols=True,
    ) as phase_pbar:
        for phase in phase_pbar:
            phase_pbar.set_postfix(current=phase, refresh=False)
            summary[phase] = build_phase_feature_artifacts(
                phase=phase,
                outdir=outdir,
                build_neighbor=build_neighbor,
            )
    return summary


def load_feature_manifest(phase: str, outdir: Path = FEATURE_OUTPUT_ROOT) -> dict[str, Any]:
    return json.loads((outdir / phase / "feature_manifest.json").read_text(encoding="utf-8"))


def load_graph_cache(phase: str, outdir: Path = FEATURE_OUTPUT_ROOT) -> GraphCache:
    manifest = load_feature_manifest(phase, outdir=outdir)
    graph_dir = outdir / phase / manifest["graph_meta"]["dir"]
    files = manifest["graph_meta"]["files"]
    return GraphCache(
        phase=phase,
        num_nodes=int(manifest["num_nodes"]),
        max_day=int(manifest["graph_meta"]["max_day"]),
        num_edge_types=int(manifest["graph_meta"]["num_edge_types"]),
        num_relations=int(manifest["graph_meta"]["num_relations"]),
        time_windows=list(manifest["graph_meta"]["time_windows"]),
        out_ptr=np.load(graph_dir / files["out_ptr"], mmap_mode="r"),
        out_neighbors=np.load(graph_dir / files["out_neighbors"], mmap_mode="r"),
        out_edge_type=np.load(graph_dir / files["out_edge_type"], mmap_mode="r"),
        out_edge_timestamp=np.load(graph_dir / files["out_edge_timestamp"], mmap_mode="r"),
        in_ptr=np.load(graph_dir / files["in_ptr"], mmap_mode="r"),
        in_neighbors=np.load(graph_dir / files["in_neighbors"], mmap_mode="r"),
        in_edge_type=np.load(graph_dir / files["in_edge_type"], mmap_mode="r"),
        in_edge_timestamp=np.load(graph_dir / files["in_edge_timestamp"], mmap_mode="r"),
        first_active=np.load(graph_dir / manifest["graph_meta"]["first_active_file"], mmap_mode="r"),
        node_time_bucket=np.load(
            graph_dir / manifest["graph_meta"]["node_time_bucket_file"],
            mmap_mode="r",
        ),
    )


def default_feature_groups(model_name: str) -> list[str]:
    if model_name == "m1_tabular":
        return ["raw_x", "missing_mask", "missing_summary"]
    if model_name == "m2_hybrid":
        return [
            "raw_x",
            "missing_mask",
            "missing_summary",
            "strong_combo",
            "graph_direction",
            "edge_type",
            "background",
            "time",
        ]
    if model_name == "m3_neighbor":
        return default_feature_groups("m2_hybrid") + ["neighbor"]
    if model_name in {"m7_utpm", "m8_utgt"}:
        return utpm_unified_feature_groups()
    if model_name in {"m4_graphsage", "m5_temporal_graphsage", "m6_temporal_gat"}:
        return default_feature_groups("m2_hybrid")
    raise KeyError(f"Unsupported model name: {model_name}")


def resolve_feature_groups(
    model_name: str,
    extra_groups: list[str] | None = None,
    feature_profile: str = "legacy",
) -> list[str]:
    profile = str(feature_profile).strip().lower()
    if profile == "utpm_unified":
        groups = list(utpm_unified_feature_groups())
    elif profile == "utpm_shift_compact":
        groups = list(utpm_shift_compact_feature_groups())
    elif profile == "utpm_shift_enhanced":
        groups = list(utpm_shift_enhanced_feature_groups())
    else:
        groups = list(default_feature_groups(model_name))
    for group_name in extra_groups or []:
        if group_name not in groups:
            groups.append(group_name)
    return groups
