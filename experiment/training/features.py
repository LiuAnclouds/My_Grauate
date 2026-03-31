from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiment.eda.analysis import compute_degree_arrays, compute_temporal_core
from experiment.eda.data_loader import PhaseData, load_phase
from experiment.training.common import FEATURE_OUTPUT_ROOT, REPO_ROOT, ensure_dir, write_json

# 特征数
RAW_FEATURE_COUNT = 17
# 边类型数
NUM_EDGE_TYPES = 11
# 时间窗口
NUM_TIME_WINDOWS = 4
STRONG_PAIRS = ((2, 3), (6, 8), (15, 16))


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
                specs.append(spec)
                continue
            if group_name in self.manifest["neighbor_groups"]:
                if self.neighbor is None:
                    raise FileNotFoundError(
                        f"{self.phase}: neighbor feature file is missing, run build_features first."
                    )
                spec = dict(self.manifest["neighbor_groups"][group_name])
                spec["source"] = "neighbor"
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
    }
    for left, right in STRONG_PAIRS:
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
    }
    if feature_name in log_exact or feature_name.startswith(log_prefixes):
        return "log"
    return "zscore"


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

    for feature_idx, feature_name in enumerate(feature_names):
        mode = _feature_normalization_type(feature_name)
        column = x_train[:, feature_idx].astype(np.float32, copy=False)
        if mode == "identity":
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
            continue
        if mode == "log":
            transformed = np.log1p(np.clip(column, 0.0, None))
            log_indices.append(feature_idx)
            log_means.append(float(np.mean(transformed)))
            log_stds.append(_stable_std(float(np.std(transformed))))
            continue
        z_indices.append(feature_idx)
        z_means.append(float(np.mean(column)))
        z_stds.append(_stable_std(float(np.std(column))))

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
    for reducer in ("mean", "max", "missing_ratio"):
        for feature_idx in range(RAW_FEATURE_COUNT):
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
            matrix[:, col + RAW_FEATURE_COUNT] = out_result
            col += 1
        col += RAW_FEATURE_COUNT
    del matrix
    return neighbor_path.name, neighbor_spans


def build_phase_feature_artifacts(
    phase: str,
    outdir: Path = FEATURE_OUTPUT_ROOT,
    build_neighbor: bool = True,
) -> dict[str, Any]:
    data = load_phase(phase, repo_root=REPO_ROOT)
    phase_dir = ensure_dir(outdir / phase)
    core_groups, _ = _group_definition()
    core_spans = _allocate_group_spans(core_groups)
    core_path = phase_dir / "core_features.npy"
    core = np.lib.format.open_memmap(
        core_path,
        mode="w+",
        dtype=np.float32,
        shape=(data.num_nodes, core_spans["time"]["end"]),
    )

    x = np.asarray(data.x, dtype=np.float32)
    missing_mask = (x == -1).astype(np.float32, copy=False)
    indegree, outdegree, total_degree = compute_degree_arrays(data)
    temporal = compute_temporal_core(data)
    first_active = temporal["first_active"].astype(np.int32, copy=False)
    last_active = temporal["last_active"].astype(np.int32, copy=False)
    active_span = temporal["active_span"].astype(np.int32, copy=False)
    time_windows = _build_edge_time_windows(data.edge_timestamp)
    node_time_bucket = _assign_node_time_bucket(first_active, time_windows)

    src = data.edge_index[:, 0]
    dst = data.edge_index[:, 1]
    background_mask = np.isin(data.y, (2, 3))

    col = 0
    core[:, col : col + RAW_FEATURE_COUNT] = x
    col += RAW_FEATURE_COUNT

    core[:, col : col + RAW_FEATURE_COUNT] = missing_mask
    col += RAW_FEATURE_COUNT

    core[:, col] = np.sum(missing_mask, axis=1, dtype=np.float32)
    col += 1

    for left, right in STRONG_PAIRS:
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

    for edge_type in range(1, NUM_EDGE_TYPES + 1):
        mask_t = data.edge_type == edge_type
        core[:, col] = np.bincount(dst[mask_t], minlength=data.num_nodes).astype(np.float32, copy=False)
        core[:, col + 1] = np.bincount(src[mask_t], minlength=data.num_nodes).astype(np.float32, copy=False)
        col += 2

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

    for edge_type in range(1, NUM_EDGE_TYPES + 1):
        mask_t = data.edge_type == edge_type
        mask_bg_in = mask_t & bg_in_mask
        mask_bg_out = mask_t & bg_out_mask
        core[:, col] = np.bincount(dst[mask_bg_in], minlength=data.num_nodes).astype(np.float32, copy=False)
        core[:, col + 1] = np.bincount(src[mask_bg_out], minlength=data.num_nodes).astype(np.float32, copy=False)
        col += 2

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
    early_total = window_total_counts[0] + window_total_counts[1]
    late_total = window_total_counts[2] + window_total_counts[3]
    core[:, col] = (early_total + 1.0) / (late_total + 1.0)
    col += 1
    del core

    neighbor_file = None
    neighbor_spans: dict[str, dict[str, Any]] = {}
    if build_neighbor:
        neighbor_file, neighbor_spans = _build_neighbor_features(
            data=data,
            phase_dir=phase_dir,
            indegree=indegree,
            outdegree=outdegree,
            x=x,
            missing_mask=missing_mask,
        )

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
    graph_dir = ensure_dir(phase_dir / "graph")
    first_active_file = graph_dir / "first_active.npy"
    node_time_bucket_file = graph_dir / "node_time_bucket.npy"
    np.save(first_active_file, first_active)
    np.save(node_time_bucket_file, node_time_bucket)

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
            "num_edge_types": NUM_EDGE_TYPES,
            "num_relations": NUM_EDGE_TYPES * 2,
            "max_day": int(data.edge_timestamp.max()),
            "time_windows": time_windows,
        },
    }
    write_json(phase_dir / "feature_manifest.json", manifest)
    return manifest


def build_feature_artifacts(
    phases: list[str],
    outdir: Path = FEATURE_OUTPUT_ROOT,
    build_neighbor: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in phases:
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
    if model_name in {"m4_graphsage", "m5_temporal_graphsage"}:
        return default_feature_groups("m2_hybrid")
    raise KeyError(f"Unsupported model name: {model_name}")
