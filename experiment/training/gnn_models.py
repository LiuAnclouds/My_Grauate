from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from experiment.training.common import (
    compute_binary_classification_metrics,
    ensure_dir,
    resolve_device,
    set_global_seed,
    write_json,
)
from experiment.training.features import (
    FeatureStore,
    GraphCache,
    HybridFeatureNormalizerState,
    default_feature_groups,
)


@dataclass(frozen=True)
class GraphPhaseContext:
    phase: str
    feature_store: FeatureStore
    graph_cache: GraphCache
    labels: np.ndarray


@dataclass(frozen=True)
class GraphModelConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.2
    feature_norm: str = "none"
    norm: str = "none"
    residual: bool = False
    ffn: bool = False
    jk: str = "last"
    edge_encoder: str = "basic"
    subgraph_head: str = "none"
    grad_clip: float = 0.0
    scheduler: str = "none"
    early_stop_patience: int = 0
    train_negative_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "dropout": float(self.dropout),
            "feature_norm": self.feature_norm,
            "norm": self.norm,
            "residual": bool(self.residual),
            "ffn": bool(self.ffn),
            "jk": self.jk,
            "edge_encoder": self.edge_encoder,
            "subgraph_head": self.subgraph_head,
            "grad_clip": float(self.grad_clip),
            "scheduler": self.scheduler,
            "early_stop_patience": int(self.early_stop_patience),
            "train_negative_ratio": float(self.train_negative_ratio),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GraphModelConfig":
        if not payload:
            return cls()
        return cls(
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-5)),
            dropout=float(payload.get("dropout", 0.2)),
            feature_norm=str(payload.get("feature_norm", "none")),
            norm=str(payload.get("norm", "none")),
            residual=bool(payload.get("residual", False)),
            ffn=bool(payload.get("ffn", False)),
            jk=str(payload.get("jk", "last")),
            edge_encoder=str(payload.get("edge_encoder", "basic")),
            subgraph_head=str(payload.get("subgraph_head", "none")),
            grad_clip=float(payload.get("grad_clip", 0.0)),
            scheduler=str(payload.get("scheduler", "none")),
            early_stop_patience=int(payload.get("early_stop_patience", 0)),
            train_negative_ratio=float(payload.get("train_negative_ratio", 0.0)),
        )

    def use_legacy_path(self) -> bool:
        return (
            self.norm == "none"
            and not self.residual
            and not self.ffn
            and self.jk == "last"
            and self.edge_encoder == "basic"
            and self.subgraph_head == "none"
        )


@dataclass(frozen=True)
class SampledSubgraph:
    node_ids: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    rel_ids: np.ndarray
    edge_timestamp: np.ndarray
    target_local_idx: np.ndarray
    node_subgraph_id: np.ndarray | None = None
    edge_subgraph_id: np.ndarray | None = None


@dataclass(frozen=True)
class TrainBatchStats:
    target_count: int
    positive_count: int
    negative_count: int

    @property
    def positive_rate(self) -> float:
        return 0.0 if self.target_count == 0 else float(self.positive_count / self.target_count)


def _sample_edge_indices(
    edge_timestamp: np.ndarray,
    fanout: int,
    rng: np.random.Generator,
    snapshot_end: int | None,
) -> np.ndarray:
    if edge_timestamp.size == 0:
        return np.empty(0, dtype=np.int32)
    if snapshot_end is not None:
        valid = np.flatnonzero(edge_timestamp <= snapshot_end)
    else:
        valid = np.arange(edge_timestamp.size, dtype=np.int32)
    if valid.size <= fanout:
        return valid.astype(np.int32, copy=False)
    choice = rng.choice(valid, size=fanout, replace=False)
    return np.sort(choice.astype(np.int32, copy=False))


def sample_relation_subgraph(
    graph: GraphCache,
    seed_nodes: np.ndarray,
    fanouts: list[int],
    rng: np.random.Generator,
    snapshot_end: int | None = None,
) -> SampledSubgraph:
    seeds = np.asarray(seed_nodes, dtype=np.int32)
    ordered_nodes: list[int] = []
    seen_nodes: set[int] = set()

    def add_nodes(nodes: np.ndarray) -> None:
        for node in nodes.tolist():
            if node not in seen_nodes:
                seen_nodes.add(node)
                ordered_nodes.append(int(node))

    add_nodes(seeds)
    frontier = seeds
    edge_records: list[tuple[int, int, int, int]] = []

    for fanout in fanouts:
        if frontier.size == 0:
            break
        next_frontier: list[np.ndarray] = []
        for center in frontier.tolist():
            in_start = int(graph.in_ptr[center])
            in_end = int(graph.in_ptr[center + 1])
            in_choice = _sample_edge_indices(
                edge_timestamp=np.asarray(graph.in_edge_timestamp[in_start:in_end]),
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
            )
            if in_choice.size:
                in_neighbors = np.asarray(graph.in_neighbors[in_start:in_end])[in_choice]
                in_type = np.asarray(graph.in_edge_type[in_start:in_end])[in_choice]
                in_time = np.asarray(graph.in_edge_timestamp[in_start:in_end])[in_choice]
                next_frontier.append(in_neighbors.astype(np.int32, copy=False))
                add_nodes(in_neighbors.astype(np.int32, copy=False))
                edge_records.extend(
                    (
                        int(src),
                        int(center),
                        int(edge_type - 1),
                        int(edge_time),
                    )
                    for src, edge_type, edge_time in zip(
                        in_neighbors.tolist(),
                        in_type.tolist(),
                        in_time.tolist(),
                        strict=True,
                    )
                )

            out_start = int(graph.out_ptr[center])
            out_end = int(graph.out_ptr[center + 1])
            out_choice = _sample_edge_indices(
                edge_timestamp=np.asarray(graph.out_edge_timestamp[out_start:out_end]),
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
            )
            if out_choice.size:
                out_neighbors = np.asarray(graph.out_neighbors[out_start:out_end])[out_choice]
                out_type = np.asarray(graph.out_edge_type[out_start:out_end])[out_choice]
                out_time = np.asarray(graph.out_edge_timestamp[out_start:out_end])[out_choice]
                next_frontier.append(out_neighbors.astype(np.int32, copy=False))
                add_nodes(out_neighbors.astype(np.int32, copy=False))
                edge_records.extend(
                    (
                        int(src),
                        int(center),
                        int(edge_type - 1 + graph.num_edge_types),
                        int(edge_time),
                    )
                    for src, edge_type, edge_time in zip(
                        out_neighbors.tolist(),
                        out_type.tolist(),
                        out_time.tolist(),
                        strict=True,
                    )
                )

        if next_frontier:
            frontier = np.unique(np.concatenate(next_frontier)).astype(np.int32, copy=False)
        else:
            frontier = np.empty(0, dtype=np.int32)

    node_ids = np.asarray(ordered_nodes, dtype=np.int32)
    global_to_local = {int(node): idx for idx, node in enumerate(node_ids.tolist())}
    target_local_idx = np.asarray(
        [global_to_local[int(node)] for node in seeds.tolist()],
        dtype=np.int64,
    )

    if edge_records:
        edge_src = np.asarray(
            [global_to_local[src] for src, _, _, _ in edge_records],
            dtype=np.int64,
        )
        edge_dst = np.asarray(
            [global_to_local[dst] for _, dst, _, _ in edge_records],
            dtype=np.int64,
        )
        rel_ids = np.asarray([rel for _, _, rel, _ in edge_records], dtype=np.int64)
        edge_timestamp = np.asarray([ts for _, _, _, ts in edge_records], dtype=np.int64)
    else:
        edge_src = np.empty(0, dtype=np.int64)
        edge_dst = np.empty(0, dtype=np.int64)
        rel_ids = np.empty(0, dtype=np.int64)
        edge_timestamp = np.empty(0, dtype=np.int64)

    return SampledSubgraph(
        node_ids=node_ids,
        edge_src=edge_src,
        edge_dst=edge_dst,
        rel_ids=rel_ids,
        edge_timestamp=edge_timestamp,
        target_local_idx=target_local_idx,
    )


def _sample_single_seed_subgraph(
    graph: GraphCache,
    seed: int,
    fanouts: list[int],
    rng: np.random.Generator,
    snapshot_end: int | None = None,
) -> SampledSubgraph:
    seed = int(seed)
    ordered_nodes = [seed]
    global_to_local = {seed: 0}
    frontier = np.asarray([seed], dtype=np.int32)
    edge_src: list[int] = []
    edge_dst: list[int] = []
    rel_ids: list[int] = []
    edge_timestamp: list[int] = []

    in_ptr = graph.in_ptr
    in_neighbors = graph.in_neighbors
    in_edge_type = graph.in_edge_type
    in_edge_timestamp = graph.in_edge_timestamp
    out_ptr = graph.out_ptr
    out_neighbors = graph.out_neighbors
    out_edge_type = graph.out_edge_type
    out_edge_timestamp = graph.out_edge_timestamp
    num_edge_types = graph.num_edge_types

    for fanout in fanouts:
        if frontier.size == 0:
            break
        next_nodes: list[int] = []
        for center in frontier.tolist():
            center = int(center)
            center_local = global_to_local[center]

            in_start = int(in_ptr[center])
            in_end = int(in_ptr[center + 1])
            in_time_slice = np.asarray(in_edge_timestamp[in_start:in_end])
            in_choice = _sample_edge_indices(
                edge_timestamp=in_time_slice,
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
            )
            if in_choice.size:
                selected_neighbors = np.asarray(in_neighbors[in_start:in_end])[in_choice]
                selected_types = np.asarray(in_edge_type[in_start:in_end])[in_choice]
                selected_times = in_time_slice[in_choice]
                for src, edge_type, edge_time in zip(
                    selected_neighbors.tolist(),
                    selected_types.tolist(),
                    selected_times.tolist(),
                    strict=True,
                ):
                    src_local = global_to_local.get(src)
                    if src_local is None:
                        src_local = len(ordered_nodes)
                        global_to_local[src] = src_local
                        ordered_nodes.append(src)
                    next_nodes.append(src)
                    edge_src.append(src_local)
                    edge_dst.append(center_local)
                    rel_ids.append(int(edge_type - 1))
                    edge_timestamp.append(int(edge_time))

            out_start = int(out_ptr[center])
            out_end = int(out_ptr[center + 1])
            out_time_slice = np.asarray(out_edge_timestamp[out_start:out_end])
            out_choice = _sample_edge_indices(
                edge_timestamp=out_time_slice,
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
            )
            if out_choice.size:
                selected_neighbors = np.asarray(out_neighbors[out_start:out_end])[out_choice]
                selected_types = np.asarray(out_edge_type[out_start:out_end])[out_choice]
                selected_times = out_time_slice[out_choice]
                for src, edge_type, edge_time in zip(
                    selected_neighbors.tolist(),
                    selected_types.tolist(),
                    selected_times.tolist(),
                    strict=True,
                ):
                    src_local = global_to_local.get(src)
                    if src_local is None:
                        src_local = len(ordered_nodes)
                        global_to_local[src] = src_local
                        ordered_nodes.append(src)
                    next_nodes.append(src)
                    edge_src.append(src_local)
                    edge_dst.append(center_local)
                    rel_ids.append(int(edge_type - 1 + num_edge_types))
                    edge_timestamp.append(int(edge_time))

        if next_nodes:
            frontier = np.unique(np.asarray(next_nodes, dtype=np.int32))
        else:
            frontier = np.empty(0, dtype=np.int32)

    return SampledSubgraph(
        node_ids=np.asarray(ordered_nodes, dtype=np.int32),
        edge_src=np.asarray(edge_src, dtype=np.int64),
        edge_dst=np.asarray(edge_dst, dtype=np.int64),
        rel_ids=np.asarray(rel_ids, dtype=np.int64),
        edge_timestamp=np.asarray(edge_timestamp, dtype=np.int64),
        target_local_idx=np.asarray([0], dtype=np.int64),
    )


def sample_batched_relation_subgraphs(
    graph: GraphCache,
    seed_nodes: np.ndarray,
    fanouts: list[int],
    rng: np.random.Generator,
    snapshot_end: int | None = None,
) -> SampledSubgraph:
    seeds = np.asarray(seed_nodes, dtype=np.int32)
    if seeds.size == 0:
        return SampledSubgraph(
            node_ids=np.empty(0, dtype=np.int32),
            edge_src=np.empty(0, dtype=np.int64),
            edge_dst=np.empty(0, dtype=np.int64),
            rel_ids=np.empty(0, dtype=np.int64),
            edge_timestamp=np.empty(0, dtype=np.int64),
            target_local_idx=np.empty(0, dtype=np.int64),
            node_subgraph_id=np.empty(0, dtype=np.int64),
            edge_subgraph_id=np.empty(0, dtype=np.int64),
        )

    node_parts: list[np.ndarray] = []
    edge_src_parts: list[np.ndarray] = []
    edge_dst_parts: list[np.ndarray] = []
    rel_parts: list[np.ndarray] = []
    edge_time_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    node_group_parts: list[np.ndarray] = []
    edge_group_parts: list[np.ndarray] = []
    node_offset = 0

    for subgraph_id, seed in enumerate(seeds.tolist()):
        subgraph = _sample_single_seed_subgraph(
            graph=graph,
            seed=seed,
            fanouts=fanouts,
            rng=rng,
            snapshot_end=snapshot_end,
        )
        node_parts.append(subgraph.node_ids)
        edge_src_parts.append(subgraph.edge_src + node_offset)
        edge_dst_parts.append(subgraph.edge_dst + node_offset)
        rel_parts.append(subgraph.rel_ids)
        edge_time_parts.append(subgraph.edge_timestamp)
        target_parts.append(subgraph.target_local_idx + node_offset)
        node_group_parts.append(
            np.full(subgraph.node_ids.shape[0], subgraph_id, dtype=np.int64)
        )
        edge_group_parts.append(
            np.full(subgraph.edge_src.shape[0], subgraph_id, dtype=np.int64)
        )
        node_offset += int(subgraph.node_ids.shape[0])

    return SampledSubgraph(
        node_ids=np.concatenate(node_parts).astype(np.int32, copy=False),
        edge_src=np.concatenate(edge_src_parts).astype(np.int64, copy=False),
        edge_dst=np.concatenate(edge_dst_parts).astype(np.int64, copy=False),
        rel_ids=np.concatenate(rel_parts).astype(np.int64, copy=False),
        edge_timestamp=np.concatenate(edge_time_parts).astype(np.int64, copy=False),
        target_local_idx=np.concatenate(target_parts).astype(np.int64, copy=False),
        node_subgraph_id=np.concatenate(node_group_parts).astype(np.int64, copy=False),
        edge_subgraph_id=np.concatenate(edge_group_parts).astype(np.int64, copy=False),
    )


class TimeEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, relative_time: torch.Tensor) -> torch.Tensor:
        return self.net(relative_time)


class SafeBatchNorm1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[0] <= 1:
            return x
        return self.norm(x)


def _make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return SafeBatchNorm1d(dim)
    return nn.Identity()


def _compute_grad_norm(parameters: Any) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().norm(2).item())
        total_sq += grad_norm * grad_norm
    return math.sqrt(total_sq)


def _dirichlet_energy(
    x: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
) -> torch.Tensor:
    if edge_src.numel() == 0:
        return x.new_tensor(0.0)
    delta = x[edge_src] - x[edge_dst]
    return (delta.pow(2).sum(dim=-1)).mean()


def _pool_mean_max(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.dim() != 2:
        raise ValueError("Expected a 2D tensor for pooling.")
    pooled_mean = values.new_zeros((num_groups, values.shape[1]))
    pooled_max = values.new_zeros((num_groups, values.shape[1]))
    if num_groups == 0 or values.shape[0] == 0:
        return pooled_mean, pooled_max

    counts = values.new_zeros((num_groups, 1))
    pooled_mean.index_add_(0, group_ids, values)
    counts.index_add_(
        0,
        group_ids,
        torch.ones((group_ids.shape[0], 1), device=values.device, dtype=values.dtype),
    )
    pooled_mean = pooled_mean / counts.clamp_min(1.0)

    if hasattr(pooled_max, "scatter_reduce_"):
        pooled_max.fill_(float("-inf"))
        pooled_max.scatter_reduce_(
            0,
            group_ids.view(-1, 1).expand(-1, values.shape[1]),
            values,
            reduce="amax",
            include_self=True,
        )
        pooled_max = torch.where(
            torch.isfinite(pooled_max),
            pooled_max,
            torch.zeros_like(pooled_max),
        )
    else:
        for group_idx in range(num_groups):
            mask = group_ids == group_idx
            if torch.any(mask):
                pooled_max[group_idx] = values[mask].max(dim=0).values
    return pooled_mean, pooled_max


class RelationSAGELayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        rel_dim: int,
        time_dim: int = 0,
    ) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, rel_dim)
        msg_in_dim = in_dim + rel_dim + time_dim
        self.msg_linear = nn.Linear(msg_in_dim, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(out_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        time_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_src.numel() == 0:
            return F.relu(self.self_linear(x))
        relation = self.relation_embedding(rel_ids)
        msg_parts = [x[edge_src], relation]
        if time_feature is not None:
            msg_parts.append(time_feature)
        msg = self.msg_linear(torch.cat(msg_parts, dim=-1))
        agg = x.new_zeros((x.shape[0], msg.shape[1]))
        agg.index_add_(0, edge_dst, msg)
        deg = x.new_zeros((x.shape[0], 1))
        deg.index_add_(
            0,
            edge_dst,
            torch.ones((edge_dst.shape[0], 1), device=x.device, dtype=x.dtype),
        )
        agg = agg / deg.clamp_min(1.0)
        out = self.self_linear(x) + self.neigh_linear(agg)
        return F.relu(out)


class ModernRelationBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float,
        norm: str,
        residual: bool,
        ffn: bool,
        edge_encoder: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.ffn_enabled = ffn
        self.gated = edge_encoder == "gated"
        self.norm1 = _make_norm(norm, hidden_dim)
        self.norm2 = _make_norm(norm, hidden_dim) if ffn else nn.Identity()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.gated
            else None
        )
        self.agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if ffn:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        else:
            self.ffn = None

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_in = x
        h = self.norm1(x)
        if edge_src.numel() == 0:
            edge_repr = h.new_zeros((0, self.hidden_dim))
            agg = h.new_zeros((h.shape[0], self.hidden_dim))
        else:
            msg = self.msg_mlp(torch.cat([h[edge_src], edge_emb], dim=-1))
            if self.gate_mlp is not None:
                gate = torch.sigmoid(
                    self.gate_mlp(torch.cat([h[edge_dst], h[edge_src], edge_emb], dim=-1))
                )
                edge_repr = gate * msg
            else:
                edge_repr = msg
            agg = h.new_zeros((h.shape[0], self.hidden_dim))
            agg.index_add_(0, edge_dst, edge_repr)
            deg = h.new_zeros((h.shape[0], 1))
            deg.index_add_(
                0,
                edge_dst,
                torch.ones((edge_dst.shape[0], 1), device=h.device, dtype=h.dtype),
            )
            agg = agg / deg.clamp_min(1.0)

        update = self.self_linear(h) + self.agg_proj(agg)
        update = self.dropout(update)
        out = h_in + update if self.residual else update

        if self.ffn is not None:
            ffn_update = self.ffn(self.norm2(out))
            ffn_update = self.dropout(ffn_update)
            out = out + ffn_update if self.residual else ffn_update

        return out, edge_repr


class RelationGraphSAGENetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_relations: int,
        rel_dim: int,
        dropout: float,
        temporal: bool,
        model_config: GraphModelConfig,
    ) -> None:
        super().__init__()
        self.temporal = temporal
        self.model_config = model_config
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_edge_types = max(num_relations // 2, 1)
        self.use_legacy = model_config.use_legacy_path()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.time_encoder = TimeEncoder(rel_dim) if temporal else None

        if self.use_legacy:
            time_dim = rel_dim if temporal else 0
            self.layers = nn.ModuleList(
                [
                    RelationSAGELayer(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        num_relations=num_relations,
                        rel_dim=rel_dim,
                        time_dim=time_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.rel_embedding = None
            self.stats_proj = None
            self.subgraph_classifier = None
            return

        self.rel_embedding = nn.Embedding(num_relations, rel_dim)
        edge_dim = rel_dim + (rel_dim if temporal else 0)
        self.layers = nn.ModuleList(
            [
                ModernRelationBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    norm=model_config.norm,
                    residual=model_config.residual,
                    ffn=model_config.ffn,
                    edge_encoder=model_config.edge_encoder,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if model_config.subgraph_head == "meanmax":
            self.stats_proj = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.subgraph_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 6, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.stats_proj = None
            self.subgraph_classifier = None

    def _build_edge_embedding(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        time_feature = None
        if self.temporal and edge_relative_time is not None and edge_relative_time.numel():
            time_feature = self.time_encoder(edge_relative_time)

        if self.use_legacy:
            return None, time_feature

        if rel_ids.numel() == 0:
            relation = x.new_zeros((0, self.rel_embedding.embedding_dim))
        else:
            relation = self.rel_embedding(rel_ids)

        if self.temporal:
            if time_feature is None:
                time_feature = relation.new_zeros((relation.shape[0], relation.shape[1]))
            edge_emb = torch.cat([relation, time_feature], dim=-1)
        else:
            edge_emb = relation
        return edge_emb, time_feature

    def _subgraph_stats(
        self,
        node_repr: torch.Tensor,
        target_local_idx: torch.Tensor,
        node_subgraph_id: torch.Tensor,
        edge_subgraph_id: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
    ) -> torch.Tensor:
        num_subgraphs = int(target_local_idx.shape[0])
        stats = node_repr.new_zeros((num_subgraphs, 6))
        stats[:, 0].index_add_(
            0,
            node_subgraph_id,
            torch.ones_like(node_subgraph_id, dtype=node_repr.dtype),
        )
        stats[:, 1].index_add_(
            0,
            edge_subgraph_id,
            torch.ones_like(edge_subgraph_id, dtype=node_repr.dtype),
        )
        if edge_relative_time is not None and edge_relative_time.numel():
            stats[:, 3].index_add_(0, edge_subgraph_id, edge_relative_time.view(-1))
            stats[:, 3] = stats[:, 3] / stats[:, 1].clamp_min(1.0)

        if rel_ids.numel():
            unique_rel_pairs = torch.unique(edge_subgraph_id * self.num_relations + rel_ids)
            unique_rel_subgraph_ids = torch.floor_divide(unique_rel_pairs, self.num_relations)
            stats[:, 2].index_add_(
                0,
                unique_rel_subgraph_ids,
                torch.ones_like(unique_rel_subgraph_ids, dtype=node_repr.dtype),
            )

            target_edge_mask = edge_dst == target_local_idx[edge_subgraph_id]
            if torch.any(target_edge_mask):
                target_subgraph_ids = edge_subgraph_id[target_edge_mask]
                target_rel_ids = rel_ids[target_edge_mask]
                inbound_mask = target_rel_ids < self.num_edge_types
                outbound_mask = ~inbound_mask
                if torch.any(inbound_mask):
                    stats[:, 4].index_add_(
                        0,
                        target_subgraph_ids[inbound_mask],
                        torch.ones_like(target_subgraph_ids[inbound_mask], dtype=node_repr.dtype),
                    )
                if torch.any(outbound_mask):
                    stats[:, 5].index_add_(
                        0,
                        target_subgraph_ids[outbound_mask],
                        torch.ones_like(target_subgraph_ids[outbound_mask], dtype=node_repr.dtype),
                    )

        stats[:, 0] = torch.log1p(stats[:, 0])
        stats[:, 1] = torch.log1p(stats[:, 1])
        stats[:, 2] = torch.log1p(stats[:, 2])
        stats[:, 4] = torch.log1p(stats[:, 4])
        stats[:, 5] = torch.log1p(stats[:, 5])
        return stats

    def _forward_subgraph_head(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        target_local_idx: torch.Tensor,
        node_subgraph_id: torch.Tensor | None,
        edge_subgraph_id: torch.Tensor | None,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
    ) -> torch.Tensor:
        if node_subgraph_id is None or edge_subgraph_id is None:
            raise ValueError("subgraph_head=meanmax requires subgraph ids.")

        num_subgraphs = int(target_local_idx.shape[0])
        target_repr = node_repr[target_local_idx]
        target_mask = torch.zeros(
            (node_repr.shape[0],),
            device=node_repr.device,
            dtype=torch.bool,
        )
        target_mask[target_local_idx] = True
        context_mask = ~target_mask
        ctx_mean, ctx_max = _pool_mean_max(
            node_repr[context_mask],
            node_subgraph_id[context_mask],
            num_subgraphs,
        )
        edge_mean, edge_max = _pool_mean_max(
            edge_repr,
            edge_subgraph_id,
            num_subgraphs,
        )
        stats = self._subgraph_stats(
            node_repr=node_repr,
            target_local_idx=target_local_idx,
            node_subgraph_id=node_subgraph_id,
            edge_subgraph_id=edge_subgraph_id,
            edge_dst=edge_dst,
            rel_ids=rel_ids,
            edge_relative_time=edge_relative_time,
        )
        fused = torch.cat(
            [
                target_repr,
                ctx_mean,
                ctx_max,
                edge_mean,
                edge_max,
                self.stats_proj(stats),
            ],
            dim=-1,
        )
        return self.subgraph_classifier(fused).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
        target_local_idx: torch.Tensor,
        node_subgraph_id: torch.Tensor | None = None,
        edge_subgraph_id: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        h = self.input_proj(x)
        h = F.relu(h) if self.use_legacy else F.gelu(h)
        h = self.input_dropout(h)

        edge_emb, time_feature = self._build_edge_embedding(x, rel_ids, edge_relative_time)
        last_edge_repr = h.new_zeros((0, self.hidden_dim))
        layer_outputs: list[torch.Tensor] = []

        if self.use_legacy:
            for layer in self.layers:
                h = layer(
                    h,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    rel_ids=rel_ids,
                    time_feature=time_feature,
                )
                h = self.input_dropout(h)
                layer_outputs.append(h)
            node_repr = layer_outputs[-1] if layer_outputs else h
            logits = self.classifier(node_repr[target_local_idx]).squeeze(-1)
        else:
            for layer in self.layers:
                h, last_edge_repr = layer(
                    h,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_emb=edge_emb,
                )
                layer_outputs.append(h)
            node_repr = (
                torch.stack(layer_outputs, dim=0).sum(dim=0)
                if self.model_config.jk == "sum" and layer_outputs
                else (layer_outputs[-1] if layer_outputs else h)
            )
            if self.model_config.subgraph_head == "meanmax":
                logits = self._forward_subgraph_head(
                    node_repr=node_repr,
                    edge_repr=last_edge_repr,
                    target_local_idx=target_local_idx,
                    node_subgraph_id=node_subgraph_id,
                    edge_subgraph_id=edge_subgraph_id,
                    edge_dst=edge_dst,
                    rel_ids=rel_ids,
                    edge_relative_time=edge_relative_time,
                )
            else:
                logits = self.classifier(node_repr[target_local_idx]).squeeze(-1)

        if not return_details:
            return logits
        diagnostics = {
            "emb_norm": float(node_repr.norm(dim=-1).mean().detach().item()),
            "dirichlet_energy": float(
                _dirichlet_energy(node_repr, edge_src=edge_src, edge_dst=edge_dst)
                .detach()
                .item()
            ),
        }
        return logits, diagnostics


class BaseGraphSAGEExperiment:
    def __init__(
        self,
        model_name: str,
        seed: int,
        input_dim: int,
        num_relations: int,
        max_day: int,
        feature_groups: list[str] | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rel_dim: int = 32,
        fanouts: list[int] | None = None,
        batch_size: int = 1024,
        epochs: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.2,
        device: str | None = None,
        temporal: bool = False,
        graph_config: GraphModelConfig | None = None,
        feature_normalizer_state: HybridFeatureNormalizerState | None = None,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.feature_groups = feature_groups or default_feature_groups(model_name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rel_dim = rel_dim
        self.fanouts = fanouts or [15, 10]
        self.batch_size = batch_size
        self.eval_batch_size = max(batch_size, 2048)
        self.epochs = epochs
        self.max_day = max_day
        self.temporal = temporal
        self.device = torch.device(resolve_device(device))
        self.feature_normalizer_state = feature_normalizer_state
        self.graph_config = graph_config or GraphModelConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
        )

        self.network = RelationGraphSAGENetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_relations=num_relations,
            rel_dim=rel_dim,
            dropout=self.graph_config.dropout,
            temporal=temporal,
            model_config=self.graph_config,
        ).to(self.device)

    def _iter_batches(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        training: bool,
        rng: np.random.Generator,
    ) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
        nodes = np.asarray(node_ids, dtype=np.int32)
        positions = np.arange(nodes.size, dtype=np.int32)
        effective_batch_size = self.batch_size if training else self.eval_batch_size
        if self.temporal:
            buckets = np.asarray(context.graph_cache.node_time_bucket[nodes], dtype=np.int8)
            batches: list[tuple[np.ndarray, np.ndarray, int | None]] = []
            for bucket_idx, window in enumerate(context.graph_cache.time_windows):
                bucket_mask = buckets == bucket_idx
                bucket_nodes = nodes[bucket_mask]
                bucket_positions = positions[bucket_mask]
                if bucket_nodes.size == 0:
                    continue
                if training:
                    order = rng.permutation(bucket_nodes.size)
                    bucket_nodes = bucket_nodes[order]
                    bucket_positions = bucket_positions[order]
                for start in range(0, bucket_nodes.size, effective_batch_size):
                    batches.append(
                        (
                            bucket_nodes[start : start + effective_batch_size],
                            bucket_positions[start : start + effective_batch_size],
                            int(window["end_day"]),
                        )
                    )
            return batches

        if training:
            order = rng.permutation(nodes.size)
            nodes = nodes[order]
            positions = positions[order]
        return [
            (
                nodes[start : start + effective_batch_size],
                positions[start : start + effective_batch_size],
                None,
            )
            for start in range(0, nodes.size, effective_batch_size)
        ]

    def _chunk_batch_partition(
        self,
        nodes: np.ndarray,
        positions: np.ndarray,
        snapshot_end: int | None,
        effective_batch_size: int,
    ) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
        return [
            (
                nodes[start : start + effective_batch_size],
                positions[start : start + effective_batch_size],
                snapshot_end,
            )
            for start in range(0, nodes.size, effective_batch_size)
        ]

    def _build_balanced_partition_batches(
        self,
        nodes: np.ndarray,
        positions: np.ndarray,
        labels: np.ndarray,
        snapshot_end: int | None,
        rng: np.random.Generator,
        effective_batch_size: int,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, int | None]], TrainBatchStats]:
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_nodes = nodes[pos_mask]
        pos_positions = positions[pos_mask]
        neg_nodes = nodes[neg_mask]
        neg_positions = positions[neg_mask]

        if pos_nodes.size == 0 or neg_nodes.size == 0:
            order = rng.permutation(nodes.size)
            shuffled_nodes = nodes[order]
            shuffled_positions = positions[order]
            return (
                self._chunk_batch_partition(
                    nodes=shuffled_nodes,
                    positions=shuffled_positions,
                    snapshot_end=snapshot_end,
                    effective_batch_size=effective_batch_size,
                ),
                TrainBatchStats(
                    target_count=int(nodes.size),
                    positive_count=int(pos_nodes.size),
                    negative_count=int(neg_nodes.size),
                ),
            )

        negative_ratio = max(float(self.graph_config.train_negative_ratio), 0.0)
        if negative_ratio <= 0.0:
            negative_ratio = float(neg_nodes.size / max(pos_nodes.size, 1))
        sampled_negatives = min(
            neg_nodes.size,
            max(1, int(math.ceil(pos_nodes.size * negative_ratio))),
        )
        neg_choice = rng.choice(neg_nodes.size, size=sampled_negatives, replace=False)
        neg_nodes = neg_nodes[neg_choice]
        neg_positions = neg_positions[neg_choice]

        pos_order = rng.permutation(pos_nodes.size)
        neg_order = rng.permutation(neg_nodes.size)
        pos_nodes = pos_nodes[pos_order]
        pos_positions = pos_positions[pos_order]
        neg_nodes = neg_nodes[neg_order]
        neg_positions = neg_positions[neg_order]

        pos_per_batch = max(1, int(math.floor(effective_batch_size / (1.0 + negative_ratio))))
        neg_per_batch = max(1, effective_batch_size - pos_per_batch)
        batches: list[tuple[np.ndarray, np.ndarray, int | None]] = []
        pos_ptr = 0
        neg_ptr = 0
        while pos_ptr < pos_nodes.size or neg_ptr < neg_nodes.size:
            batch_node_parts: list[np.ndarray] = []
            batch_position_parts: list[np.ndarray] = []

            if pos_ptr < pos_nodes.size:
                next_pos_ptr = min(pos_ptr + pos_per_batch, pos_nodes.size)
                batch_node_parts.append(pos_nodes[pos_ptr:next_pos_ptr])
                batch_position_parts.append(pos_positions[pos_ptr:next_pos_ptr])
                pos_ptr = next_pos_ptr

            if neg_ptr < neg_nodes.size:
                next_neg_ptr = min(neg_ptr + neg_per_batch, neg_nodes.size)
                batch_node_parts.append(neg_nodes[neg_ptr:next_neg_ptr])
                batch_position_parts.append(neg_positions[neg_ptr:next_neg_ptr])
                neg_ptr = next_neg_ptr

            batch_nodes = np.concatenate(batch_node_parts, axis=0)
            batch_positions = np.concatenate(batch_position_parts, axis=0)
            order = rng.permutation(batch_nodes.size)
            batches.append(
                (
                    batch_nodes[order],
                    batch_positions[order],
                    snapshot_end,
                )
            )

        return (
            batches,
            TrainBatchStats(
                target_count=int(pos_nodes.size + neg_nodes.size),
                positive_count=int(pos_nodes.size),
                negative_count=int(neg_nodes.size),
            ),
        )

    def _build_train_batches(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, int | None]], TrainBatchStats]:
        nodes = np.asarray(node_ids, dtype=np.int32)
        positions = np.arange(nodes.size, dtype=np.int32)
        effective_batch_size = self.batch_size
        if self.graph_config.train_negative_ratio <= 0.0:
            batches = self._iter_batches(
                context=context,
                node_ids=nodes,
                training=True,
                rng=rng,
            )
            labels = context.labels[nodes].astype(np.int8, copy=False)
            pos_count = int(np.sum(labels == 1))
            neg_count = int(np.sum(labels == 0))
            return (
                batches,
                TrainBatchStats(
                    target_count=int(nodes.size),
                    positive_count=pos_count,
                    negative_count=neg_count,
                ),
            )

        if self.temporal:
            buckets = np.asarray(context.graph_cache.node_time_bucket[nodes], dtype=np.int8)
            batches: list[tuple[np.ndarray, np.ndarray, int | None]] = []
            total_pos = 0
            total_neg = 0
            total_targets = 0
            for bucket_idx, window in enumerate(context.graph_cache.time_windows):
                bucket_mask = buckets == bucket_idx
                bucket_nodes = nodes[bucket_mask]
                bucket_positions = positions[bucket_mask]
                if bucket_nodes.size == 0:
                    continue
                bucket_labels = context.labels[bucket_nodes].astype(np.int8, copy=False)
                bucket_batches, bucket_stats = self._build_balanced_partition_batches(
                    nodes=bucket_nodes,
                    positions=bucket_positions,
                    labels=bucket_labels,
                    snapshot_end=int(window["end_day"]),
                    rng=rng,
                    effective_batch_size=effective_batch_size,
                )
                batches.extend(bucket_batches)
                total_targets += bucket_stats.target_count
                total_pos += bucket_stats.positive_count
                total_neg += bucket_stats.negative_count
            return (
                batches,
                TrainBatchStats(
                    target_count=total_targets,
                    positive_count=total_pos,
                    negative_count=total_neg,
                ),
            )

        labels = context.labels[nodes].astype(np.int8, copy=False)
        batches, stats = self._build_balanced_partition_batches(
            nodes=nodes,
            positions=positions,
            labels=labels,
            snapshot_end=None,
            rng=rng,
            effective_batch_size=effective_batch_size,
        )
        return batches, stats

    def _sample_batch_subgraph(
        self,
        graph: GraphCache,
        batch_nodes: np.ndarray,
        rng: np.random.Generator,
        snapshot_end: int | None,
    ) -> SampledSubgraph:
        sampler = (
            sample_batched_relation_subgraphs
            if self.graph_config.subgraph_head == "meanmax"
            else sample_relation_subgraph
        )
        return sampler(
            graph=graph,
            seed_nodes=batch_nodes,
            fanouts=self.fanouts,
            rng=rng,
            snapshot_end=snapshot_end,
        )

    def _tensorize_subgraph(
        self,
        context: GraphPhaseContext,
        subgraph: SampledSubgraph,
        snapshot_end: int | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        x_np = context.feature_store.take_rows(subgraph.node_ids)
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        edge_src = torch.as_tensor(subgraph.edge_src, dtype=torch.long, device=self.device)
        edge_dst = torch.as_tensor(subgraph.edge_dst, dtype=torch.long, device=self.device)
        rel_ids = torch.as_tensor(subgraph.rel_ids, dtype=torch.long, device=self.device)
        target_idx = torch.as_tensor(
            subgraph.target_local_idx,
            dtype=torch.long,
            device=self.device,
        )

        edge_relative_time = None
        if self.temporal and subgraph.edge_timestamp.size:
            snapshot = snapshot_end if snapshot_end is not None else self.max_day
            relative_time = (
                snapshot - subgraph.edge_timestamp.astype(np.float32, copy=False)
            ) / max(float(self.max_day), 1.0)
            relative_time = np.clip(relative_time, 0.0, 1.0)
            edge_relative_time = torch.as_tensor(
                relative_time.reshape(-1, 1),
                dtype=torch.float32,
                device=self.device,
            )

        node_subgraph_id = None
        if subgraph.node_subgraph_id is not None:
            node_subgraph_id = torch.as_tensor(
                subgraph.node_subgraph_id,
                dtype=torch.long,
                device=self.device,
            )

        edge_subgraph_id = None
        if subgraph.edge_subgraph_id is not None:
            edge_subgraph_id = torch.as_tensor(
                subgraph.edge_subgraph_id,
                dtype=torch.long,
                device=self.device,
            )

        return (
            x,
            edge_src,
            edge_dst,
            rel_ids,
            edge_relative_time,
            target_idx,
            node_subgraph_id,
            edge_subgraph_id,
        )

    def fit(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
    ) -> dict[str, float]:
        set_global_seed(self.seed)
        train_ids = np.asarray(train_ids, dtype=np.int32)
        val_ids = np.asarray(val_ids, dtype=np.int32)
        train_labels = context.labels[train_ids].astype(np.float32, copy=False)
        val_labels = context.labels[val_ids].astype(np.int8, copy=False)

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.graph_config.learning_rate,
            weight_decay=self.graph_config.weight_decay,
        )
        scheduler = None
        if self.graph_config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
            )

        pos_count = float(np.sum(train_labels == 1))
        neg_count = float(np.sum(train_labels == 0))
        effective_neg_count = neg_count
        if self.graph_config.train_negative_ratio > 0.0:
            effective_neg_count = min(
                neg_count,
                float(math.ceil(pos_count * self.graph_config.train_negative_ratio)),
            )
        pos_weight = torch.tensor(
            [effective_neg_count / max(pos_count, 1.0)],
            dtype=torch.float32,
            device=self.device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_auc = -math.inf
        best_epoch = -1
        epochs_without_improvement = 0
        epoch_rng = np.random.default_rng(self.seed)

        with tqdm(
            range(1, self.epochs + 1),
            desc=f"{self.model_name}:seed{self.seed}:epochs",
            unit="epoch",
            dynamic_ncols=True,
        ) as epoch_pbar:
            for epoch in epoch_pbar:
                self.network.train()
                batch_losses: list[float] = []
                batch_subgraph_nodes: list[float] = []
                batch_subgraph_edges: list[float] = []
                batch_emb_norm: list[float] = []
                batch_dirichlet: list[float] = []
                batch_grad_norm: list[float] = []

                train_batches, train_batch_stats = self._build_train_batches(
                    context=context,
                    node_ids=train_ids,
                    rng=epoch_rng,
                )
                with tqdm(
                    train_batches,
                    desc=f"{self.model_name}:seed{self.seed}:train:{epoch}/{self.epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False,
                ) as batch_pbar:
                    for batch_nodes, _, snapshot_end in batch_pbar:
                        subgraph = self._sample_batch_subgraph(
                            graph=context.graph_cache,
                            batch_nodes=batch_nodes,
                            rng=epoch_rng,
                            snapshot_end=snapshot_end,
                        )
                        (
                            x,
                            edge_src,
                            edge_dst,
                            rel_ids,
                            edge_relative_time,
                            target_idx,
                            node_subgraph_id,
                            edge_subgraph_id,
                        ) = self._tensorize_subgraph(
                            context=context,
                            subgraph=subgraph,
                            snapshot_end=snapshot_end,
                        )

                        y_batch = torch.as_tensor(
                            context.labels[batch_nodes],
                            dtype=torch.float32,
                            device=self.device,
                        )

                        optimizer.zero_grad(set_to_none=True)
                        logits, diagnostics = self.network(
                            x=x,
                            edge_src=edge_src,
                            edge_dst=edge_dst,
                            rel_ids=rel_ids,
                            edge_relative_time=edge_relative_time,
                            target_local_idx=target_idx,
                            node_subgraph_id=node_subgraph_id,
                            edge_subgraph_id=edge_subgraph_id,
                            return_details=True,
                        )
                        loss = criterion(logits, y_batch)
                        loss.backward()

                        if self.graph_config.grad_clip > 0:
                            grad_norm = float(
                                torch.nn.utils.clip_grad_norm_(
                                    self.network.parameters(),
                                    self.graph_config.grad_clip,
                                ).item()
                            )
                        else:
                            grad_norm = _compute_grad_norm(self.network.parameters())

                        optimizer.step()
                        batch_losses.append(float(loss.detach().item()))
                        batch_subgraph_nodes.append(float(subgraph.node_ids.shape[0]))
                        batch_subgraph_edges.append(float(subgraph.edge_src.shape[0]))
                        batch_emb_norm.append(float(diagnostics["emb_norm"]))
                        batch_dirichlet.append(float(diagnostics["dirichlet_energy"]))
                        batch_grad_norm.append(float(grad_norm))
                        batch_pbar.set_postfix(
                            loss=f"{batch_losses[-1]:.4f}",
                            nodes=int(subgraph.node_ids.shape[0]),
                            edges=int(subgraph.edge_src.shape[0]),
                            refresh=False,
                        )

                val_prob = self.predict_proba(
                    context=context,
                    node_ids=val_ids,
                    batch_seed=self.seed + 1000,
                    progress_desc=f"{self.model_name}:seed{self.seed}:val:{epoch}/{self.epochs}",
                )
                val_metrics = compute_binary_classification_metrics(val_labels, val_prob)
                val_auc = val_metrics["auc"]
                if scheduler is not None:
                    scheduler.step(1.0 - val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.network.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                current_lr = float(optimizer.param_groups[0]["lr"])
                epoch_pbar.set_postfix(
                    val_auc=f"{val_auc:.4f}",
                    best_epoch=best_epoch,
                    lr=f"{current_lr:.2e}",
                    refresh=False,
                )
                tqdm.write(
                    f"[{self.model_name}] epoch={epoch} "
                    f"train_loss={np.mean(batch_losses):.6f} "
                    f"val_auc={val_auc:.6f} "
                    f"val_pr_auc={val_metrics['pr_auc']:.6f} "
                    f"val_ap={val_metrics['ap']:.6f} "
                    f"train_targets={train_batch_stats.target_count} "
                    f"train_pos={train_batch_stats.positive_count} "
                    f"train_neg={train_batch_stats.negative_count} "
                    f"train_pos_rate={train_batch_stats.positive_rate:.4f} "
                    f"avg_subgraph_nodes={np.mean(batch_subgraph_nodes):.2f} "
                    f"avg_subgraph_edges={np.mean(batch_subgraph_edges):.2f} "
                    f"emb_norm={np.mean(batch_emb_norm):.6f} "
                    f"dirichlet={np.mean(batch_dirichlet):.6f} "
                    f"grad_norm={np.mean(batch_grad_norm):.6f} "
                    f"best_epoch={best_epoch} "
                    f"lr={current_lr:.6g} "
                    f"loss_pos_weight={float(pos_weight.item()):.4f}"
                )

                if (
                    self.graph_config.early_stop_patience > 0
                    and epochs_without_improvement >= self.graph_config.early_stop_patience
                ):
                    tqdm.write(
                        f"[{self.model_name}] early_stop epoch={epoch} "
                        f"patience={self.graph_config.early_stop_patience}"
                    )
                    break

        if best_state is None:
            raise RuntimeError(f"{self.model_name}: failed to capture a best checkpoint.")
        self.network.load_state_dict(best_state)
        return {
            "val_auc": float(best_val_auc),
            "best_epoch": float(best_epoch),
            "loss_pos_weight": float(pos_weight.item()),
        }

    @torch.no_grad()
    def predict_proba(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        batch_seed: int | None = None,
        progress_desc: str | None = None,
    ) -> np.ndarray:
        self.network.eval()
        node_ids = np.asarray(node_ids, dtype=np.int32)
        rng = np.random.default_rng(self.seed if batch_seed is None else batch_seed)
        probabilities = np.zeros(node_ids.shape[0], dtype=np.float32)
        batches = self._iter_batches(
            context=context,
            node_ids=node_ids,
            training=False,
            rng=rng,
        )
        desc = progress_desc or f"{self.model_name}:seed{self.seed}:{context.phase}:predict"
        processed = 0
        with tqdm(
            batches,
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ) as batch_pbar:
            for batch_nodes, batch_positions, snapshot_end in batch_pbar:
                subgraph = self._sample_batch_subgraph(
                    graph=context.graph_cache,
                    batch_nodes=batch_nodes,
                    rng=rng,
                    snapshot_end=snapshot_end,
                )
                (
                    x,
                    edge_src,
                    edge_dst,
                    rel_ids,
                    edge_relative_time,
                    target_idx,
                    node_subgraph_id,
                    edge_subgraph_id,
                ) = self._tensorize_subgraph(
                    context=context,
                    subgraph=subgraph,
                    snapshot_end=snapshot_end,
                )
                logits = self.network(
                    x=x,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    rel_ids=rel_ids,
                    edge_relative_time=edge_relative_time,
                    target_local_idx=target_idx,
                    node_subgraph_id=node_subgraph_id,
                    edge_subgraph_id=edge_subgraph_id,
                )
                batch_prob = (
                    torch.sigmoid(logits)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                probabilities[batch_positions] = batch_prob
                processed += int(batch_positions.size)
                batch_pbar.set_postfix(done=f"{processed}/{node_ids.size}", refresh=False)
        return probabilities

    def save(self, run_dir: Path) -> None:
        ensure_dir(run_dir)
        torch.save(self.network.state_dict(), run_dir / "model.pt")
        metadata = {
            "model_name": self.model_name,
            "seed": self.seed,
            "feature_groups": self.feature_groups,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "rel_dim": self.rel_dim,
            "fanouts": self.fanouts,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "epochs": self.epochs,
            "learning_rate": self.graph_config.learning_rate,
            "weight_decay": self.graph_config.weight_decay,
            "dropout": self.graph_config.dropout,
            "max_day": self.max_day,
            "temporal": self.temporal,
            "graph_model_config": self.graph_config.to_dict(),
            "feature_normalizer_state": (
                None
                if self.feature_normalizer_state is None
                else self.feature_normalizer_state.to_dict()
            ),
        }
        write_json(run_dir / "model_meta.json", metadata)

    @classmethod
    def load(
        cls,
        run_dir: Path,
        input_dim: int,
        num_relations: int,
        device: str | None = None,
    ) -> "BaseGraphSAGEExperiment":
        meta = json.loads((run_dir / "model_meta.json").read_text(encoding="utf-8"))
        graph_config_payload = meta.get("graph_model_config")
        if graph_config_payload is not None:
            graph_config = GraphModelConfig.from_dict(graph_config_payload)
        else:
            graph_config = GraphModelConfig(
                learning_rate=float(meta.get("learning_rate", 1e-3)),
                weight_decay=float(meta.get("weight_decay", 1e-5)),
                dropout=float(meta.get("dropout", 0.2)),
            )

        instance = cls(
            model_name=meta["model_name"],
            seed=int(meta["seed"]),
            input_dim=input_dim,
            num_relations=num_relations,
            max_day=int(meta["max_day"]),
            feature_groups=list(meta["feature_groups"]),
            hidden_dim=int(meta["hidden_dim"]),
            num_layers=int(meta["num_layers"]),
            rel_dim=int(meta["rel_dim"]),
            fanouts=list(meta["fanouts"]),
            batch_size=int(meta["batch_size"]),
            epochs=int(meta["epochs"]),
            device=device,
            temporal=bool(meta.get("temporal", False)),
            graph_config=graph_config,
            feature_normalizer_state=HybridFeatureNormalizerState.from_dict(
                meta.get("feature_normalizer_state")
            ),
        )
        instance.eval_batch_size = int(meta.get("eval_batch_size", max(instance.batch_size, 2048)))
        state_dict = torch.load(
            run_dir / "model.pt",
            map_location=instance.device,
            weights_only=True,
        )
        instance.network.load_state_dict(state_dict)
        return instance


class RelationGraphSAGEExperiment(BaseGraphSAGEExperiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["temporal"] = False
        super().__init__(*args, **kwargs)


class TemporalRelationGraphSAGEExperiment(BaseGraphSAGEExperiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["temporal"] = True
        super().__init__(*args, **kwargs)
