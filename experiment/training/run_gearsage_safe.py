from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (
    MODEL_OUTPUT_ROOT,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    load_phase_arrays,
    save_prediction_npz,
    set_global_seed,
    write_json,
)


@dataclass(frozen=True)
class ProcessedPhaseGraph:
    phase: str
    x: np.ndarray
    y: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_type: np.ndarray
    edge_timestamp: np.ndarray
    edge_direct: np.ndarray
    out_ptr: np.ndarray
    out_neighbors: np.ndarray
    label_feature_slice: tuple[int, int]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True)
class ExactSubgraph:
    node_ids: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_type: np.ndarray
    edge_timestamp: np.ndarray
    edge_direct: np.ndarray
    target_local_idx: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe GEARSage-style training on thesis splits.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "gearsage_safe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-seed-batch-size", type=int, default=4096)
    parser.add_argument("--eval-seed-batch-size", type=int, default=4096)
    parser.add_argument("--update-mode", choices=("plain", "gru"), default="plain")
    parser.add_argument("--initial-neighbor-decay", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _build_known_codes(phase: str, labels: np.ndarray, split) -> np.ndarray:
    codes = np.full(labels.shape[0], 4, dtype=np.int8)
    codes[labels == 2] = 2
    codes[labels == 3] = 3
    if phase == "phase1":
        train_ids = np.asarray(split.train_ids, dtype=np.int32)
        fg = labels[train_ids]
        codes[train_ids[fg == 0]] = 0
        codes[train_ids[fg == 1]] = 1
    return codes


def _add_degree_feature(x: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray) -> np.ndarray:
    indegree = np.bincount(edge_dst, minlength=x.shape[0]).astype(np.float32, copy=False)
    outdegree = np.bincount(edge_src, minlength=x.shape[0]).astype(np.float32, copy=False)
    return np.concatenate([x, indegree[:, None], outdegree[:, None]], axis=1).astype(np.float32, copy=False)


def _add_official_gearsage_stats(
    x: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_type: np.ndarray,
    edge_timestamp: np.ndarray,
) -> np.ndarray:
    num_nodes = x.shape[0]
    indegree_hint = np.bincount(edge_dst, minlength=num_nodes).astype(np.float32, copy=False)

    ts_sum = np.bincount(
        edge_src,
        weights=edge_timestamp.astype(np.float32, copy=False),
        minlength=num_nodes,
    ).astype(np.float32, copy=False)
    ts_sum = np.log1p(ts_sum).astype(np.float32, copy=False)

    ts_max = np.zeros(num_nodes, dtype=np.float32)
    np.maximum.at(ts_max, edge_src, edge_timestamp.astype(np.float32, copy=False))
    np.maximum.at(ts_max, edge_dst, edge_timestamp.astype(np.float32, copy=False))
    ts_max = np.log1p(ts_max).astype(np.float32, copy=False)

    type_hist = np.zeros((num_nodes, 11), dtype=np.float32)
    for current_type in range(1, 12):
        type_mask = edge_type == current_type
        if np.any(type_mask):
            type_hist[:, current_type - 1] = np.bincount(
                edge_src[type_mask],
                minlength=num_nodes,
            ).astype(np.float32, copy=False)

    return np.concatenate(
        [x, indegree_hint[:, None], ts_sum[:, None], ts_max[:, None], type_hist],
        axis=1,
    ).astype(np.float32, copy=False)


def _add_cos_sim_sum(x: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray) -> np.ndarray:
    src_x = x[edge_src]
    dst_x = x[edge_dst]
    src_norm = np.linalg.norm(src_x, axis=1)
    dst_norm = np.linalg.norm(dst_x, axis=1)
    denom = np.clip(src_norm * dst_norm, 1e-6, None)
    sim = np.sum(src_x * dst_x, axis=1) / denom
    sim_sum = np.bincount(edge_src, weights=sim.astype(np.float32, copy=False), minlength=x.shape[0]).astype(
        np.float32,
        copy=False,
    )
    return np.concatenate([x, sim_sum[:, None]], axis=1).astype(np.float32, copy=False)


def _to_undirected(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_type: np.ndarray,
    edge_timestamp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = np.concatenate([edge_src, edge_dst]).astype(np.int32, copy=False)
    col = np.concatenate([edge_dst, edge_src]).astype(np.int32, copy=False)
    edge_attr = np.concatenate([edge_type, edge_type]).astype(np.int16, copy=False)
    edge_time = np.concatenate([edge_timestamp, edge_timestamp]).astype(np.int32, copy=False)
    return row, col, edge_attr, edge_time


def _add_feature_flag(x: np.ndarray, raw_dim: int) -> np.ndarray:
    feature_flag = np.zeros((x.shape[0], raw_dim), dtype=np.float32)
    raw = x[:, :raw_dim]
    feature_flag[raw == -1] = 1.0
    x_filled = x.copy()
    x_filled[x_filled == -1] = 0.0
    return np.concatenate([x_filled, feature_flag], axis=1).astype(np.float32, copy=False)


def _add_safe_label_counts(
    x: np.ndarray,
    unique_src: np.ndarray,
    unique_dst: np.ndarray,
    known_codes: np.ndarray,
) -> np.ndarray:
    num_nodes = x.shape[0]
    known_fg = ((known_codes == 0) | (known_codes == 1)).astype(np.int64, copy=False)
    known_bg = ((known_codes == 2) | (known_codes == 3)).astype(np.int64, copy=False)
    fg_count = (
        np.bincount(unique_src, weights=known_fg[unique_dst], minlength=num_nodes)
        + np.bincount(unique_dst, weights=known_fg[unique_src], minlength=num_nodes)
    ).astype(np.float32, copy=False)
    bg_count = (
        np.bincount(unique_src, weights=known_bg[unique_dst], minlength=num_nodes)
        + np.bincount(unique_dst, weights=known_bg[unique_src], minlength=num_nodes)
    ).astype(np.float32, copy=False)
    return np.concatenate([x, fg_count[:, None], bg_count[:, None]], axis=1).astype(np.float32, copy=False)


def _add_safe_label_feature(x: np.ndarray, known_codes: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    fg = ((known_codes == 0) | (known_codes == 1)).astype(np.float32, copy=False)
    bg2 = (known_codes == 2).astype(np.float32, copy=False)
    bg3 = (known_codes == 3).astype(np.float32, copy=False)
    start = int(x.shape[1])
    x_aug = np.concatenate([x, fg[:, None], bg2[:, None], bg3[:, None]], axis=1).astype(np.float32, copy=False)
    return x_aug, (start, start + 3)


def _build_out_csr(num_nodes: int, edge_src: np.ndarray, edge_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((edge_dst, edge_src))
    src_sorted = edge_src[order].astype(np.int32, copy=False)
    dst_sorted = edge_dst[order].astype(np.int32, copy=False)
    counts = np.bincount(src_sorted, minlength=num_nodes).astype(np.int64, copy=False)
    out_ptr = np.empty(num_nodes + 1, dtype=np.int64)
    out_ptr[0] = 0
    np.cumsum(counts, out=out_ptr[1:])
    return out_ptr, dst_sorted


def build_processed_phase_graph(phase: str, split) -> ProcessedPhaseGraph:
    arrays = load_phase_arrays(
        phase,
        keys=("x", "y", "edge_index", "edge_type", "edge_timestamp"),
    )
    x_raw = np.asarray(arrays["x"], dtype=np.float32)
    y = np.asarray(arrays["y"], dtype=np.int64)
    edge_index = np.asarray(arrays["edge_index"], dtype=np.int32)
    edge_type = np.asarray(arrays["edge_type"], dtype=np.int16).reshape(-1)
    edge_timestamp = np.asarray(arrays["edge_timestamp"], dtype=np.int32).reshape(-1)
    edge_src = edge_index[:, 0].astype(np.int32, copy=False)
    edge_dst = edge_index[:, 1].astype(np.int32, copy=False)

    x = _add_official_gearsage_stats(
        x_raw,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_type=edge_type,
        edge_timestamp=edge_timestamp,
    )
    x = _add_degree_feature(x, edge_src=edge_src, edge_dst=edge_dst)
    x = _add_cos_sim_sum(x, edge_src=edge_src, edge_dst=edge_dst)

    und_src, und_dst, und_type, und_time = _to_undirected(
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_type=edge_type,
        edge_timestamp=edge_timestamp,
    )
    unique_mask = und_src < und_dst
    unique_src = und_src[unique_mask].astype(np.int32, copy=False)
    unique_dst = und_dst[unique_mask].astype(np.int32, copy=False)
    unique_type = und_type[unique_mask].astype(np.int16, copy=False)
    unique_time = und_time[unique_mask].astype(np.int32, copy=False)

    final_src = np.concatenate([unique_src, unique_dst]).astype(np.int32, copy=False)
    final_dst = np.concatenate([unique_dst, unique_src]).astype(np.int32, copy=False)
    final_type = np.concatenate([unique_type, unique_type]).astype(np.int16, copy=False)
    final_time = np.concatenate([unique_time, unique_time]).astype(np.int32, copy=False)
    final_direct = np.concatenate(
        [
            np.zeros(unique_src.shape[0], dtype=np.int64),
            np.ones(unique_src.shape[0], dtype=np.int64),
        ]
    )

    known_codes = _build_known_codes(phase=phase, labels=y.astype(np.int8, copy=False), split=split)
    x = _add_feature_flag(x, raw_dim=x_raw.shape[1])
    x = _add_safe_label_counts(x, unique_src=unique_src, unique_dst=unique_dst, known_codes=known_codes)
    x, label_feature_slice = _add_safe_label_feature(x, known_codes=known_codes)

    out_ptr, out_neighbors = _build_out_csr(
        num_nodes=x.shape[0],
        edge_src=final_src,
        edge_dst=final_dst,
    )

    return ProcessedPhaseGraph(
        phase=phase,
        x=x.astype(np.float32, copy=False),
        y=y.astype(np.int64, copy=False),
        edge_src=final_src,
        edge_dst=final_dst,
        edge_type=final_type.astype(np.int64, copy=False),
        edge_timestamp=final_time.astype(np.int64, copy=False),
        edge_direct=final_direct.astype(np.int64, copy=False),
        out_ptr=out_ptr,
        out_neighbors=out_neighbors.astype(np.int32, copy=False),
        label_feature_slice=label_feature_slice,
    )


def exact_k_hop_subgraph(graph: ProcessedPhaseGraph, seed_nodes: np.ndarray, hops: int) -> ExactSubgraph:
    seeds = np.asarray(seed_nodes, dtype=np.int32)
    seen = np.zeros(graph.num_nodes, dtype=bool)
    seen[seeds] = True
    frontier = seeds
    for _ in range(max(hops, 0)):
        if frontier.size == 0:
            break
        neighbor_parts: list[np.ndarray] = []
        for node in frontier.tolist():
            start = int(graph.out_ptr[node])
            end = int(graph.out_ptr[node + 1])
            if end > start:
                neighbor_parts.append(graph.out_neighbors[start:end])
        if not neighbor_parts:
            break
        neighbors = np.concatenate(neighbor_parts).astype(np.int32, copy=False)
        unseen = neighbors[~seen[neighbors]]
        if unseen.size == 0:
            frontier = np.empty(0, dtype=np.int32)
            continue
        frontier = np.unique(unseen).astype(np.int32, copy=False)
        seen[frontier] = True

    node_ids = np.flatnonzero(seen).astype(np.int32, copy=False)
    local_id = np.full(graph.num_nodes, -1, dtype=np.int32)
    local_id[node_ids] = np.arange(node_ids.shape[0], dtype=np.int32)
    edge_mask = seen[graph.edge_src] & seen[graph.edge_dst]
    edge_src = local_id[graph.edge_src[edge_mask]].astype(np.int32, copy=False)
    edge_dst = local_id[graph.edge_dst[edge_mask]].astype(np.int32, copy=False)
    target_local_idx = local_id[seeds].astype(np.int32, copy=False)
    return ExactSubgraph(
        node_ids=node_ids,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_type=graph.edge_type[edge_mask].astype(np.int64, copy=False),
        edge_timestamp=graph.edge_timestamp[edge_mask].astype(np.int64, copy=False),
        edge_direct=graph.edge_direct[edge_mask].astype(np.int64, copy=False),
        target_local_idx=target_local_idx,
    )


class TimeEncoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.w = nn.Linear(1, dimension)
        base = torch.from_numpy(1 / 10 ** np.linspace(0, 1.5, dimension)).float().reshape(dimension, 1)
        self.w.weight = nn.Parameter(base)
        self.w.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.w(torch.log(t.float() + 1.0).unsqueeze(1)))


class SimpleSAGEConv(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_m = nn.Linear(node_dim + edge_dim * 2, out_dim)
        self.lin_r = nn.Linear(node_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_t: torch.Tensor,
        include_self: bool = True,
    ) -> torch.Tensor:
        row, col = edge_index
        msg = torch.cat([x[col], edge_attr, edge_t], dim=1)
        agg = x.new_zeros((x.size(0), msg.size(1)))
        agg.index_add_(0, row, msg)
        out = self.lin_m(agg)
        if include_self:
            out = 0.5 * out + self.lin_r(x)
        return out


class GEARSageSafe(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_attr_channels: int = 50,
        time_channels: int = 50,
        num_layers: int = 3,
        dropout: float = 0.3,
        update_mode: str = "plain",
        initial_neighbor_decay: float = 0.85,
    ) -> None:
        super().__init__()
        self.update_mode = str(update_mode)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        if self.update_mode == "gru":
            self.input_proj = nn.Linear(in_channels, hidden_channels)
            self.updates = nn.ModuleList()
            init_decay = float(np.clip(initial_neighbor_decay, 1e-3, 1.0 - 1e-3))
            self.decay_logits = nn.Parameter(
                torch.full((num_layers,), float(math.log(init_decay / (1.0 - init_decay))))
            )
            for _ in range(num_layers):
                self.convs.append(
                    SimpleSAGEConv(
                        node_dim=hidden_channels,
                        edge_dim=edge_attr_channels,
                        out_dim=hidden_channels,
                    )
                )
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                self.updates.append(nn.GRUCell(hidden_channels, hidden_channels))
            self.classifier = nn.Linear(hidden_channels, out_channels)
        else:
            self.input_proj = None
            self.updates = None
            self.decay_logits = None
            self.classifier = None
            for idx in range(num_layers):
                node_dim = in_channels if idx == 0 else hidden_channels
                out_dim = out_channels if idx == num_layers - 1 else hidden_channels
                self.convs.append(SimpleSAGEConv(node_dim=node_dim, edge_dim=edge_attr_channels, out_dim=out_dim))
                self.bns.append(nn.BatchNorm1d(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.emb_type = nn.Embedding(12, edge_attr_channels)
        self.emb_direction = nn.Embedding(2, edge_attr_channels)
        self.t_enc = TimeEncoder(time_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.lin_m.reset_parameters()
            conv.lin_r.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.input_proj is not None:
            self.input_proj.reset_parameters()
        if self.updates is not None:
            for gru in self.updates:
                gru.reset_parameters()
        if self.classifier is not None:
            self.classifier.reset_parameters()
        nn.init.xavier_uniform_(self.emb_type.weight)
        nn.init.xavier_uniform_(self.emb_direction.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_t: torch.Tensor,
        edge_d: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr = self.emb_type(edge_attr) + self.emb_direction(edge_d)
        edge_t = self.t_enc(edge_t)
        if self.update_mode == "gru":
            state = self.input_proj(x)
            for idx, conv in enumerate(self.convs):
                msg = conv(state, edge_index, edge_attr, edge_t, include_self=False)
                msg = self.bns[idx](msg)
                msg = self.activation(msg)
                msg = msg * torch.sigmoid(self.decay_logits[idx])
                state = self.updates[idx](msg, state)
                if idx != len(self.convs) - 1:
                    state = self.dropout(state)
            logits = self.classifier(state)
            return F.log_softmax(logits, dim=-1)

        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr, edge_t, include_self=True)
            x = self.bns[idx](x)
            if idx != len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return F.log_softmax(x, dim=-1)


def _iter_seed_batches(node_ids: np.ndarray, batch_size: int, rng: np.random.Generator | None, shuffle: bool) -> list[np.ndarray]:
    nodes = np.asarray(node_ids, dtype=np.int32)
    if shuffle and rng is not None:
        order = rng.permutation(nodes.shape[0])
        nodes = nodes[order]
    return [nodes[start : start + batch_size] for start in range(0, nodes.shape[0], batch_size)]


def _mask_target_label_features(x: torch.Tensor, target_idx: torch.Tensor, feature_slice: tuple[int, int]) -> None:
    start, end = feature_slice
    if start < end and target_idx.numel():
        x[target_idx, start:end] = 0.0


def _predict_scores(
    model: GEARSageSafe,
    graph: ProcessedPhaseGraph,
    node_ids: np.ndarray,
    hops: int,
    batch_size: int,
    device: torch.device,
    progress_desc: str | None = None,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    iterator = _iter_seed_batches(node_ids=node_ids, batch_size=batch_size, rng=None, shuffle=False)
    pbar = tqdm(iterator, desc=progress_desc, leave=False, dynamic_ncols=True) if progress_desc else iterator
    with torch.no_grad():
        for batch_nodes in pbar:
            subgraph = exact_k_hop_subgraph(graph=graph, seed_nodes=batch_nodes, hops=hops)
            x = torch.as_tensor(graph.x[subgraph.node_ids], dtype=torch.float32, device=device)
            edge_index = torch.as_tensor(
                np.stack([subgraph.edge_src, subgraph.edge_dst], axis=0),
                dtype=torch.long,
                device=device,
            )
            edge_attr = torch.as_tensor(subgraph.edge_type, dtype=torch.long, device=device)
            edge_t = torch.as_tensor(subgraph.edge_timestamp, dtype=torch.float32, device=device)
            edge_d = torch.as_tensor(subgraph.edge_direct, dtype=torch.long, device=device)
            target_idx = torch.as_tensor(subgraph.target_local_idx, dtype=torch.long, device=device)
            _mask_target_label_features(x, target_idx=target_idx, feature_slice=graph.label_feature_slice)
            out = model(x, edge_index, edge_attr, edge_t, edge_d)
            prob = out[target_idx].exp()[:, 1].detach().cpu().numpy().astype(np.float32, copy=False)
            outputs.append(prob)
    if progress_desc:
        pbar.close()
    return np.concatenate(outputs).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    set_global_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    split = load_experiment_split()
    phase1_graph = build_processed_phase_graph("phase1", split=split)
    phase2_graph = build_processed_phase_graph("phase2", split=split)

    train_ids = np.asarray(split.train_ids, dtype=np.int32)
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    train_pos = train_ids[phase1_graph.y[train_ids] == 1]
    train_neg = train_ids[phase1_graph.y[train_ids] == 0]
    val_labels = (phase1_graph.y[val_ids] == 1).astype(np.int8, copy=False)
    external_labels = (phase2_graph.y[external_ids] == 1).astype(np.int8, copy=False)

    model = GEARSageSafe(
        in_channels=int(phase1_graph.x.shape[1]),
        hidden_channels=int(args.hidden_dim),
        out_channels=2,
        num_layers=int(args.layers),
        dropout=float(args.dropout),
        update_mode=str(args.update_mode),
        initial_neighbor_decay=float(args.initial_neighbor_decay),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    run_dir = ensure_dir(args.outdir / args.run_name)
    log_path = run_dir / "train.log"
    log_path.write_text("", encoding="utf-8")
    history: list[dict[str, Any]] = []

    best_state: dict[str, torch.Tensor] | None = None
    best_val_auc = -math.inf
    best_epoch = -1
    stale_epochs = 0
    train_rng = np.random.default_rng(args.seed)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        sampled_neg = train_rng.choice(train_neg, size=train_pos.shape[0], replace=False)
        epoch_nodes = np.concatenate([train_pos, sampled_neg]).astype(np.int32, copy=False)
        batch_list = _iter_seed_batches(
            node_ids=epoch_nodes,
            batch_size=int(args.train_seed_batch_size),
            rng=train_rng,
            shuffle=True,
        )
        batch_losses: list[float] = []
        with tqdm(batch_list, desc=f"gearsage_safe:train:{epoch}/{args.epochs}", leave=False, dynamic_ncols=True) as pbar:
            for batch_nodes in pbar:
                subgraph = exact_k_hop_subgraph(graph=phase1_graph, seed_nodes=batch_nodes, hops=int(args.hops))
                x = torch.as_tensor(phase1_graph.x[subgraph.node_ids], dtype=torch.float32, device=device)
                edge_index = torch.as_tensor(
                    np.stack([subgraph.edge_src, subgraph.edge_dst], axis=0),
                    dtype=torch.long,
                    device=device,
                )
                edge_attr = torch.as_tensor(subgraph.edge_type, dtype=torch.long, device=device)
                edge_t = torch.as_tensor(subgraph.edge_timestamp, dtype=torch.float32, device=device)
                edge_d = torch.as_tensor(subgraph.edge_direct, dtype=torch.long, device=device)
                target_idx = torch.as_tensor(subgraph.target_local_idx, dtype=torch.long, device=device)
                target_y = torch.as_tensor((phase1_graph.y[batch_nodes] == 1).astype(np.int64), dtype=torch.long, device=device)
                _mask_target_label_features(x, target_idx=target_idx, feature_slice=phase1_graph.label_feature_slice)

                optimizer.zero_grad(set_to_none=True)
                out = model(x, edge_index, edge_attr, edge_t, edge_d)
                loss = F.nll_loss(out[target_idx], target_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                batch_losses.append(float(loss.detach().item()))
                pbar.set_postfix(loss=f"{batch_losses[-1]:.4f}", nodes=int(subgraph.node_ids.size), edges=int(subgraph.edge_src.size))
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        val_prob = _predict_scores(
            model=model,
            graph=phase1_graph,
            node_ids=val_ids,
            hops=int(args.hops),
            batch_size=int(args.eval_seed_batch_size),
            device=device,
            progress_desc=f"gearsage_safe:val:{epoch}/{args.epochs}",
        )
        val_auc = float(roc_auc_score(val_labels, val_prob))
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)) if batch_losses else math.nan,
            "val_auc": val_auc,
        }

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            external_prob = _predict_scores(
                model=model,
                graph=phase2_graph,
                node_ids=external_ids,
                hops=int(args.hops),
                batch_size=int(args.eval_seed_batch_size),
                device=device,
                progress_desc=f"gearsage_safe:external:{epoch}/{args.epochs}",
            )
            external_auc = float(roc_auc_score(external_labels, external_prob))
            record["external_auc_on_best"] = external_auc
            save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, val_labels, val_prob)
            save_prediction_npz(run_dir / "phase2_external_predictions.npz", external_ids, external_labels, external_prob)
        else:
            stale_epochs += 1

        history.append(record)
        line = json.dumps(record, ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")
        print(line)

        if stale_epochs >= int(args.patience):
            print(f"[gearsage_safe] early_stop epoch={epoch} patience={args.patience}")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")

    model.load_state_dict(best_state)
    val_prob = _predict_scores(
        model=model,
        graph=phase1_graph,
        node_ids=val_ids,
        hops=int(args.hops),
        batch_size=int(args.eval_seed_batch_size),
        device=device,
        progress_desc="gearsage_safe:final_val",
    )
    external_prob = _predict_scores(
        model=model,
        graph=phase2_graph,
        node_ids=external_ids,
        hops=int(args.hops),
        batch_size=int(args.eval_seed_batch_size),
        device=device,
        progress_desc="gearsage_safe:final_external",
    )
    val_metrics = compute_binary_classification_metrics(val_labels, val_prob)
    external_metrics = compute_binary_classification_metrics(external_labels, external_prob)
    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, val_labels, val_prob)
    save_prediction_npz(run_dir / "phase2_external_predictions.npz", external_ids, external_labels, external_prob)
    torch.save(model.state_dict(), run_dir / "model.pt")

    summary = {
        "model_name": "gearsage_safe",
        "run_name": args.run_name,
        "phase1_train_size": int(train_ids.size),
        "phase1_val_size": int(val_ids.size),
        "phase2_external_size": int(external_ids.size),
        "phase1_val_auc_mean": float(val_metrics["auc"]),
        "phase1_val_pr_auc_mean": float(val_metrics["pr_auc"]),
        "phase1_val_ap_mean": float(val_metrics["ap"]),
        "phase2_external_auc_mean": float(external_metrics["auc"]),
        "phase2_external_pr_auc_mean": float(external_metrics["pr_auc"]),
        "phase2_external_ap_mean": float(external_metrics["ap"]),
        "seed_metrics": [
            {
                "seed": int(args.seed),
                "val_auc": float(val_metrics["auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
                "val_ap": float(val_metrics["ap"]),
                "external_auc": float(external_metrics["auc"]),
                "external_pr_auc": float(external_metrics["pr_auc"]),
                "external_ap": float(external_metrics["ap"]),
                "best_epoch": float(best_epoch),
                "train_log_path": str(log_path),
            }
        ],
        "config": {
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "dropout": float(args.dropout),
            "hops": int(args.hops),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "train_seed_batch_size": int(args.train_seed_batch_size),
            "eval_seed_batch_size": int(args.eval_seed_batch_size),
            "update_mode": str(args.update_mode),
            "initial_neighbor_decay": float(args.initial_neighbor_decay),
            "device": str(device),
        },
        "history": history,
        "phase1_val_predictions": str(run_dir / "phase1_val_predictions.npz"),
        "phase2_external_predictions": str(run_dir / "phase2_external_predictions.npz"),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Training finished: {run_dir}")


if __name__ == "__main__":
    main()
