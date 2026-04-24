from __future__ import annotations

from typing import Any

import numpy as np

from data_processing.core.contracts import PreparedPhaseContract


def load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Preparing Elliptic-style datasets requires pandas. Run inside the Graph conda environment."
        ) from exc
    return pd


def map_elliptic_binary_labels(raw_classes: Any) -> np.ndarray:
    pd = load_pandas()
    labels = pd.Series(raw_classes).astype(str).str.strip().str.lower()
    mapped = np.full(labels.shape[0], -100, dtype=np.int32)
    mapped[labels.to_numpy() == "1"] = 1
    mapped[labels.to_numpy() == "2"] = 0
    return mapped


def build_edge_arrays(edges, node_ids: np.ndarray, time_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(node_ids.tolist())}
    raw_src = edges["txId1"].to_numpy(dtype=np.int64, copy=False)
    raw_dst = edges["txId2"].to_numpy(dtype=np.int64, copy=False)
    src_idx = np.fromiter(
        (node_to_idx.get(int(tx_id), -1) for tx_id in raw_src),
        dtype=np.int32,
        count=raw_src.shape[0],
    )
    dst_idx = np.fromiter(
        (node_to_idx.get(int(tx_id), -1) for tx_id in raw_dst),
        dtype=np.int32,
        count=raw_dst.shape[0],
    )
    valid_edge = (src_idx >= 0) & (dst_idx >= 0)
    edge_index = np.column_stack([src_idx[valid_edge], dst_idx[valid_edge]]).astype(np.int32, copy=False)
    edge_timestamp = np.maximum(time_steps[edge_index[:, 0]], time_steps[edge_index[:, 1]]).astype(
        np.int32,
        copy=False,
    )
    return edge_index, edge_timestamp


def build_chronological_node_contracts(
    *,
    x: np.ndarray,
    y: np.ndarray,
    time_steps: np.ndarray,
    edge_index: np.ndarray,
    edge_timestamp: np.ndarray,
    phase1_max_step: int,
) -> tuple[PreparedPhaseContract, PreparedPhaseContract]:
    phase1 = _build_phase1_contract(
        x=x,
        y=y,
        time_steps=time_steps,
        edge_index=edge_index,
        edge_timestamp=edge_timestamp,
        phase1_max_step=phase1_max_step,
    )
    phase2 = _build_phase2_contract(
        x=x,
        y=y,
        time_steps=time_steps,
        edge_index=edge_index,
        edge_timestamp=edge_timestamp,
        phase1_max_step=phase1_max_step,
    )
    return phase1, phase2


def build_full_graph_contract(
    *,
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_timestamp: np.ndarray,
) -> PreparedPhaseContract:
    train_mask = np.flatnonzero(np.isin(y, (0, 1))).astype(np.int32, copy=False)
    test_mask = np.flatnonzero(y == -100).astype(np.int32, copy=False)
    return PreparedPhaseContract(
        x=x.astype(np.float32, copy=False),
        y=y.astype(np.int32, copy=False),
        edge_index=edge_index.astype(np.int32, copy=False),
        edge_type=np.ones(edge_index.shape[0], dtype=np.int16),
        edge_timestamp=edge_timestamp.astype(np.int32, copy=False),
        train_mask=train_mask,
        test_mask=test_mask,
    )


def _build_phase1_contract(
    x: np.ndarray,
    y: np.ndarray,
    time_steps: np.ndarray,
    edge_index: np.ndarray,
    edge_timestamp: np.ndarray,
    phase1_max_step: int,
) -> PreparedPhaseContract:
    phase1_nodes = np.flatnonzero(time_steps <= int(phase1_max_step)).astype(np.int32, copy=False)
    reindex = np.full(time_steps.shape[0], -1, dtype=np.int32)
    reindex[phase1_nodes] = np.arange(phase1_nodes.shape[0], dtype=np.int32)
    phase1_edge_mask = (
        (edge_timestamp <= int(phase1_max_step))
        & (reindex[edge_index[:, 0]] >= 0)
        & (reindex[edge_index[:, 1]] >= 0)
    )
    phase1_edge_index = np.column_stack(
        [
            reindex[edge_index[phase1_edge_mask, 0]],
            reindex[edge_index[phase1_edge_mask, 1]],
        ]
    ).astype(np.int32, copy=False)
    phase1_y = y[phase1_nodes].astype(np.int32, copy=False)
    train_mask = np.flatnonzero(np.isin(phase1_y, (0, 1))).astype(np.int32, copy=False)
    test_mask = np.flatnonzero(phase1_y == -100).astype(np.int32, copy=False)
    return PreparedPhaseContract(
        x=x[phase1_nodes].astype(np.float32, copy=False),
        y=phase1_y,
        edge_index=phase1_edge_index,
        edge_type=np.ones(phase1_edge_index.shape[0], dtype=np.int16),
        edge_timestamp=edge_timestamp[phase1_edge_mask].astype(np.int32, copy=False),
        train_mask=train_mask,
        test_mask=test_mask,
    )


def _build_phase2_contract(
    x: np.ndarray,
    y: np.ndarray,
    time_steps: np.ndarray,
    edge_index: np.ndarray,
    edge_timestamp: np.ndarray,
    phase1_max_step: int,
) -> PreparedPhaseContract:
    external_eval_mask = (time_steps > int(phase1_max_step)) & np.isin(y, (0, 1))
    phase2_y = np.full(y.shape[0], -100, dtype=np.int32)
    phase2_y[external_eval_mask] = y[external_eval_mask]
    train_mask = np.flatnonzero(external_eval_mask).astype(np.int32, copy=False)
    test_mask = np.flatnonzero(~external_eval_mask).astype(np.int32, copy=False)
    return PreparedPhaseContract(
        x=x.astype(np.float32, copy=False),
        y=phase2_y,
        edge_index=edge_index.astype(np.int32, copy=False),
        edge_type=np.ones(edge_index.shape[0], dtype=np.int16),
        edge_timestamp=edge_timestamp.astype(np.int32, copy=False),
        train_mask=train_mask,
        test_mask=test_mask,
    )


def phase_summary(contract: PreparedPhaseContract) -> dict[str, int | float]:
    train_labels = contract.y[contract.train_mask]
    return {
        "num_nodes": contract.num_nodes,
        "num_edges": contract.num_edges,
        "raw_feature_count": int(contract.x.shape[1]),
        "train_size": int(contract.train_mask.size),
        "test_size": int(contract.test_mask.size),
        "positive_count": int(np.sum(train_labels == 1)),
        "negative_count": int(np.sum(train_labels == 0)),
        "positive_rate": float(np.mean(train_labels == 1)) if train_labels.size else 0.0,
    }
