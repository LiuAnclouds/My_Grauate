from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PreparedGraphContract:
    x: np.ndarray
    y: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_timestamp: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[0])


PreparedPhaseContract = PreparedGraphContract


def _flatten(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape(-1)


def validate_prepared_graph(phase: str, contract: PreparedGraphContract) -> None:
    x = np.asarray(contract.x, dtype=np.float32)
    y = _flatten(contract.y, np.int32)
    edge_index = np.asarray(contract.edge_index, dtype=np.int32)
    edge_type = _flatten(contract.edge_type, np.int16)
    edge_timestamp = _flatten(contract.edge_timestamp, np.int32)
    train_mask = _flatten(contract.train_mask, np.int32)
    test_mask = _flatten(contract.test_mask, np.int32)

    if x.ndim != 2:
        raise AssertionError(f"{phase}: x must be 2D, got {x.shape}")
    if y.shape[0] != x.shape[0]:
        raise AssertionError(f"{phase}: y/x node size mismatch")
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise AssertionError(f"{phase}: edge_index must be (N_edge, 2), got {edge_index.shape}")
    if edge_type.shape[0] != edge_index.shape[0]:
        raise AssertionError(f"{phase}: edge_type length mismatch")
    if edge_timestamp.shape[0] != edge_index.shape[0]:
        raise AssertionError(f"{phase}: edge_timestamp length mismatch")

    num_nodes = x.shape[0]
    if edge_index.size:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise AssertionError(f"{phase}: edge_index contains out-of-range node ids")
    if train_mask.size:
        if int(train_mask.min()) < 0 or int(train_mask.max()) >= num_nodes:
            raise AssertionError(f"{phase}: train_mask contains out-of-range node ids")
    if test_mask.size:
        if int(test_mask.min()) < 0 or int(test_mask.max()) >= num_nodes:
            raise AssertionError(f"{phase}: test_mask contains out-of-range node ids")

    if np.unique(train_mask).size != train_mask.size:
        raise AssertionError(f"{phase}: duplicated ids in train_mask")
    if np.unique(test_mask).size != test_mask.size:
        raise AssertionError(f"{phase}: duplicated ids in test_mask")
    if np.intersect1d(train_mask, test_mask, assume_unique=True).size:
        raise AssertionError(f"{phase}: train_mask and test_mask overlap")

    train_labels = np.unique(y[train_mask]) if train_mask.size else np.empty(0, dtype=np.int32)
    if train_labels.size and not set(train_labels.tolist()).issubset({0, 1}):
        raise AssertionError(
            f"{phase}: train labels must be subset of {{0, 1}}, got {train_labels.tolist()}"
        )
    if test_mask.size and not np.all(y[test_mask] == -100):
        raise AssertionError(f"{phase}: test labels must all be -100")


def save_prepared_graph(path: Path, contract: PreparedGraphContract) -> None:
    validate_prepared_graph(path.stem, contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.asarray(contract.x, dtype=np.float32),
        y=_flatten(contract.y, np.int32),
        edge_index=np.asarray(contract.edge_index, dtype=np.int32),
        edge_type=_flatten(contract.edge_type, np.int16),
        edge_timestamp=_flatten(contract.edge_timestamp, np.int32),
        train_mask=_flatten(contract.train_mask, np.int32),
        test_mask=_flatten(contract.test_mask, np.int32),
    )


validate_prepared_phase = validate_prepared_graph
save_prepared_phase = save_prepared_graph
