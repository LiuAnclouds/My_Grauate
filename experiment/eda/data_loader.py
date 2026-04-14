from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment.datasets.registry import get_active_dataset_spec, get_dataset_spec


ACTIVE_DATASET_SPEC = get_active_dataset_spec()
PHASE_FILENAMES = dict(ACTIVE_DATASET_SPEC.phase_filenames)
LABEL_NAMES = dict(ACTIVE_DATASET_SPEC.label_names)


@dataclass(frozen=True)
class PhaseData:
    phase: str
    path: Path
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


def resolve_dataset_path(phase: str, repo_root: Path | None = None) -> Path:
    if phase not in PHASE_FILENAMES:
        raise ValueError(f"Unsupported phase: {phase}")

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    spec = get_active_dataset_spec()
    dataset_root = repo_root / spec.dataset_root_relative
    matches = sorted(dataset_root.rglob(spec.phase_filenames[phase]))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {spec.phase_filenames[phase]} under {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple matches for {spec.phase_filenames[phase]}: {matches}"
        )
    return matches[0]


def _flatten_array(values: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    flat = values.reshape(-1)
    if dtype is not None:
        return flat.astype(dtype, copy=False)
    return flat


def _validate_phase_data(data: PhaseData) -> None:
    if data.x.ndim != 2:
        raise AssertionError(f"{data.phase}: x must be 2D, got {data.x.shape}")
    if data.y.shape[0] != data.x.shape[0]:
        raise AssertionError(f"{data.phase}: y/x node size mismatch")
    if data.edge_index.ndim != 2 or data.edge_index.shape[1] != 2:
        raise AssertionError(
            f"{data.phase}: edge_index must be (N_edge, 2), got {data.edge_index.shape}"
        )

    edge_count = data.edge_index.shape[0]
    if data.edge_type.shape[0] != edge_count or data.edge_timestamp.shape[0] != edge_count:
        raise AssertionError(f"{data.phase}: edge arrays length mismatch")

    train_unique = np.unique(data.train_mask)
    test_unique = np.unique(data.test_mask)
    if train_unique.size != data.train_mask.size:
        raise AssertionError(f"{data.phase}: duplicated ids in train_mask")
    if test_unique.size != data.test_mask.size:
        raise AssertionError(f"{data.phase}: duplicated ids in test_mask")
    if np.intersect1d(train_unique, test_unique, assume_unique=True).size:
        raise AssertionError(f"{data.phase}: train_mask and test_mask overlap")

    train_labels = np.unique(data.y[data.train_mask])
    if not set(train_labels.tolist()).issubset({0, 1}):
        raise AssertionError(
            f"{data.phase}: train labels must be subset of {{0, 1}}, got {train_labels}"
        )
    if not np.all(data.y[data.test_mask] == -100):
        raise AssertionError(f"{data.phase}: test labels must all be -100")


def load_phase(phase: str, repo_root: Path | None = None) -> PhaseData:
    phase_path = resolve_dataset_path(phase, repo_root=repo_root)
    npz = np.load(phase_path, allow_pickle=False)

    phase_data = PhaseData(
        phase=phase,
        path=phase_path,
        x=np.asarray(npz["x"]),
        y=_flatten_array(np.asarray(npz["y"]), dtype=np.int32),
        edge_index=np.asarray(npz["edge_index"], dtype=np.int32),
        edge_type=_flatten_array(np.asarray(npz["edge_type"]), dtype=np.int16),
        edge_timestamp=_flatten_array(np.asarray(npz["edge_timestamp"]), dtype=np.int32),
        train_mask=_flatten_array(np.asarray(npz["train_mask"]), dtype=np.int32),
        test_mask=_flatten_array(np.asarray(npz["test_mask"]), dtype=np.int32),
    )
    _validate_phase_data(phase_data)
    return phase_data
