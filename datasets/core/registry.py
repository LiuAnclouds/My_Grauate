from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DATASET_ENV_VAR = "GRADPROJ_ACTIVE_DATASET"
DEFAULT_DATASET_NAME = "xinye_dgraph"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    dataset_root_relative: str
    phase_filenames: dict[str, str]
    label_names: dict[int, str]
    default_artifacts: tuple[str, ...] = ("phase1", "phase2")
    split_style: str = "two_phase"
    split_train_artifact: str = "phase1"
    split_val_artifact: str = "phase1"
    split_external_artifact: str | None = "phase2"
    background_labels: tuple[int, ...] = ()
    strong_pairs: tuple[tuple[int, int], ...] = ()
    output_namespace: str | None = None
    description: str = ""

    @property
    def uses_legacy_output_layout(self) -> bool:
        return self.output_namespace is None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "xinye_dgraph": DatasetSpec(
        name="xinye_dgraph",
        display_name="XinYe DGraph",
        dataset_root_relative="datasets/raw/xinye_dgraph",
        phase_filenames={
            "graph": "phase1_gdata.npz",
            "phase1": "phase1_gdata.npz",
            "phase2": "phase2_gdata.npz",
        },
        label_names={
            -100: "test_holdout",
            0: "normal",
            1: "fraud",
            2: "background_2",
            3: "background_3",
        },
        default_artifacts=("graph",),
        split_style="single_graph",
        split_train_artifact="graph",
        split_val_artifact="graph",
        split_external_artifact=None,
        background_labels=(2, 3),
        strong_pairs=((2, 3), (6, 8), (15, 16)),
        output_namespace=None,
        description="Single-graph view over XinYe official phase1 for unified time-based train/val experiments.",
    ),
    "ellipticpp_transactions": DatasetSpec(
        name="ellipticpp_transactions",
        display_name="Elliptic++ Transactions",
        dataset_root_relative="datasets/raw/ellipticpp_transactions/prepared",
        phase_filenames={
            "graph": "graph_gdata.npz",
            "phase1": "phase1_gdata.npz",
            "phase2": "phase2_gdata.npz",
        },
        label_names={
            -100: "unknown",
            0: "licit",
            1: "illicit",
        },
        default_artifacts=("graph",),
        split_style="single_graph",
        split_train_artifact="graph",
        split_val_artifact="graph",
        split_external_artifact=None,
        background_labels=(),
        strong_pairs=(),
        output_namespace="ellipticpp_transactions",
        description=(
            "Single full Elliptic++ transaction graph with unified time-based train/val split."
        ),
    ),
    "elliptic_transactions": DatasetSpec(
        name="elliptic_transactions",
        display_name="Elliptic Transactions",
        dataset_root_relative="datasets/raw/elliptic_transactions/prepared",
        phase_filenames={
            "graph": "graph_gdata.npz",
            "phase1": "phase1_gdata.npz",
            "phase2": "phase2_gdata.npz",
        },
        label_names={
            -100: "unknown",
            0: "licit",
            1: "illicit",
        },
        default_artifacts=("graph",),
        split_style="single_graph",
        split_train_artifact="graph",
        split_val_artifact="graph",
        split_external_artifact=None,
        background_labels=(),
        strong_pairs=(),
        output_namespace="elliptic_transactions",
        description=(
            "Single full Elliptic Bitcoin AML graph with unified time-based train/val split."
        ),
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise KeyError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}") from exc


def get_active_dataset_name() -> str:
    return os.environ.get(DATASET_ENV_VAR, DEFAULT_DATASET_NAME).strip() or DEFAULT_DATASET_NAME


def get_active_dataset_spec() -> DatasetSpec:
    return get_dataset_spec(get_active_dataset_name())


def resolve_output_roots(repo_root: Path) -> tuple[Path, Path]:
    spec = get_active_dataset_spec()
    outputs_root = repo_root / "outputs"
    if spec.uses_legacy_output_layout:
        return outputs_root / "eda", outputs_root / "training"
    dataset_root = outputs_root / spec.output_namespace
    return dataset_root / "eda", dataset_root / "training"
