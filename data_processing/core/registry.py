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
    description: str = ""


DATASET_SPECS: dict[str, DatasetSpec] = {
    "xinye_dgraph": DatasetSpec(
        name="xinye_dgraph",
        display_name="XinYe DGraph",
        dataset_root_relative="data/raw/xinye_dgraph",
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
        description="Single-graph view over XinYe official phase1 for unified time-based train/val experiments.",
    ),
    "xinye_dgraph_trainval": DatasetSpec(
        name="xinye_dgraph_trainval",
        display_name="XinYe DGraph Train/Val Induced",
        dataset_root_relative="data/raw/xinye_dgraph_trainval",
        phase_filenames={
            "graph": "graph_gdata.npz",
            "phase1": "graph_gdata.npz",
            "phase2": "graph_gdata.npz",
        },
        label_names={
            0: "normal",
            1: "fraud",
        },
        default_artifacts=("graph",),
        split_style="single_graph",
        split_train_artifact="graph",
        split_val_artifact="graph",
        split_external_artifact=None,
        background_labels=(),
        strong_pairs=((2, 3), (6, 8), (15, 16)),
        description=(
            "XinYe phase1 induced subgraph restricted to the current train and validation "
            "nodes, used to test whether full-graph background structure adds noise."
        ),
    ),
    "xinye_dgraph_tv_ctx500k": DatasetSpec(
        name="xinye_dgraph_tv_ctx500k",
        display_name="XinYe DGraph Train/Val + 500k Context",
        dataset_root_relative="data/raw/xinye_dgraph_tv_ctx500k",
        phase_filenames={
            "graph": "graph_gdata.npz",
            "phase1": "graph_gdata.npz",
            "phase2": "graph_gdata.npz",
        },
        label_names={
            -100: "unlabeled_context",
            0: "normal",
            1: "fraud",
        },
        default_artifacts=("graph",),
        split_style="single_graph",
        split_train_artifact="graph",
        split_val_artifact="graph",
        split_external_artifact=None,
        background_labels=(),
        strong_pairs=((2, 3), (6, 8), (15, 16)),
        description="XinYe train/val nodes plus top-ranked one-hop context nodes.",
    ),
    "xinye_dgraph_tv_ctx800k": DatasetSpec(
        name="xinye_dgraph_tv_ctx800k",
        display_name="XinYe DGraph Train/Val + 800k Context",
        dataset_root_relative="data/raw/xinye_dgraph_tv_ctx800k",
        phase_filenames={
            "graph": "graph_gdata.npz",
            "phase1": "graph_gdata.npz",
            "phase2": "graph_gdata.npz",
        },
        label_names={
            -100: "unlabeled_context",
            0: "normal",
            1: "fraud",
        },
        default_artifacts=("graph",),
        split_style="single_graph",
        split_train_artifact="graph",
        split_val_artifact="graph",
        split_external_artifact=None,
        background_labels=(),
        strong_pairs=((2, 3), (6, 8), (15, 16)),
        description="XinYe train/val nodes plus top-ranked one-hop context nodes.",
    ),
    "xinye_dgraph_tv_1hop": DatasetSpec(
        name="xinye_dgraph_tv_1hop",
        display_name="XinYe DGraph Train/Val + Full One-Hop",
        dataset_root_relative="data/raw/xinye_dgraph_tv_1hop",
        phase_filenames={
            "graph": "graph_gdata.npz",
            "phase1": "graph_gdata.npz",
            "phase2": "graph_gdata.npz",
        },
        label_names={
            -100: "unlabeled_context",
            0: "normal",
            1: "fraud",
        },
        default_artifacts=("graph",),
        split_style="single_graph",
        split_train_artifact="graph",
        split_val_artifact="graph",
        split_external_artifact=None,
        background_labels=(),
        strong_pairs=((2, 3), (6, 8), (15, 16)),
        description="XinYe train/val nodes plus all one-hop context nodes.",
    ),
    "xinye_dgraph_tv_trainpast": DatasetSpec(
        name="xinye_dgraph_tv_trainpast",
        display_name="XinYe DGraph Train/Val + Train-Past Context",
        dataset_root_relative="data/raw/xinye_dgraph_tv_trainpast",
        phase_filenames={
            "graph": "graph_gdata.npz",
            "phase1": "graph_gdata.npz",
            "phase2": "graph_gdata.npz",
        },
        label_names={
            -100: "unlabeled_context",
            0: "normal",
            1: "fraud",
        },
        default_artifacts=("graph",),
        split_style="single_graph",
        split_train_artifact="graph",
        split_val_artifact="graph",
        split_external_artifact=None,
        background_labels=(),
        strong_pairs=((2, 3), (6, 8), (15, 16)),
        description=(
            "XinYe train/val nodes plus one-hop context selected only through training "
            "anchors and pre-threshold edges."
        ),
    ),
    "ellipticpp_transactions": DatasetSpec(
        name="ellipticpp_transactions",
        display_name="Elliptic++ Transactions",
        dataset_root_relative="data/raw/ellipticpp_transactions/prepared",
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
        description=(
            "Single full Elliptic++ transaction graph with unified time-based train/val split."
        ),
    ),
    "elliptic_transactions": DatasetSpec(
        name="elliptic_transactions",
        display_name="Elliptic Transactions",
        dataset_root_relative="data/raw/elliptic_transactions/prepared",
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
    return outputs_root / "analysis" / spec.name, outputs_root / "features" / spec.name
