from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.core.contracts import save_prepared_phase
from data_processing.core.downloads import download_file
from data_processing.core.elliptic import (
    build_chronological_node_contracts,
    build_edge_arrays,
    build_full_graph_contract,
    load_pandas,
    map_elliptic_binary_labels,
    phase_summary,
)


SOURCE_URLS = {
    "classes": "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_classes.csv",
    "edgelist": "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_edgelist.csv",
    "features": "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_features.csv",
}
RAW_FILE_NAMES = {
    "classes": "elliptic_txs_classes.csv",
    "edgelist": "elliptic_txs_edgelist.csv",
    "features": "elliptic_txs_features.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the original Elliptic Bitcoin AML dataset into the shared "
            "phase1/phase2 anti-fraud contract."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "elliptic_transactions" / "raw",
        help="Directory for raw CSV files.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "elliptic_transactions" / "prepared",
        help="Directory where phase1_gdata.npz and phase2_gdata.npz will be written.",
    )
    parser.add_argument(
        "--phase1-max-step",
        type=int,
        default=34,
        help="Last time step included in phase1.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw CSV files even when local copies already exist.",
    )
    parser.add_argument(
        "--hf-mirror",
        action="append",
        default=["https://hf-mirror.com"],
        help="Hugging Face mirror base URL. Can be passed multiple times.",
    )
    return parser.parse_args()


def _download_raw_files(raw_dir: Path, force: bool, hf_mirror_bases: list[str]) -> dict[str, Path]:
    paths = {name: raw_dir / filename for name, filename in RAW_FILE_NAMES.items()}
    for name, url in SOURCE_URLS.items():
        download_file(
            url,
            paths[name],
            force=force,
            hf_mirror_bases=hf_mirror_bases,
            allow_direct_hf=False,
            retries=3,
            timeout_seconds=600,
        )
    return paths


def _load_raw_tables(paths: dict[str, Path]):
    pd = load_pandas()
    classes = pd.read_csv(paths["classes"], dtype={"txId": "int64", "class": "string"})
    edges = pd.read_csv(paths["edgelist"], dtype={"txId1": "int64", "txId2": "int64"})
    feature_col_count = int(pd.read_csv(paths["features"], header=None, nrows=1).shape[1])
    feature_dtype = {0: "int64", 1: "int16"}
    feature_dtype.update({idx: "float32" for idx in range(2, feature_col_count)})
    features = pd.read_csv(paths["features"], header=None, dtype=feature_dtype, low_memory=False)
    return classes, edges, features


def main() -> None:
    args = parse_args()
    raw_paths = _download_raw_files(
        args.raw_dir,
        force=bool(args.force_download),
        hf_mirror_bases=list(args.hf_mirror),
    )
    classes, edges, features = _load_raw_tables(raw_paths)

    node_ids = features.iloc[:, 0].to_numpy(dtype=np.int64, copy=False)
    time_steps = features.iloc[:, 1].to_numpy(dtype=np.int32, copy=False)
    x = features.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)

    class_series = classes.set_index("txId").reindex(node_ids)["class"]
    if class_series.isna().any():
        missing = int(class_series.isna().sum())
        raise RuntimeError(f"Missing class labels for {missing} transactions in elliptic_txs_features.csv.")
    labels = map_elliptic_binary_labels(class_series)
    edge_index, edge_timestamp = build_edge_arrays(edges, node_ids=node_ids, time_steps=time_steps)
    graph = build_full_graph_contract(
        x=x,
        y=labels,
        edge_index=edge_index,
        edge_timestamp=edge_timestamp,
    )
    phase1, phase2 = build_chronological_node_contracts(
        x=x,
        y=labels,
        time_steps=time_steps,
        edge_index=edge_index,
        edge_timestamp=edge_timestamp,
        phase1_max_step=int(args.phase1_max_step),
    )

    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    save_prepared_phase(args.prepared_dir / "graph_gdata.npz", graph)
    save_prepared_phase(args.prepared_dir / "phase1_gdata.npz", phase1)
    save_prepared_phase(args.prepared_dir / "phase2_gdata.npz", phase2)

    metadata = {
        "dataset": "elliptic_transactions",
        "source_urls": SOURCE_URLS,
        "class_mapping": {
            "1": "illicit -> 1",
            "2": "licit -> 0",
            "unknown": "unknown -> -100",
        },
        "phase1_max_step": int(args.phase1_max_step),
        "raw_feature_count": int(x.shape[1]),
        "includes_time_step_feature": True,
        "graph": phase_summary(graph),
        "phase2_full_history_graph": True,
        "phase1": phase_summary(phase1),
        "phase2": phase_summary(phase2),
    }
    (args.prepared_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"graph": metadata["graph"], "phase1": metadata["phase1"], "phase2": metadata["phase2"]}, indent=2))


if __name__ == "__main__":
    main()
