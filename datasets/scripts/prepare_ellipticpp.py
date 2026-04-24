from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.core.contracts import save_prepared_phase
from datasets.core.downloads import download_file
from datasets.core.elliptic import (
    build_chronological_node_contracts,
    build_edge_arrays,
    build_full_graph_contract,
    load_pandas,
    map_elliptic_binary_labels,
    phase_summary,
)


SOURCE_URLS = {
    "classes": "https://media.githubusercontent.com/media/git-disl/EllipticPlusPlus/main/Transactions%20Dataset/txs_classes.csv",
    "edgelist": "https://media.githubusercontent.com/media/git-disl/EllipticPlusPlus/main/Transactions%20Dataset/txs_edgelist.csv",
    "features": "https://media.githubusercontent.com/media/git-disl/EllipticPlusPlus/main/Transactions%20Dataset/txs_features.csv",
}
RAW_FILE_NAMES = {
    "classes": "txs_classes.csv",
    "edgelist": "txs_edgelist.csv",
    "features": "txs_features.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Elliptic++ Transactions into the shared phase1/phase2 dynamic-graph "
            "anti-fraud contract."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "raw" / "ellipticpp_transactions" / "raw",
        help="Directory for downloaded raw CSV files.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "raw" / "ellipticpp_transactions" / "prepared",
        help="Directory where phase1_gdata.npz and phase2_gdata.npz will be written.",
    )
    parser.add_argument(
        "--phase1-max-step",
        type=int,
        default=34,
        help=(
            "Last time step included in phase1. The default follows the common Elliptic "
            "early-history split and leaves later labeled nodes for phase2 external evaluation."
        ),
    )
    parser.add_argument(
        "--drop-time-step-feature",
        action="store_true",
        help="Exclude the original 'Time step' column from x. Edge timestamps are always retained.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw CSV files even when local copies already exist.",
    )
    return parser.parse_args()


def _download_raw_files(raw_dir: Path, force: bool) -> dict[str, Path]:
    paths = {name: raw_dir / filename for name, filename in RAW_FILE_NAMES.items()}
    for name, url in SOURCE_URLS.items():
        download_file(url, paths[name], force=force, retries=3, timeout_seconds=600)
    return paths


def _load_raw_tables(paths: dict[str, Path], include_time_step_feature: bool):
    pd = load_pandas()
    feature_header = pd.read_csv(paths["features"], nrows=0)
    feature_dtype = {
        column: "float32"
        for column in feature_header.columns
        if column not in {"txId", "Time step"}
    }
    feature_dtype.update({"txId": "int64", "Time step": "int16"})
    features = pd.read_csv(paths["features"], dtype=feature_dtype, low_memory=False)
    classes = pd.read_csv(paths["classes"], dtype={"txId": "int64", "class": "int8"})
    edges = pd.read_csv(paths["edgelist"], dtype={"txId1": "int64", "txId2": "int64"})

    feature_columns = [column for column in features.columns if column != "txId"]
    if not include_time_step_feature:
        feature_columns = [column for column in feature_columns if column != "Time step"]
    return features, classes, edges, feature_columns


def main() -> None:
    args = parse_args()
    raw_paths = _download_raw_files(args.raw_dir, force=bool(args.force_download))
    features, classes, edges, feature_columns = _load_raw_tables(
        raw_paths,
        include_time_step_feature=not bool(args.drop_time_step_feature),
    )

    node_ids = features["txId"].to_numpy(dtype=np.int64, copy=False)
    time_steps = features["Time step"].to_numpy(dtype=np.int32, copy=False)
    class_series = classes.set_index("txId").reindex(node_ids)["class"]
    if class_series.isna().any():
        missing = int(class_series.isna().sum())
        raise RuntimeError(f"Missing class labels for {missing} transactions in txs_features.csv.")
    labels = map_elliptic_binary_labels(class_series)
    x = features[feature_columns].to_numpy(dtype=np.float32, copy=True)
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
        "dataset": "ellipticpp_transactions",
        "source_urls": SOURCE_URLS,
        "class_mapping": {
            "1": "illicit -> 1",
            "2": "licit -> 0",
            "3": "unknown -> -100",
        },
        "phase1_max_step": int(args.phase1_max_step),
        "feature_columns": feature_columns,
        "graph": phase_summary(graph),
        "phase2_full_history_graph": True,
        "phase1": phase_summary(phase1),
        "phase2": phase_summary(phase2),
    }
    (args.prepared_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: metadata[key] for key in ("graph", "phase1", "phase2")}, indent=2))


if __name__ == "__main__":
    main()
