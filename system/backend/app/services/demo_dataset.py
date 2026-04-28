from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import EXPERIMENT_ROOT
from app.models import DatasetUpload, GraphEdge, PersonNode, ProcessingTask
from app.services.synthetic_identity import synthetic_person_for_node


OFFICIAL_DATASET_LABELS = {
    "xinye_dgraph": "XinYe validation nodes",
    "elliptic_transactions": "Elliptic validation nodes",
    "ellipticpp_transactions": "Elliptic++ validation nodes",
}


def seed_official_validation_dataset(
    *,
    db: Session,
    dataset_name: str,
    limit: int = 120,
) -> DatasetUpload:
    if dataset_name not in OFFICIAL_DATASET_LABELS:
        supported = ", ".join(sorted(OFFICIAL_DATASET_LABELS))
        raise ValueError(f"unsupported official dataset: {dataset_name}. Supported: {supported}")

    existing = db.scalar(
        select(DatasetUpload).where(
            DatasetUpload.name == f"demo_{dataset_name}_validation",
            DatasetUpload.status == "official_validation",
        )
    )
    if existing is not None:
        return existing

    parameter_file = EXPERIMENT_ROOT / "configs" / "train" / f"{dataset_name}.json"
    train_payload = json.loads(parameter_file.read_text(encoding="utf-8-sig"))["train"]
    feature_dir = _resolve_experiment_path(str(train_payload["feature_dir"]))
    analysis_root = EXPERIMENT_ROOT / "outputs" / "analysis" / dataset_name
    split = json.loads((analysis_root / "recommended_split.json").read_text(encoding="utf-8-sig"))
    val_ids = _load_validation_ids(analysis_root=analysis_root, split=split)
    node_ids = np.asarray(val_ids[: max(1, int(limit))], dtype=np.int32)

    dataset = DatasetUpload(
        name=f"demo_{dataset_name}_validation",
        original_filename=f"{dataset_name}_validation.csv",
        storage_path=str(analysis_root / "recommended_split.json"),
        row_count=int(node_ids.size),
        status="official_validation",
        mapping_json={
            "source": "official_validation",
            "source_dataset": dataset_name,
            "source_node_ids": [int(value) for value in node_ids.tolist()],
            "feature_dir": str(feature_dir),
            "model": "full_dyrift_gnn/dyrift_gnn",
        },
    )
    db.add(dataset)
    db.flush()

    _insert_validation_nodes(db=db, dataset_id=dataset.id, node_ids=node_ids, feature_dir=feature_dir)
    inserted_edges = _insert_validation_edges(
        db=db,
        dataset_id=dataset.id,
        node_ids=node_ids,
        feature_dir=feature_dir,
    )
    if inserted_edges == 0:
        _insert_sequence_edges(db=db, dataset_id=dataset.id, node_ids=node_ids)

    db.add(
        ProcessingTask(
            dataset_id=dataset.id,
            task_type="feature_processing",
            status="completed",
            progress=1.0,
            current_step="official_feature_cache",
            message="Official validation nodes were loaded from the existing feature cache.",
        )
    )
    db.commit()
    db.refresh(dataset)
    return dataset


def _resolve_experiment_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = EXPERIMENT_ROOT / path
    return path.resolve()


def _load_validation_ids(*, analysis_root: Path, split: dict[str, Any]) -> np.ndarray:
    if "val_split" in split:
        return np.load(analysis_root / split["val_split"]["id_path"])
    if "phase1_time_split" in split:
        return np.load(analysis_root / split["phase1_time_split"]["val_id_path"])
    raise KeyError(f"Unsupported split format: {analysis_root / 'recommended_split.json'}")


def _insert_validation_nodes(
    *,
    db: Session,
    dataset_id: int,
    node_ids: np.ndarray,
    feature_dir: Path,
) -> None:
    feature_path = feature_dir / "graph" / "core_features.npy"
    core_features = np.load(feature_path, mmap_mode="r")
    for node_id in node_ids.tolist():
        node_key = str(int(node_id))
        person = synthetic_person_for_node(node_key)
        feature_row = np.asarray(core_features[int(node_id)], dtype=np.float32)
        feature_json = {
            f"core_feature_{index:02d}": _json_float(value)
            for index, value in enumerate(feature_row[:12].tolist())
        }
        db.add(
            PersonNode(
                dataset_id=dataset_id,
                node_id=node_key,
                raw_json={
                    "source_node_id": int(node_id),
                    "source": "official_validation",
                },
                feature_json=feature_json,
                **person,
            )
        )


def _insert_validation_edges(
    *,
    db: Session,
    dataset_id: int,
    node_ids: np.ndarray,
    feature_dir: Path,
) -> int:
    graph_root = feature_dir / "graph" / "graph"
    out_ptr = np.load(graph_root / "out_ptr.npy", mmap_mode="r")
    out_neighbors = np.load(graph_root / "out_neighbors.npy", mmap_mode="r")
    edge_type_path = graph_root / "out_edge_type.npy"
    edge_time_path = graph_root / "out_edge_timestamp.npy"
    out_edge_type = np.load(edge_type_path, mmap_mode="r") if edge_type_path.exists() else None
    out_edge_time = np.load(edge_time_path, mmap_mode="r") if edge_time_path.exists() else None
    node_set = {int(value) for value in node_ids.tolist()}
    inserted = 0
    for source in node_ids.tolist():
        start = int(out_ptr[int(source)])
        end = int(out_ptr[int(source) + 1])
        for edge_index in range(start, end):
            target = int(out_neighbors[edge_index])
            if target not in node_set:
                continue
            db.add(
                GraphEdge(
                    dataset_id=dataset_id,
                    source_id=str(int(source)),
                    target_id=str(target),
                    edge_type=(
                        f"rel_{int(out_edge_type[edge_index])}"
                        if out_edge_type is not None
                        else "official_relation"
                    ),
                    timestamp=(
                        str(int(out_edge_time[edge_index]))
                        if out_edge_time is not None
                        else None
                    ),
                )
            )
            inserted += 1
            if inserted >= 500:
                return inserted
    return inserted


def _insert_sequence_edges(*, db: Session, dataset_id: int, node_ids: np.ndarray) -> None:
    values = [str(int(value)) for value in node_ids.tolist()]
    for index in range(max(0, len(values) - 1)):
        db.add(
            GraphEdge(
                dataset_id=dataset_id,
                source_id=values[index],
                target_id=values[index + 1],
                edge_type="validation_sequence",
            )
        )


def _json_float(value: Any) -> float:
    if value is None:
        return 0.0
    result = float(value)
    if not np.isfinite(result):
        return 0.0
    return result
