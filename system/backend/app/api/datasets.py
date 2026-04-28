from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_ROOT
from app.database import get_db
from app.models import DatasetUpload, GraphEdge, InferenceResult, PersonNode, ProcessingTask
from app.schemas import (
    DatasetSummary,
    GraphEdgeItem,
    GraphNode,
    GraphResponse,
    InferenceResultItem,
    InferenceRunResponse,
    TaskResponse,
)
from app.services.demo_dataset import OFFICIAL_DATASET_LABELS, seed_official_validation_dataset
from app.services.inference_runner import (
    replace_inference_results,
    run_feature_fallback,
    run_full_model_subprocess,
)
from app.services.llm_mapper import heuristic_mapping
from app.services.synthetic_identity import synthetic_person_for_node


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetSummary])
def list_datasets(db: Session = Depends(get_db)) -> list[DatasetSummary]:
    datasets = db.scalars(select(DatasetUpload).order_by(DatasetUpload.created_at.desc())).all()
    return [
        DatasetSummary(
            id=item.id,
            name=item.name,
            original_filename=item.original_filename,
            row_count=item.row_count,
            status=item.status,
            created_at=item.created_at,
        )
        for item in datasets
    ]


@router.post("/upload", response_model=DatasetSummary)
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)) -> DatasetSummary:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only csv files are supported")
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    storage_path = UPLOAD_ROOT / safe_name
    counter = 1
    while storage_path.exists():
        storage_path = UPLOAD_ROOT / f"{storage_path.stem}_{counter}{storage_path.suffix}"
        counter += 1
    with storage_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    frame = pd.read_csv(storage_path)
    if frame.empty:
        raise HTTPException(status_code=400, detail="csv file is empty")
    mapping = heuristic_mapping(list(frame.columns))
    dataset = DatasetUpload(
        name=storage_path.stem,
        original_filename=safe_name,
        storage_path=str(storage_path),
        row_count=int(frame.shape[0]),
        status="normalized",
        mapping_json=mapping,
    )
    db.add(dataset)
    db.flush()

    _insert_nodes(db=db, dataset_id=dataset.id, frame=frame, mapping=mapping)
    _insert_edges(db=db, dataset_id=dataset.id, frame=frame, mapping=mapping)
    task = ProcessingTask(
        dataset_id=dataset.id,
        task_type="feature_processing",
        status="pending",
        progress=0.0,
        current_step="uploaded",
        message="CSV normalized into database records.",
    )
    db.add(task)
    db.commit()
    db.refresh(dataset)
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        original_filename=dataset.original_filename,
        row_count=dataset.row_count,
        status=dataset.status,
        created_at=dataset.created_at,
    )


@router.post("/demo/{dataset_name}", response_model=DatasetSummary)
def create_demo_dataset(
    dataset_name: str,
    limit: int = 120,
    db: Session = Depends(get_db),
) -> DatasetSummary:
    if dataset_name not in OFFICIAL_DATASET_LABELS:
        raise HTTPException(status_code=404, detail="official demo dataset not found")
    try:
        dataset = seed_official_validation_dataset(
            db=db,
            dataset_name=dataset_name,
            limit=max(20, min(int(limit), 180)),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        original_filename=dataset.original_filename,
        row_count=dataset.row_count,
        status=dataset.status,
        created_at=dataset.created_at,
    )


@router.get("/{dataset_id}/graph", response_model=GraphResponse)
def get_graph(dataset_id: int, db: Session = Depends(get_db)) -> GraphResponse:
    dataset = db.get(DatasetUpload, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    nodes = db.scalars(select(PersonNode).where(PersonNode.dataset_id == dataset_id).limit(300)).all()
    edges = db.scalars(select(GraphEdge).where(GraphEdge.dataset_id == dataset_id).limit(600)).all()
    inference_records = db.scalars(
        select(InferenceResult).where(InferenceResult.dataset_id == dataset_id)
    ).all()
    inference_by_node = {record.node_id: record for record in inference_records}
    graph_nodes = [
        _graph_node_from_record(node=node, inference=inference_by_node.get(node.node_id))
        for node in nodes
    ]
    graph_edges = [
        GraphEdgeItem(
            id=str(edge.id),
            source=edge.source_id,
            target=edge.target_id,
            edge_type=edge.edge_type,
            amount=edge.amount,
            timestamp=edge.timestamp,
        )
        for edge in edges
    ]
    return GraphResponse(dataset_id=dataset_id, nodes=graph_nodes, edges=graph_edges)


@router.post("/{dataset_id}/feature-task", response_model=TaskResponse)
def create_feature_task(dataset_id: int, db: Session = Depends(get_db)) -> TaskResponse:
    if db.get(DatasetUpload, dataset_id) is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    task = ProcessingTask(
        dataset_id=dataset_id,
        task_type="feature_processing",
        status="running",
        progress=0.1,
        current_step="graph_preview",
        message="Feature processing task created. Model-aligned processing will be connected next.",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return TaskResponse(
        id=task.id,
        dataset_id=task.dataset_id,
        task_type=task.task_type,
        status=task.status,
        progress=task.progress,
        current_step=task.current_step,
        message=task.message,
    )


@router.post("/{dataset_id}/infer", response_model=InferenceRunResponse)
def run_inference(dataset_id: int, db: Session = Depends(get_db)) -> InferenceRunResponse:
    dataset = db.get(DatasetUpload, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    nodes = db.scalars(select(PersonNode).where(PersonNode.dataset_id == dataset_id)).all()
    if not nodes:
        raise HTTPException(status_code=400, detail="dataset has no nodes")

    mapping = dict(dataset.mapping_json or {})
    source_dataset = mapping.get("source_dataset")
    source_node_ids = [str(value) for value in mapping.get("source_node_ids") or []]
    try:
        if mapping.get("source") == "official_validation" and source_dataset and source_node_ids:
            raw_scores = run_full_model_subprocess(
                dataset_name=str(source_dataset),
                node_ids=source_node_ids,
            )
        else:
            raw_scores = run_feature_fallback(nodes)
        stored = replace_inference_results(db=db, dataset_id=dataset_id, scores=raw_scores)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    db.add(
        ProcessingTask(
            dataset_id=dataset_id,
            task_type="inference",
            status="completed",
            progress=1.0,
            current_step="full_model_inference",
            message="DyRIFT-TGAT inference finished and results were saved.",
        )
    )
    db.commit()
    return _build_inference_response(db=db, dataset_id=dataset_id, records=stored)


@router.get("/{dataset_id}/inference-results", response_model=list[InferenceResultItem])
def list_inference_results(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> list[InferenceResultItem]:
    if db.get(DatasetUpload, dataset_id) is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    records = db.scalars(
        select(InferenceResult)
        .where(InferenceResult.dataset_id == dataset_id)
        .order_by(InferenceResult.risk_score.desc())
    ).all()
    return _result_items(db=db, dataset_id=dataset_id, records=records)


def _graph_node_from_record(node: PersonNode, inference: InferenceResult | None) -> GraphNode:
    color = "#2f80ed"
    risk_score = None
    risk_label = None
    if inference is not None:
        risk_score = float(inference.risk_score)
        risk_label = inference.risk_label
        color = "#d83b3b" if inference.risk_label == "suspicious" else "#2e9d57"
    return GraphNode(
        id=node.node_id,
        label=node.display_name,
        region=node.region,
        occupation=node.occupation,
        size=max(18, min(48, 18 + len(node.feature_json) + int((risk_score or 0.0) * 12))),
        color=color,
        risk_score=risk_score,
        risk_label=risk_label,
    )


def _build_inference_response(
    *,
    db: Session,
    dataset_id: int,
    records: list[InferenceResult],
) -> InferenceRunResponse:
    items = _result_items(db=db, dataset_id=dataset_id, records=records)
    abnormal = sum(1 for item in items if item.risk_label == "suspicious")
    normal = len(items) - abnormal
    return InferenceRunResponse(
        dataset_id=dataset_id,
        total_nodes=len(items),
        abnormal_nodes=abnormal,
        normal_nodes=normal,
        message=f"Inference finished: {abnormal} suspicious nodes, {normal} normal nodes.",
        results=items[:30],
    )


def _result_items(
    *,
    db: Session,
    dataset_id: int,
    records: list[InferenceResult],
) -> list[InferenceResultItem]:
    node_map = {
        node.node_id: node
        for node in db.scalars(select(PersonNode).where(PersonNode.dataset_id == dataset_id)).all()
    }
    items: list[InferenceResultItem] = []
    for record in sorted(records, key=lambda item: item.risk_score, reverse=True):
        node = node_map.get(record.node_id)
        explanation = dict(record.explanation_json or {})
        items.append(
            InferenceResultItem(
                node_id=record.node_id,
                display_name=node.display_name if node is not None else record.node_id,
                id_number=node.id_number if node is not None else "",
                region=node.region if node is not None else "",
                occupation=node.occupation if node is not None else "",
                risk_score=float(record.risk_score),
                risk_label=record.risk_label,
                reason=str(explanation.get("reason") or ""),
            )
        )
    return items


def _insert_nodes(*, db: Session, dataset_id: int, frame: pd.DataFrame, mapping: dict[str, Any]) -> None:
    node_column = mapping["node_id"]
    feature_columns = mapping.get("feature_columns") or []
    seen: set[str] = set()
    for row in frame.head(5000).to_dict(orient="records"):
        node_id = str(row.get(node_column))
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        person = synthetic_person_for_node(node_id)
        features = {
            column: _json_value(row.get(column))
            for column in feature_columns
            if _is_numeric(row.get(column))
        }
        db.add(
            PersonNode(
                dataset_id=dataset_id,
                node_id=node_id,
                raw_json={key: _json_value(value) for key, value in row.items()},
                feature_json=features,
                **person,
            )
        )


def _insert_edges(*, db: Session, dataset_id: int, frame: pd.DataFrame, mapping: dict[str, Any]) -> None:
    source_column = mapping.get("source_id")
    target_column = mapping.get("target_id")
    timestamp_column = mapping.get("timestamp")
    amount_column = mapping.get("amount")
    edge_type_column = mapping.get("edge_type")
    node_column = mapping["node_id"]

    if source_column and target_column:
        for row in frame.head(8000).to_dict(orient="records"):
            source = str(row.get(source_column))
            target = str(row.get(target_column))
            if not source or not target:
                continue
            db.add(
                GraphEdge(
                    dataset_id=dataset_id,
                    source_id=source,
                    target_id=target,
                    edge_type=str(row.get(edge_type_column) or "relation"),
                    amount=_float_or_none(row.get(amount_column)) if amount_column else None,
                    timestamp=str(row.get(timestamp_column)) if timestamp_column else None,
                )
            )
        return

    node_ids = [str(value) for value in frame[node_column].head(300).tolist()]
    for index in range(max(0, len(node_ids) - 1)):
        db.add(
            GraphEdge(
                dataset_id=dataset_id,
                source_id=node_ids[index],
                target_id=node_ids[index + 1],
                edge_type="preview_sequence",
            )
        )


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
        return pd.notna(value)
    except (TypeError, ValueError):
        return False


def _float_or_none(value: Any) -> float | None:
    return float(value) if _is_numeric(value) else None


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
