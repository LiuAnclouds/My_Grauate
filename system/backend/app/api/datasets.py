from __future__ import annotations

import shutil
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_ROOT
from app.database import get_db
from app.models import DatasetUpload, GraphEdge, InferenceResult, PersonNode, ProcessingEvent, ProcessingTask
from app.schemas import (
    DatasetSummary,
    DatasetUpdateRequest,
    GraphEdgeItem,
    GraphNode,
    GraphResponse,
    InferenceResultItem,
    InferenceRunResponse,
    MappingResponse,
    ProcessingEventItem,
    TaskResponse,
    TaskTimelineResponse,
)
from app.services.demo_dataset import OFFICIAL_DATASET_LABELS, OFFICIAL_DATASET_PROFILES, seed_official_validation_dataset
from app.services.inference_runner import (
    replace_inference_results,
    run_feature_fallback,
    run_full_model_subprocess,
)
from app.services.llm_mapper import infer_mapping
from app.services.pipeline_story import (
    append_event,
    build_dataset_summary,
    build_feature_task_story,
    build_graph_task_story,
    build_inference_story,
    create_task,
    latest_task,
    reset_task_events,
    timeline_events,
)
from app.services.synthetic_identity import synthetic_person_for_node


router = APIRouter(prefix="/datasets", tags=["datasets"])


FRAUD_EVENT_PROFILES = [
    "异常资金快进快出",
    "多账户协同套现",
    "虚假商户交易团伙",
    "高频小额试探交易",
    "跨区域异常转移",
    "账号接管风险事件",
    "疑似洗钱链路扩散",
    "空壳企业关联交易",
    "设备集群异常注册",
    "黑灰产中介撮合",
]


@router.get("", response_model=list[DatasetSummary])
def list_datasets(db: Session = Depends(get_db)) -> list[DatasetSummary]:
    datasets = db.scalars(select(DatasetUpload).order_by(DatasetUpload.created_at.desc())).all()
    return [_dataset_summary(item, db=db) for item in datasets]


@router.post("/upload", response_model=DatasetSummary)
def upload_dataset(
    file: UploadFile = File(...),
    network_name: str | None = Form(None),
    event_name: str | None = Form(None),
    use_llm: bool = Form(False),
    db: Session = Depends(get_db),
) -> DatasetSummary:
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
    mapping, mapping_method, mapping_message = infer_mapping(
        headers=list(frame.columns),
        sample_rows=frame.head(5).to_dict(orient="records"),
        use_llm=bool(use_llm),
    )
    mapping = {
        **mapping,
        "mapping_method": mapping_method,
        "mapping_message": mapping_message,
        "source": "uploaded_csv",
    }
    business_name = _normalize_business_name(network_name) or _business_name_for_upload(storage_path.stem)
    fraud_event = _normalize_event_name(event_name, seed=storage_path.stem)
    dataset = DatasetUpload(
        name=storage_path.stem,
        original_filename=safe_name,
        storage_path=str(storage_path),
        row_count=int(frame.shape[0]),
        status="normalized",
        mapping_json=mapping,
        summary_json={},
    )
    db.add(dataset)
    db.flush()

    business_id = _business_id_for(dataset.id)
    inserted_nodes = _insert_nodes(
        db=db,
        dataset_id=dataset.id,
        frame=frame,
        mapping=mapping,
        business_id=business_id,
        fraud_event=fraud_event,
    )
    inserted_edges = _insert_edges(db=db, dataset_id=dataset.id, frame=frame, mapping=mapping)
    dataset.summary_json = build_dataset_summary(
        node_count=inserted_nodes,
        edge_count=inserted_edges,
        mapping=mapping,
        source="uploaded_csv",
    )
    dataset.summary_json.update(
        {
            "business_id": business_id,
            "business_name": business_name,
            "source_description": "用户接入业务网络",
            "technical_name": storage_path.stem,
            "ingest_mode": "uploaded_csv",
            "fraud_event": fraud_event,
            "risk_object_type": "person",
            "identity_mode": "synthetic_person",
        }
    )
    task = create_task(
        db=db,
        dataset_id=dataset.id,
        task_type="feature_processing",
        status="pending",
        progress=0.0,
        current_step="uploaded",
        message="数据资产已完成入库，等待启动分析任务。",
        summary=dataset.summary_json,
    )
    append_event(
        db=db,
        dataset_id=dataset.id,
        task_id=task.id,
        stage="ingestion",
        step_key="csv_uploaded",
        title="数据资产接入完成",
        detail=f"已识别 {inserted_nodes} 个对象、{inserted_edges} 条关系，解析方式：{mapping_method}。",
        progress=0.02,
        metrics=dataset.summary_json,
    )
    db.commit()
    db.refresh(dataset)
    return _dataset_summary(dataset, db=db)


@router.patch("/{dataset_id}", response_model=DatasetSummary)
def update_dataset(
    dataset_id: int,
    payload: DatasetUpdateRequest,
    db: Session = Depends(get_db),
) -> DatasetSummary:
    dataset = db.get(DatasetUpload, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    summary = dict(dataset.summary_json or {})
    summary["business_name"] = payload.business_name.strip()
    dataset.summary_json = summary
    db.commit()
    db.refresh(dataset)
    return _dataset_summary(dataset, db=db)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)) -> dict[str, str]:
    dataset = db.get(DatasetUpload, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    for model in (InferenceResult, GraphEdge, PersonNode):
        db.execute(delete(model).where(model.dataset_id == dataset_id))
    db.execute(delete(ProcessingEvent).where(ProcessingEvent.dataset_id == dataset_id))
    db.execute(delete(ProcessingTask).where(ProcessingTask.dataset_id == dataset_id))
    storage_path = Path(dataset.storage_path)
    db.delete(dataset)
    db.commit()
    if storage_path.exists() and storage_path.is_file() and storage_path.parent == UPLOAD_ROOT:
        storage_path.unlink(missing_ok=True)
    return {"message": "business network deleted"}


@router.get("/{dataset_id}/mapping", response_model=MappingResponse)
def get_mapping(dataset_id: int, db: Session = Depends(get_db)) -> MappingResponse:
    dataset = db.get(DatasetUpload, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    mapping = dict(dataset.mapping_json or {})
    return MappingResponse(
        dataset_id=dataset.id,
        mapping=mapping,
        method=str(mapping.get("mapping_method") or mapping.get("source") or "stored"),
        message=str(mapping.get("mapping_message") or "字段映射已保存。"),
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
    return _dataset_summary(dataset, db=db)


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
    timeline = timeline_events(db=db, dataset_id=dataset_id)
    focus_node_ids = {
        event.focus_node_id for event in timeline if event.focus_node_id is not None
    }
    focus_pairs = {
        (event.focus_node_id, neighbor)
        for event in timeline
        if event.focus_node_id is not None
        for neighbor in event.focus_neighbor_ids
    }
    graph_nodes = [
        _graph_node_from_record(
            node=node,
            inference=inference_by_node.get(node.node_id),
            is_recent_focus=node.node_id in focus_node_ids,
        )
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
            highlighted=(edge.source_id, edge.target_id) in focus_pairs
            or (edge.target_id, edge.source_id) in focus_pairs,
        )
        for edge in edges
    ]
    return GraphResponse(
        dataset_id=dataset_id,
        nodes=graph_nodes,
        edges=graph_edges,
        summary=_presentation_summary(dataset),
    )


@router.post("/{dataset_id}/graph-task", response_model=TaskResponse)
def create_graph_task(dataset_id: int, db: Session = Depends(get_db)) -> TaskResponse:
    dataset = db.get(DatasetUpload, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    nodes = db.scalars(select(PersonNode).where(PersonNode.dataset_id == dataset_id)).all()
    edges = db.scalars(select(GraphEdge).where(GraphEdge.dataset_id == dataset_id)).all()
    if not nodes:
        raise HTTPException(status_code=400, detail="dataset has no nodes")

    reset_task_events(db=db, dataset_id=dataset_id, task_type="graph_construction")
    task = create_task(
        db=db,
        dataset_id=dataset_id,
        task_type="graph_construction",
        status="running",
        progress=0.08,
        current_step="start_graph_construction",
        message="正在构建关系图谱。",
        summary=_presentation_summary(dataset),
    )
    build_graph_task_story(db=db, dataset_id=dataset_id, task=task, nodes=nodes, edges=edges)
    dataset.status = "graph_ready"
    db.commit()
    db.refresh(task)
    return _task_response(task)


@router.post("/{dataset_id}/feature-task", response_model=TaskResponse)
def create_feature_task(dataset_id: int, db: Session = Depends(get_db)) -> TaskResponse:
    dataset = db.get(DatasetUpload, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    nodes = db.scalars(select(PersonNode).where(PersonNode.dataset_id == dataset_id)).all()
    edges = db.scalars(select(GraphEdge).where(GraphEdge.dataset_id == dataset_id)).all()
    reset_task_events(db=db, dataset_id=dataset_id, task_type="feature_processing")
    task = create_task(
        db=db,
        dataset_id=dataset_id,
        task_type="feature_processing",
        status="running",
        progress=0.06,
        current_step="start_feature_processing",
        message="已启动分析准备流程，正在组织关系网络与特征输入。",
        summary=_presentation_summary(dataset),
    )
    build_feature_task_story(db=db, dataset_id=dataset_id, task=task, nodes=nodes, edges=edges)
    dataset.status = "feature_ready"
    db.commit()
    db.refresh(task)
    return _task_response(task)


@router.get("/{dataset_id}/timeline", response_model=TaskTimelineResponse)
def get_timeline(dataset_id: int, db: Session = Depends(get_db)) -> TaskTimelineResponse:
    if db.get(DatasetUpload, dataset_id) is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    task = latest_task(db=db, dataset_id=dataset_id)
    events = timeline_events(db=db, dataset_id=dataset_id, task_id=task.id if task is not None else None)
    return TaskTimelineResponse(
        dataset_id=dataset_id,
        task=_task_response(task) if task is not None else None,
        events=[
            ProcessingEventItem(
                id=event.id,
                stage=event.stage,
                step_key=event.step_key,
                title=event.title,
                detail=event.detail,
                progress=event.progress,
                focus_node_id=event.focus_node_id,
                focus_neighbor_ids=[str(value) for value in (event.focus_neighbor_ids or [])],
                top_features=[str(value) for value in (event.top_features or [])],
                metrics=dict(event.metrics_json or {}),
                created_at=event.created_at,
            )
            for event in events
        ],
    )


@router.post("/{dataset_id}/infer", response_model=InferenceRunResponse)
def run_inference(dataset_id: int, db: Session = Depends(get_db)) -> InferenceRunResponse:
    dataset = db.get(DatasetUpload, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    nodes = db.scalars(select(PersonNode).where(PersonNode.dataset_id == dataset_id)).all()
    if not nodes:
        raise HTTPException(status_code=400, detail="dataset has no nodes")

    reset_task_events(db=db, dataset_id=dataset_id, task_type="inference")
    task = create_task(
        db=db,
        dataset_id=dataset_id,
        task_type="inference",
        status="running",
        progress=0.04,
        current_step="loading_model",
        message="正在执行风险识别并生成对象级结果。",
        summary=_presentation_summary(dataset),
    )

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
        build_inference_story(db=db, dataset_id=dataset_id, task=task, records=stored, nodes=nodes)
        dataset.status = "inference_completed"
    except Exception as exc:
        task.status = "failed"
        task.progress = 1.0
        task.current_step = "failed"
        task.message = f"风险识别失败：{exc}"
        append_event(
            db=db,
            dataset_id=dataset_id,
            task_id=task.id,
            stage="inference",
            step_key="failed",
            title="风险识别失败",
            detail=str(exc),
            progress=1.0,
            metrics={"error": str(exc)},
        )
        db.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    db.commit()
    return _build_inference_response(db=db, dataset_id=dataset_id, records=stored, task_id=task.id)


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


def _dataset_summary(dataset: DatasetUpload, *, db: Session | None = None) -> DatasetSummary:
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        original_filename=dataset.original_filename,
        row_count=dataset.row_count,
        status=dataset.status,
        created_at=dataset.created_at,
        summary=_presentation_summary(dataset, db=db),
    )


def _presentation_summary(dataset: DatasetUpload, *, db: Session | None = None) -> dict[str, Any]:
    summary = dict(dataset.summary_json or {})
    mapping = dict(dataset.mapping_json or {})
    if mapping.get("source") == "official_validation":
        dataset_key = str(mapping.get("source_dataset") or "")
        profile = OFFICIAL_DATASET_PROFILES.get(dataset_key)
        if profile is not None:
            summary.setdefault("business_name", profile["business_name"])
            summary.setdefault("source_description", profile["source_description"])
            summary.setdefault("technical_name", dataset_key)
            summary.setdefault("technical_label", profile["technical_label"])
    else:
        summary.setdefault("business_name", f"业务网络 {_business_id_for(dataset.id)}")
        summary.setdefault("business_id", _business_id_for(dataset.id))
        summary.setdefault("source_description", "用户接入业务网络")
        summary.setdefault("technical_name", dataset.name)
        summary.setdefault("risk_object_type", "person")
    summary.setdefault("business_id", _business_id_for(dataset.id))
    summary.setdefault("risk_object_type", "person")
    if db is not None:
        if summary.get("node_count") is None:
            summary["node_count"] = int(
                db.scalar(select(func.count()).select_from(PersonNode).where(PersonNode.dataset_id == dataset.id)) or 0
            )
        if summary.get("edge_count") is None:
            summary["edge_count"] = int(
                db.scalar(select(func.count()).select_from(GraphEdge).where(GraphEdge.dataset_id == dataset.id)) or 0
            )
    return summary


def _task_response(task: Any) -> TaskResponse:
    return TaskResponse(
        id=task.id,
        dataset_id=task.dataset_id,
        task_type=task.task_type,
        status=task.status,
        progress=task.progress,
        current_step=task.current_step,
        message=task.message,
        summary=dict(task.summary_json or {}),
    )


def _graph_node_from_record(
    *,
    node: PersonNode,
    inference: InferenceResult | None,
    is_recent_focus: bool,
) -> GraphNode:
    color = "#3b82f6" if is_recent_focus else "#5b6b7f"
    risk_score = None
    risk_label = None
    if inference is not None:
        risk_score = float(inference.risk_score)
        risk_label = inference.risk_label
        color = "#d14343" if inference.risk_label == "suspicious" else "#2f9e62"
    if is_recent_focus:
        color = "#2f80ed"
    source_type = str((node.raw_json or {}).get("source") or "dataset")
    return GraphNode(
        id=node.node_id,
        label=node.display_name,
        region=node.region,
        occupation=node.occupation,
        size=max(18, min(54, 18 + len(node.feature_json) + int((risk_score or 0.0) * 16))),
        color=color,
        risk_score=risk_score,
        risk_label=risk_label,
        source_type=source_type,
        feature_count=len(node.feature_json or {}),
    )


def _build_inference_response(
    *,
    db: Session,
    dataset_id: int,
    records: list[InferenceResult],
    task_id: int | None,
) -> InferenceRunResponse:
    items = _result_items(db=db, dataset_id=dataset_id, records=records)
    abnormal = sum(1 for item in items if item.risk_label == "suspicious")
    normal = len(items) - abnormal
    return InferenceRunResponse(
        dataset_id=dataset_id,
        total_nodes=len(items),
        abnormal_nodes=abnormal,
        normal_nodes=normal,
        message=f"风险识别完成：高风险对象 {abnormal} 个，低风险对象 {normal} 个。",
        results=items[:30],
        task_id=task_id,
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
                support_neighbors=[
                    str(value) for value in (explanation.get("support_neighbors") or [])
                ],
                top_features=[str(value) for value in (explanation.get("top_features") or [])],
            )
        )
    return items


def _business_id_for(dataset_id: int) -> str:
    return f"BN-{dataset_id:04d}"


def _business_name_for_upload(_raw_name: str) -> str:
    return "未命名业务网络"


def _normalize_business_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text[:80] if text else None


def _normalize_event_name(value: str | None, *, seed: str) -> str:
    if value is not None and value.strip():
        return value.strip()[:80]
    index = _stable_bucket(seed, len(FRAUD_EVENT_PROFILES))
    return FRAUD_EVENT_PROFILES[index]


def _gang_id_for(*, dataset_id: int, node_id: str, fraud_event: str) -> str:
    bucket = _stable_bucket(f"{dataset_id}:{fraud_event}:{node_id}", 12) + 1
    return f"FG-{dataset_id:04d}-{bucket:02d}"


def _stable_bucket(value: str, size: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % size


def _insert_nodes(
    *,
    db: Session,
    dataset_id: int,
    frame: pd.DataFrame,
    mapping: dict[str, Any],
    business_id: str,
    fraud_event: str,
) -> int:
    node_column = mapping["node_id"]
    feature_columns = mapping.get("feature_columns") or []
    seen: set[str] = set()
    inserted = 0
    for row in frame.head(5000).to_dict(orient="records"):
        raw_node_id = row.get(node_column)
        if raw_node_id is None or pd.isna(raw_node_id):
            continue
        node_id = str(raw_node_id)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        person = _person_payload_for_row(row=row, mapping=mapping, node_id=node_id)
        features = {
            column: _json_value(row.get(column))
            for column in feature_columns
            if _is_numeric(row.get(column))
        }
        object_id = f"P-{dataset_id:04d}-{inserted + 1:06d}"
        db.add(
            PersonNode(
                dataset_id=dataset_id,
                node_id=node_id,
                raw_json={
                    "source": "uploaded_csv",
                    "object_type": "person",
                    "business_id": business_id,
                    "person_object_id": object_id,
                    "fraud_event": fraud_event,
                    "gang_id": _gang_id_for(dataset_id=dataset_id, node_id=node_id, fraud_event=fraud_event),
                    "original_row": {key: _json_value(value) for key, value in row.items()},
                },
                feature_json=features,
                **person,
            )
        )
        inserted += 1
    return inserted


def _insert_edges(*, db: Session, dataset_id: int, frame: pd.DataFrame, mapping: dict[str, Any]) -> int:
    source_column = mapping.get("source_id")
    target_column = mapping.get("target_id")
    timestamp_column = mapping.get("timestamp")
    amount_column = mapping.get("amount")
    edge_type_column = mapping.get("edge_type")
    node_column = mapping["node_id"]
    inserted = 0

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
            inserted += 1
        return inserted

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
        inserted += 1
    return inserted


def _person_payload_for_row(*, row: dict[str, Any], mapping: dict[str, Any], node_id: str) -> dict[str, str]:
    display_columns = [str(value) for value in (mapping.get("display_columns") or [])]
    generated = synthetic_person_for_node(node_id)
    value_map = {column.lower(): row.get(column) for column in display_columns}
    return {
        "display_name": str(value_map.get("display_name") or value_map.get("name") or generated["display_name"]),
        "id_number": str(value_map.get("id_number") or generated["id_number"]),
        "phone": str(value_map.get("phone") or generated["phone"]),
        "region": str(value_map.get("region") or value_map.get("city") or generated["region"]),
        "occupation": str(value_map.get("occupation") or generated["occupation"]),
    }


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
