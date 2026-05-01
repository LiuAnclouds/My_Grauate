from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import GraphEdge, InferenceResult, PersonNode, ProcessingEvent, ProcessingTask


def build_dataset_summary(*, node_count: int, edge_count: int, mapping: dict[str, Any], source: str) -> dict[str, Any]:
    feature_columns = [str(value) for value in (mapping.get("feature_columns") or [])]
    return {
        "source": source,
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns[:12],
        "mapping_method": str(mapping.get("mapping_method") or mapping.get("source") or "stored"),
    }


def create_task(
    *,
    db: Session,
    dataset_id: int,
    task_type: str,
    status: str,
    progress: float,
    current_step: str,
    message: str,
    summary: dict[str, Any] | None = None,
) -> ProcessingTask:
    task = ProcessingTask(
        dataset_id=dataset_id,
        task_type=task_type,
        status=status,
        progress=progress,
        current_step=current_step,
        message=message,
        summary_json=summary or {},
        updated_at=datetime.utcnow(),
    )
    db.add(task)
    db.flush()
    return task


def update_task(
    *,
    task: ProcessingTask,
    status: str,
    progress: float,
    current_step: str,
    message: str,
    summary: dict[str, Any] | None = None,
) -> ProcessingTask:
    task.status = status
    task.progress = progress
    task.current_step = current_step
    task.message = message
    task.summary_json = summary or task.summary_json or {}
    task.updated_at = datetime.utcnow()
    return task


def append_event(
    *,
    db: Session,
    dataset_id: int,
    task_id: int | None,
    stage: str,
    step_key: str,
    title: str,
    detail: str,
    progress: float,
    focus_node_id: str | None = None,
    focus_neighbor_ids: list[str] | None = None,
    top_features: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> ProcessingEvent:
    event = ProcessingEvent(
        dataset_id=dataset_id,
        task_id=task_id,
        stage=stage,
        step_key=step_key,
        title=title,
        detail=detail,
        progress=progress,
        focus_node_id=focus_node_id,
        focus_neighbor_ids=[str(value) for value in (focus_neighbor_ids or [])],
        top_features=[str(value) for value in (top_features or [])],
        metrics_json=metrics or {},
    )
    db.add(event)
    return event


def reset_task_events(*, db: Session, dataset_id: int, task_type: str) -> None:
    task_ids = [
        value
        for value in db.scalars(
            select(ProcessingTask.id).where(
                ProcessingTask.dataset_id == dataset_id,
                ProcessingTask.task_type == task_type,
            )
        ).all()
    ]
    if task_ids:
        db.query(ProcessingEvent).filter(ProcessingEvent.task_id.in_(task_ids)).delete(
            synchronize_session=False
        )


def latest_task(*, db: Session, dataset_id: int) -> ProcessingTask | None:
    tasks = db.scalars(
        select(ProcessingTask)
        .where(ProcessingTask.dataset_id == dataset_id)
        .order_by(ProcessingTask.updated_at.desc(), ProcessingTask.id.desc())
    ).all()
    if not tasks:
        return None
    task_ids = [task.id for task in tasks]
    event_task_ids = set(
        db.scalars(
            select(ProcessingEvent.task_id)
            .where(ProcessingEvent.dataset_id == dataset_id, ProcessingEvent.task_id.in_(task_ids))
        ).all()
    )
    for task in tasks:
        if task.id in event_task_ids:
            return task
    return tasks[0]


def timeline_events(*, db: Session, dataset_id: int, task_id: int | None = None) -> list[ProcessingEvent]:
    query = select(ProcessingEvent).where(ProcessingEvent.dataset_id == dataset_id)
    if task_id is not None:
        query = query.where(ProcessingEvent.task_id == task_id)
    return db.scalars(query.order_by(ProcessingEvent.created_at.asc(), ProcessingEvent.id.asc())).all()


def build_feature_task_story(
    *,
    db: Session,
    dataset_id: int,
    task: ProcessingTask,
    nodes: list[PersonNode],
    edges: list[GraphEdge],
) -> None:
    feature_counts = [len(node.feature_json or {}) for node in nodes]
    avg_feature_count = round(float(np.mean(feature_counts)) if feature_counts else 0.0, 2)
    top_regions = Counter(node.region for node in nodes).most_common(3)
    sample_node = nodes[0] if nodes else None
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="feature_processing",
        step_key="load_nodes",
        title="装载对象数据",
        detail=f"已读取 {len(nodes)} 个对象，正在整理分析所需字段。",
        progress=0.12,
        focus_node_id=sample_node.node_id if sample_node else None,
        top_features=list((sample_node.feature_json or {}).keys())[:4] if sample_node else [],
        metrics={"nodes": len(nodes), "avg_feature_count": avg_feature_count},
    )
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="feature_processing",
        step_key="build_graph",
        title="组织关系网络",
        detail=f"已装载 {len(edges)} 条关系，正在形成邻域结构。",
        progress=0.32,
        focus_node_id=sample_node.node_id if sample_node else None,
        focus_neighbor_ids=[edge.target_id for edge in edges[:4]],
        metrics={"edges": len(edges)},
    )
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="feature_processing",
        step_key="normalize",
        title="准备对象特征",
        detail="已按当前分析配置整理特征目录与标准化输入。",
        progress=0.56,
        focus_node_id=sample_node.node_id if sample_node else None,
        top_features=list((sample_node.feature_json or {}).keys())[:6] if sample_node else [],
        metrics={"avg_feature_count": avg_feature_count, "regions": dict(top_regions)},
    )
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="feature_processing",
        step_key="temporal_context",
        title="编码时序上下文",
        detail="正在整理时间窗口与邻域上下文，准备进入风险识别。",
        progress=0.81,
        focus_node_id=sample_node.node_id if sample_node else None,
        focus_neighbor_ids=[edge.target_id for edge in edges[:6]],
        metrics={"sampled_regions": [name for name, _ in top_regions]},
    )
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="feature_processing",
        step_key="ready",
        title="分析输入准备完成",
        detail="对象特征与关系上下文已准备完毕，可以启动风险识别。",
        progress=1.0,
        focus_node_id=sample_node.node_id if sample_node else None,
        top_features=list((sample_node.feature_json or {}).keys())[:6] if sample_node else [],
        metrics={"ready": True},
    )
    update_task(
        task=task,
        status="completed",
        progress=1.0,
        current_step="feature_ready",
        message="分析输入准备完成，可以开始风险识别。",
        summary={"nodes": len(nodes), "edges": len(edges), "avg_feature_count": avg_feature_count},
    )


def build_graph_task_story(
    *,
    db: Session,
    dataset_id: int,
    task: ProcessingTask,
    nodes: list[PersonNode],
    edges: list[GraphEdge],
) -> None:
    sample_node = nodes[0] if nodes else None
    sample_neighbors = [edge.target_id for edge in edges[:6]]
    top_edge_types = Counter(edge.edge_type for edge in edges).most_common(3)
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="graph_construction",
        step_key="load_objects",
        title="装载风险对象",
        detail=f"已读取 {len(nodes)} 个风险对象，准备组织关系结构。",
        progress=0.18,
        focus_node_id=sample_node.node_id if sample_node else None,
        metrics={"nodes": len(nodes)},
    )
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="graph_construction",
        step_key="load_relations",
        title="装载关系边",
        detail=f"已读取 {len(edges)} 条关系边，正在整理指向与类型。",
        progress=0.42,
        focus_node_id=sample_node.node_id if sample_node else None,
        focus_neighbor_ids=sample_neighbors[:4],
        metrics={"edges": len(edges), "edge_types": dict(top_edge_types)},
    )
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="graph_construction",
        step_key="layout_ready",
        title="计算图谱布局",
        detail="已完成节点尺寸、关系方向和风险视觉样式准备。",
        progress=0.76,
        focus_node_id=sample_node.node_id if sample_node else None,
        focus_neighbor_ids=sample_neighbors,
        metrics={"layout": "ready"},
    )
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="graph_construction",
        step_key="graph_ready",
        title="关系图谱构建完成",
        detail="关系图谱已可查看，后续研判会在图中同步标记风险对象。",
        progress=1.0,
        focus_node_id=sample_node.node_id if sample_node else None,
        focus_neighbor_ids=sample_neighbors,
        metrics={"ready": True, "nodes": len(nodes), "edges": len(edges)},
    )
    update_task(
        task=task,
        status="completed",
        progress=1.0,
        current_step="graph_ready",
        message="关系图谱构建完成，可以进入图谱查看。",
        summary={"nodes": len(nodes), "edges": len(edges), "ready": True},
    )


def build_inference_story(
    *,
    db: Session,
    dataset_id: int,
    task: ProcessingTask,
    records: list[InferenceResult],
    nodes: list[PersonNode],
) -> None:
    node_map = {node.node_id: node for node in nodes}
    ranked = sorted(records, key=lambda item: item.risk_score, reverse=True)
    abnormal = [record for record in ranked if record.risk_label == "suspicious"]
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="inference",
        step_key="load_checkpoint",
        title="加载识别配置",
        detail="已复用实验目录中的训练配置与最佳权重文件。",
        progress=0.08,
        metrics={"records": len(records)},
    )
    for index, record in enumerate(ranked[:6], start=1):
        node = node_map.get(record.node_id)
        explanation = dict(record.explanation_json or {})
        append_event(
            db=db,
            dataset_id=dataset_id,
            task_id=task.id,
            stage="inference",
            step_key=f"aggregate_{index}",
            title=f"分析对象 {record.node_id}",
            detail=str(explanation.get("reason") or "系统正在聚合关联对象与关键特征。"),
            progress=min(0.18 + index * 0.1, 0.82),
            focus_node_id=record.node_id,
            focus_neighbor_ids=[str(value) for value in explanation.get("support_neighbors") or []],
            top_features=[str(value) for value in explanation.get("top_features") or []],
            metrics={
                "risk_score": round(float(record.risk_score), 4),
                "risk_label": record.risk_label,
                "display_name": node.display_name if node is not None else record.node_id,
            },
        )
    append_event(
        db=db,
        dataset_id=dataset_id,
        task_id=task.id,
        stage="inference",
        step_key="finalize",
        title="生成风险结果",
        detail=f"风险识别完成，共定位 {len(abnormal)} 个高风险对象，结果已写入数据库。",
        progress=1.0,
        focus_node_id=abnormal[0].node_id if abnormal else (ranked[0].node_id if ranked else None),
        focus_neighbor_ids=[
            str(value)
            for value in ((abnormal[0].explanation_json or {}).get("support_neighbors") or [])
        ]
        if abnormal
        else [],
        top_features=[
            str(value)
            for value in ((abnormal[0].explanation_json or {}).get("top_features") or [])
        ]
        if abnormal
        else [],
        metrics={"abnormal": len(abnormal), "normal": len(records) - len(abnormal)},
    )
    update_task(
        task=task,
        status="completed",
        progress=1.0,
        current_step="risk_inference_completed",
        message="风险识别完成，对象级结果已生成。",
        summary={"abnormal": len(abnormal), "normal": len(records) - len(abnormal)},
    )
