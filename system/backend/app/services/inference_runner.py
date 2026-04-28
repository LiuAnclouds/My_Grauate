from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import BACKEND_ROOT, DATA_ROOT, settings
from app.models import GraphEdge, InferenceResult, PersonNode


def run_full_model_subprocess(
    *,
    dataset_name: str,
    node_ids: list[str],
) -> list[dict[str, Any]]:
    request_root = DATA_ROOT / "inference_requests"
    request_root.mkdir(parents=True, exist_ok=True)
    request_path = request_root / f"{dataset_name}_nodes.json"
    output_path = request_root / f"{dataset_name}_predictions.json"
    request_path.write_text(
        json.dumps({"node_ids": [int(value) for value in node_ids]}, ensure_ascii=False),
        encoding="utf-8",
    )
    if output_path.exists():
        output_path.unlink()
    command = [
        sys.executable,
        str(BACKEND_ROOT / "scripts" / "predict_validation.py"),
        "--dataset",
        dataset_name,
        "--node-ids-file",
        str(request_path),
        "--output-json",
        str(output_path),
        "--device",
        settings.inference_device,
    ]
    completed = subprocess.run(
        command,
        cwd=BACKEND_ROOT.parents[1],
        capture_output=True,
        text=True,
        timeout=int(settings.inference_timeout_seconds),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "full model inference failed: "
            f"stdout={completed.stdout[-1000:]} stderr={completed.stderr[-2000:]}"
        )
    payload = json.loads(output_path.read_text(encoding="utf-8-sig"))
    return list(payload["results"])


def run_feature_fallback(nodes: list[PersonNode]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for node in nodes:
        values = [
            float(value)
            for value in (node.feature_json or {}).values()
            if isinstance(value, (int, float))
        ]
        score = _sigmoid(float(np.mean(values) if values else 0.0))
        results.append({"node_id": node.node_id, "risk_score": score})
    return results


def replace_inference_results(
    *,
    db: Session,
    dataset_id: int,
    scores: list[dict[str, Any]],
) -> list[InferenceResult]:
    db.query(InferenceResult).filter(InferenceResult.dataset_id == dataset_id).delete()
    score_values = np.asarray([float(item["risk_score"]) for item in scores], dtype=np.float32)
    threshold = float(np.quantile(score_values, 0.85)) if score_values.size else 1.0
    threshold = max(0.5, threshold)
    node_map = {
        node.node_id: node
        for node in db.query(PersonNode).filter(PersonNode.dataset_id == dataset_id).all()
    }
    edge_map = _collect_neighbor_map(db=db, dataset_id=dataset_id)
    stored: list[InferenceResult] = []
    for item in scores:
        node_id = str(item["node_id"])
        score = float(item["risk_score"])
        label = "suspicious" if score >= threshold else "normal"
        node = node_map.get(node_id)
        explanation = _build_explanation(
            node=node,
            neighbor_ids=edge_map.get(node_id, []),
            score=score,
            label=label,
        )
        record = InferenceResult(
            dataset_id=dataset_id,
            node_id=node_id,
            risk_score=score,
            risk_label=label,
            explanation_json=explanation,
        )
        db.add(record)
        stored.append(record)
    db.commit()
    for record in stored:
        db.refresh(record)
    return stored


def _collect_neighbor_map(*, db: Session, dataset_id: int) -> dict[str, list[str]]:
    neighbor_map: dict[str, list[str]] = {}
    edges = db.query(GraphEdge).filter(GraphEdge.dataset_id == dataset_id).limit(2000).all()
    for edge in edges:
        neighbor_map.setdefault(edge.source_id, []).append(edge.target_id)
        neighbor_map.setdefault(edge.target_id, []).append(edge.source_id)
    return {key: values[:6] for key, values in neighbor_map.items()}


def _build_explanation(
    *,
    node: PersonNode | None,
    neighbor_ids: list[str],
    score: float,
    label: str,
) -> dict[str, Any]:
    feature_json = dict(node.feature_json or {}) if node is not None else {}
    top_features = sorted(
        (
            (key, abs(float(value)))
            for key, value in feature_json.items()
            if isinstance(value, (int, float))
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    if label == "suspicious":
        reason = "风险分数较高，模型在时间邻域聚合后将该节点列为重点核查对象。"
    else:
        reason = "风险分数处于当前数据集较低区间，暂未列为异常节点。"
    return {
        "reason": reason,
        "support_neighbors": neighbor_ids,
        "top_features": [name for name, _ in top_features],
        "score": score,
    }


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))
