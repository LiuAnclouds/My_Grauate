from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment.training.common import load_prediction_npz, resolve_prediction_path
from experiment.training.xgb_utils import binary_score_from_softprob


def _scatter_known_prediction_rows(
    target_matrix: np.ndarray,
    *,
    target_ids: np.ndarray,
    bundle: dict[str, np.ndarray],
    feature_values: np.ndarray,
) -> int:
    position = {
        int(node_id): idx for idx, node_id in enumerate(np.asarray(bundle["node_ids"], dtype=np.int32).tolist())
    }
    matched_target: list[int] = []
    matched_source: list[int] = []
    for node_id in np.asarray(target_ids, dtype=np.int32).tolist():
        source_idx = position.get(int(node_id))
        if source_idx is None:
            continue
        matched_target.append(int(node_id))
        matched_source.append(int(source_idx))
    if not matched_target:
        return 0
    target_matrix[np.asarray(matched_target, dtype=np.int32)] = feature_values[
        np.asarray(matched_source, dtype=np.int64)
    ]
    return len(matched_target)


def transform_target_context_prediction_features(
    score: np.ndarray,
    transform: str,
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if str(transform) == "raw":
        return values.astype(np.float32, copy=False)
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    if clipped.shape[1] == 1:
        return np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)
    return np.log(clipped).astype(np.float32, copy=False)


def load_target_context_prediction_features(
    *,
    prediction_dir: Path,
    prediction_transform: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    phase1_num_nodes: int,
    phase2_num_nodes: int,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    run_dir = Path(prediction_dir)
    train_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase1_train"))
    val_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase1_val"))
    train_features = transform_target_context_prediction_features(
        train_bundle["probability"],
        prediction_transform,
    )
    val_features = transform_target_context_prediction_features(
        val_bundle["probability"],
        prediction_transform,
    )
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError(
            f"{run_dir}: train/val prediction feature dims mismatch "
            f"{train_features.shape[1]} vs {val_features.shape[1]}."
        )

    phase1_features = np.zeros((int(phase1_num_nodes), int(train_features.shape[1])), dtype=np.float32)
    _scatter_known_prediction_rows(
        phase1_features,
        target_ids=train_ids,
        bundle=train_bundle,
        feature_values=train_features,
    )
    _scatter_known_prediction_rows(
        phase1_features,
        target_ids=val_ids,
        bundle=val_bundle,
        feature_values=val_features,
    )

    phase2_features = None
    if external_ids.size:
        external_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase2_external"))
        external_features = transform_target_context_prediction_features(
            external_bundle["probability"],
            prediction_transform,
        )
        if external_features.shape[1] != train_features.shape[1]:
            raise ValueError(
                f"{run_dir}: train/external prediction feature dims mismatch "
                f"{train_features.shape[1]} vs {external_features.shape[1]}."
            )
        phase2_features = np.zeros((int(phase2_num_nodes), int(train_features.shape[1])), dtype=np.float32)
        _scatter_known_prediction_rows(
            phase2_features,
            target_ids=external_ids,
            bundle=external_bundle,
            feature_values=external_features,
        )

    feature_names = [f"teacher_pred_{idx}" for idx in range(int(train_features.shape[1]))]
    return phase1_features, phase2_features, feature_names


def coerce_teacher_binary_score(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    if values.ndim == 1:
        return values.astype(np.float32, copy=False)
    if values.ndim == 2 and values.shape[1] == 1:
        return values.reshape(-1).astype(np.float32, copy=False)
    if values.ndim == 2 and values.shape[1] == 4:
        return binary_score_from_softprob(values)
    if values.ndim == 2 and values.shape[1] == 2:
        denom = np.clip(values[:, 0] + values[:, 1], 1e-6, None)
        return (values[:, 1] / denom).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported teacher prediction shape: {values.shape}")


def load_teacher_distill_targets(
    *,
    prediction_dir: Path,
    train_ids: np.ndarray,
    phase1_num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    run_dir = Path(prediction_dir)
    train_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase1_train"))
    score = coerce_teacher_binary_score(train_bundle["probability"])
    targets = np.zeros((int(phase1_num_nodes),), dtype=np.float32)
    mask = np.zeros((int(phase1_num_nodes),), dtype=bool)
    position = {
        int(node_id): idx
        for idx, node_id in enumerate(np.asarray(train_bundle["node_ids"], dtype=np.int32).tolist())
    }
    matched_target: list[int] = []
    matched_source: list[int] = []
    for node_id in np.asarray(train_ids, dtype=np.int32).tolist():
        source_idx = position.get(int(node_id))
        if source_idx is None:
            continue
        matched_target.append(int(node_id))
        matched_source.append(int(source_idx))
    if matched_target:
        matched_target_idx = np.asarray(matched_target, dtype=np.int32)
        matched_source_idx = np.asarray(matched_source, dtype=np.int64)
        targets[matched_target_idx] = score[matched_source_idx]
        mask[matched_target_idx] = True
    return targets, mask
