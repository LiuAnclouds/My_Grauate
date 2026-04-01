from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from experiment.eda.data_loader import resolve_dataset_path


REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_OUTPUT_ROOT = REPO_ROOT / "experiment" / "outputs" / "eda"
TRAINING_OUTPUT_ROOT = REPO_ROOT / "experiment" / "outputs" / "training"
FEATURE_OUTPUT_ROOT = TRAINING_OUTPUT_ROOT / "features"
MODEL_OUTPUT_ROOT = TRAINING_OUTPUT_ROOT / "models"
BLEND_OUTPUT_ROOT = TRAINING_OUTPUT_ROOT / "blends"


@dataclass(frozen=True)
class ExperimentSplit:
    train_ids: np.ndarray
    val_ids: np.ndarray
    external_ids: np.ndarray
    threshold_day: int
    external_phase: str = "phase2"


@dataclass(frozen=True)
class PredictionArtifacts:
    val_path: Path
    external_path: Path
    summary_path: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    labels = np.unique(y_true)
    if labels.size < 2:
        raise ValueError("AUC requires both positive and negative samples.")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    labels = np.unique(y_true)
    if labels.size < 2:
        raise ValueError("Average precision requires both positive and negative samples.")
    return float(average_precision_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    labels = np.unique(y_true)
    if labels.size < 2:
        raise ValueError("PR-AUC requires both positive and negative samples.")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(np.trapz(precision[::-1], recall[::-1]))


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    return {
        "auc": safe_auc(y_true, y_score),
        "ap": safe_average_precision(y_true, y_score),
        "pr_auc": safe_pr_auc(y_true, y_score),
    }


def load_experiment_split(eda_root: Path = EDA_OUTPUT_ROOT) -> ExperimentSplit:
    summary = read_json(eda_root / "recommended_split.json")
    phase1 = summary["phase1_time_split"]
    phase2 = summary["phase2_external_eval"]
    return ExperimentSplit(
        train_ids=np.load(eda_root / phase1["train_id_path"]),
        val_ids=np.load(eda_root / phase1["val_id_path"]),
        external_ids=np.load(eda_root / phase2["id_path"]),
        threshold_day=int(phase1["threshold_day"]),
        external_phase="phase2",
    )


def load_phase_arrays(
    phase: str,
    keys: tuple[str, ...] = ("y", "train_mask", "test_mask"),
) -> dict[str, np.ndarray]:
    phase_path = resolve_dataset_path(phase, repo_root=REPO_ROOT)
    npz = np.load(phase_path, allow_pickle=False)
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        values = np.asarray(npz[key])
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        arrays[key] = values
    return arrays


def slice_node_ids(node_ids: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or node_ids.size <= limit:
        return node_ids
    rng = np.random.default_rng(seed)
    choice = rng.choice(node_ids.size, size=limit, replace=False)
    return np.sort(node_ids[choice].astype(np.int32, copy=False))


def save_prediction_npz(
    path: Path,
    node_ids: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        node_ids=np.asarray(node_ids, dtype=np.int32),
        y_true=np.asarray(y_true, dtype=np.int8),
        probability=np.asarray(probabilities, dtype=np.float32),
    )


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
