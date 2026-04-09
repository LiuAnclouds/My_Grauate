from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment.training.common import ensure_dir


def rank_norm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    return ranks


def binary_score_from_softprob(prob: np.ndarray) -> np.ndarray:
    prob_arr = np.asarray(prob, dtype=np.float32).reshape(-1, 4)
    foreground = np.clip(prob_arr[:, 0] + prob_arr[:, 1], 1e-6, None)
    return (prob_arr[:, 1] / foreground).astype(np.float32, copy=False)


def multiclass_binary_auc(predt: np.ndarray, dmatrix: Any) -> tuple[str, float]:
    labels = np.asarray(dmatrix.get_label(), dtype=np.int8)
    prob = np.asarray(predt, dtype=np.float32).reshape(labels.shape[0], 4)
    score = binary_score_from_softprob(prob)
    return "binary_auc", float(roc_auc_score(labels, score))


def build_multiclass_bg_sample_weight(
    y_train: np.ndarray,
    *,
    fraud_weight_scale: float,
    background_weight: float,
    time_weight_half_life_days: float = 0.0,
    time_weight_floor: float = 0.25,
    train_first_active: np.ndarray | None = None,
    threshold_day: int | None = None,
) -> dict[str, object]:
    y_arr = np.asarray(y_train, dtype=np.int32)
    count0 = float(np.sum(y_arr == 0))
    count1 = float(np.sum(y_arr == 1))
    count2 = float(np.sum(y_arr == 2))
    count3 = float(np.sum(y_arr == 3))
    class_weight = {
        0: 1.0,
        1: float(fraud_weight_scale) * count0 / max(count1, 1.0),
        2: float(background_weight) * count0 / max(count2, 1.0),
        3: float(background_weight) * count0 / max(count3, 1.0),
    }
    sample_weight = np.asarray([class_weight[int(label)] for label in y_arr], dtype=np.float32)
    time_weight_payload: dict[str, float | None | bool] = {
        "enabled": False,
        "half_life_days": None,
        "floor": None,
    }
    if (
        float(time_weight_half_life_days) > 0.0
        and train_first_active is not None
        and threshold_day is not None
    ):
        age = np.clip(
            float(threshold_day) - np.asarray(train_first_active, dtype=np.float32),
            0.0,
            None,
        )
        decay = np.power(np.float32(0.5), age / float(time_weight_half_life_days)).astype(
            np.float32,
            copy=False,
        )
        floor = float(np.clip(time_weight_floor, 0.0, 1.0))
        sample_weight *= (floor + (1.0 - floor) * decay).astype(np.float32, copy=False)
        time_weight_payload = {
            "enabled": True,
            "half_life_days": float(time_weight_half_life_days),
            "floor": floor,
        }
    mean_weight = float(np.mean(sample_weight, dtype=np.float64))
    if mean_weight > 0.0:
        sample_weight /= mean_weight
    return {
        "sample_weight": sample_weight,
        "class_weight": {str(key): float(value) for key, value in class_weight.items()},
        "time_weight": time_weight_payload,
    }


def write_feature_importance_csv(
    booster: Any,
    feature_names: list[str],
    path: Path,
    *,
    importance_type: str = "gain",
) -> None:
    scores = booster.get_score(importance_type=importance_type)
    rows = [
        {
            "feature_name": feature_name,
            "gain": float(scores.get(f"f{idx}", 0.0)),
        }
        for idx, feature_name in enumerate(feature_names)
    ]
    rows.sort(key=lambda row: row["gain"], reverse=True)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["feature_name", "gain"])
        writer.writeheader()
        writer.writerows(rows)
