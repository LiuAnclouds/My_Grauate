from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from data_processing.core.registry import resolve_output_roots
from analysis.data_loader import resolve_dataset_path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_OUTPUT_ROOT, FEATURE_OUTPUT_ROOT = resolve_output_roots(REPO_ROOT)
TRAIN_OUTPUT_ROOT = REPO_ROOT / "outputs" / "train"
EPOCH_METRIC_COLUMNS = ("epoch", "train_loss", "train_auc", "val_loss", "val_auc")


@dataclass(frozen=True)
class ExperimentSplit:
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_pool_ids: np.ndarray
    external_ids: np.ndarray
    threshold_day: int
    train_phase: str = "phase1"
    val_phase: str = "phase1"
    external_phase: str | None = "phase2"
    split_style: str = "two_phase"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_clean_epoch_metrics(output_path: Path, source_paths: list[Path]) -> Path:
    rows_by_epoch: dict[int, list[dict[str, float | None]]] = {}
    for source_path in source_paths:
        if not source_path.exists() or source_path.stat().st_size == 0:
            continue
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw_row in csv.DictReader(handle):
                epoch_raw = raw_row.get("epoch")
                if epoch_raw is None or str(epoch_raw).strip() == "":
                    continue
                epoch = int(float(str(epoch_raw)))
                rows_by_epoch.setdefault(epoch, []).append(
                    {
                        "train_loss": _coerce_metric(raw_row, "train_loss"),
                        "train_auc": _coerce_metric(raw_row, "train_auc", "train-auc"),
                        "val_loss": _coerce_metric(raw_row, "val_loss", "val-logloss"),
                        "val_auc": _coerce_metric(raw_row, "val_auc", "val-auc"),
                    }
                )

    clean_rows: list[dict[str, str]] = []
    for epoch in sorted(rows_by_epoch):
        epoch_rows = rows_by_epoch[epoch]
        clean_rows.append(
            {
                "epoch": str(epoch),
                "train_loss": _format_optional_metric(_mean_available(epoch_rows, "train_loss")),
                "train_auc": _format_optional_metric(_mean_available(epoch_rows, "train_auc")),
                "val_loss": _format_optional_metric(_mean_available(epoch_rows, "val_loss")),
                "val_auc": _format_optional_metric(_mean_available(epoch_rows, "val_auc")),
            }
        )

    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EPOCH_METRIC_COLUMNS))
        writer.writeheader()
        writer.writerows(clean_rows)
    return output_path


def _coerce_metric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        return float(value)
    return None


def _mean_available(rows: list[dict[str, float | None]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


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


def load_experiment_split(analysis_root: Path = ANALYSIS_OUTPUT_ROOT) -> ExperimentSplit:
    summary = read_json(analysis_root / "recommended_split.json")
    if "train_split" in summary and "val_split" in summary:
        train = summary["train_split"]
        val = summary["val_split"]
        test_pool = summary.get("test_pool") or summary.get("unlabeled_pool")
        external = summary.get("external_eval")
        test_pool_ids = (
            np.load(analysis_root / test_pool["id_path"])
            if test_pool is not None and test_pool.get("id_path")
            else np.empty(0, dtype=np.int32)
        )
        external_ids = (
            np.load(analysis_root / external["id_path"])
            if external is not None and external.get("id_path")
            else np.empty(0, dtype=np.int32)
        )
        return ExperimentSplit(
            train_ids=np.load(analysis_root / train["id_path"]),
            val_ids=np.load(analysis_root / val["id_path"]),
            test_pool_ids=np.asarray(test_pool_ids, dtype=np.int32),
            external_ids=np.asarray(external_ids, dtype=np.int32),
            threshold_day=int(summary.get("threshold_day", train["threshold_day"])),
            train_phase=str(summary.get("train_phase", "graph")),
            val_phase=str(summary.get("val_phase", summary.get("train_phase", "graph"))),
            external_phase=summary.get("external_phase"),
            split_style=str(summary.get("split_style", "single_graph")),
        )

    if "phase1_time_split" in summary:
        phase1 = summary["phase1_time_split"]
        phase2 = summary.get("phase2_external_eval", {})
        external_path = phase2.get("id_path")
        external_ids = (
            np.load(analysis_root / external_path)
            if external_path
            else np.empty(0, dtype=np.int32)
        )
        return ExperimentSplit(
            train_ids=np.load(analysis_root / phase1["train_id_path"]),
            val_ids=np.load(analysis_root / phase1["val_id_path"]),
            test_pool_ids=np.empty(0, dtype=np.int32),
            external_ids=np.asarray(external_ids, dtype=np.int32),
            threshold_day=int(phase1["threshold_day"]),
            train_phase="phase1",
            val_phase="phase1",
            external_phase="phase2" if external_path else None,
            split_style="two_phase",
        )
    raise KeyError(f"Unsupported recommended_split.json format under {analysis_root}")


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


def resolve_prediction_path(run_dir: Path, split_name: str) -> Path:
    candidates = (
        run_dir / f"{split_name}_avg_predictions.npz",
        run_dir / f"{split_name}_predictions.npz",
        run_dir / f"{split_name}_blend_predictions.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{run_dir}: missing prediction file for {split_name}. "
        f"Expected one of {[path.name for path in candidates]}."
    )


def load_prediction_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    files = set(data.files)

    node_key = "node_ids" if "node_ids" in files else "ids"
    label_key = "y_true" if "y_true" in files else "labels"
    score_key = next(
        (
            key
            for key in (
                "probability",
                "probabilities",
                "score",
                "scores",
                "avg_probability",
                "avg_probabilities",
                "avg_score",
                "avg_scores",
            )
            if key in files
        ),
        None,
    )
    if score_key is None:
        raise KeyError(f"{path}: unsupported prediction archive keys {sorted(files)}")

    score = np.asarray(data[score_key], dtype=np.float32)
    if score.ndim == 2 and score.shape[1] == 1:
        score = score.reshape(-1)
    return {
        "node_ids": np.asarray(data[node_key], dtype=np.int32),
        "y_true": np.asarray(data[label_key], dtype=np.int8),
        "probability": score,
    }


def align_prediction_bundle(
    bundle: dict[str, np.ndarray],
    ref_node_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    node_ids = np.asarray(bundle["node_ids"], dtype=np.int32)
    ref_ids = np.asarray(ref_node_ids, dtype=np.int32)
    if np.array_equal(node_ids, ref_ids):
        return {
            "node_ids": node_ids,
            "y_true": np.asarray(bundle["y_true"], dtype=np.int8, copy=False),
            "probability": np.asarray(bundle["probability"], dtype=np.float32, copy=False),
        }

    position = {int(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}
    aligned_idx = np.asarray([position[int(node_id)] for node_id in ref_ids], dtype=np.int64)
    return {
        "node_ids": ref_ids,
        "y_true": np.asarray(bundle["y_true"][aligned_idx], dtype=np.int8, copy=False),
        "probability": np.asarray(bundle["probability"][aligned_idx], dtype=np.float32, copy=False),
    }


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
