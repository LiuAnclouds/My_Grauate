from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import get_active_dataset_spec
from experiment.training.common import (
    BLEND_OUTPUT_ROOT,
    align_prediction_bundle,
    compute_binary_classification_metrics,
    ensure_dir,
    load_prediction_npz,
    resolve_prediction_path,
    save_prediction_npz,
    write_json,
)
from experiment.training.thesis_contract import (
    OFFICIAL_BACKBONE_FEATURE_PROFILE,
    OFFICIAL_BACKBONE_MODEL,
    OFFICIAL_BACKBONE_PRESET,
    OFFICIAL_HYBRID_BLEND_ALPHA,
    OFFICIAL_HYBRID_SECONDARY_MODEL,
)


ACTIVE_DATASET_SPEC = get_active_dataset_spec()


def _path_repr(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _coerce_binary_score(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float32)
    if values.ndim == 1:
        return values.astype(np.float32, copy=False)
    if values.ndim == 2 and values.shape[1] == 1:
        return values.reshape(-1).astype(np.float32, copy=False)
    if values.ndim == 2 and values.shape[1] == 2:
        return values[:, 1].astype(np.float32, copy=False)
    if values.ndim == 2:
        return values[:, 1:].sum(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported prediction shape: {values.shape}")


def _load_split_bundle(run_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = load_prediction_npz(resolve_prediction_path(run_dir, split_name))
    node_ids = np.asarray(bundle["node_ids"], dtype=np.int32)
    labels = np.asarray(bundle["y_true"], dtype=np.int8)
    probability = _coerce_binary_score(bundle["probability"])
    return node_ids, labels, probability


def _align_scores(
    *,
    reference_node_ids: np.ndarray,
    bundle_node_ids: np.ndarray,
    bundle_score: np.ndarray,
) -> np.ndarray:
    if np.array_equal(reference_node_ids, bundle_node_ids):
        return np.asarray(bundle_score, dtype=np.float32, copy=False)
    bundle = {
        "node_ids": np.asarray(bundle_node_ids, dtype=np.int32),
        "y_true": np.zeros(bundle_node_ids.shape[0], dtype=np.int8),
        "probability": np.asarray(bundle_score, dtype=np.float32),
    }
    aligned = align_prediction_bundle(bundle, np.asarray(reference_node_ids, dtype=np.int32))
    return np.asarray(aligned["probability"], dtype=np.float32, copy=False)


def _align_bundle_to_reference(
    *,
    reference_node_ids: np.ndarray,
    reference_labels: np.ndarray,
    reference_probability: np.ndarray,
    target_node_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.array_equal(reference_node_ids, target_node_ids):
        return (
            np.asarray(reference_node_ids, dtype=np.int32, copy=False),
            np.asarray(reference_labels, dtype=np.int8, copy=False),
            np.asarray(reference_probability, dtype=np.float32, copy=False),
        )
    bundle = {
        "node_ids": np.asarray(reference_node_ids, dtype=np.int32),
        "y_true": np.asarray(reference_labels, dtype=np.int8),
        "probability": np.asarray(reference_probability, dtype=np.float32),
    }
    aligned = align_prediction_bundle(bundle, np.asarray(target_node_ids, dtype=np.int32))
    return (
        np.asarray(aligned["node_ids"], dtype=np.int32, copy=False),
        np.asarray(aligned["y_true"], dtype=np.int8, copy=False),
        np.asarray(aligned["probability"], dtype=np.float32, copy=False),
    )


def _shared_train_alignment(
    *,
    base_node_ids: np.ndarray,
    base_labels: np.ndarray,
    base_probability: np.ndarray,
    secondary_node_ids: np.ndarray,
    secondary_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_ids = np.asarray(base_node_ids, dtype=np.int32)
    secondary_ids = np.asarray(secondary_node_ids, dtype=np.int32)
    shared_ids = np.intersect1d(base_ids, secondary_ids, assume_unique=False)
    if shared_ids.size == 0:
        raise ValueError("Base train predictions and secondary train predictions have no shared node ids.")
    aligned_ids, aligned_labels, aligned_base_prob = _align_bundle_to_reference(
        reference_node_ids=base_ids,
        reference_labels=base_labels,
        reference_probability=base_probability,
        target_node_ids=shared_ids,
    )
    aligned_secondary_prob = _align_scores(
        reference_node_ids=shared_ids,
        bundle_node_ids=secondary_ids,
        bundle_score=secondary_probability,
    )
    return aligned_ids, aligned_labels, aligned_base_prob, aligned_secondary_prob


def _maybe_binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float] | None:
    label_arr = np.asarray(labels, dtype=np.int32)
    score_arr = np.asarray(probability, dtype=np.float32)
    valid_mask = np.isin(label_arr, (0, 1))
    if not np.any(valid_mask):
        return None
    label_arr = label_arr[valid_mask]
    score_arr = score_arr[valid_mask]
    if np.unique(label_arr).size < 2:
        return None
    return compute_binary_classification_metrics(label_arr, score_arr)


def _blend_probability(
    *,
    gnn_probability: np.ndarray,
    secondary_probability: np.ndarray,
    blend_alpha: float,
) -> np.ndarray:
    alpha = float(blend_alpha)
    gnn_logit = logit(np.clip(np.asarray(gnn_probability, dtype=np.float32), 1e-6, 1.0 - 1e-6))
    secondary_logit = logit(np.clip(np.asarray(secondary_probability, dtype=np.float32), 1e-6, 1.0 - 1e-6))
    return expit((1.0 - alpha) * gnn_logit + alpha * secondary_logit).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blend one saved thesis GNN backbone run with one saved official graphprop residual run. "
            "The hybrid keeps the shared dynamic GNN backbone and applies a fixed logit-space correction "
            "from the leakage-safe phase1-train graph propagation branch."
        )
    )
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--secondary-run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outdir", type=Path, default=BLEND_OUTPUT_ROOT)
    parser.add_argument("--blend-alpha", type=float, default=OFFICIAL_HYBRID_BLEND_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_run_dir = Path(args.base_run_dir).resolve()
    secondary_run_dir = Path(args.secondary_run_dir).resolve()

    base_summary = json.loads((base_run_dir / "summary.json").read_text(encoding="utf-8"))
    secondary_summary = json.loads((secondary_run_dir / "summary.json").read_text(encoding="utf-8"))

    base_dataset = str(base_summary.get("dataset") or "")
    if base_dataset and base_dataset != ACTIVE_DATASET_SPEC.name:
        raise ValueError(
            "Base run dataset mismatch: "
            f"expected `{ACTIVE_DATASET_SPEC.name}`, got `{base_dataset}` from {base_run_dir}."
        )
    secondary_dataset = str(secondary_summary.get("dataset") or ACTIVE_DATASET_SPEC.name)
    if secondary_dataset != ACTIVE_DATASET_SPEC.name:
        raise ValueError(
            "Secondary run dataset mismatch: "
            f"expected `{ACTIVE_DATASET_SPEC.name}`, got `{secondary_dataset}` from {secondary_run_dir}."
        )
    if str(base_summary.get("model_name")) != OFFICIAL_BACKBONE_MODEL:
        raise ValueError(f"The official thesis hybrid expects an `{OFFICIAL_BACKBONE_MODEL}` backbone run.")
    base_preset = str(base_summary.get("preset") or OFFICIAL_BACKBONE_PRESET)
    if base_preset != OFFICIAL_BACKBONE_PRESET:
        raise ValueError(
            f"The official thesis hybrid is locked to the unified `{OFFICIAL_BACKBONE_PRESET}` backbone."
        )
    feature_profile = str(base_summary.get("feature_profile") or OFFICIAL_BACKBONE_FEATURE_PROFILE)
    if feature_profile != OFFICIAL_BACKBONE_FEATURE_PROFILE:
        raise ValueError(
            f"The official thesis hybrid requires the shared `{OFFICIAL_BACKBONE_FEATURE_PROFILE}` feature contract."
        )

    train_ids, y_train, gnn_train_prob = _load_split_bundle(base_run_dir, "phase1_train")
    val_ids, y_val, gnn_val_prob = _load_split_bundle(base_run_dir, "phase1_val")
    sec_train_ids, _, sec_train_prob_raw = _load_split_bundle(secondary_run_dir, "phase1_train")
    sec_val_ids, _, sec_val_prob_raw = _load_split_bundle(secondary_run_dir, "phase1_val")
    aligned_train_ids, aligned_train_labels, aligned_gnn_train_prob, sec_train_prob = _shared_train_alignment(
        base_node_ids=train_ids,
        base_labels=y_train,
        base_probability=gnn_train_prob,
        secondary_node_ids=sec_train_ids,
        secondary_probability=sec_train_prob_raw,
    )
    sec_val_prob = _align_scores(
        reference_node_ids=val_ids,
        bundle_node_ids=sec_val_ids,
        bundle_score=sec_val_prob_raw,
    )

    blended_train_prob = _blend_probability(
        gnn_probability=aligned_gnn_train_prob,
        secondary_probability=sec_train_prob,
        blend_alpha=float(args.blend_alpha),
    )
    blended_val_prob = _blend_probability(
        gnn_probability=gnn_val_prob,
        secondary_probability=sec_val_prob,
        blend_alpha=float(args.blend_alpha),
    )

    gnn_train_metrics = _maybe_binary_metrics(aligned_train_labels, aligned_gnn_train_prob)
    gnn_val_metrics = _maybe_binary_metrics(y_val, gnn_val_prob)
    secondary_train_metrics = _maybe_binary_metrics(aligned_train_labels, sec_train_prob)
    secondary_val_metrics = _maybe_binary_metrics(y_val, sec_val_prob)
    blended_train_metrics = _maybe_binary_metrics(aligned_train_labels, blended_train_prob)
    blended_val_metrics = _maybe_binary_metrics(y_val, blended_val_prob)

    run_dir = ensure_dir(Path(args.outdir) / args.run_name)
    save_prediction_npz(
        run_dir / "phase1_train_blend_predictions.npz",
        aligned_train_ids,
        aligned_train_labels,
        blended_train_prob,
    )
    save_prediction_npz(run_dir / "phase1_val_blend_predictions.npz", val_ids, y_val, blended_val_prob)

    summary = {
        "model_name": "thesis_gnn_primary_hybrid",
        "run_name": args.run_name,
        "dataset": ACTIVE_DATASET_SPEC.name,
        "feature_profile": feature_profile,
        "base_run_dir": _path_repr(base_run_dir),
        "secondary_run_dir": _path_repr(secondary_run_dir),
        "base_preset": base_preset,
        "blend_alpha": float(args.blend_alpha),
        "hybrid_principle": (
            "UTPM dynamic GNN backbone + leakage-safe phase1-train graphprop residual head "
            "under one shared feature contract"
        ),
        "dataset_isolation": True,
        "cross_dataset_training": False,
        "same_backbone_across_datasets": True,
        "same_secondary_architecture_across_datasets": True,
        "secondary_training_split": "phase1_train_only",
        "leakage_safeguards": [
            "The GNN backbone run must belong to the active dataset namespace.",
            "The graphprop residual run is fit only on the active dataset's phase1_train split.",
            "Validation rows are only used for evaluation of saved prediction bundles.",
            "Both branches consume the same dataset-scoped unified feature pipeline and never mix datasets.",
        ],
        "secondary_model": {
            "family": OFFICIAL_HYBRID_SECONDARY_MODEL,
            "source_summary_path": _path_repr(secondary_run_dir / "summary.json"),
        },
        "full_backbone_train_size": int(train_ids.size),
        "shared_train_size": int(aligned_train_ids.size),
        "val_size": int(val_ids.size),
        "gnn_train_auc": None if gnn_train_metrics is None else float(gnn_train_metrics["auc"]),
        "gnn_val_auc": None if gnn_val_metrics is None else float(gnn_val_metrics["auc"]),
        "secondary_train_auc": None if secondary_train_metrics is None else float(secondary_train_metrics["auc"]),
        "secondary_val_auc": None if secondary_val_metrics is None else float(secondary_val_metrics["auc"]),
        "train_auc_mean": None if blended_train_metrics is None else float(blended_train_metrics["auc"]),
        "val_auc_mean": None if blended_val_metrics is None else float(blended_val_metrics["auc"]),
        "train_pr_auc_mean": None if blended_train_metrics is None else float(blended_train_metrics["pr_auc"]),
        "val_pr_auc_mean": None if blended_val_metrics is None else float(blended_val_metrics["pr_auc"]),
        "train_ap_mean": None if blended_train_metrics is None else float(blended_train_metrics["ap"]),
        "val_ap_mean": None if blended_val_metrics is None else float(blended_val_metrics["ap"]),
        "phase1_train_avg_predictions": _path_repr(run_dir / "phase1_train_blend_predictions.npz"),
        "phase1_val_avg_predictions": _path_repr(run_dir / "phase1_val_blend_predictions.npz"),
        "gnn_backbone_train_prediction_path": _path_repr(resolve_prediction_path(base_run_dir, "phase1_train")),
        "gnn_backbone_val_prediction_path": _path_repr(resolve_prediction_path(base_run_dir, "phase1_val")),
        "secondary_train_prediction_path": _path_repr(resolve_prediction_path(secondary_run_dir, "phase1_train")),
        "secondary_val_prediction_path": _path_repr(resolve_prediction_path(secondary_run_dir, "phase1_val")),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        "[thesis_hybrid_blend] "
        f"dataset={ACTIVE_DATASET_SPEC.name} "
        f"base_run={_path_repr(base_run_dir)} "
        f"secondary_run={_path_repr(secondary_run_dir)} "
        f"blend_alpha={float(args.blend_alpha):.2f} "
        f"gnn_val_auc={summary['gnn_val_auc']:.6f} "
        f"secondary_val_auc={summary['secondary_val_auc']:.6f} "
        f"hybrid_val_auc={summary['val_auc_mean']:.6f}"
    )
    print(f"Hybrid blend finished: {run_dir}")


if __name__ == "__main__":
    main()
