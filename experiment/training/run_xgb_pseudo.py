from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (
    MODEL_OUTPUT_ROOT,
    compute_binary_classification_metrics,
    ensure_dir,
    load_prediction_npz,
    load_experiment_split,
    load_phase_arrays,
    resolve_prediction_path,
    save_prediction_npz,
    set_global_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a second-stage GPU XGBoost using phase1 labels plus high-confidence "
            "pseudo labels from an existing phase2 prediction run."
        )
    )
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        required=True,
        help="Existing model run directory with summary.json and phase2_external predictions.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Output run name under experiment/outputs/training/models/xgboost_gpu/.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=MODEL_OUTPUT_ROOT / "xgboost_gpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-child-weight", type=float, default=6.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--pseudo-pos-threshold", type=float, default=0.97)
    parser.add_argument("--pseudo-neg-threshold", type=float, default=0.03)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    parser.add_argument(
        "--pseudo-neg-ratio-cap",
        type=float,
        default=20.0,
        help="Keep at most this many pseudo negatives per pseudo positive.",
    )
    parser.add_argument(
        "--max-pseudo-nodes",
        type=int,
        default=200000,
        help="Optional cap on total pseudo-labeled phase2 nodes kept for training.",
    )
    return parser.parse_args()

def _select_pseudo_indices(
    probabilities: np.ndarray,
    pos_threshold: float,
    neg_threshold: float,
    neg_ratio_cap: float,
    max_nodes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    pos_idx = np.flatnonzero(probabilities >= pos_threshold).astype(np.int32, copy=False)
    neg_idx = np.flatnonzero(probabilities <= neg_threshold).astype(np.int32, copy=False)

    if pos_idx.size and neg_idx.size:
        neg_cap = int(np.ceil(pos_idx.size * max(neg_ratio_cap, 1.0)))
        if neg_idx.size > neg_cap:
            neg_order = np.argsort(probabilities[neg_idx], kind="stable")
            neg_idx = neg_idx[neg_order[:neg_cap]].astype(np.int32, copy=False)

    if max_nodes is not None and (pos_idx.size + neg_idx.size) > max_nodes:
        pos_score = probabilities[pos_idx]
        neg_score = 1.0 - probabilities[neg_idx]
        merged_idx = np.concatenate([pos_idx, neg_idx]).astype(np.int32, copy=False)
        merged_label = np.concatenate(
            [
                np.ones(pos_idx.size, dtype=np.int8),
                np.zeros(neg_idx.size, dtype=np.int8),
            ]
        )
        merged_score = np.concatenate([pos_score, neg_score]).astype(np.float32, copy=False)
        top_order = np.argsort(-merged_score, kind="stable")[:max_nodes]
        kept_idx = merged_idx[top_order]
        kept_label = merged_label[top_order]
        pos_idx = kept_idx[kept_label == 1].astype(np.int32, copy=False)
        neg_idx = kept_idx[kept_label == 0].astype(np.int32, copy=False)

    return pos_idx, neg_idx


def main() -> None:
    args = parse_args()
    import xgboost as xgb

    set_global_seed(args.seed)
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)

    summary = json.loads((args.base_run_dir / "summary.json").read_text(encoding="utf-8"))
    cache_dir_value = summary.get("cache_dir")
    if not cache_dir_value:
        raise KeyError(f"{args.base_run_dir}: summary.json does not expose cache_dir.")
    cache_dir = Path(cache_dir_value)

    x_train = np.load(cache_dir / "phase1_train.npy", mmap_mode="r")
    x_val = np.load(cache_dir / "phase1_val.npy", mmap_mode="r")
    x_external = np.load(cache_dir / "phase2_external.npy", mmap_mode="r")

    external_bundle = load_prediction_npz(resolve_prediction_path(args.base_run_dir, "phase2_external"))
    if not np.array_equal(external_bundle["node_ids"], split.external_ids):
        raise AssertionError("phase2 external node ids do not align with recommended split.")

    pseudo_pos_idx, pseudo_neg_idx = _select_pseudo_indices(
        probabilities=external_bundle["probability"],
        pos_threshold=float(args.pseudo_pos_threshold),
        neg_threshold=float(args.pseudo_neg_threshold),
        neg_ratio_cap=float(args.pseudo_neg_ratio_cap),
        max_nodes=int(args.max_pseudo_nodes) if args.max_pseudo_nodes is not None else None,
    )
    if pseudo_pos_idx.size == 0 or pseudo_neg_idx.size == 0:
        raise RuntimeError(
            "Pseudo-labeled phase2 selection is empty. Relax thresholds or increase max_pseudo_nodes."
        )

    pseudo_idx = np.concatenate([pseudo_pos_idx, pseudo_neg_idx]).astype(np.int32, copy=False)
    pseudo_y = np.concatenate(
        [
            np.ones(pseudo_pos_idx.size, dtype=np.int8),
            np.zeros(pseudo_neg_idx.size, dtype=np.int8),
        ]
    )
    pseudo_conf = np.concatenate(
        [
            external_bundle["probability"][pseudo_pos_idx],
            1.0 - external_bundle["probability"][pseudo_neg_idx],
        ]
    ).astype(np.float32, copy=False)
    pseudo_weight = (float(args.pseudo_weight) * np.clip(pseudo_conf, 0.5, 1.0)).astype(
        np.float32,
        copy=False,
    )

    y_train = phase1_y[split.train_ids]
    y_val = phase1_y[split.val_ids]
    y_external = phase2_y[split.external_ids]

    combined_x = np.concatenate(
        [
            np.asarray(x_train, dtype=np.float32),
            np.asarray(x_external[pseudo_idx], dtype=np.float32),
        ],
        axis=0,
    )
    combined_y = np.concatenate([y_train, pseudo_y]).astype(np.int8, copy=False)
    combined_weight = np.concatenate(
        [
            np.ones(y_train.shape[0], dtype=np.float32),
            pseudo_weight,
        ]
    ).astype(np.float32, copy=False)

    pos_count = float(np.sum(combined_y == 1))
    neg_count = float(np.sum(combined_y == 0))
    scale_pos_weight = neg_count / max(pos_count, 1.0)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": args.device,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": args.early_stopping_rounds,
        "random_state": args.seed,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        combined_x,
        combined_y,
        sample_weight=combined_weight,
        eval_set=[(np.asarray(x_val, dtype=np.float32), y_val)],
        verbose=50,
    )

    val_prob = model.predict_proba(np.asarray(x_val, dtype=np.float32))[:, 1].astype(np.float32, copy=False)
    external_prob = model.predict_proba(np.asarray(x_external, dtype=np.float32))[:, 1].astype(
        np.float32,
        copy=False,
    )
    val_metrics = compute_binary_classification_metrics(y_val, val_prob)
    external_metrics = compute_binary_classification_metrics(y_external, external_prob)

    run_dir = ensure_dir(args.outdir / args.run_name)
    save_prediction_npz(run_dir / "phase1_val_predictions.npz", split.val_ids, y_val, val_prob)
    save_prediction_npz(
        run_dir / "phase2_external_predictions.npz",
        split.external_ids,
        y_external,
        external_prob,
    )
    model.save_model(run_dir / "model.json")

    summary_payload = {
        "model": "xgboost_gpu_pseudo",
        "run_name": args.run_name,
        "seed": args.seed,
        "source_run_dir": str(args.base_run_dir),
        "source_cache_dir": str(cache_dir),
        "phase1_train_size": int(split.train_ids.size),
        "phase1_val_size": int(split.val_ids.size),
        "phase2_external_size": int(split.external_ids.size),
        "pseudo_positive_count": int(pseudo_pos_idx.size),
        "pseudo_negative_count": int(pseudo_neg_idx.size),
        "pseudo_positive_threshold": float(args.pseudo_pos_threshold),
        "pseudo_negative_threshold": float(args.pseudo_neg_threshold),
        "pseudo_weight_mean": float(np.mean(pseudo_weight)),
        "best_iteration": int(getattr(model, "best_iteration", 0)),
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "params": params,
    }
    write_json(run_dir / "summary.json", summary_payload)
    print(
        f"[xgboost_gpu_pseudo] run={args.run_name} "
        f"pseudo_pos={pseudo_pos_idx.size} "
        f"pseudo_neg={pseudo_neg_idx.size} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f}"
    )


if __name__ == "__main__":
    main()
