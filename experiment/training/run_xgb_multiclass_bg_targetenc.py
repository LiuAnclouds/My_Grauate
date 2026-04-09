from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (
    FEATURE_OUTPUT_ROOT,
    MODEL_OUTPUT_ROOT,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    load_phase_arrays,
    save_prediction_npz,
    set_global_seed,
    write_json,
)
from experiment.training.features import load_feature_manifest, load_graph_cache
from experiment.training.run_xgb_multiclass_bg import (
    _binary_score_from_softprob,
    _build_feature_matrices,
    _build_sample_weight,
    _multiclass_binary_auc,
)
from experiment.training.xgb.multiclass_bg_runtime import (
    build_historical_multiclass_bg_split,
    train_multiclass_bg_xgb,
)


RAW_FEATURE_COUNT = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU XGBoost multiclass training with historical background supervision plus raw-feature target encoding.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--feature-model",
        choices=("m2_hybrid", "m3_neighbor"),
        default="m3_neighbor",
    )
    parser.add_argument(
        "--extra-groups",
        nargs="*",
        default=(),
        help="Optional extra offline feature groups appended to the feature_model.",
    )
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "xgboost_gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=300)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--fraud-weight-scale", type=float, default=1.0)
    parser.add_argument("--time-weight-half-life-days", type=float, default=0.0)
    parser.add_argument("--time-weight-floor", type=float, default=0.25)
    parser.add_argument("--te-bins", type=int, default=32)
    parser.add_argument("--te-smoothing", type=float, default=200.0)
    return parser.parse_args()


def _load_raw_feature_rows(
    phase: str,
    node_ids: np.ndarray,
    feature_dir: Path,
) -> np.ndarray:
    manifest = load_feature_manifest(phase, outdir=feature_dir)
    core = np.load(feature_dir / phase / manifest["core_file"], mmap_mode="r")
    raw_spec = manifest["core_groups"]["raw_x"]
    rows = np.asarray(node_ids, dtype=np.int32)
    return np.asarray(core[rows, raw_spec["start"] : raw_spec["end"]], dtype=np.float32)


def _stable_quantile_edges(
    values: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    valid = values[values != -1.0]
    if valid.size == 0:
        return np.empty(0, dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, num=max(int(n_bins), 2) + 1, dtype=np.float64)[1:-1]
    if quantiles.size == 0:
        return np.empty(0, dtype=np.float32)
    edges = np.quantile(valid, quantiles).astype(np.float32, copy=False)
    if edges.size == 0:
        return edges
    return np.unique(edges)


def _digitize_with_missing(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.zeros(values.shape[0], dtype=np.int32)
    valid_mask = values != -1.0
    if np.any(valid_mask):
        bins[valid_mask] = np.digitize(values[valid_mask], edges, right=False).astype(np.int32) + 1
    return bins


def _safe_prob(
    numerator: np.ndarray,
    denominator: np.ndarray,
    prior: float,
    smoothing: float,
) -> np.ndarray:
    return ((numerator + smoothing * prior) / np.maximum(denominator + smoothing, 1.0)).astype(
        np.float32,
        copy=False,
    )


def _logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)


def _build_target_encoding_features(
    raw_train: np.ndarray,
    y_train: np.ndarray,
    raw_val: np.ndarray,
    raw_external: np.ndarray,
    n_bins: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    external_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    global_fg_mask = np.isin(y_train, (0, 1))
    global_fg_pos_rate = float(np.mean(y_train[global_fg_mask] == 1)) if np.any(global_fg_mask) else 0.5
    for feature_idx in range(RAW_FEATURE_COUNT):
        train_values = np.asarray(raw_train[:, feature_idx], dtype=np.float32)
        val_values = np.asarray(raw_val[:, feature_idx], dtype=np.float32)
        external_values = np.asarray(raw_external[:, feature_idx], dtype=np.float32)

        edges = _stable_quantile_edges(train_values, n_bins=n_bins)
        train_bin = _digitize_with_missing(train_values, edges)
        val_bin = _digitize_with_missing(val_values, edges)
        external_bin = _digitize_with_missing(external_values, edges)
        num_states = int(max(train_bin.max(initial=0), val_bin.max(initial=0), external_bin.max(initial=0))) + 1

        total_count = np.bincount(train_bin, minlength=num_states).astype(np.float32, copy=False)
        fg_count = np.bincount(train_bin[global_fg_mask], minlength=num_states).astype(np.float32, copy=False)
        fraud_count = np.bincount(train_bin[y_train == 1], minlength=num_states).astype(np.float32, copy=False)
        total_train = total_count[train_bin]
        fg_train = fg_count[train_bin]
        fraud_train = fraud_count[train_bin]

        total_train_adj = total_train - 1.0
        fg_train_adj = fg_train - np.isin(y_train, (0, 1)).astype(np.float32)
        fraud_train_adj = fraud_train - (y_train == 1).astype(np.float32)

        te_train_fg_prob = _safe_prob(
            numerator=fraud_train_adj,
            denominator=fg_train_adj,
            prior=global_fg_pos_rate,
            smoothing=smoothing,
        )
        te_val_fg_prob = _safe_prob(
            numerator=fraud_count[val_bin],
            denominator=fg_count[val_bin],
            prior=global_fg_pos_rate,
            smoothing=smoothing,
        )
        te_external_fg_prob = _safe_prob(
            numerator=fraud_count[external_bin],
            denominator=fg_count[external_bin],
            prior=global_fg_pos_rate,
            smoothing=smoothing,
        )

        train_support = np.log1p(np.maximum(total_train_adj, 0.0)).astype(np.float32, copy=False)
        val_support = np.log1p(total_count[val_bin]).astype(np.float32, copy=False)
        external_support = np.log1p(total_count[external_bin]).astype(np.float32, copy=False)

        train_blocks.extend(
            [
                te_train_fg_prob.reshape(-1, 1),
                _logit(te_train_fg_prob).reshape(-1, 1),
                train_support.reshape(-1, 1),
            ]
        )
        val_blocks.extend(
            [
                te_val_fg_prob.reshape(-1, 1),
                _logit(te_val_fg_prob).reshape(-1, 1),
                val_support.reshape(-1, 1),
            ]
        )
        external_blocks.extend(
            [
                te_external_fg_prob.reshape(-1, 1),
                _logit(te_external_fg_prob).reshape(-1, 1),
                external_support.reshape(-1, 1),
            ]
        )
        feature_names.extend(
            [
                f"te_x{feature_idx}_fg_prob",
                f"te_x{feature_idx}_fg_logit",
                f"te_x{feature_idx}_support_log",
            ]
        )

    return (
        np.concatenate(train_blocks, axis=1).astype(np.float32, copy=False),
        np.concatenate(val_blocks, axis=1).astype(np.float32, copy=False),
        np.concatenate(external_blocks, axis=1).astype(np.float32, copy=False),
        feature_names,
    )


def _write_feature_importance(
    booster,
    feature_names: list[str],
    path: Path,
) -> None:
    scores = booster.get_score(importance_type="gain")
    rows = []
    for idx, feature_name in enumerate(feature_names):
        rows.append({"feature_name": feature_name, "gain": float(scores.get(f"f{idx}", 0.0))})
    rows.sort(key=lambda row: row["gain"], reverse=True)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["feature_name", "gain"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    phase1_graph = load_graph_cache("phase1", outdir=args.feature_dir)
    first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)
    split_data = build_historical_multiclass_bg_split(
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        first_active=first_active,
        include_future_background=False,
    )

    x_train, x_val, x_external, feature_names = _build_feature_matrices(
        feature_dir=args.feature_dir,
        feature_model=args.feature_model,
        extra_groups=list(args.extra_groups),
        train_ids=split_data.historical_ids,
        val_ids=split_data.val_ids,
        external_ids=split_data.external_ids,
    )
    raw_train = _load_raw_feature_rows("phase1", split_data.historical_ids, args.feature_dir)
    raw_val = _load_raw_feature_rows("phase1", split_data.val_ids, args.feature_dir)
    raw_external = _load_raw_feature_rows("phase2", split_data.external_ids, args.feature_dir)

    te_train, te_val, te_external, te_feature_names = _build_target_encoding_features(
        raw_train=raw_train,
        y_train=split_data.y_train,
        raw_val=raw_val,
        raw_external=raw_external,
        n_bins=int(args.te_bins),
        smoothing=float(args.te_smoothing),
    )
    x_train = np.concatenate([x_train, te_train], axis=1).astype(np.float32, copy=False)
    x_val = np.concatenate([x_val, te_val], axis=1).astype(np.float32, copy=False)
    x_external = np.concatenate([x_external, te_external], axis=1).astype(np.float32, copy=False)
    feature_names = list(feature_names) + list(te_feature_names)

    run_dir = ensure_dir(args.outdir / args.run_name)
    train_multiclass_bg_xgb(
        args,
        split_data=split_data,
        x_train=x_train,
        x_val=x_val,
        x_external=x_external,
        feature_names=feature_names,
        run_dir=run_dir,
        model_name="xgboost_gpu_multiclass_bg_targetenc",
        summary_extra={
            "feature_model": args.feature_model,
            "extra_groups": list(args.extra_groups),
            "target_encoding": {
                "te_bins": int(args.te_bins),
                "te_smoothing": float(args.te_smoothing),
                "te_feature_dim": int(len(te_feature_names)),
            },
        },
    )


if __name__ == "__main__":
    main()
