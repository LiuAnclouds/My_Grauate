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
from experiment.training.features import FeatureStore, resolve_feature_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU XGBoost with covariate-shift weighting from unlabeled val features.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-model", choices=("m2_hybrid", "m3_neighbor"), default="m3_neighbor")
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
    parser.add_argument("--n-estimators", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=400)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--weight-clip-min", type=float, default=0.25)
    parser.add_argument("--weight-clip-max", type=float, default=6.0)
    parser.add_argument("--weight-mix", type=float, default=1.0, help="0 means no adaptation, 1 means full adapted weights.")
    parser.add_argument("--domain-n-estimators", type=int, default=1200)
    parser.add_argument("--domain-learning-rate", type=float, default=0.03)
    parser.add_argument("--domain-max-depth", type=int, default=6)
    parser.add_argument("--domain-min-child-weight", type=float, default=20.0)
    parser.add_argument("--max-train-nodes", type=int, default=None)
    parser.add_argument("--max-val-nodes", type=int, default=None)
    parser.add_argument("--max-external-nodes", type=int, default=None)
    return parser.parse_args()


def _slice_node_ids(node_ids: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or node_ids.size <= limit:
        return np.asarray(node_ids, dtype=np.int32)
    rng = np.random.default_rng(seed)
    choice = rng.choice(node_ids.size, size=limit, replace=False)
    return np.sort(node_ids[choice].astype(np.int32, copy=False))


def _load_feature_slices(
    feature_dir: Path,
    feature_model: str,
    extra_groups: list[str] | None,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    groups = resolve_feature_groups(feature_model, extra_groups)
    phase1_store = FeatureStore("phase1", groups, outdir=feature_dir)
    phase2_store = FeatureStore("phase2", groups, outdir=feature_dir)
    x_train = phase1_store.take_rows(train_ids).astype(np.float32, copy=False)
    x_val = phase1_store.take_rows(val_ids).astype(np.float32, copy=False)
    x_external = phase2_store.take_rows(external_ids).astype(np.float32, copy=False)
    return x_train, x_val, x_external, list(phase1_store.feature_names)


def _booster_params(args: argparse.Namespace, scale_pos_weight: float) -> dict[str, float | int | str]:
    return {
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


def _domain_params(args: argparse.Namespace) -> dict[str, float | int | str]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": args.device,
        "learning_rate": args.domain_learning_rate,
        "n_estimators": args.domain_n_estimators,
        "max_depth": args.domain_max_depth,
        "min_child_weight": args.domain_min_child_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": args.max_bin,
        "early_stopping_rounds": 100,
        "random_state": args.seed + 1000,
    }


def main() -> None:
    args = parse_args()
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    set_global_seed(args.seed)
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)

    train_ids = _slice_node_ids(split.train_ids, args.max_train_nodes, seed=args.seed + 11)
    val_ids = _slice_node_ids(split.val_ids, args.max_val_nodes, seed=args.seed + 17)
    external_ids = _slice_node_ids(split.external_ids, args.max_external_nodes, seed=args.seed + 29)

    x_train, x_val, x_external, feature_names = _load_feature_slices(
        feature_dir=args.feature_dir,
        feature_model=args.feature_model,
        extra_groups=list(args.extra_groups),
        train_ids=train_ids,
        val_ids=val_ids,
        external_ids=external_ids,
    )
    y_train = phase1_y[train_ids]
    y_val = phase1_y[val_ids]
    y_external = phase2_y[external_ids]

    domain_x = np.concatenate([x_train, x_val], axis=0).astype(np.float32, copy=False)
    domain_y = np.concatenate(
        [
            np.zeros(x_train.shape[0], dtype=np.int8),
            np.ones(x_val.shape[0], dtype=np.int8),
        ]
    )
    rng = np.random.default_rng(args.seed + 404)
    perm = rng.permutation(domain_x.shape[0])
    domain_x = domain_x[perm]
    domain_y = domain_y[perm]
    split_at = int(domain_x.shape[0] * 0.85)
    domain_train_x = domain_x[:split_at]
    domain_train_y = domain_y[:split_at]
    domain_holdout_x = domain_x[split_at:]
    domain_holdout_y = domain_y[split_at:]

    domain_model = xgb.XGBClassifier(**_domain_params(args))
    domain_model.fit(
        domain_train_x,
        domain_train_y,
        eval_set=[(domain_holdout_x, domain_holdout_y)],
        verbose=50,
    )
    domain_holdout_prob = domain_model.predict_proba(domain_holdout_x)[:, 1]
    domain_auc = float(roc_auc_score(domain_holdout_y, domain_holdout_prob))

    train_domain_prob = domain_model.predict_proba(x_train)[:, 1].astype(np.float32, copy=False)
    odds = train_domain_prob / np.clip(1.0 - train_domain_prob, 1e-4, None)
    adapted = np.power(np.clip(odds, 1e-4, None), args.weight_power).astype(np.float32, copy=False)
    adapted = np.clip(adapted, args.weight_clip_min, args.weight_clip_max)
    sample_weight = ((1.0 - args.weight_mix) + args.weight_mix * adapted).astype(np.float32, copy=False)
    sample_weight /= float(np.mean(sample_weight))

    pos_count = float(np.sum(y_train == 1))
    neg_count = float(np.sum(y_train == 0))
    scale_pos_weight = neg_count / max(pos_count, 1.0)
    model = xgb.XGBClassifier(**_booster_params(args, scale_pos_weight=scale_pos_weight))
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(x_val, y_val)],
        verbose=50,
    )

    val_prob = model.predict_proba(x_val)[:, 1].astype(np.float32, copy=False)
    external_prob = model.predict_proba(x_external)[:, 1].astype(np.float32, copy=False)
    val_metrics = compute_binary_classification_metrics(y_val, val_prob)
    external_metrics = compute_binary_classification_metrics(y_external, external_prob)

    run_dir = ensure_dir(args.outdir / args.run_name)
    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, y_val, val_prob)
    save_prediction_npz(run_dir / "phase2_external_predictions.npz", external_ids, y_external, external_prob)
    domain_model.save_model(run_dir / "domain_model.json")
    model.save_model(run_dir / "model.json")

    summary = {
        "model": "xgboost_gpu_covshift",
        "run_name": args.run_name,
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "seed": args.seed,
        "feature_dim": int(len(feature_names)),
        "domain_auc": domain_auc,
        "weight_power": float(args.weight_power),
        "weight_clip_min": float(args.weight_clip_min),
        "weight_clip_max": float(args.weight_clip_max),
        "weight_mix": float(args.weight_mix),
        "sample_weight_summary": {
            "min": float(np.min(sample_weight)),
            "p50": float(np.median(sample_weight)),
            "p90": float(np.quantile(sample_weight, 0.9)),
            "p99": float(np.quantile(sample_weight, 0.99)),
            "max": float(np.max(sample_weight)),
            "mean": float(np.mean(sample_weight)),
        },
        "best_iteration": int(getattr(model, "best_iteration", 0)),
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "params": _booster_params(args, scale_pos_weight=scale_pos_weight),
        "domain_params": _domain_params(args),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[xgboost_gpu_covshift] run={args.run_name} "
        f"domain_auc={domain_auc:.6f} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f}"
    )


if __name__ == "__main__":
    main()
