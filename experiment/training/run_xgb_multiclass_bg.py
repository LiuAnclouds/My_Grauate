from __future__ import annotations

import argparse
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
from experiment.training.features import FeatureStore, load_graph_cache, resolve_feature_groups
from experiment.training.xgb_utils import (
    binary_score_from_softprob,
    build_multiclass_bg_sample_weight,
    multiclass_binary_auc,
    write_feature_importance_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU XGBoost multiclass training with historical background supervision.",
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
    parser.add_argument(
        "--background-weight",
        type=float,
        default=0.25,
        help="Relative total weight assigned to each background class vs. normal class total weight.",
    )
    parser.add_argument(
        "--fraud-weight-scale",
        type=float,
        default=1.0,
        help="Multiplier on the normal/fraud balancing ratio.",
    )
    parser.add_argument(
        "--time-weight-half-life-days",
        type=float,
        default=0.0,
        help="If > 0, exponentially upweight recent historical nodes toward the split threshold.",
    )
    parser.add_argument(
        "--time-weight-floor",
        type=float,
        default=0.25,
        help="Minimum recency weight when --time-weight-half-life-days > 0.",
    )
    parser.add_argument(
        "--include-future-background",
        action="store_true",
        help=(
            "Include all phase1 background nodes (labels 2/3) regardless of activation day, "
            "while still restricting 0/1 supervision to the historical side of the split."
        ),
    )
    parser.add_argument(
        "--min-train-first-active-day",
        type=int,
        default=0,
        help="Optional lower bound on phase1 historical node activation day used for training.",
    )
    return parser.parse_args()


def _build_feature_matrices(
    feature_dir: Path,
    feature_model: str,
    extra_groups: list[str],
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


def _multiclass_binary_auc(predt: np.ndarray, dmatrix) -> tuple[str, float]:
    return multiclass_binary_auc(predt, dmatrix)


def _binary_score_from_softprob(prob: np.ndarray) -> np.ndarray:
    return binary_score_from_softprob(prob)


def _build_sample_weight(
    y_train: np.ndarray,
    args: argparse.Namespace,
    train_first_active: np.ndarray | None = None,
    threshold_day: int | None = None,
) -> dict[str, float | dict[str, float]]:
    return build_multiclass_bg_sample_weight(
        y_train,
        fraud_weight_scale=float(args.fraud_weight_scale),
        background_weight=float(args.background_weight),
        time_weight_half_life_days=float(args.time_weight_half_life_days),
        time_weight_floor=float(args.time_weight_floor),
        train_first_active=train_first_active,
        threshold_day=threshold_day,
    )


def _write_feature_importance(
    booster,
    feature_names: list[str],
    path: Path,
) -> None:
    write_feature_importance_csv(booster, feature_names, path)


def main() -> None:
    args = parse_args()
    import xgboost as xgb

    set_global_seed(args.seed)
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    phase1_graph = load_graph_cache("phase1", outdir=args.feature_dir)
    first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)

    min_train_day = int(max(args.min_train_first_active_day, 0))
    if args.include_future_background:
        train_mask = (
            ((first_active <= int(split.threshold_day)) & (first_active >= min_train_day) & np.isin(phase1_y, (0, 1)))
            | ((first_active >= min_train_day) & np.isin(phase1_y, (2, 3)))
        )
    else:
        train_mask = (
            (first_active <= int(split.threshold_day))
            & (first_active >= min_train_day)
            & np.isin(phase1_y, (0, 1, 2, 3))
        )
    historical_ids = np.flatnonzero(train_mask).astype(np.int32, copy=False)
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)

    x_train, x_val, x_external, feature_names = _build_feature_matrices(
        feature_dir=args.feature_dir,
        feature_model=args.feature_model,
        extra_groups=list(args.extra_groups),
        train_ids=historical_ids,
        val_ids=val_ids,
        external_ids=external_ids,
    )
    y_train = phase1_y[historical_ids].astype(np.int32, copy=False)
    y_val = phase1_y[val_ids].astype(np.int32, copy=False)
    y_external = phase2_y[external_ids].astype(np.int32, copy=False)
    train_first_active = first_active[historical_ids].astype(np.int32, copy=False)

    sample_weight_payload = _build_sample_weight(
        y_train,
        args,
        train_first_active=train_first_active,
        threshold_day=int(split.threshold_day),
    )
    sample_weight = sample_weight_payload["sample_weight"]

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(x_val, label=y_val)
    dexternal = xgb.DMatrix(x_external, label=y_external)

    params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "tree_method": "hist",
        "device": args.device,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "seed": args.seed,
        "disable_default_eval_metric": 1,
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(args.n_estimators),
        evals=[(dval, "validation")],
        custom_metric=_multiclass_binary_auc,
        maximize=True,
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=50,
    )

    val_prob = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1)).reshape(-1, 4)
    external_prob = booster.predict(
        dexternal,
        iteration_range=(0, booster.best_iteration + 1),
    ).reshape(-1, 4)
    val_score = _binary_score_from_softprob(val_prob)
    external_score = _binary_score_from_softprob(external_prob)
    val_metrics = compute_binary_classification_metrics(y_val, val_score)
    external_metrics = compute_binary_classification_metrics(y_external, external_score)

    run_dir = ensure_dir(args.outdir / args.run_name)
    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, y_val, val_score)
    save_prediction_npz(run_dir / "phase2_external_predictions.npz", external_ids, y_external, external_score)
    booster.save_model(run_dir / "model.json")
    _write_feature_importance(booster, feature_names, run_dir / "feature_importance.csv")

    summary = {
        "model": "xgboost_gpu_multiclass_bg",
        "run_name": args.run_name,
        "seed": args.seed,
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "include_future_background": bool(args.include_future_background),
        "min_train_first_active_day": min_train_day,
        "feature_dim": int(len(feature_names)),
        "threshold_day": int(split.threshold_day),
        "historical_train_size": int(historical_ids.size),
        "historical_train_label_counts": {
            str(label): int(np.sum(y_train == label))
            for label in (0, 1, 2, 3)
        },
        "class_weight": sample_weight_payload["class_weight"],
        "time_weight": sample_weight_payload["time_weight"],
        "best_iteration": int(booster.best_iteration),
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "params": params,
        "prediction_paths": {
            "phase1_val": str(run_dir / "phase1_val_predictions.npz"),
            "phase2_external": str(run_dir / "phase2_external_predictions.npz"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[xgboost_gpu_multiclass_bg] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f} "
        f"best_iteration={summary['best_iteration']}"
    )


if __name__ == "__main__":
    main()
