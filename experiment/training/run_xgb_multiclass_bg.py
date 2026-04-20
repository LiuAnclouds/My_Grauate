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
from experiment.training.xgb.domain_adaptation import (
    add_domain_weight_args,
    build_domain_adaptation_weights_from_args,
)
from experiment.training.xgb_utils import (
    binary_score_from_softprob,
    build_multiclass_bg_sample_weight,
    multiclass_binary_auc,
    write_feature_importance_csv,
)


def _split_phase_names(split) -> tuple[str, str, str]:
    train_phase = str(split.train_phase)
    val_phase = str(split.val_phase)
    external_phase = str(split.external_phase) if split.external_phase else val_phase
    return train_phase, val_phase, external_phase


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
    add_domain_weight_args(parser)
    return parser.parse_args()


def _build_feature_matrices(
    feature_dir: Path,
    feature_model: str,
    extra_groups: list[str],
    train_phase: str,
    val_phase: str,
    external_phase: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    groups = resolve_feature_groups(feature_model, extra_groups)
    train_store = FeatureStore(train_phase, groups, outdir=feature_dir)
    if val_phase == train_phase:
        val_store = train_store
    else:
        val_store = FeatureStore(val_phase, groups, outdir=feature_dir)
    x_train = train_store.take_rows(train_ids).astype(np.float32, copy=False)
    x_val = val_store.take_rows(val_ids).astype(np.float32, copy=False)
    if external_ids.size:
        if external_phase == train_phase:
            external_store = train_store
        elif external_phase == val_phase:
            external_store = val_store
        else:
            external_store = FeatureStore(external_phase, groups, outdir=feature_dir)
        x_external = external_store.take_rows(external_ids).astype(np.float32, copy=False)
    else:
        x_external = np.empty((0, x_train.shape[1]), dtype=np.float32)
    return x_train, x_val, x_external, list(train_store.feature_names)


def _multiclass_binary_auc(predt: np.ndarray, dmatrix) -> tuple[str, float]:
    return multiclass_binary_auc(predt, dmatrix)


def _binary_score_from_softprob(prob: np.ndarray) -> np.ndarray:
    return binary_score_from_softprob(prob)


def _safe_metrics_or_none(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float] | None:
    labels = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(y_score, dtype=np.float32)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return compute_binary_classification_metrics(labels, score)


def _build_historical_ids(
    *,
    split,
    phase1_y: np.ndarray,
    first_active: np.ndarray,
    min_train_day: int,
    include_future_background: bool,
) -> tuple[np.ndarray, str]:
    if str(split.split_style) == "single_graph":
        split_train_ids = np.asarray(split.train_ids, dtype=np.int32)
        if min_train_day > 0:
            train_mask = np.asarray(first_active[split_train_ids], dtype=np.int32) >= int(min_train_day)
            historical_ids = split_train_ids[train_mask].astype(np.int32, copy=False)
            return historical_ids, "split_train_ids_recent_start"
        return split_train_ids, "split_train_ids"

    if include_future_background:
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
    return historical_ids, "threshold_mask"


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
    train_phase, val_phase, external_phase = _split_phase_names(split)
    phase1_y = np.asarray(load_phase_arrays(train_phase, keys=("y",))["y"], dtype=np.int8)
    phase1_graph = load_graph_cache(train_phase, outdir=args.feature_dir)
    if val_phase == train_phase:
        val_y = phase1_y
    else:
        val_y = np.asarray(load_phase_arrays(val_phase, keys=("y",))["y"], dtype=np.int8)
    if external_phase == train_phase:
        phase2_y = phase1_y
    elif external_phase == val_phase:
        phase2_y = val_y
    else:
        phase2_y = np.asarray(load_phase_arrays(external_phase, keys=("y",))["y"], dtype=np.int8)
    first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)

    min_train_day = int(max(args.min_train_first_active_day, 0))
    historical_ids, historical_selection_mode = _build_historical_ids(
        split=split,
        phase1_y=phase1_y,
        first_active=first_active,
        min_train_day=min_train_day,
        include_future_background=bool(args.include_future_background),
    )
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    overlap = np.intersect1d(historical_ids, val_ids)
    if overlap.size:
        raise ValueError(
            "Historical train ids overlap validation ids. "
            f"split_style={split.split_style} overlap={overlap.size}"
        )
    external_ids = np.asarray(split.external_ids, dtype=np.int32)

    x_train, x_val, x_external, feature_names = _build_feature_matrices(
        feature_dir=args.feature_dir,
        feature_model=args.feature_model,
        extra_groups=list(args.extra_groups),
        train_phase=train_phase,
        val_phase=val_phase,
        external_phase=external_phase,
        train_ids=historical_ids,
        val_ids=val_ids,
        external_ids=external_ids,
    )
    y_train = phase1_y[historical_ids].astype(np.int32, copy=False)
    y_val = val_y[val_ids].astype(np.int32, copy=False)
    y_external = phase2_y[external_ids].astype(np.int32, copy=False)
    train_first_active = first_active[historical_ids].astype(np.int32, copy=False)
    num_class = max(
        2,
        int(np.max(y_train)) + 1 if y_train.size else 2,
        int(np.max(y_val)) + 1 if y_val.size else 2,
        int(np.max(y_external)) + 1 if y_external.size else 2,
    )

    sample_weight_payload = _build_sample_weight(
        y_train,
        args,
        train_first_active=train_first_active,
        threshold_day=int(split.threshold_day),
    )
    sample_weight = np.asarray(sample_weight_payload["sample_weight"], dtype=np.float32)
    domain_weight, domain_weight_payload = build_domain_adaptation_weights_from_args(
        args,
        x_train=x_train,
        x_val=x_val,
    )
    sample_weight *= np.asarray(domain_weight, dtype=np.float32)
    mean_weight = float(np.mean(sample_weight, dtype=np.float64))
    if mean_weight > 0.0:
        sample_weight /= mean_weight
    sample_weight_payload["sample_weight"] = sample_weight
    sample_weight_payload["domain_weight"] = domain_weight_payload

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(x_val, label=y_val)
    dexternal = xgb.DMatrix(x_external, label=y_external) if external_ids.size else None

    params = {
        "objective": "multi:softprob",
        "num_class": int(num_class),
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

    val_prob = booster.predict(
        dval,
        iteration_range=(0, booster.best_iteration + 1),
    ).reshape(-1, int(num_class))
    val_score = _binary_score_from_softprob(val_prob)
    val_metrics = compute_binary_classification_metrics(y_val, val_score)
    if dexternal is not None:
        external_prob = booster.predict(
            dexternal,
            iteration_range=(0, booster.best_iteration + 1),
        ).reshape(-1, int(num_class))
        external_score = _binary_score_from_softprob(external_prob)
        external_metrics = _safe_metrics_or_none(y_external, external_score)
    else:
        external_score = np.empty(0, dtype=np.float32)
        external_metrics = None

    run_dir = ensure_dir(args.outdir / args.run_name)
    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, y_val, val_score)
    if external_ids.size:
        save_prediction_npz(run_dir / "phase2_external_predictions.npz", external_ids, y_external, external_score)
    booster.save_model(run_dir / "model.json")
    _write_feature_importance(booster, feature_names, run_dir / "feature_importance.csv")

    summary = {
        "model": "xgboost_gpu_multiclass_bg",
        "run_name": args.run_name,
        "seed": args.seed,
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "train_phase": train_phase,
        "val_phase": val_phase,
        "external_phase": None if split.external_phase is None else str(split.external_phase),
        "num_class": int(num_class),
        "historical_selection_mode": historical_selection_mode,
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
        "domain_weight": sample_weight_payload["domain_weight"],
        "best_iteration": int(booster.best_iteration),
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "params": params,
        "prediction_paths": {
            "phase1_val": str(run_dir / "phase1_val_predictions.npz"),
        },
    }
    if external_ids.size:
        summary["prediction_paths"]["phase2_external"] = str(run_dir / "phase2_external_predictions.npz")
    write_json(run_dir / "summary.json", summary)
    external_auc_text = "n/a" if external_metrics is None else f"{external_metrics['auc']:.6f}"
    print(
        f"[xgboost_gpu_multiclass_bg] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_auc_text} "
        f"best_iteration={summary['best_iteration']}"
    )


if __name__ == "__main__":
    main()
