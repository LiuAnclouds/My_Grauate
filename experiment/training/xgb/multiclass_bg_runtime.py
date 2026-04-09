from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiment.training.common import (
    compute_binary_classification_metrics,
    ensure_dir,
    save_prediction_npz,
    write_json,
)
from experiment.training.xgb_utils import (
    binary_score_from_softprob,
    build_multiclass_bg_sample_weight,
    multiclass_binary_auc,
    write_feature_importance_csv,
)


@dataclass(frozen=True)
class HistoricalMulticlassBgSplit:
    threshold_day: int
    historical_ids: np.ndarray
    val_ids: np.ndarray
    external_ids: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_external: np.ndarray
    train_first_active: np.ndarray


def build_historical_multiclass_bg_split(
    split,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    first_active: np.ndarray,
    *,
    include_future_background: bool = False,
) -> HistoricalMulticlassBgSplit:
    threshold_day = int(split.threshold_day)
    labels1 = np.asarray(phase1_y, dtype=np.int8)
    labels2 = np.asarray(phase2_y, dtype=np.int8)
    first_active_arr = np.asarray(first_active, dtype=np.int32)
    if include_future_background:
        train_mask = (
            ((first_active_arr <= threshold_day) & np.isin(labels1, (0, 1)))
            | np.isin(labels1, (2, 3))
        )
    else:
        train_mask = (first_active_arr <= threshold_day) & np.isin(labels1, (0, 1, 2, 3))

    historical_ids = np.flatnonzero(train_mask).astype(np.int32, copy=False)
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    return HistoricalMulticlassBgSplit(
        threshold_day=threshold_day,
        historical_ids=historical_ids,
        val_ids=val_ids,
        external_ids=external_ids,
        y_train=labels1[historical_ids].astype(np.int32, copy=False),
        y_val=labels1[val_ids].astype(np.int32, copy=False),
        y_external=labels2[external_ids].astype(np.int32, copy=False),
        train_first_active=first_active_arr[historical_ids].astype(np.int32, copy=False),
    )


def build_multiclass_bg_params(args: Any) -> dict[str, float | int | str]:
    return {
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


def train_multiclass_bg_xgb(
    args: Any,
    *,
    split_data: HistoricalMulticlassBgSplit,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_external: np.ndarray,
    feature_names: list[str],
    run_dir: Path,
    model_name: str,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import xgboost as xgb

    sample_weight_payload = build_multiclass_bg_sample_weight(
        split_data.y_train,
        fraud_weight_scale=float(args.fraud_weight_scale),
        background_weight=float(args.background_weight),
        time_weight_half_life_days=float(args.time_weight_half_life_days),
        time_weight_floor=float(args.time_weight_floor),
        train_first_active=split_data.train_first_active,
        threshold_day=split_data.threshold_day,
    )
    dtrain = xgb.DMatrix(
        np.asarray(x_train, dtype=np.float32),
        label=split_data.y_train,
        weight=np.asarray(sample_weight_payload["sample_weight"], dtype=np.float32),
    )
    dval = xgb.DMatrix(np.asarray(x_val, dtype=np.float32), label=split_data.y_val)
    dexternal = xgb.DMatrix(np.asarray(x_external, dtype=np.float32), label=split_data.y_external)

    params = build_multiclass_bg_params(args)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(args.n_estimators),
        evals=[(dval, "validation")],
        custom_metric=multiclass_binary_auc,
        maximize=True,
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=50,
    )
    best_iteration = int(booster.best_iteration)
    val_prob = booster.predict(dval, iteration_range=(0, best_iteration + 1)).reshape(-1, 4)
    external_prob = booster.predict(
        dexternal,
        iteration_range=(0, best_iteration + 1),
    ).reshape(-1, 4)
    val_score = binary_score_from_softprob(val_prob)
    external_score = binary_score_from_softprob(external_prob)
    val_metrics = compute_binary_classification_metrics(split_data.y_val, val_score)
    external_metrics = compute_binary_classification_metrics(split_data.y_external, external_score)

    run_path = ensure_dir(run_dir)
    save_prediction_npz(
        run_path / "phase1_val_predictions.npz",
        split_data.val_ids,
        split_data.y_val,
        val_score,
    )
    save_prediction_npz(
        run_path / "phase2_external_predictions.npz",
        split_data.external_ids,
        split_data.y_external,
        external_score,
    )
    booster.save_model(run_path / "model.json")
    write_feature_importance_csv(booster, feature_names, run_path / "feature_importance.csv")

    summary: dict[str, Any] = {
        "model": model_name,
        "run_name": getattr(args, "run_name", run_path.name),
        "seed": int(args.seed),
        "feature_dim": int(len(feature_names)),
        "threshold_day": int(split_data.threshold_day),
        "historical_train_size": int(split_data.historical_ids.size),
        "historical_train_label_counts": {
            str(label): int(np.sum(split_data.y_train == label))
            for label in (0, 1, 2, 3)
        },
        "class_weight": sample_weight_payload["class_weight"],
        "time_weight": sample_weight_payload["time_weight"],
        "best_iteration": best_iteration,
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "params": params,
        "prediction_paths": {
            "phase1_val": str(run_path / "phase1_val_predictions.npz"),
            "phase2_external": str(run_path / "phase2_external_predictions.npz"),
        },
    }
    if summary_extra:
        summary.update(summary_extra)
    write_json(run_path / "summary.json", summary)
    print(
        f"[{model_name}] run={summary['run_name']} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f} "
        f"best_iteration={best_iteration}"
    )
    return summary
