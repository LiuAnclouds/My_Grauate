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
from experiment.training.xgb.domain_adaptation import build_domain_adaptation_weights_from_args


@dataclass(frozen=True)
class HistoricalMulticlassBgSplit:
    threshold_day: int
    min_train_first_active_day: int
    historical_selection_mode: str
    historical_ids: np.ndarray
    val_ids: np.ndarray
    external_ids: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_external: np.ndarray
    train_first_active: np.ndarray


def _safe_binary_metrics_or_none(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float] | None:
    label_arr = np.asarray(y_true, dtype=np.int32)
    score_arr = np.asarray(y_score, dtype=np.float32)
    valid_mask = np.isin(label_arr, (0, 1))
    if not np.any(valid_mask):
        return None
    label_arr = label_arr[valid_mask]
    score_arr = score_arr[valid_mask]
    if np.unique(label_arr).size < 2:
        return None
    return compute_binary_classification_metrics(label_arr, score_arr)


def build_historical_multiclass_bg_split(
    split,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    first_active: np.ndarray,
    *,
    include_future_background: bool = False,
    min_train_first_active_day: int = 0,
) -> HistoricalMulticlassBgSplit:
    threshold_day = int(split.threshold_day)
    min_train_day = int(max(min_train_first_active_day, 0))
    labels1 = np.asarray(phase1_y, dtype=np.int8)
    labels2 = np.asarray(phase2_y, dtype=np.int8)
    first_active_arr = np.asarray(first_active, dtype=np.int32)
    if str(split.split_style) == "single_graph":
        split_train_ids = np.asarray(split.train_ids, dtype=np.int32)
        if min_train_day > 0:
            train_mask = first_active_arr[split_train_ids] >= min_train_day
            historical_ids = split_train_ids[train_mask].astype(np.int32, copy=False)
            selection_mode = "split_train_ids_recent_start"
        else:
            historical_ids = split_train_ids
            selection_mode = "split_train_ids"
    elif include_future_background:
        train_mask = (
            ((first_active_arr <= threshold_day) & (first_active_arr >= min_train_day) & np.isin(labels1, (0, 1)))
            | ((first_active_arr >= min_train_day) & np.isin(labels1, (2, 3)))
        )
        historical_ids = np.flatnonzero(train_mask).astype(np.int32, copy=False)
        selection_mode = "threshold_mask"
    else:
        train_mask = (
            (first_active_arr <= threshold_day)
            & (first_active_arr >= min_train_day)
            & np.isin(labels1, (0, 1, 2, 3))
        )
        historical_ids = np.flatnonzero(train_mask).astype(np.int32, copy=False)
        selection_mode = "threshold_mask"

    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    overlap = np.intersect1d(historical_ids, val_ids)
    if overlap.size:
        raise ValueError(
            "Historical train ids overlap validation ids. "
            f"split_style={split.split_style} overlap={overlap.size}"
        )
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    return HistoricalMulticlassBgSplit(
        threshold_day=threshold_day,
        min_train_first_active_day=min_train_day,
        historical_selection_mode=selection_mode,
        historical_ids=historical_ids,
        val_ids=val_ids,
        external_ids=external_ids,
        y_train=labels1[historical_ids].astype(np.int32, copy=False),
        y_val=labels1[val_ids].astype(np.int32, copy=False),
        y_external=labels2[external_ids].astype(np.int32, copy=False),
        train_first_active=first_active_arr[historical_ids].astype(np.int32, copy=False),
    )


def build_multiclass_bg_params(args: Any) -> dict[str, float | int | str]:
    num_class = int(getattr(args, "num_class", 4))
    return {
        "objective": "multi:softprob",
        "num_class": num_class,
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
    sample_weight = np.asarray(sample_weight_payload["sample_weight"], dtype=np.float32)
    domain_weight, domain_weight_payload = build_domain_adaptation_weights_from_args(
        args,
        x_train=np.asarray(x_train, dtype=np.float32),
        x_val=np.asarray(x_val, dtype=np.float32),
    )
    sample_weight *= np.asarray(domain_weight, dtype=np.float32)
    mean_weight = float(np.mean(sample_weight, dtype=np.float64))
    if mean_weight > 0.0:
        sample_weight /= mean_weight
    sample_weight_payload["sample_weight"] = sample_weight
    sample_weight_payload["domain_weight"] = domain_weight_payload
    dtrain = xgb.DMatrix(
        np.asarray(x_train, dtype=np.float32),
        label=split_data.y_train,
        weight=sample_weight,
    )
    dval = xgb.DMatrix(np.asarray(x_val, dtype=np.float32), label=split_data.y_val)
    dexternal = (
        xgb.DMatrix(np.asarray(x_external, dtype=np.float32), label=split_data.y_external)
        if split_data.external_ids.size
        else None
    )
    num_class = max(
        2,
        int(np.max(split_data.y_train)) + 1 if split_data.y_train.size else 2,
        int(np.max(split_data.y_val)) + 1 if split_data.y_val.size else 2,
        int(np.max(split_data.y_external)) + 1 if split_data.y_external.size else 2,
    )
    setattr(args, "num_class", int(num_class))

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
    train_prob = booster.predict(dtrain, iteration_range=(0, best_iteration + 1)).reshape(-1, int(num_class))
    train_score = binary_score_from_softprob(train_prob)
    train_metrics = _safe_binary_metrics_or_none(split_data.y_train, train_score)
    val_prob = booster.predict(dval, iteration_range=(0, best_iteration + 1)).reshape(-1, int(num_class))
    val_score = binary_score_from_softprob(val_prob)
    val_metrics = _safe_binary_metrics_or_none(split_data.y_val, val_score)
    if dexternal is not None:
        external_prob = booster.predict(
            dexternal,
            iteration_range=(0, best_iteration + 1),
        ).reshape(-1, int(num_class))
        external_score = binary_score_from_softprob(external_prob)
        external_metrics = _safe_binary_metrics_or_none(split_data.y_external, external_score)
    else:
        external_score = np.empty(0, dtype=np.float32)
        external_metrics = None

    run_path = ensure_dir(run_dir)
    save_prediction_npz(
        run_path / "phase1_train_predictions.npz",
        split_data.historical_ids,
        split_data.y_train,
        train_score,
    )
    save_prediction_npz(
        run_path / "phase1_val_predictions.npz",
        split_data.val_ids,
        split_data.y_val,
        val_score,
    )
    if split_data.external_ids.size:
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
        "num_class": int(num_class),
        "threshold_day": int(split_data.threshold_day),
        "min_train_first_active_day": int(split_data.min_train_first_active_day),
        "historical_selection_mode": str(split_data.historical_selection_mode),
        "historical_train_size": int(split_data.historical_ids.size),
        "historical_train_label_counts": {
            str(label): int(np.sum(split_data.y_train == label))
            for label in (0, 1, 2, 3)
        },
        "class_weight": sample_weight_payload["class_weight"],
        "time_weight": sample_weight_payload["time_weight"],
        "domain_weight": sample_weight_payload["domain_weight"],
        "best_iteration": best_iteration,
        "phase1_train_metrics": train_metrics,
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "params": params,
        "prediction_paths": {
            "phase1_train": str(run_path / "phase1_train_predictions.npz"),
            "phase1_val": str(run_path / "phase1_val_predictions.npz"),
        },
    }
    if split_data.external_ids.size:
        summary["prediction_paths"]["phase2_external"] = str(run_path / "phase2_external_predictions.npz")
    if summary_extra:
        summary.update(summary_extra)
    write_json(run_path / "summary.json", summary)
    external_auc_text = "n/a" if external_metrics is None else f"{external_metrics['auc']:.6f}"
    print(
        f"[{model_name}] run={summary['run_name']} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_auc_text} "
        f"best_iteration={best_iteration}"
    )
    return summary
