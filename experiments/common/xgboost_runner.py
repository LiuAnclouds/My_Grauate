from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from dyrift.features.features import FeatureStore, build_hybrid_feature_normalizer, resolve_feature_groups
from dyrift.utils.common import (
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    set_global_seed,
    write_clean_epoch_metrics,
)

from .contracts import DatasetPlan, ExperimentConfig, resolve_dataset_output_roots


def run_xgboost_dataset(
    *,
    config: ExperimentConfig,
    plan: DatasetPlan,
    dataset_dir: Path,
    seeds: list[int],
) -> Path:
    xgb_spec = config.runner_spec
    model_display_name = str(xgb_spec.get("model_display_name") or config.display_name)
    include_target_context = bool(xgb_spec.get("include_target_context", False))
    params = _build_xgboost_params(spec=xgb_spec, plan=plan)
    num_boost_round = int(xgb_spec.get("num_boost_round", 400))
    early_stopping_rounds = int(xgb_spec.get("early_stopping_rounds", 5))

    dataset_analysis_root, _ = resolve_dataset_output_roots(plan.dataset_name)
    split = load_experiment_split(analysis_root=dataset_analysis_root)
    train_ids = np.asarray(split.train_ids, dtype=np.int32)
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    test_pool_ids = np.asarray(split.test_pool_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)

    feature_groups = resolve_feature_groups("dyrift_gnn", feature_profile=plan.feature_profile)
    normalizer_state = build_hybrid_feature_normalizer(
        phase=split.train_phase,
        selected_groups=feature_groups,
        train_ids=train_ids,
        outdir=plan.feature_dir,
    )
    train_store = FeatureStore(split.train_phase, feature_groups, outdir=plan.feature_dir, normalizer_state=normalizer_state)
    phase1_store = FeatureStore(split.val_phase, feature_groups, outdir=plan.feature_dir, normalizer_state=normalizer_state)
    phase2_store = FeatureStore(split.external_phase, feature_groups, outdir=plan.feature_dir, normalizer_state=normalizer_state) if split.external_phase else None

    train_x = train_store.take_rows(train_ids)
    val_x = phase1_store.take_rows(val_ids)
    test_pool_x = phase1_store.take_rows(test_pool_ids) if test_pool_ids.size else np.empty((0, train_x.shape[1]), dtype=np.float32)
    external_x = (
        phase2_store.take_rows(external_ids)
        if phase2_store is not None and external_ids.size
        else np.empty((0, train_x.shape[1]), dtype=np.float32)
    )

    if include_target_context and plan.target_context_groups:
        target_normalizer = build_hybrid_feature_normalizer(
            phase=split.train_phase,
            selected_groups=plan.target_context_groups,
            train_ids=train_ids,
            outdir=plan.feature_dir,
        )
        train_target_store = FeatureStore(
            split.train_phase,
            plan.target_context_groups,
            outdir=plan.feature_dir,
            normalizer_state=target_normalizer,
        )
        phase1_target_store = FeatureStore(
            split.val_phase,
            plan.target_context_groups,
            outdir=plan.feature_dir,
            normalizer_state=target_normalizer,
        )
        phase2_target_store = (
            FeatureStore(
                split.external_phase,
                plan.target_context_groups,
                outdir=plan.feature_dir,
                normalizer_state=target_normalizer,
            )
            if split.external_phase
            else None
        )
        train_x = np.concatenate([train_x, train_target_store.take_rows(train_ids)], axis=1)
        val_x = np.concatenate([val_x, phase1_target_store.take_rows(val_ids)], axis=1)
        if test_pool_ids.size:
            test_pool_x = np.concatenate([test_pool_x, phase1_target_store.take_rows(test_pool_ids)], axis=1)
        if phase2_target_store is not None and external_ids.size:
            external_x = np.concatenate([external_x, phase2_target_store.take_rows(external_ids)], axis=1)

    from dyrift.models.runtime import build_runtime  # local import to reuse correct label contract
    from dyrift.models.engine import GraphModelConfig

    runtime = build_runtime(
        feature_dir=plan.feature_dir,
        model_name="dyrift_gnn",
        split=split,
        train_ids=train_ids,
        graph_config=GraphModelConfig(feature_norm="hybrid"),
        feature_profile=plan.feature_profile,
        target_context_groups=plan.target_context_groups,
    )
    train_labels = np.asarray(runtime.phase1_context.labels[train_ids], dtype=np.int8)
    val_labels = np.asarray(runtime.phase1_context.labels[val_ids], dtype=np.int8)
    test_pool_labels = np.asarray(runtime.phase1_context.labels[test_pool_ids], dtype=np.int8)
    external_labels = (
        np.asarray(runtime.phase2_context.labels[external_ids], dtype=np.int8)
        if external_ids.size
        else np.empty(0, dtype=np.int8)
    )

    print(
        "[experiment:xgboost] "
        f"experiment={config.experiment_name} "
        f"dataset={plan.dataset_name} "
        f"model={model_display_name} "
        f"features={train_x.shape[1]} "
        f"target_context={'on' if include_target_context and plan.target_context_groups else 'off'}"
    )

    if len(seeds) != 1:
        raise ValueError(
            "Official flat output layout supports exactly one seed per run. "
            "Use a separate experiment name for multi-seed studies."
        )

    for seed in seeds:
        set_global_seed(int(seed))
        run_artifact_dir = ensure_dir(dataset_dir)
        seed_params = dict(params)
        seed_params["seed"] = int(seed)

        pos_count = max(int(np.sum(train_labels == 1)), 1)
        neg_count = max(int(np.sum(train_labels == 0)), 1)
        seed_params.setdefault("scale_pos_weight", float(neg_count / pos_count))

        dtrain = xgb.DMatrix(train_x, label=train_labels)
        dval = xgb.DMatrix(val_x, label=val_labels)
        evals_result: dict[str, dict[str, list[float]]] = {}
        booster = xgb.train(
            params=seed_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        trained_rounds = _resolve_trained_rounds(
            booster=booster,
            evals_result=evals_result,
            fallback_rounds=int(num_boost_round),
        )
        epoch_rows = _build_epoch_rows(
            booster=booster,
            train_x=train_x,
            train_y=train_labels,
            val_x=val_x,
            val_y=val_labels,
            test_pool_x=test_pool_x,
            test_pool_y=test_pool_labels,
            external_x=external_x,
            external_y=external_labels,
            trained_rounds=trained_rounds,
        )
        _write_csv(run_artifact_dir / "epoch_metrics.csv", epoch_rows)

        best_epoch_row = max(epoch_rows, key=lambda row: float(row["val_auc"]))
        best_iteration = int(best_epoch_row["epoch"])
        train_prob = _predict_round(booster, train_x, best_iteration)
        val_prob = _predict_round(booster, val_x, best_iteration)
        test_pool_prob = _predict_round(booster, test_pool_x, best_iteration) if test_pool_ids.size else None
        external_prob = _predict_round(booster, external_x, best_iteration) if external_ids.size else None

        booster.save_model(str(run_artifact_dir / "model.json"))

        test_metrics = None if test_pool_prob is None else _maybe_compute_binary_metrics(test_pool_labels, test_pool_prob)
        external_metrics = None if external_prob is None else _maybe_compute_binary_metrics(external_labels, external_prob)

        train_auc = float(compute_binary_classification_metrics(train_labels, train_prob)["auc"])
        val_auc = float(compute_binary_classification_metrics(val_labels, val_prob)["auc"])
        print(
            f"[experiment:{config.experiment_name}] dataset={plan.dataset_name} seed={seed} "
            f"train_auc={train_auc:.6f} val_auc={val_auc:.6f} "
            f"test_auc={_format_metric(None if test_metrics is None else test_metrics['auc'])}"
        )

    epoch_metrics_path = write_clean_epoch_metrics(
        dataset_dir / "epoch_metrics.csv",
        [dataset_dir / "epoch_metrics.csv"],
    )
    return epoch_metrics_path


def _build_xgboost_params(*, spec: dict[str, Any], plan: DatasetPlan) -> dict[str, Any]:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5.0,
        "lambda": 1.0,
        "alpha": 0.0,
    }
    params.update(spec.get("params", {}))
    return params


def _resolve_trained_rounds(
    *,
    booster: xgb.Booster,
    evals_result: dict[str, dict[str, list[float]]],
    fallback_rounds: int,
) -> int:
    for metric_map in evals_result.values():
        for values in metric_map.values():
            if values:
                return int(len(values))

    num_boosted_rounds = getattr(booster, "num_boosted_rounds", None)
    if callable(num_boosted_rounds):
        resolved = int(num_boosted_rounds())
        if resolved > 0:
            return resolved
    return int(fallback_rounds)


def _build_epoch_rows(
    *,
    booster: xgb.Booster,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_pool_x: np.ndarray,
    test_pool_y: np.ndarray,
    external_x: np.ndarray,
    external_y: np.ndarray,
    trained_rounds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in range(1, int(trained_rounds) + 1):
        train_prob = _predict_round(booster, train_x, epoch)
        val_prob = _predict_round(booster, val_x, epoch)
        test_prob = _predict_round(booster, test_pool_x, epoch) if test_pool_x.size else None
        external_prob = _predict_round(booster, external_x, epoch) if external_x.size else None
        train_auc = compute_binary_classification_metrics(train_y, train_prob)["auc"]
        val_auc = compute_binary_classification_metrics(val_y, val_prob)["auc"]
        train_loss = _binary_log_loss(train_y, train_prob)
        val_loss = _binary_log_loss(val_y, val_prob)
        test_metrics = None if test_prob is None else _maybe_compute_binary_metrics(test_pool_y, test_prob)
        external_metrics = None if external_prob is None else _maybe_compute_binary_metrics(external_y, external_prob)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_auc": float(train_auc),
                "val_auc": float(val_auc),
                "test_auc": None if test_metrics is None else float(test_metrics["auc"]),
                "external_auc": None if external_metrics is None else float(external_metrics["auc"]),
            }
        )
    return rows


def _binary_log_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)), dtype=np.float64))


def _predict_round(booster: xgb.Booster, x: np.ndarray, round_count: int) -> np.ndarray:
    return booster.predict(
        xgb.DMatrix(x),
        iteration_range=(0, int(round_count)),
    ).astype(np.float32, copy=False)


def _maybe_compute_binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float] | None:
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


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
