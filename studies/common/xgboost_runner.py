from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from features.features import FeatureStore, build_hybrid_feature_normalizer, resolve_feature_groups
from utils.common import (
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    save_prediction_npz,
    set_global_seed,
    write_json,
)

from .contracts import DatasetPlan, REPO_ROOT, StudyConfig, resolve_dataset_output_roots


def run_xgboost_dataset(
    *,
    study: StudyConfig,
    plan: DatasetPlan,
    dataset_dir: Path,
    seeds: list[int],
) -> Path:
    xgb_spec = study.runner_spec
    model_display_name = str(xgb_spec.get("model_display_name") or study.display_name)
    include_target_context = bool(xgb_spec.get("include_target_context", False))
    params = _build_xgboost_params(spec=xgb_spec, plan=plan)
    num_boost_round = int(xgb_spec.get("num_boost_round", 400))
    early_stopping_rounds = int(xgb_spec.get("early_stopping_rounds", 40))

    dataset_eda_root, _ = resolve_dataset_output_roots(plan.dataset_name)
    split = load_experiment_split(eda_root=dataset_eda_root)
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

    from models.runtime import build_runtime  # local import to reuse correct label contract
    from models.engine import GraphModelConfig

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

    train_predictions: list[np.ndarray] = []
    val_predictions: list[np.ndarray] = []
    test_pool_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []
    seed_metrics: list[dict[str, Any]] = []

    print(
        "[study:xgboost] "
        f"study={study.study_name} "
        f"dataset={plan.dataset_name} "
        f"model={model_display_name} "
        f"features={train_x.shape[1]} "
        f"target_context={'on' if include_target_context and plan.target_context_groups else 'off'}"
    )

    for seed in seeds:
        set_global_seed(int(seed))
        seed_dir = ensure_dir(dataset_dir / f"seed_{int(seed)}")
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

        trained_rounds = int(getattr(booster, "best_iteration", -1)) + 1
        if trained_rounds <= 0:
            trained_rounds = int(num_boost_round)
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
        _write_csv(seed_dir / "epoch_metrics.csv", epoch_rows)

        best_epoch_row = max(epoch_rows, key=lambda row: float(row["val_auc"]))
        best_iteration = int(best_epoch_row["epoch"])
        train_prob = _predict_round(booster, train_x, best_iteration)
        val_prob = _predict_round(booster, val_x, best_iteration)
        test_pool_prob = _predict_round(booster, test_pool_x, best_iteration) if test_pool_ids.size else None
        external_prob = _predict_round(booster, external_x, best_iteration) if external_ids.size else None

        save_prediction_npz(seed_dir / "phase1_train_predictions.npz", train_ids, train_labels, train_prob)
        save_prediction_npz(seed_dir / "phase1_val_predictions.npz", val_ids, val_labels, val_prob)
        if test_pool_prob is not None:
            save_prediction_npz(seed_dir / "test_pool_predictions.npz", test_pool_ids, test_pool_labels, test_pool_prob)
        if external_prob is not None:
            save_prediction_npz(seed_dir / "phase2_external_predictions.npz", external_ids, external_labels, external_prob)

        fit_summary = {
            "best_epoch": int(best_iteration),
            "trained_epochs": int(trained_rounds),
            "best_score": float(booster.best_score) if getattr(booster, "best_score", None) is not None else None,
            "early_stopping_rounds": int(early_stopping_rounds),
            "params": seed_params,
        }
        write_json(seed_dir / "fit_summary.json", fit_summary)
        booster.save_model(str(seed_dir / "model.json"))

        train_predictions.append(train_prob)
        val_predictions.append(val_prob)
        if test_pool_prob is not None:
            test_pool_predictions.append(test_pool_prob)
        if external_prob is not None:
            external_predictions.append(external_prob)

        test_metrics = None if test_pool_prob is None else _maybe_compute_binary_metrics(test_pool_labels, test_pool_prob)
        external_metrics = None if external_prob is None else _maybe_compute_binary_metrics(external_labels, external_prob)

        seed_metrics.append(
            {
                "study_name": study.study_name,
                "dataset": plan.dataset_name,
                "dataset_display_name": plan.dataset_display_name,
                "model_display_name": model_display_name,
                "seed": int(seed),
                "train_auc": float(compute_binary_classification_metrics(train_labels, train_prob)["auc"]),
                "val_auc": float(compute_binary_classification_metrics(val_labels, val_prob)["auc"]),
                "test_auc": None if test_metrics is None else float(test_metrics["auc"]),
                "external_auc": None if external_metrics is None else float(external_metrics["auc"]),
                "best_epoch": int(best_iteration),
                "trained_epochs": int(trained_rounds),
                "fit_summary_path": _path_repr(seed_dir / "fit_summary.json"),
                "epoch_metrics_path": _path_repr(seed_dir / "epoch_metrics.csv"),
                "model_path": _path_repr(seed_dir / "model.json"),
            }
        )

    train_avg_path = _save_average_predictions(dataset_dir, "phase1_train", train_ids, train_labels, train_predictions)
    val_avg_path = _save_average_predictions(dataset_dir, "phase1_val", val_ids, val_labels, val_predictions)
    test_avg_path = (
        _save_average_predictions(dataset_dir, "test_pool", test_pool_ids, test_pool_labels, test_pool_predictions)
        if test_pool_predictions
        else None
    )
    external_avg_path = (
        _save_average_predictions(dataset_dir, "phase2_external", external_ids, external_labels, external_predictions)
        if external_predictions
        else None
    )

    seed_overview_path = dataset_dir / "seed_overview.csv"
    _write_csv(seed_overview_path, seed_metrics)
    epoch_merged_path = dataset_dir / "epoch_metrics_merged.csv"
    _merge_epoch_metrics(
        source_rows=seed_metrics,
        output_path=epoch_merged_path,
        dataset_name=plan.dataset_name,
        study_name=study.study_name,
    )

    summary = {
        "study_name": study.study_name,
        "display_name": study.display_name,
        "study_type": study.study_type,
        "runner": "xgboost_same_input",
        "dataset": plan.dataset_name,
        "dataset_display_name": plan.dataset_display_name,
        "model_display_name": model_display_name,
        "feature_profile": plan.feature_profile,
        "feature_dir": _path_repr(plan.feature_dir),
        "include_target_context": include_target_context,
        "dataset_plan": plan.to_summary_payload(),
        "xgboost_params": params,
        "num_boost_round": int(num_boost_round),
        "early_stopping_rounds": int(early_stopping_rounds),
        "seeds": [int(seed) for seed in seeds],
        "train_size": int(train_ids.size),
        "val_size": int(val_ids.size),
        "test_pool_size": int(test_pool_ids.size),
        "external_size": int(external_ids.size),
        "train_auc_mean": _metric_mean(seed_metrics, "train_auc"),
        "train_auc_std": _metric_std(seed_metrics, "train_auc"),
        "val_auc_mean": _metric_mean(seed_metrics, "val_auc"),
        "val_auc_std": _metric_std(seed_metrics, "val_auc"),
        "test_auc_mean": _metric_mean(seed_metrics, "test_auc"),
        "test_auc_std": _metric_std(seed_metrics, "test_auc"),
        "external_auc_mean": _metric_mean(seed_metrics, "external_auc"),
        "external_auc_std": _metric_std(seed_metrics, "external_auc"),
        "best_epoch_mean": _metric_mean(seed_metrics, "best_epoch"),
        "trained_epochs_mean": _metric_mean(seed_metrics, "trained_epochs"),
        "phase1_train_avg_predictions": _path_repr(train_avg_path),
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "test_pool_avg_predictions": None if test_avg_path is None else _path_repr(test_avg_path),
        "phase2_external_avg_predictions": None if external_avg_path is None else _path_repr(external_avg_path),
        "seed_overview_path": _path_repr(seed_overview_path),
        "epoch_metrics_merged_path": _path_repr(epoch_merged_path),
        "seed_metrics": seed_metrics,
        "no_leakage_notes": [
            "Feature normalization is fit on train nodes only.",
            "The XGBoost baseline consumes the same cached UTPM inputs without graphprop or teacher outputs.",
            "Validation labels are used only for early stopping and evaluation, not for feature fitting.",
            "Each dataset is trained in isolation under its own dataset-scoped caches.",
        ],
    }
    summary_path = dataset_dir / "summary.json"
    write_json(summary_path, summary)
    return summary_path


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
        test_metrics = None if test_prob is None else _maybe_compute_binary_metrics(test_pool_y, test_prob)
        external_metrics = None if external_prob is None else _maybe_compute_binary_metrics(external_y, external_prob)
        rows.append(
            {
                "epoch": epoch,
                "train_auc": float(train_auc),
                "val_auc": float(val_auc),
                "test_auc": None if test_metrics is None else float(test_metrics["auc"]),
                "external_auc": None if external_metrics is None else float(external_metrics["auc"]),
            }
        )
    return rows


def _predict_round(booster: xgb.Booster, x: np.ndarray, round_count: int) -> np.ndarray:
    return booster.predict(
        xgb.DMatrix(x),
        iteration_range=(0, int(round_count)),
    ).astype(np.float32, copy=False)


def _save_average_predictions(
    run_dir: Path,
    split_name: str,
    node_ids: np.ndarray,
    labels: np.ndarray,
    predictions: list[np.ndarray],
) -> Path:
    mean_pred = np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float32, copy=False)
    path = run_dir / f"{split_name}_avg_predictions.npz"
    save_prediction_npz(path, node_ids=node_ids, y_true=labels, probabilities=mean_pred)
    return path


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


def _metric_mean(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in metrics if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _metric_std(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in metrics if row.get(key) is not None]
    if not values:
        return None
    return float(np.std(values))


def _path_repr(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _merge_epoch_metrics(
    *,
    source_rows: list[dict[str, Any]],
    output_path: Path,
    dataset_name: str,
    study_name: str,
) -> None:
    merged_rows: list[dict[str, Any]] = []
    for row in source_rows:
        epoch_metrics_path = REPO_ROOT / str(row["epoch_metrics_path"])
        if not epoch_metrics_path.exists():
            continue
        with epoch_metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for epoch_row in reader:
                merged_rows.append(
                    {
                        "study_name": study_name,
                        "dataset": dataset_name,
                        "seed": row["seed"],
                        **epoch_row,
                    }
                )
    _write_csv(output_path, merged_rows)
