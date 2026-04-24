from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from dyrift.models.graph import get_experiment_cls
from dyrift.models.presets import _BASE_GRAPH_CONFIG, apply_cfg_overrides, build_graph_cfg
from dyrift.models.runtime import build_runtime
from dyrift.utils.common import (
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    save_prediction_npz,
    set_global_seed,
    slice_node_ids,
    write_clean_epoch_metrics,
    write_json,
)

from .contracts import DatasetPlan, ExperimentConfig, resolve_dataset_output_roots


def run_graph_dataset(
    *,
    config: ExperimentConfig,
    plan: DatasetPlan,
    dataset_dir: Path,
    seeds: list[int],
    device: str,
) -> Path:
    graph_spec = config.runner_spec
    model_key = str(graph_spec["engine_target"])
    model_display_name = str(graph_spec.get("model_display_name") or config.display_name)
    backbone_display_name = str(graph_spec.get("backbone_display_name") or model_display_name)
    graph_config = _resolve_graph_config(graph_spec=graph_spec, plan=plan)
    target_context_groups = _resolve_target_context_groups(graph_spec=graph_spec, plan=plan)

    dataset_analysis_root, _ = resolve_dataset_output_roots(plan.dataset_name)
    split = load_experiment_split(analysis_root=dataset_analysis_root)
    train_ids = slice_node_ids(split.train_ids, None, seed=11)
    val_ids = slice_node_ids(split.val_ids, None, seed=17)
    test_pool_ids = slice_node_ids(split.test_pool_ids, None, seed=23)
    external_ids = slice_node_ids(split.external_ids, None, seed=29)

    runtime = build_runtime(
        feature_dir=plan.feature_dir,
        model_name=model_key,
        split=split,
        train_ids=train_ids,
        graph_config=graph_config,
        feature_profile=plan.feature_profile,
        target_context_groups=target_context_groups,
    )
    graph_config = runtime.graph_config
    phase1_context = runtime.phase1_context
    phase2_context = runtime.phase2_context

    train_labels = np.asarray(phase1_context.labels[train_ids], dtype=np.int8)
    val_labels = np.asarray(phase1_context.labels[val_ids], dtype=np.int8)
    test_pool_labels = np.asarray(phase1_context.labels[test_pool_ids], dtype=np.int8)
    external_labels = (
        np.asarray(phase2_context.labels[external_ids], dtype=np.int8)
        if external_ids.size
        else np.empty(0, dtype=np.int8)
    )

    experiment_cls = get_experiment_cls(model_key)
    train_predictions: list[np.ndarray] = []
    val_predictions: list[np.ndarray] = []
    test_pool_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []

    print(
        "[experiment:graph] "
        f"experiment={config.experiment_name} "
        f"dataset={plan.dataset_name} "
        f"model={model_display_name} "
        f"backbone={backbone_display_name} "
        f"feature_profile={plan.feature_profile} "
        f"target_context={'on' if target_context_groups else 'off'} "
        f"train={train_ids.size} val={val_ids.size} test_pool={test_pool_ids.size} external={external_ids.size}"
    )

    with tqdm(
        seeds,
        desc=f"experiment:{config.experiment_name}:{plan.dataset_short}",
        unit="seed",
        dynamic_ncols=True,
    ) as seed_pbar:
        for seed in seed_pbar:
            set_global_seed(int(seed))
            seed_dir = ensure_dir(dataset_dir / f"seed_{int(seed)}")
            trainer = experiment_cls(
                model_name=model_display_name,
                seed=int(seed),
                input_dim=runtime.input_dim,
                num_relations=runtime.num_relations,
                max_day=runtime.global_max_day,
                feature_groups=runtime.feature_groups,
                hidden_dim=plan.hidden_dim,
                num_layers=len(plan.fanouts),
                rel_dim=plan.rel_dim,
                fanouts=list(plan.fanouts),
                batch_size=plan.batch_size,
                epochs=plan.epochs,
                device=device,
                graph_config=graph_config,
                feature_normalizer_state=runtime.feature_normalizer_state,
                target_context_input_dim=runtime.target_context_input_dim,
                target_context_feature_groups=runtime.target_context_feature_groups,
                target_context_normalizer_state=runtime.target_context_normalizer_state,
            )
            fit_metrics = trainer.fit(
                context=phase1_context,
                train_ids=train_ids,
                val_ids=val_ids,
                test_pool_ids=test_pool_ids,
                artifact_dir=seed_dir,
            )
            trained_epochs = int(len(trainer.training_history))
            _write_enriched_fit_metrics(
                path=seed_dir / "fit_metrics.json",
                payload={**fit_metrics, "trained_epochs": trained_epochs},
            )

            train_prob = trainer.predict_proba(
                phase1_context,
                train_ids,
                batch_seed=int(seed) + 500,
                progress_desc=f"{config.experiment_name}:seed{seed}:train",
            )
            val_prob = trainer.predict_proba(
                phase1_context,
                val_ids,
                batch_seed=int(seed) + 1000,
                progress_desc=f"{config.experiment_name}:seed{seed}:val",
            )
            test_pool_prob = (
                trainer.predict_proba(
                    phase1_context,
                    test_pool_ids,
                    batch_seed=int(seed) + 1500,
                    progress_desc=f"{config.experiment_name}:seed{seed}:test_pool",
                )
                if test_pool_ids.size
                else None
            )
            external_prob = (
                trainer.predict_proba(
                    phase2_context,
                    external_ids,
                    batch_seed=int(seed) + 2000,
                    progress_desc=f"{config.experiment_name}:seed{seed}:external",
                )
                if external_ids.size
                else None
            )

            train_metrics = compute_binary_classification_metrics(train_labels, train_prob)
            val_metrics = compute_binary_classification_metrics(val_labels, val_prob)
            test_metrics = (
                None if test_pool_prob is None else _maybe_compute_binary_metrics(test_pool_labels, test_pool_prob)
            )
            external_metrics = (
                None if external_prob is None else _maybe_compute_binary_metrics(external_labels, external_prob)
            )

            trainer.save(seed_dir)
            save_prediction_npz(seed_dir / "phase1_train_predictions.npz", train_ids, train_labels, train_prob)
            save_prediction_npz(seed_dir / "phase1_val_predictions.npz", val_ids, val_labels, val_prob)
            if test_pool_prob is not None:
                save_prediction_npz(
                    seed_dir / "test_pool_predictions.npz",
                    test_pool_ids,
                    test_pool_labels,
                    test_pool_prob,
                )
            if external_prob is not None:
                save_prediction_npz(
                    seed_dir / "phase2_external_predictions.npz",
                    external_ids,
                    external_labels,
                    external_prob,
                )

            train_predictions.append(train_prob)
            val_predictions.append(val_prob)
            if test_pool_prob is not None:
                test_pool_predictions.append(test_pool_prob)
            if external_prob is not None:
                external_predictions.append(external_prob)

            metric_row = {
                "experiment_name": config.experiment_name,
                "dataset": plan.dataset_name,
                "dataset_display_name": plan.dataset_display_name,
                "model_display_name": model_display_name,
                "seed": int(seed),
                "train_auc": float(train_metrics["auc"]),
                "val_auc": float(val_metrics["auc"]),
                "test_auc": None if test_metrics is None else float(test_metrics["auc"]),
                "external_auc": None if external_metrics is None else float(external_metrics["auc"]),
                "best_epoch": int(fit_metrics["best_epoch"]),
                "trained_epochs": trained_epochs,
            }
            seed_pbar.set_postfix(val_auc=f"{metric_row['val_auc']:.4f}", refresh=False)
            tqdm.write(
                f"[experiment:{config.experiment_name}] dataset={plan.dataset_name} seed={seed} "
                f"train_auc={metric_row['train_auc']:.6f} val_auc={metric_row['val_auc']:.6f} "
                f"test_auc={_format_metric(metric_row['test_auc'])}"
            )

    _save_average_predictions(
        dataset_dir,
        "phase1_train",
        train_ids,
        train_labels,
        train_predictions,
    )
    _save_average_predictions(
        dataset_dir,
        "phase1_val",
        val_ids,
        val_labels,
        val_predictions,
    )
    if test_pool_predictions:
        _save_average_predictions(dataset_dir, "test_pool", test_pool_ids, test_pool_labels, test_pool_predictions)
    if external_predictions:
        _save_average_predictions(dataset_dir, "phase2_external", external_ids, external_labels, external_predictions)

    epoch_metrics_path = write_clean_epoch_metrics(
        dataset_dir / "epoch_metrics.csv",
        [dataset_dir / f"seed_{int(seed)}" / "epoch_metrics.csv" for seed in seeds],
    )
    return epoch_metrics_path


def _resolve_graph_config(*, graph_spec: dict[str, Any], plan: DatasetPlan) -> GraphModelConfig:
    source = str(graph_spec.get("config_source", "preset")).strip().lower()
    if source == "preset":
        graph_config = build_graph_cfg(
            str(graph_spec["engine_target"]),
            str(graph_spec.get("preset")),
        )
    elif source == "base_graph":
        graph_config = replace(_BASE_GRAPH_CONFIG)
    else:
        raise ValueError(f"Unsupported graph config source: {source}")

    combined_overrides = list(plan.graph_config_overrides) + [str(value) for value in graph_spec.get("graph_config_overrides", [])]
    graph_config, _ = apply_cfg_overrides(graph_config, combined_overrides)
    runtime_updates: dict[str, Any] = {}
    if plan.learning_rate is not None:
        runtime_updates["learning_rate"] = float(plan.learning_rate)
    if plan.weight_decay is not None:
        runtime_updates["weight_decay"] = float(plan.weight_decay)
    if plan.dropout is not None:
        runtime_updates["dropout"] = float(plan.dropout)
    if runtime_updates:
        graph_config = replace(graph_config, **runtime_updates)
    return graph_config


def _resolve_target_context_groups(*, graph_spec: dict[str, Any], plan: DatasetPlan) -> list[str]:
    raw_groups = graph_spec.get("target_context_groups", "__dataset__")
    if raw_groups == "__dataset__":
        return list(plan.target_context_groups)
    if raw_groups in (None, "none", "disabled"):
        return []
    if not isinstance(raw_groups, list):
        raise ValueError("graph.target_context_groups must be a list, null, or `__dataset__`.")
    return [str(value) for value in raw_groups]


def _write_enriched_fit_metrics(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


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


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
