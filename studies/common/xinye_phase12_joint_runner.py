from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from datasets.core.registry import DATASET_ENV_VAR
from features.features import GraphCache
from models.engine import GraphPhaseContext
from models.graph import get_experiment_cls
from models.presets import apply_cfg_overrides, build_graph_cfg
from models.runtime import build_runtime
from studies.common.contracts import REPO_ROOT, StudyConfig, resolve_dataset_output_roots
from studies.common.graph_runner import (
    _metric_mean,
    _metric_std,
    _path_repr,
    _save_average_predictions,
    _write_csv,
)
from utils.common import (
    ExperimentSplit,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    load_phase_arrays,
    save_prediction_npz,
    set_global_seed,
    slice_node_ids,
    write_json,
)


class _MergedFeatureStore:
    def __init__(self, left_store, right_store, right_offset: int, phase: str) -> None:
        self.left_store = left_store
        self.right_store = right_store
        self.right_offset = int(right_offset)
        self.phase = str(phase)
        self.phase_dir = Path(".")
        if int(left_store.input_dim) != int(right_store.input_dim):
            raise ValueError("Merged feature stores require the same input dimension.")
        self.input_dim = int(left_store.input_dim)

    def take_rows(self, node_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        output = np.empty((rows.shape[0], self.input_dim), dtype=np.float32)
        left_mask = rows < self.right_offset
        if np.any(left_mask):
            output[left_mask] = self.left_store.take_rows(rows[left_mask].astype(np.int32, copy=False))
        if np.any(~left_mask):
            output[~left_mask] = self.right_store.take_rows(
                (rows[~left_mask] - self.right_offset).astype(np.int32, copy=False)
            )
        return output


class _MergedSamplingProfile:
    def __init__(self, left_profile: np.ndarray | None, right_profile: np.ndarray | None, right_offset: int) -> None:
        self.left_profile = left_profile
        self.right_profile = right_profile
        self.right_offset = int(right_offset)
        self._dtype = np.float32
        self._dim = self._resolve_dim()

    def _resolve_dim(self) -> int:
        for profile in (self.left_profile, self.right_profile):
            if profile is not None:
                return int(profile.shape[1])
        return 0

    def __getitem__(self, index):
        rows = np.asarray(index, dtype=np.int64)
        scalar = rows.ndim == 0
        rows = rows.reshape(-1)
        output = np.empty((rows.shape[0], self._dim), dtype=self._dtype)
        left_mask = rows < self.right_offset
        if np.any(left_mask):
            output[left_mask] = np.asarray(
                self.left_profile[rows[left_mask].astype(np.int32, copy=False)],
                dtype=self._dtype,
            )
        if np.any(~left_mask):
            output[~left_mask] = np.asarray(
                self.right_profile[(rows[~left_mask] - self.right_offset).astype(np.int32, copy=False)],
                dtype=self._dtype,
            )
        if scalar:
            return output[0]
        return output


def run_xinye_phase12_joint_study(study_dir: Path) -> None:
    args = _parse_args()
    from studies.common.contracts import load_study_config, resolve_dataset_plan

    study = load_study_config(study_dir)
    if study.datasets != ["xinye_dgraph"]:
        raise ValueError("This supplementary runner only supports the XinYe dataset.")
    plan = resolve_dataset_plan(study, "xinye_dgraph")

    os.environ[DATASET_ENV_VAR] = plan.dataset_name
    for key, value in plan.feature_env.items():
        os.environ[key] = str(value)

    dataset_dir = ensure_dir(study.output_root / plan.dataset_name)
    if args.skip_existing and (dataset_dir / "summary.json").exists():
        print(f"[study] skip existing dataset result: {dataset_dir / 'summary.json'}")
        return

    if args.build_features:
        _build_feature_cache(plan.feature_dir, plan.dataset_name, plan.feature_env, "graph")
        _build_feature_cache(plan.feature_dir, plan.dataset_name, plan.feature_env, "phase2")

    seeds = [int(seed) for seed in (args.seeds or study.seeds)]
    dataset_summary_path = run_xinye_phase12_joint_dataset(
        study=study,
        dataset_dir=dataset_dir,
        seeds=seeds,
        device=args.device,
    )
    print(f"[study] dataset summary written to {dataset_summary_path}")
    _aggregate_single_dataset(study=study, dataset_dir=dataset_dir, seeds=seeds)


def run_xinye_phase12_joint_dataset(
    *,
    study: StudyConfig,
    dataset_dir: Path,
    seeds: list[int],
    device: str,
) -> Path:
    from studies.common.contracts import resolve_dataset_plan

    plan = resolve_dataset_plan(study, "xinye_dgraph")
    graph_spec = study.runner_spec
    model_key = str(graph_spec["engine_target"])
    model_display_name = str(graph_spec.get("model_display_name") or study.display_name)
    backbone_display_name = str(graph_spec.get("backbone_display_name") or "TRGT")
    target_context_groups = list(plan.target_context_groups)

    dataset_eda_root, _ = resolve_dataset_output_roots(plan.dataset_name)
    split = load_experiment_split(eda_root=dataset_eda_root)
    split_with_phase2 = ExperimentSplit(
        train_ids=np.asarray(split.train_ids, dtype=np.int32),
        val_ids=np.asarray(split.val_ids, dtype=np.int32),
        test_pool_ids=np.asarray(split.test_pool_ids, dtype=np.int32),
        external_ids=np.asarray(load_phase_arrays("phase2", keys=("train_mask",))["train_mask"], dtype=np.int32),
        threshold_day=int(split.threshold_day),
        train_phase=str(split.train_phase),
        val_phase=str(split.val_phase),
        external_phase="phase2",
        split_style=str(split.split_style),
    )

    phase1_train_ids = slice_node_ids(split_with_phase2.train_ids, None, seed=11)
    phase1_val_ids = slice_node_ids(split_with_phase2.val_ids, None, seed=17)
    phase1_test_pool_ids = slice_node_ids(split_with_phase2.test_pool_ids, None, seed=23)
    phase2_train_ids = slice_node_ids(split_with_phase2.external_ids, None, seed=29)
    phase2_test_pool_ids = np.asarray(load_phase_arrays("phase2", keys=("test_mask",))["test_mask"], dtype=np.int32)

    graph_config = _resolve_graph_config(graph_spec=graph_spec, plan=plan)
    runtime = build_runtime(
        feature_dir=plan.feature_dir,
        model_name=model_key,
        split=split_with_phase2,
        train_ids=phase1_train_ids,
        graph_config=graph_config,
        feature_profile=plan.feature_profile,
        target_context_groups=target_context_groups,
    )
    graph_config = runtime.graph_config
    if runtime.phase1_context.graph_cache.num_relations != runtime.phase2_context.graph_cache.num_relations:
        raise ValueError("phase1/phase2 relation dimensions do not match for joint XinYe study.")

    joint_context, right_offset = _merge_phase_contexts(
        runtime.phase1_context,
        runtime.phase2_context,
    )

    joint_train_ids = np.concatenate(
        [
            np.asarray(phase1_train_ids, dtype=np.int32),
            np.asarray(phase2_train_ids + right_offset, dtype=np.int32),
        ]
    ).astype(np.int32, copy=False)
    joint_test_pool_ids = np.concatenate(
        [
            np.asarray(phase1_test_pool_ids, dtype=np.int32),
            np.asarray(phase2_test_pool_ids + right_offset, dtype=np.int32),
        ]
    ).astype(np.int32, copy=False)

    phase1_train_labels = np.asarray(runtime.phase1_context.labels[phase1_train_ids], dtype=np.int8)
    phase1_val_labels = np.asarray(runtime.phase1_context.labels[phase1_val_ids], dtype=np.int8)
    phase1_test_pool_labels = np.asarray(runtime.phase1_context.labels[phase1_test_pool_ids], dtype=np.int8)
    phase2_train_labels = np.asarray(runtime.phase2_context.labels[phase2_train_ids], dtype=np.int8)

    experiment_cls = get_experiment_cls(model_key)
    train_predictions: list[np.ndarray] = []
    val_predictions: list[np.ndarray] = []
    test_pool_predictions: list[np.ndarray] = []
    phase2_train_predictions: list[np.ndarray] = []
    seed_metrics: list[dict[str, Any]] = []

    print(
        "[study:xinye_phase12_joint] "
        f"study={study.study_name} "
        f"model={model_display_name} "
        f"backbone={backbone_display_name} "
        f"phase1_train={phase1_train_ids.size} "
        f"phase2_train={phase2_train_ids.size} "
        f"joint_train={joint_train_ids.size} "
        f"phase1_val={phase1_val_ids.size} "
        f"joint_test_pool={joint_test_pool_ids.size}"
    )

    for seed in seeds:
        set_global_seed(int(seed))
        seed_dir = ensure_dir(dataset_dir / f"seed_{int(seed)}")
        experiment = experiment_cls(
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
        fit_metrics = experiment.fit(
            context=joint_context,
            train_ids=joint_train_ids,
            val_ids=phase1_val_ids,
            test_pool_ids=joint_test_pool_ids,
            artifact_dir=seed_dir,
        )
        trained_epochs = int(len(experiment.training_history))

        phase1_train_prob = experiment.predict_proba(
            joint_context,
            phase1_train_ids,
            batch_seed=int(seed) + 500,
            progress_desc=f"{study.study_name}:seed{seed}:phase1_train",
        )
        phase1_val_prob = experiment.predict_proba(
            joint_context,
            phase1_val_ids,
            batch_seed=int(seed) + 1000,
            progress_desc=f"{study.study_name}:seed{seed}:phase1_val",
        )
        phase1_test_pool_prob = experiment.predict_proba(
            joint_context,
            phase1_test_pool_ids,
            batch_seed=int(seed) + 1500,
            progress_desc=f"{study.study_name}:seed{seed}:phase1_test_pool",
        )
        phase2_train_prob = experiment.predict_proba(
            joint_context,
            np.asarray(phase2_train_ids + right_offset, dtype=np.int32),
            batch_seed=int(seed) + 2000,
            progress_desc=f"{study.study_name}:seed{seed}:phase2_train",
        )

        train_metrics = compute_binary_classification_metrics(phase1_train_labels, phase1_train_prob)
        val_metrics = compute_binary_classification_metrics(phase1_val_labels, phase1_val_prob)
        test_metrics = _maybe_compute_binary_metrics(phase1_test_pool_labels, phase1_test_pool_prob)
        phase2_train_metrics = _maybe_compute_binary_metrics(phase2_train_labels, phase2_train_prob)

        experiment.save(seed_dir)
        save_prediction_npz(seed_dir / "phase1_train_predictions.npz", phase1_train_ids, phase1_train_labels, phase1_train_prob)
        save_prediction_npz(seed_dir / "phase1_val_predictions.npz", phase1_val_ids, phase1_val_labels, phase1_val_prob)
        save_prediction_npz(
            seed_dir / "test_pool_predictions.npz",
            phase1_test_pool_ids,
            phase1_test_pool_labels,
            phase1_test_pool_prob,
        )
        save_prediction_npz(
            seed_dir / "phase2_train_predictions.npz",
            phase2_train_ids,
            phase2_train_labels,
            phase2_train_prob,
        )

        train_predictions.append(phase1_train_prob)
        val_predictions.append(phase1_val_prob)
        test_pool_predictions.append(phase1_test_pool_prob)
        phase2_train_predictions.append(phase2_train_prob)

        seed_row = {
            "study_name": study.study_name,
            "dataset": plan.dataset_name,
            "dataset_display_name": plan.dataset_display_name,
            "model_display_name": model_display_name,
            "seed": int(seed),
            "phase1_train_auc": float(train_metrics["auc"]),
            "phase1_val_auc": float(val_metrics["auc"]),
            "phase1_test_auc": None if test_metrics is None else float(test_metrics["auc"]),
            "phase2_train_auc": None if phase2_train_metrics is None else float(phase2_train_metrics["auc"]),
            "joint_train_size": int(joint_train_ids.size),
            "best_epoch": int(fit_metrics["best_epoch"]),
            "trained_epochs": int(trained_epochs),
            "fit_summary_path": _path_repr(seed_dir / "fit_summary.json"),
            "epoch_metrics_path": _path_repr(seed_dir / "epoch_metrics.csv"),
            "train_log_path": _path_repr(seed_dir / "train.log"),
            "curve_path": _path_repr(seed_dir / "training_curves.png"),
        }
        seed_metrics.append(seed_row)
        print(
            f"[study:{study.study_name}] seed={seed} "
            f"phase1_val_auc={seed_row['phase1_val_auc']:.6f} "
            f"phase2_train_auc={_format_metric(seed_row['phase2_train_auc'])} "
            f"best_epoch={seed_row['best_epoch']}"
        )

    train_avg_path = _save_average_predictions(
        dataset_dir,
        "phase1_train",
        phase1_train_ids,
        phase1_train_labels,
        train_predictions,
    )
    val_avg_path = _save_average_predictions(
        dataset_dir,
        "phase1_val",
        phase1_val_ids,
        phase1_val_labels,
        val_predictions,
    )
    test_avg_path = _save_average_predictions(
        dataset_dir,
        "test_pool",
        phase1_test_pool_ids,
        phase1_test_pool_labels,
        test_pool_predictions,
    )
    phase2_train_avg_path = _save_average_predictions(
        dataset_dir,
        "phase2_train",
        phase2_train_ids,
        phase2_train_labels,
        phase2_train_predictions,
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
        "runner": study.runner,
        "dataset": plan.dataset_name,
        "dataset_display_name": plan.dataset_display_name,
        "model_display_name": model_display_name,
        "backbone_display_name": backbone_display_name,
        "implementation_target": model_key,
        "feature_profile": plan.feature_profile,
        "feature_dir": _path_repr(plan.feature_dir),
        "dataset_plan": plan.to_summary_payload(),
        "dataset_profile_path": _path_repr(study.dataset_profile_path),
        "target_context_groups": list(target_context_groups),
        "graph_config": graph_config.to_dict(),
        "seeds": [int(seed) for seed in seeds],
        "phase1_train_size": int(phase1_train_ids.size),
        "phase2_train_size": int(phase2_train_ids.size),
        "joint_train_size": int(joint_train_ids.size),
        "val_size": int(phase1_val_ids.size),
        "phase1_test_pool_size": int(phase1_test_pool_ids.size),
        "joint_test_pool_size": int(joint_test_pool_ids.size),
        "phase1_train_auc_mean": _metric_mean(seed_metrics, "phase1_train_auc"),
        "phase1_train_auc_std": _metric_std(seed_metrics, "phase1_train_auc"),
        "phase1_val_auc_mean": _metric_mean(seed_metrics, "phase1_val_auc"),
        "phase1_val_auc_std": _metric_std(seed_metrics, "phase1_val_auc"),
        "phase1_test_auc_mean": _metric_mean(seed_metrics, "phase1_test_auc"),
        "phase1_test_auc_std": _metric_std(seed_metrics, "phase1_test_auc"),
        "phase2_train_auc_mean": _metric_mean(seed_metrics, "phase2_train_auc"),
        "phase2_train_auc_std": _metric_std(seed_metrics, "phase2_train_auc"),
        "best_epoch_mean": _metric_mean(seed_metrics, "best_epoch"),
        "trained_epochs_mean": _metric_mean(seed_metrics, "trained_epochs"),
        "phase1_train_avg_predictions": _path_repr(train_avg_path),
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "test_pool_avg_predictions": _path_repr(test_avg_path),
        "phase2_train_avg_predictions": _path_repr(phase2_train_avg_path),
        "seed_overview_path": _path_repr(seed_overview_path),
        "epoch_metrics_merged_path": _path_repr(epoch_merged_path),
        "seed_metrics": seed_metrics,
        "warning_notes": [
            "This supplementary study injects official phase2 labeled nodes into joint training, so it is not the leakage-free thesis mainline.",
            "Validation remains fixed on the official phase1 internal time-aware val split.",
            "The merged graph is a disjoint union of phase1 and phase2; no cross-phase synthetic edges are introduced.",
        ],
    }
    summary_path = dataset_dir / "summary.json"
    write_json(summary_path, summary)
    return summary_path


def _merge_phase_contexts(
    left_context: GraphPhaseContext,
    right_context: GraphPhaseContext,
) -> tuple[GraphPhaseContext, int]:
    right_offset = int(left_context.graph_cache.num_nodes)
    joint_graph_cache = _merge_graph_caches(left_context.graph_cache, right_context.graph_cache)
    joint_feature_store = _MergedFeatureStore(
        left_context.feature_store,
        right_context.feature_store,
        right_offset,
        phase="graph_plus_phase2",
    )
    joint_target_context_store = None
    if left_context.target_context_store is not None and right_context.target_context_store is not None:
        joint_target_context_store = _MergedFeatureStore(
            left_context.target_context_store,
            right_context.target_context_store,
            right_offset,
            phase="graph_plus_phase2_target_context",
        )

    joint_labels = np.concatenate(
        [
            np.asarray(left_context.labels, dtype=np.int8),
            np.asarray(right_context.labels, dtype=np.int8),
        ]
    ).astype(np.int8, copy=False)
    joint_known_label_codes = None
    if left_context.known_label_codes is not None and right_context.known_label_codes is not None:
        joint_known_label_codes = np.concatenate(
            [
                np.asarray(left_context.known_label_codes, dtype=np.int8),
                np.asarray(right_context.known_label_codes, dtype=np.int8),
            ]
        ).astype(np.int8, copy=False)

    joint_sampling_profile = None
    if left_context.sampling_profile is not None and right_context.sampling_profile is not None:
        joint_sampling_profile = _MergedSamplingProfile(
            left_context.sampling_profile,
            right_context.sampling_profile,
            right_offset,
        )

    historical_background_ids = None
    if left_context.historical_background_ids is not None or right_context.historical_background_ids is not None:
        left_bg = (
            np.asarray(left_context.historical_background_ids, dtype=np.int32)
            if left_context.historical_background_ids is not None
            else np.empty(0, dtype=np.int32)
        )
        right_bg = (
            np.asarray(right_context.historical_background_ids, dtype=np.int32) + right_offset
            if right_context.historical_background_ids is not None
            else np.empty(0, dtype=np.int32)
        )
        historical_background_ids = np.concatenate([left_bg, right_bg]).astype(np.int32, copy=False)

    return (
        GraphPhaseContext(
            phase="graph_plus_phase2",
            feature_store=joint_feature_store,
            graph_cache=joint_graph_cache,
            labels=joint_labels,
            target_context_store=joint_target_context_store,
            known_label_codes=joint_known_label_codes,
            reference_day=None,
            historical_background_ids=historical_background_ids,
            sampling_profile=joint_sampling_profile,
        ),
        right_offset,
    )


def _merge_graph_caches(left: GraphCache, right: GraphCache) -> GraphCache:
    if int(left.num_edge_types) != int(right.num_edge_types):
        raise ValueError("Merged XinYe graph cache requires identical edge-type counts.")
    left_node_count = int(left.num_nodes)
    left_out_edge_count = int(left.out_neighbors.shape[0])
    left_in_edge_count = int(left.in_neighbors.shape[0])
    left_time_window_count = len(left.time_windows)

    merged_time_windows = [dict(window) for window in left.time_windows]
    for idx, window in enumerate(right.time_windows, start=1):
        merged_window = dict(window)
        merged_window["window_idx"] = left_time_window_count + idx
        merged_time_windows.append(merged_window)

    return GraphCache(
        phase="graph_plus_phase2",
        num_nodes=int(left.num_nodes + right.num_nodes),
        max_day=max(int(left.max_day), int(right.max_day)),
        num_edge_types=int(left.num_edge_types),
        num_relations=int(left.num_relations),
        time_windows=merged_time_windows,
        out_ptr=np.concatenate(
            [
                np.asarray(left.out_ptr, dtype=np.int64),
                np.asarray(right.out_ptr[1:], dtype=np.int64) + left_out_edge_count,
            ]
        ).astype(np.int64, copy=False),
        out_neighbors=np.concatenate(
            [
                np.asarray(left.out_neighbors, dtype=np.int32),
                np.asarray(right.out_neighbors, dtype=np.int32) + left_node_count,
            ]
        ).astype(np.int32, copy=False),
        out_edge_type=np.concatenate(
            [
                np.asarray(left.out_edge_type, dtype=np.int32),
                np.asarray(right.out_edge_type, dtype=np.int32),
            ]
        ).astype(np.int32, copy=False),
        out_edge_timestamp=np.concatenate(
            [
                np.asarray(left.out_edge_timestamp, dtype=np.int32),
                np.asarray(right.out_edge_timestamp, dtype=np.int32),
            ]
        ).astype(np.int32, copy=False),
        in_ptr=np.concatenate(
            [
                np.asarray(left.in_ptr, dtype=np.int64),
                np.asarray(right.in_ptr[1:], dtype=np.int64) + left_in_edge_count,
            ]
        ).astype(np.int64, copy=False),
        in_neighbors=np.concatenate(
            [
                np.asarray(left.in_neighbors, dtype=np.int32),
                np.asarray(right.in_neighbors, dtype=np.int32) + left_node_count,
            ]
        ).astype(np.int32, copy=False),
        in_edge_type=np.concatenate(
            [
                np.asarray(left.in_edge_type, dtype=np.int32),
                np.asarray(right.in_edge_type, dtype=np.int32),
            ]
        ).astype(np.int32, copy=False),
        in_edge_timestamp=np.concatenate(
            [
                np.asarray(left.in_edge_timestamp, dtype=np.int32),
                np.asarray(right.in_edge_timestamp, dtype=np.int32),
            ]
        ).astype(np.int32, copy=False),
        first_active=np.concatenate(
            [
                np.asarray(left.first_active, dtype=np.int32),
                np.asarray(right.first_active, dtype=np.int32),
            ]
        ).astype(np.int32, copy=False),
        node_time_bucket=np.concatenate(
            [
                np.asarray(left.node_time_bucket, dtype=np.int8),
                np.asarray(right.node_time_bucket, dtype=np.int8) + left_time_window_count,
            ]
        ).astype(np.int8, copy=False),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the supplementary XinYe phase1+phase2 joint-train / phase1-val study.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--build-features", action="store_true", help="Build graph and phase2 feature caches before running.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if the dataset summary already exists.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed override.")
    return parser.parse_args()


def _build_feature_cache(feature_dir: Path, dataset_name: str, feature_env: dict[str, str], phase: str) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "mainline.py"),
        "build_features",
        "--phase",
        str(phase),
        "--outdir",
        str(feature_dir),
    ]
    env = os.environ.copy()
    env[DATASET_ENV_VAR] = dataset_name
    for key, value in feature_env.items():
        env[key] = str(value)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _resolve_graph_config(*, graph_spec: dict[str, Any], plan) -> Any:
    graph_config = build_graph_cfg(
        str(graph_spec["engine_target"]),
        str(graph_spec.get("preset")),
    )
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


def _aggregate_single_dataset(*, study: StudyConfig, dataset_dir: Path, seeds: list[int]) -> None:
    summary_path = dataset_dir / "summary.json"
    summary = _read_json(summary_path)
    results = [
        {
            "dataset": summary["dataset"],
            "dataset_display_name": summary["dataset_display_name"],
            "display_name": summary["display_name"],
            "runner": summary["runner"],
            "val_auc_mean": summary.get("phase1_val_auc_mean"),
            "val_auc_std": summary.get("phase1_val_auc_std"),
            "test_auc_mean": summary.get("phase1_test_auc_mean"),
            "trained_epochs_mean": summary.get("trained_epochs_mean"),
            "summary_path": _path_repr(summary_path),
        }
    ]
    output_root = study.output_root
    ensure_dir(output_root)
    auc_summary_path = output_root / "auc_summary.csv"
    seed_overview_path = output_root / "seed_overview.csv"
    epoch_metrics_path = output_root / "epoch_metrics_all.csv"
    _write_csv(auc_summary_path, results)
    _write_csv(seed_overview_path, _read_csv_rows(dataset_dir / "seed_overview.csv"))
    _write_csv(epoch_metrics_path, _read_csv_rows(dataset_dir / "epoch_metrics_merged.csv"))
    write_json(
        output_root / "summary.json",
        {
            "study_name": study.study_name,
            "display_name": study.display_name,
            "study_type": study.study_type,
            "runner": study.runner,
            "description": study.description,
            "datasets": list(study.datasets),
            "dataset_profile_path": _path_repr(study.dataset_profile_path),
            "config_path": _path_repr(study.config_path),
            "output_root": _path_repr(output_root),
            "seeds": [int(seed) for seed in seeds],
            "auc_summary_path": _path_repr(auc_summary_path),
            "seed_overview_path": _path_repr(seed_overview_path),
            "epoch_metrics_path": _path_repr(epoch_metrics_path),
            "results": results,
        },
    )


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


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


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
