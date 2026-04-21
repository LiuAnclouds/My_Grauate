from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import get_active_dataset_spec
from experiment.training.common import (
    FEATURE_OUTPUT_ROOT,
    MODEL_OUTPUT_ROOT,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    save_prediction_npz,
    set_global_seed,
    slice_node_ids,
    write_json,
)
from experiment.training.features import build_feature_artifacts
from experiment.training.graph_runtime import resolve_graph_experiment_class
from experiment.training.gnn_models import GraphModelConfig
from experiment.training.thesis_contract import (
    OFFICIAL_BACKBONE_FEATURE_PROFILE,
    OFFICIAL_BACKBONE_MODEL,
    OFFICIAL_MAINLINE_BATCH_SIZE,
    OFFICIAL_MAINLINE_FANOUTS,
    OFFICIAL_MAINLINE_HIDDEN_DIM,
    OFFICIAL_MAINLINE_REL_DIM,
    OFFICIAL_TARGET_CONTEXT_GROUPS,
    TRANSFORMER_BACKBONE_MODEL,
    TRANSFORMER_BACKBONE_PRESET,
    TRANSFORMER_BACKBONE_TEACHER_PRESET,
)
from experiment.training.thesis_runtime import prepare_thesis_runtime
from experiment.training.thesis_presets import (
    apply_graph_config_overrides,
    build_thesis_graph_config,
    default_thesis_preset,
    supported_thesis_presets,
)


ACTIVE_DATASET_SPEC = get_active_dataset_spec()


def _path_repr(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _coerce_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


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


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


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


def _maybe_compute_binary_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float] | None:
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


def parse_args() -> argparse.Namespace:
    dataset_spec = get_active_dataset_spec()
    build_phase_choices = (*dataset_spec.phase_filenames.keys(), "both")
    default_build_phase = "both" if len(dataset_spec.default_artifacts) > 1 else dataset_spec.default_artifacts[0]

    parser = argparse.ArgumentParser(
        description="Unified thesis mainline for dynamic-graph anti-fraud experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build_features",
        help="Build the unified UTPM feature cache and graph cache.",
    )
    build_parser.add_argument(
        "--phase",
        default=default_build_phase,
        choices=build_phase_choices,
        help="Which dataset artifact to build. `both` expands to the dataset default artifact set.",
    )
    build_parser.add_argument(
        "--outdir",
        type=Path,
        default=FEATURE_OUTPUT_ROOT,
        help="Output directory for feature caches.",
    )
    build_parser.add_argument(
        "--with-neighbor",
        action="store_true",
        help="Also build offline neighbor features. The thesis mainline does not require them.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train the clean thesis mainline with unified train/val/test_pool evaluation.",
    )
    train_parser.add_argument(
        "--model",
        choices=("m5_temporal_graphsage", "m7_utpm", "m8_utgt"),
        default=OFFICIAL_BACKBONE_MODEL,
        help=(
            "`m5_temporal_graphsage` is the unified baseline; "
            "`m7_utpm` is the legacy stable GraphSAGE thesis backbone; "
            "`m8_utgt` is the transformer-style thesis backbone and the recommended final-result family."
        ),
    )
    train_parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Named thesis preset. "
            "`m5_temporal_graphsage`: unified_baseline. "
            "`m7_utpm`: utpm_temporal_shift_v4 (legacy stable backbone). "
            f"`{TRANSFORMER_BACKBONE_MODEL}`: {TRANSFORMER_BACKBONE_PRESET} (pure UTGT) or "
            f"`{TRANSFORMER_BACKBONE_TEACHER_PRESET}` (teacher-guided recommended backbone)."
        ),
    )
    train_parser.add_argument(
        "--run-name",
        default="thesis_mainline",
        help="Subdirectory name for this experiment run.",
    )
    train_parser.add_argument(
        "--feature-profile",
        choices=("utpm_unified", "utpm_shift_compact", "utpm_shift_enhanced"),
        default=OFFICIAL_BACKBONE_FEATURE_PROFILE,
        help="Unified feature contract used by the thesis mainline.",
    )
    train_parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_OUTPUT_ROOT,
        help="Feature cache directory from build_features.",
    )
    train_parser.add_argument(
        "--target-context-prediction-dir",
        type=Path,
        default=None,
        help="Optional dataset-local prediction directory used as auxiliary target-context teacher features.",
    )
    train_parser.add_argument(
        "--target-context-prediction-transform",
        choices=("raw", "logit"),
        default="raw",
        help="Transform applied to target-context teacher predictions before fusion.",
    )
    train_parser.add_argument(
        "--teacher-distill-prediction-dir",
        type=Path,
        default=None,
        help="Optional dataset-local prediction directory used as fixed teacher distillation targets.",
    )
    train_parser.add_argument(
        "--outdir",
        type=Path,
        default=MODEL_OUTPUT_ROOT,
        help="Output directory for model checkpoints and predictions.",
    )
    train_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 52, 62],
        help="Random seeds used for multi-seed evaluation.",
    )
    train_parser.add_argument(
        "--max-train-nodes",
        type=int,
        default=None,
        help="Optional train node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--max-val-nodes",
        type=int,
        default=None,
        help="Optional validation node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--max-test-pool-nodes",
        type=int,
        default=None,
        help="Optional test_pool node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--max-external-nodes",
        type=int,
        default=None,
        help="Optional legacy phase2 external node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--device",
        default=None,
        help="Torch device for graph models, e.g. cuda or cpu.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=16,
        help="Epochs for graph models.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=OFFICIAL_MAINLINE_BATCH_SIZE,
        help="Batch size for graph models.",
    )
    train_parser.add_argument(
        "--fanouts",
        type=int,
        nargs="+",
        default=list(OFFICIAL_MAINLINE_FANOUTS),
        help="Neighbor fanouts per layer for graph models.",
    )
    train_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=OFFICIAL_MAINLINE_HIDDEN_DIM,
        help="Hidden dimension for graph models.",
    )
    train_parser.add_argument(
        "--rel-dim",
        type=int,
        default=OFFICIAL_MAINLINE_REL_DIM,
        help="Relation embedding dimension for graph models.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional learning-rate override on top of the selected thesis preset.",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Optional weight-decay override on top of the selected thesis preset.",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional dropout override on top of the selected thesis preset.",
    )
    train_parser.add_argument(
        "--graph-config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Low-level GraphModelConfig override, repeatable. Example: --graph-config-override pseudo_contrastive_weight=0.0",
    )
    return parser.parse_args()


def _prepare_split_ids(args: argparse.Namespace):
    split = load_experiment_split()
    train_ids = slice_node_ids(split.train_ids, args.max_train_nodes, seed=11)
    val_ids = slice_node_ids(split.val_ids, args.max_val_nodes, seed=17)
    test_pool_ids = slice_node_ids(split.test_pool_ids, args.max_test_pool_nodes, seed=23)
    external_ids = slice_node_ids(split.external_ids, args.max_external_nodes, seed=29)
    return split, train_ids, val_ids, test_pool_ids, external_ids


def _build_graph_config(args: argparse.Namespace) -> GraphModelConfig:
    preset_name = args.preset or default_thesis_preset(args.model)
    if preset_name not in supported_thesis_presets(args.model):
        supported = ", ".join(supported_thesis_presets(args.model))
        raise ValueError(
            f"Unsupported preset `{preset_name}` for `{args.model}`. Supported presets: {supported}"
        )
    config = build_thesis_graph_config(args.model, preset_name)
    config, applied_overrides = apply_graph_config_overrides(config, args.graph_config_override)
    runtime_updates: dict[str, Any] = {}
    if args.learning_rate is not None:
        runtime_updates["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        runtime_updates["weight_decay"] = float(args.weight_decay)
    if args.dropout is not None:
        runtime_updates["dropout"] = float(args.dropout)
    if runtime_updates:
        config = replace(config, **runtime_updates)
        applied_overrides = {**applied_overrides, **runtime_updates}
    args.resolved_preset = preset_name
    args.applied_graph_config_overrides = applied_overrides
    return config


def _mainline_metadata(model_name: str, graph_config: GraphModelConfig | None = None) -> dict[str, Any]:
    if str(model_name) in {"m7_utpm", "m8_utgt"}:
        aux_regularizer = "pseudo-contrastive test-pool consistency"
        if graph_config is not None and bool(graph_config.pseudo_contrastive_time_balanced):
            aux_regularizer = "time-balanced pseudo-contrastive test-pool consistency"
        if str(model_name) == "m8_utgt":
            main_innovation = "UTGT unified temporal-relation graph transformer"
        else:
            main_innovation = "UTPM unified temporal-relation prototype encoder"
        if graph_config is not None and bool(graph_config.known_label_feature):
            main_innovation = f"{main_innovation} with masked label-context channel"
        if graph_config is not None and bool(graph_config.target_context_fusion == "drift_residual"):
            main_innovation = f"{main_innovation} with drift-residual adaptation"
        if graph_config is not None and bool(graph_config.target_context_fusion == "drift_mix"):
            main_innovation = f"{main_innovation} with adaptive drift-mixture adaptation"
        if graph_config is not None and bool(graph_config.target_context_fusion == "drift_uncertainty_mix"):
            main_innovation = f"{main_innovation} with uncertainty-aware drift adaptation"
        if graph_config is not None and str(graph_config.target_context_fusion) != "none":
            main_innovation = f"{main_innovation} and temporal-normality bridge context"
        return {
            "thesis_mainline_enabled": True,
            "main_innovation": main_innovation,
            "aux_regularizer": aux_regularizer,
        }
    return {
        "thesis_mainline_enabled": False,
        "main_innovation": "UTPM unified temporal-relation encoder",
        "aux_regularizer": None,
    }


def run_build_features(args: argparse.Namespace) -> None:
    phases = list(ACTIVE_DATASET_SPEC.default_artifacts) if args.phase == "both" else [args.phase]
    summary = build_feature_artifacts(
        phases=phases,
        outdir=args.outdir,
        build_neighbor=bool(args.with_neighbor),
    )
    write_json(args.outdir / "build_summary.json", summary)
    print(f"Feature build finished: {args.outdir}")


def run_train(args: argparse.Namespace) -> None:
    split, train_ids, val_ids, test_pool_ids, external_ids = _prepare_split_ids(args)
    if split.train_phase != split.val_phase:
        raise NotImplementedError("The thesis mainline requires train/val to come from the same graph.")

    graph_config = _build_graph_config(args)
    target_context_prediction_dir = _coerce_optional_path(args.target_context_prediction_dir)
    teacher_distill_prediction_dir = _coerce_optional_path(args.teacher_distill_prediction_dir)
    teacher_guidance_requested = (
        target_context_prediction_dir is not None
        or teacher_distill_prediction_dir is not None
        or float(graph_config.teacher_distill_weight) > 0.0
    )
    if str(args.model) == OFFICIAL_BACKBONE_MODEL and teacher_guidance_requested:
        raise ValueError(
            "The legacy `m7_utpm` backbone is kept teacher-free. "
            "Use `m8_utgt` for teacher-guided thesis runs."
        )
    if float(graph_config.teacher_distill_weight) > 0.0 and teacher_distill_prediction_dir is None:
        raise ValueError(
            "teacher_distill_weight > 0 requires --teacher-distill-prediction-dir "
            "so the thesis runtime can load fixed teacher targets."
        )
    metadata = _mainline_metadata(args.model, graph_config)
    runtime = prepare_thesis_runtime(
        feature_dir=args.feature_dir,
        model_name=args.model,
        split=split,
        train_ids=train_ids,
        graph_config=graph_config,
        feature_profile=args.feature_profile,
        target_context_prediction_dir=target_context_prediction_dir,
        target_context_prediction_transform=args.target_context_prediction_transform,
        teacher_distill_prediction_dir=teacher_distill_prediction_dir,
    )
    graph_config = runtime.graph_config
    phase1_context = runtime.phase1_context
    phase2_context = runtime.phase2_context

    run_dir = ensure_dir(args.outdir / args.model / args.run_name)
    input_dim = runtime.input_dim
    num_relations = runtime.num_relations
    global_max_day = runtime.global_max_day
    train_labels = np.asarray(phase1_context.labels[train_ids], dtype=np.int8)
    val_labels = np.asarray(phase1_context.labels[val_ids], dtype=np.int8)
    test_pool_labels = np.asarray(phase1_context.labels[test_pool_ids], dtype=np.int8)
    external_labels = (
        np.asarray(phase2_context.labels[external_ids], dtype=np.int8)
        if external_ids.size
        else np.empty(0, dtype=np.int8)
    )

    train_predictions: list[np.ndarray] = []
    val_predictions: list[np.ndarray] = []
    test_pool_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    experiment_cls = resolve_graph_experiment_class(args.model)

    print(
        "[thesis_mainline] "
        f"dataset={ACTIVE_DATASET_SPEC.name} "
        f"model={args.model} "
        f"preset={getattr(args, 'resolved_preset', default_thesis_preset(args.model))} "
        f"feature_profile={args.feature_profile} "
        f"context_bridge={','.join(runtime.target_context_feature_groups) if runtime.target_context_feature_groups else 'none'} "
        f"train={train_ids.size} val={val_ids.size} test_pool={test_pool_ids.size} external={external_ids.size}"
    )

    with tqdm(
        args.seeds,
        desc=f"train:{args.model}:seeds",
        unit="seed",
        dynamic_ncols=True,
    ) as seed_pbar:
        for seed in seed_pbar:
            set_global_seed(seed)
            seed_dir = ensure_dir(run_dir / f"seed_{seed}")
            experiment = experiment_cls(
                model_name=args.model,
                seed=seed,
                input_dim=input_dim,
                num_relations=num_relations,
                max_day=global_max_day,
                feature_groups=runtime.feature_groups,
                hidden_dim=args.hidden_dim,
                num_layers=len(args.fanouts),
                rel_dim=args.rel_dim,
                fanouts=list(args.fanouts),
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=args.device,
                graph_config=graph_config,
                feature_normalizer_state=runtime.feature_normalizer_state,
                target_context_input_dim=runtime.target_context_input_dim,
                target_context_feature_groups=runtime.target_context_feature_groups,
                target_context_normalizer_state=runtime.target_context_normalizer_state,
            )
            fit_metrics = experiment.fit(
                context=phase1_context,
                train_ids=train_ids,
                val_ids=val_ids,
                test_pool_ids=test_pool_ids,
                artifact_dir=seed_dir,
            )
            train_prob = experiment.predict_proba(
                phase1_context,
                train_ids,
                batch_seed=seed + 500,
                progress_desc=f"{args.model}:seed{seed}:phase1_train",
            )
            val_prob = experiment.predict_proba(
                phase1_context,
                val_ids,
                batch_seed=seed + 1000,
                progress_desc=f"{args.model}:seed{seed}:phase1_val",
            )
            test_pool_prob = (
                experiment.predict_proba(
                    phase1_context,
                    test_pool_ids,
                    batch_seed=seed + 1500,
                    progress_desc=f"{args.model}:seed{seed}:test_pool",
                )
                if test_pool_ids.size
                else None
            )
            external_prob = (
                experiment.predict_proba(
                    phase2_context,
                    external_ids,
                    batch_seed=seed + 2000,
                    progress_desc=f"{args.model}:seed{seed}:phase2_external",
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

            experiment.save(seed_dir)
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
            metrics.append(
                {
                    "seed": seed,
                    "train_auc": train_metrics["auc"],
                    "train_pr_auc": train_metrics["pr_auc"],
                    "train_ap": train_metrics["ap"],
                    "val_auc": val_metrics["auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_ap": val_metrics["ap"],
                    "test_auc": None if test_metrics is None else test_metrics["auc"],
                    "test_pr_auc": None if test_metrics is None else test_metrics["pr_auc"],
                    "test_ap": None if test_metrics is None else test_metrics["ap"],
                    "external_auc": None if external_metrics is None else external_metrics["auc"],
                    "external_pr_auc": None if external_metrics is None else external_metrics["pr_auc"],
                    "external_ap": None if external_metrics is None else external_metrics["ap"],
                    "best_epoch": fit_metrics["best_epoch"],
                    "loss_pos_weight": fit_metrics["loss_pos_weight"],
                    "train_log_path": _path_repr(seed_dir / "train.log"),
                    "epoch_metrics_path": _path_repr(seed_dir / "epoch_metrics.csv"),
                    "curve_path": _path_repr(seed_dir / "training_curves.png"),
                }
            )
            seed_pbar.set_postfix(
                train_auc=f"{train_metrics['auc']:.4f}",
                val_auc=f"{val_metrics['auc']:.4f}",
                test_auc=_format_metric(None if test_metrics is None else test_metrics["auc"]),
                refresh=False,
            )
            tqdm.write(
                f"[{args.model}] seed={seed} "
                f"train_auc={train_metrics['auc']:.6f} "
                f"val_auc={val_metrics['auc']:.6f} "
                f"test_auc={_format_metric(None if test_metrics is None else test_metrics['auc'])} "
                f"legacy_external_auc={_format_metric(None if external_metrics is None else external_metrics['auc'])}"
            )

    train_avg_path = _save_average_predictions(run_dir, "phase1_train", train_ids, train_labels, train_predictions)
    val_avg_path = _save_average_predictions(run_dir, "phase1_val", val_ids, val_labels, val_predictions)
    test_avg_path = (
        _save_average_predictions(run_dir, "test_pool", test_pool_ids, test_pool_labels, test_pool_predictions)
        if test_pool_predictions
        else None
    )
    external_avg_path = (
        _save_average_predictions(
            run_dir,
            "phase2_external",
            external_ids,
            external_labels,
            external_predictions,
        )
        if external_predictions
        else None
    )

    leakage_safeguards = [
        "Feature normalization is fit on split-train nodes only.",
        "Known-label artifacts are built with the time threshold and never consume future labels.",
        "Validation and test_pool predictions are inference-only; their labels never flow back into training.",
        "Each dataset is trained in isolation under its own dataset-scoped cache and output directory.",
        "The target-context bridge reuses only dataset-local feature caches and train-fit normalizers.",
    ]
    if teacher_guidance_requested:
        leakage_safeguards.append(
            "Teacher predictions are fixed dataset-local inference outputs from phase1-train-fitted models; "
            "they are loaded read-only and never refit on validation labels inside the GNN run."
        )
    summary = {
        "model_name": args.model,
        "run_name": args.run_name,
        "dataset": ACTIVE_DATASET_SPEC.name,
        "dataset_display_name": ACTIVE_DATASET_SPEC.display_name,
        "preset": getattr(args, "resolved_preset", default_thesis_preset(args.model)),
        "graph_config_overrides": getattr(args, "applied_graph_config_overrides", {}),
        "feature_profile": args.feature_profile,
        "official_eval_contract": ["train", "val", "test_pool"],
        "dataset_isolation": True,
        "cross_dataset_training": False,
        "leakage_safeguards": leakage_safeguards,
        "thesis_mainline_enabled": metadata["thesis_mainline_enabled"],
        "main_innovation": metadata["main_innovation"],
        "aux_regularizer": metadata["aux_regularizer"],
        "feature_groups": phase1_context.feature_store.selected_groups,
        "target_context_feature_groups": list(runtime.target_context_feature_groups),
        "target_context_bridge_enabled": bool(runtime.target_context_feature_groups),
        "target_context_bridge_family": (
            "temporal_normality_bridge"
            if runtime.target_context_feature_groups
            else "none"
        ),
        "teacher_guidance": {
            "enabled": bool(teacher_guidance_requested),
            "preset_teacher_candidate": bool(
                getattr(args, "resolved_preset", None) == TRANSFORMER_BACKBONE_TEACHER_PRESET
            ),
            "target_context_prediction_dir": (
                None if target_context_prediction_dir is None else _path_repr(target_context_prediction_dir)
            ),
            "target_context_prediction_transform": str(args.target_context_prediction_transform),
            "teacher_distill_prediction_dir": (
                None if teacher_distill_prediction_dir is None else _path_repr(teacher_distill_prediction_dir)
            ),
            "teacher_distill_weight": float(graph_config.teacher_distill_weight),
            "hard_negative_teacher_blend": float(graph_config.hard_negative_teacher_blend),
            "target_aux_feature_count": (
                0 if phase1_context.target_aux_feature_names is None else len(phase1_context.target_aux_feature_names)
            ),
            "distill_train_coverage": (
                None
                if phase1_context.distill_target_mask is None or train_ids.size == 0
                else float(np.mean(np.asarray(phase1_context.distill_target_mask[train_ids], dtype=np.float32)))
            ),
        },
        "split_style": split.split_style,
        "train_phase": split.train_phase,
        "val_phase": split.val_phase,
        "test_pool_phase": split.train_phase,
        "external_phase": split.external_phase,
        "seeds": list(args.seeds),
        "train_size": int(train_ids.size),
        "val_size": int(val_ids.size),
        "test_pool_size": int(test_pool_ids.size),
        "external_size": int(external_ids.size),
        "train_auc_mean": _metric_mean(metrics, "train_auc"),
        "train_auc_std": _metric_std(metrics, "train_auc"),
        "train_pr_auc_mean": _metric_mean(metrics, "train_pr_auc"),
        "train_pr_auc_std": _metric_std(metrics, "train_pr_auc"),
        "train_ap_mean": _metric_mean(metrics, "train_ap"),
        "train_ap_std": _metric_std(metrics, "train_ap"),
        "val_auc_mean": _metric_mean(metrics, "val_auc"),
        "val_auc_std": _metric_std(metrics, "val_auc"),
        "val_pr_auc_mean": _metric_mean(metrics, "val_pr_auc"),
        "val_pr_auc_std": _metric_std(metrics, "val_pr_auc"),
        "val_ap_mean": _metric_mean(metrics, "val_ap"),
        "val_ap_std": _metric_std(metrics, "val_ap"),
        "test_auc_mean": _metric_mean(metrics, "test_auc"),
        "test_auc_std": _metric_std(metrics, "test_auc"),
        "test_pr_auc_mean": _metric_mean(metrics, "test_pr_auc"),
        "test_pr_auc_std": _metric_std(metrics, "test_pr_auc"),
        "test_ap_mean": _metric_mean(metrics, "test_ap"),
        "test_ap_std": _metric_std(metrics, "test_ap"),
        "external_auc_mean": _metric_mean(metrics, "external_auc"),
        "external_auc_std": _metric_std(metrics, "external_auc"),
        "external_pr_auc_mean": _metric_mean(metrics, "external_pr_auc"),
        "external_pr_auc_std": _metric_std(metrics, "external_pr_auc"),
        "external_ap_mean": _metric_mean(metrics, "external_ap"),
        "external_ap_std": _metric_std(metrics, "external_ap"),
        "train_prediction_path": _path_repr(train_avg_path),
        "val_prediction_path": _path_repr(val_avg_path),
        "test_prediction_path": None if test_avg_path is None else _path_repr(test_avg_path),
        "legacy_external_prediction_path": (
            None if external_avg_path is None else _path_repr(external_avg_path)
        ),
        "seed_metrics": metrics,
        "phase1_train_avg_predictions": _path_repr(train_avg_path),
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "test_pool_avg_predictions": None if test_avg_path is None else _path_repr(test_avg_path),
        "phase2_external_avg_predictions": (
            None if external_avg_path is None else _path_repr(external_avg_path)
        ),
        "graph_config": {
            "input_dim": input_dim,
            "num_relations": num_relations,
            "hidden_dim": args.hidden_dim,
            "rel_dim": args.rel_dim,
            "fanouts": list(args.fanouts),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": args.device,
            "aggregator_type": (
                "attention" if str(args.model) == TRANSFORMER_BACKBONE_MODEL else "sage"
            ),
            "official_target_context_groups": list(OFFICIAL_TARGET_CONTEXT_GROUPS),
            **graph_config.to_dict(),
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Training finished: {run_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "build_features":
        run_build_features(args)
        return
    if args.command == "train":
        run_train(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
