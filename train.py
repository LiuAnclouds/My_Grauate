from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dyrift.data_processing.core.registry import get_active_dataset_spec
from dyrift.config_loader import TrainParameters, resolve_train_parameters
from dyrift.models.engine import GraphModelConfig
from dyrift.models.presets import (
    apply_cfg_overrides,
    build_graph_cfg,
    list_presets,
)
from dyrift.models.runtime import build_runtime
from dyrift.models.spec import (
    DYRIFT_GNN_MODEL,
    DYRIFT_MODEL_SHORT_NAME,
    OFFICIAL_TRAIN_EPOCHS,
    TGAT_BACKBONE_SHORT_NAME,
    TRANSFORMER_BACKBONE_DEPLOY_PRESET,
    TRANSFORMER_BACKBONE_MODEL,
    TRANSFORMER_BACKBONE_PRESET,
)
from dyrift.features.features import build_feature_artifacts
from dyrift.models.graph import get_experiment_cls
from dyrift.utils.common import (
    FEATURE_OUTPUT_ROOT,
    TRAIN_OUTPUT_ROOT,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    save_prediction_npz,
    set_global_seed,
    slice_node_ids,
    write_clean_epoch_metrics,
    write_json,
)


ACTIVE_DATASET_SPEC = get_active_dataset_spec()


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
        description="Single-dataset DyRIFT-TGAT training entrypoint."
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
        help="Also build offline neighbor features.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train one dataset and write outputs under outputs/train/<experiment>/<model>/<dataset>.",
    )
    train_parser.add_argument(
        "--parameter-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON train-parameter file. CLI values override JSON values. "
            "Core train hyperparameters have no thesis defaults in this entrypoint."
        ),
    )
    train_parser.add_argument(
        "--model",
        choices=("m5_temporal_graphsage", "m7_utpm", DYRIFT_GNN_MODEL),
        default=None,
        help=(
            "`m5_temporal_graphsage` is the unified baseline; "
            "`m7_utpm` is the legacy stable GraphSAGE thesis backbone; "
            f"`{DYRIFT_GNN_MODEL}` is the {DYRIFT_MODEL_SHORT_NAME} method with the "
            f"{TGAT_BACKBONE_SHORT_NAME} backbone."
        ),
    )
    train_parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Named thesis preset. "
            "`m5_temporal_graphsage`: unified_baseline. "
            "`m7_utpm`: utpm_temporal_shift_v4 (legacy stable backbone). "
            f"`{TRANSFORMER_BACKBONE_MODEL}`: {TRANSFORMER_BACKBONE_PRESET} (TGAT base) or "
            f"`{TRANSFORMER_BACKBONE_DEPLOY_PRESET}` (deployable DyRIFT-TGAT)."
        ),
    )
    train_parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment folder name under outputs/train.",
    )
    train_parser.add_argument(
        "--run-name",
        default=None,
        help="Human-readable run label stored in metadata.",
    )
    train_parser.add_argument(
        "--feature-profile",
        choices=(
            "utpm_unified",
            "utpm_shift_compact",
            "utpm_shift_enhanced",
            "utpm_shift_history",
            "utpm_shift_fused",
            "utpm_shift_fused_rawmask",
        ),
        default=None,
        help="Unified feature contract used by the training run.",
    )
    train_parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help="Feature cache directory from build_features.",
    )
    train_parser.add_argument(
        "--target-context-groups",
        nargs="*",
        default=None,
        help=(
            "Optional explicit target-context feature groups for the internal prior bridge. "
            "When omitted, no hidden thesis group default is injected. "
            "Pass `none` to disable the internal target-context feature branch."
        ),
    )
    train_parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Base output directory. The final path is <outdir>/<experiment>/<model>/<dataset>.",
    )
    train_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
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
        default=None,
        help="Epochs for graph models.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for graph models.",
    )
    train_parser.add_argument(
        "--fanouts",
        type=int,
        nargs="+",
        default=None,
        help="Neighbor fanouts per layer for graph models.",
    )
    train_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for graph models.",
    )
    train_parser.add_argument(
        "--rel-dim",
        type=int,
        default=None,
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
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve train parameters and print them without loading data or training.",
    )
    train_parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist per-split prediction NPZ files for optional offline analysis.",
    )
    train_parser.add_argument(
        "--skip-train-predictions",
        action="store_true",
        help="When --save-predictions is used, skip phase1_train prediction exports.",
    )
    train_parser.add_argument(
        "--skip-test-pool-predictions",
        action="store_true",
        help="When --save-predictions is used, skip test_pool prediction exports.",
    )
    return parser.parse_args()


def _prepare_split_ids(args: argparse.Namespace):
    split = load_experiment_split()
    train_ids = slice_node_ids(split.train_ids, args.max_train_nodes, seed=11)
    val_ids = slice_node_ids(split.val_ids, args.max_val_nodes, seed=17)
    test_pool_ids = slice_node_ids(split.test_pool_ids, args.max_test_pool_nodes, seed=23)
    external_ids = slice_node_ids(split.external_ids, args.max_external_nodes, seed=29)
    return split, train_ids, val_ids, test_pool_ids, external_ids


def _apply_train_parameters(args: argparse.Namespace) -> TrainParameters:
    params = resolve_train_parameters(
        args=args,
        default_epochs=OFFICIAL_TRAIN_EPOCHS,
        default_outdir=TRAIN_OUTPUT_ROOT,
    )
    args.experiment_name = params.experiment_name
    args.model = params.model
    args.preset = params.preset
    args.run_name = params.run_name
    args.feature_profile = params.feature_profile
    args.feature_dir = params.feature_dir
    args.target_context_groups = params.target_context_groups
    args.outdir = params.outdir
    args.seeds = list(params.seeds)
    args.epochs = int(params.epochs)
    args.batch_size = int(params.batch_size)
    args.hidden_dim = int(params.hidden_dim)
    args.rel_dim = int(params.rel_dim)
    args.fanouts = list(params.fanouts)
    args.device = params.device
    args.learning_rate = params.learning_rate
    args.weight_decay = params.weight_decay
    args.dropout = params.dropout
    args.graph_config_override = list(params.graph_config_overrides)
    args.resolved_train_parameters = params
    return params


def _build_graph_config(args: argparse.Namespace) -> GraphModelConfig:
    preset_name = str(args.preset)
    if preset_name not in list_presets(args.model):
        supported = ", ".join(list_presets(args.model))
        raise ValueError(
            f"Unsupported preset `{preset_name}` for `{args.model}`. Supported presets: {supported}"
        )
    config = build_graph_cfg(args.model, preset_name)
    config, applied_overrides = apply_cfg_overrides(config, args.graph_config_override)
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


def run_build_features(args: argparse.Namespace) -> None:
    phases = list(ACTIVE_DATASET_SPEC.default_artifacts) if args.phase == "both" else [args.phase]
    summary = build_feature_artifacts(
        phases=phases,
        outdir=args.outdir,
        build_neighbor=bool(args.with_neighbor),
    )
    write_json(args.outdir / "feature_build.json", summary)
    print(f"Feature build finished: {args.outdir}")


def run_train(args: argparse.Namespace) -> None:
    params = _apply_train_parameters(args)
    if bool(getattr(args, "dry_run", False)):
        print(json.dumps({"train_parameters": params.to_dict()}, indent=2, ensure_ascii=False))
        return
    split, train_ids, val_ids, test_pool_ids, external_ids = _prepare_split_ids(args)
    if split.train_phase != split.val_phase:
        raise NotImplementedError("Training requires train/val to come from the same graph.")

    graph_config = _build_graph_config(args)
    if args.target_context_groups is None:
        target_context_groups = []
    else:
        normalized_target_context_groups = [
            str(value).strip() for value in args.target_context_groups if str(value).strip()
        ]
        if len(normalized_target_context_groups) == 1 and normalized_target_context_groups[0].lower() == "none":
            target_context_groups = []
        else:
            target_context_groups = normalized_target_context_groups
    runtime = build_runtime(
        feature_dir=args.feature_dir,
        model_name=args.model,
        split=split,
        train_ids=train_ids,
        graph_config=graph_config,
        feature_profile=args.feature_profile,
        target_context_groups=target_context_groups,
    )
    graph_config = runtime.graph_config
    phase1_context = runtime.phase1_context
    phase2_context = runtime.phase2_context

    run_dir = ensure_dir(args.outdir / args.experiment_name / args.model / ACTIVE_DATASET_SPEC.name)
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
    save_predictions = bool(getattr(args, "save_predictions", False))
    experiment_cls = get_experiment_cls(args.model)

    print(
        "[train] "
        f"experiment={args.experiment_name} "
        f"dataset={ACTIVE_DATASET_SPEC.name} "
        f"model={args.model} "
        f"preset={getattr(args, 'resolved_preset', args.preset)} "
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
            experiment.fit(
                context=phase1_context,
                train_ids=train_ids,
                val_ids=val_ids,
                test_pool_ids=test_pool_ids,
                artifact_dir=seed_dir,
            )
            experiment.save(seed_dir)
            train_prob = None
            if save_predictions and not bool(getattr(args, "skip_train_predictions", False)):
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
                if save_predictions
                and test_pool_ids.size
                and not bool(getattr(args, "skip_test_pool_predictions", False))
                else None
            )
            external_prob = (
                experiment.predict_proba(
                    phase2_context,
                    external_ids,
                    batch_seed=seed + 2000,
                    progress_desc=f"{args.model}:seed{seed}:phase2_external",
                )
                if save_predictions and external_ids.size
                else None
            )

            train_metrics = (
                compute_binary_classification_metrics(train_labels, train_prob)
                if train_prob is not None
                else None
            )
            val_metrics = compute_binary_classification_metrics(val_labels, val_prob)
            test_metrics = (
                None if test_pool_prob is None else _maybe_compute_binary_metrics(test_pool_labels, test_pool_prob)
            )
            external_metrics = (
                None if external_prob is None else _maybe_compute_binary_metrics(external_labels, external_prob)
            )

            if save_predictions:
                if train_prob is not None:
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

            if save_predictions and train_prob is not None:
                train_predictions.append(train_prob)
            if save_predictions:
                val_predictions.append(val_prob)
            if save_predictions and test_pool_prob is not None:
                test_pool_predictions.append(test_pool_prob)
            if save_predictions and external_prob is not None:
                external_predictions.append(external_prob)
            train_auc_str = "skipped" if train_metrics is None else f"{train_metrics['auc']:.6f}"
            seed_pbar.set_postfix(
                train_auc="skipped" if train_metrics is None else f"{train_metrics['auc']:.4f}",
                val_auc=f"{val_metrics['auc']:.4f}",
                test_auc=_format_metric(None if test_metrics is None else test_metrics["auc"]),
                refresh=False,
            )
            tqdm.write(
                f"[{args.model}] seed={seed} "
                f"train_auc={train_auc_str} "
                f"val_auc={val_metrics['auc']:.6f} "
                f"test_auc={_format_metric(None if test_metrics is None else test_metrics['auc'])} "
                f"legacy_external_auc={_format_metric(None if external_metrics is None else external_metrics['auc'])}"
            )

    train_avg_path = (
        _save_average_predictions(run_dir, "phase1_train", train_ids, train_labels, train_predictions)
        if save_predictions and train_predictions
        else None
    )
    val_avg_path = (
        _save_average_predictions(run_dir, "phase1_val", val_ids, val_labels, val_predictions)
        if save_predictions and val_predictions
        else None
    )
    test_avg_path = (
        _save_average_predictions(run_dir, "test_pool", test_pool_ids, test_pool_labels, test_pool_predictions)
        if save_predictions and test_pool_predictions
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
        if save_predictions and external_predictions
        else None
    )
    clean_epoch_path = write_clean_epoch_metrics(
        run_dir / "epoch_metrics.csv",
        [run_dir / f"seed_{int(seed)}" / "epoch_metrics.csv" for seed in args.seeds],
    )

    print(f"Training finished: {run_dir}")
    print(f"Epoch metrics: {clean_epoch_path}")


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
