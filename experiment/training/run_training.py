from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (
    BLEND_OUTPUT_ROOT,
    FEATURE_OUTPUT_ROOT,
    MODEL_OUTPUT_ROOT,
    align_prediction_bundle,
    compute_binary_classification_metrics,
    ensure_dir,
    load_prediction_npz,
    load_experiment_split,
    load_phase_arrays,
    resolve_prediction_path,
    save_prediction_npz,
    set_global_seed,
    slice_node_ids,
    write_json,
)
from experiment.training.features import (
    FeatureStore,
    build_feature_artifacts,
    build_hybrid_feature_normalizer,
    resolve_feature_groups,
)
from experiment.training.graph_runtime import (
    build_graph_label_artifacts,
    make_graph_contexts,
    resolve_graph_experiment_class,
)
from experiment.training.gbdt_models import LightGBMExperiment
from experiment.training.gnn_models import (
    GraphModelConfig,
)


DEFAULT_SEEDS = [42, 52, 62]
PROMOTION_DELTA_MIN = 0.003
PROMOTION_EXTERNAL_DROP_MAX = 0.001


def _path_repr(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training CLI for the DGraph anti-fraud benchmark framework."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build_features",
        help="Build offline feature caches and graph caches.",
    )
    build_parser.add_argument(
        "--phase",
        default="both",
        choices=("phase1", "phase2", "both"),
        help="Which phase to build.",
    )
    build_parser.add_argument(
        "--outdir",
        type=Path,
        default=FEATURE_OUTPUT_ROOT,
        help="Output directory for feature caches.",
    )
    build_parser.add_argument(
        "--skip-neighbor",
        action="store_true",
        help="Skip 1-hop neighbor aggregation features for a faster build.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train one model family and produce validation/external predictions.",
    )
    train_parser.add_argument(
        "--model",
        required=True,
        choices=(
            "m1_tabular",
            "m2_hybrid",
            "m3_neighbor",
            "m4_graphsage",
            "m5_temporal_graphsage",
            "m6_temporal_gat",
        ),
        help="Model family to train.",
    )
    train_parser.add_argument(
        "--run-name",
        default="default",
        help="Subdirectory name for this experiment run.",
    )
    train_parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_OUTPUT_ROOT,
        help="Feature cache directory from build_features.",
    )
    train_parser.add_argument(
        "--outdir",
        type=Path,
        default=MODEL_OUTPUT_ROOT,
        help="Output directory for model checkpoints and predictions.",
    )
    train_parser.add_argument(
        "--extra-groups",
        nargs="*",
        default=(),
        help=(
            "Optional extra offline feature groups appended on top of the model default. "
            "Useful for temporal_snapshot / temporal_recent / temporal_relation_recent."
        ),
    )
    train_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds used for the 3-seed evaluation rule.",
    )
    train_parser.add_argument(
        "--max-train-nodes",
        type=int,
        default=None,
        help="Optional node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--max-val-nodes",
        type=int,
        default=None,
        help="Optional validation node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--max-external-nodes",
        type=int,
        default=None,
        help="Optional phase2 external node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--device",
        default=None,
        help="Torch device for graph models, e.g. cuda or cpu.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Epochs for graph models.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for graph models.",
    )
    train_parser.add_argument(
        "--fanouts",
        type=int,
        nargs="+",
        default=[15, 10],
        help="Neighbor fanouts per layer for graph models.",
    )
    train_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for graph models.",
    )
    train_parser.add_argument(
        "--rel-dim",
        type=int,
        default=32,
        help="Relation embedding dimension for graph models.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for graph models.",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for graph models.",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout for graph models.",
    )
    train_parser.add_argument(
        "--feature-norm",
        choices=("none", "hybrid"),
        default="none",
        help="Feature normalization recipe for graph models.",
    )
    train_parser.add_argument(
        "--norm",
        choices=("none", "layer", "batch"),
        default="none",
        help="Hidden-state normalization for modern graph blocks.",
    )
    train_parser.add_argument(
        "--residual",
        action="store_true",
        help="Enable residual connections in modern graph blocks.",
    )
    train_parser.add_argument(
        "--ffn",
        action="store_true",
        help="Enable FFN layers in modern graph blocks.",
    )
    train_parser.add_argument(
        "--jk",
        choices=("last", "sum"),
        default="last",
        help="Jumping knowledge mode for graph models.",
    )
    train_parser.add_argument(
        "--edge-encoder",
        choices=("basic", "gated"),
        default="basic",
        help="Edge-aware message encoder for modern graph blocks.",
    )
    train_parser.add_argument(
        "--subgraph-head",
        choices=("none", "meanmax"),
        default="none",
        help="Subgraph fusion head for graph models.",
    )
    train_parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Gradient clipping threshold; 0 disables clipping.",
    )
    train_parser.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default="none",
        help="Learning-rate scheduler for graph models.",
    )
    train_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stopping patience; 0 disables early stopping.",
    )
    train_parser.add_argument(
        "--train-negative-ratio",
        type=float,
        default=0.0,
        help=(
            "If > 0, use only train_ids for balanced target-node sampling: keep all "
            "positives and sample at most this many negatives per positive each epoch."
        ),
    )
    train_parser.add_argument(
        "--negative-sampler",
        choices=("random", "hard", "mixed"),
        default="random",
        help="Train-time negative target sampler used when --train-negative-ratio > 0.",
    )
    train_parser.add_argument(
        "--hard-negative-mix",
        type=float,
        default=0.5,
        help="For negative-sampler=mixed, fraction of sampled train negatives drawn from the hard pool.",
    )
    train_parser.add_argument(
        "--hard-negative-warmup-epochs",
        type=int,
        default=1,
        help="Epochs to keep pure random negative sampling before hard-negative mining starts.",
    )
    train_parser.add_argument(
        "--hard-negative-refresh",
        type=int,
        default=2,
        help="Refresh hard-negative pools every N epochs after warmup.",
    )
    train_parser.add_argument(
        "--hard-negative-candidate-cap",
        type=int,
        default=100000,
        help="Maximum number of negative candidates scored per partition when mining hard negatives.",
    )
    train_parser.add_argument(
        "--hard-negative-candidate-multiplier",
        type=float,
        default=4.0,
        help="Mine roughly this many negative candidates relative to the epoch negative budget.",
    )
    train_parser.add_argument(
        "--hard-negative-pool-multiplier",
        type=float,
        default=2.0,
        help="Keep a hard-negative pool this many times larger than the epoch negative budget.",
    )
    train_parser.add_argument(
        "--loss-type",
        choices=("bce", "focal", "bce_ranking", "focal_ranking"),
        default="bce",
        help="Supervised loss for graph models.",
    )
    train_parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma. Used when loss-type contains focal.",
    )
    train_parser.add_argument(
        "--focal-alpha",
        type=float,
        default=-1.0,
        help="Optional focal alpha in [0,1]. Negative values disable alpha balancing.",
    )
    train_parser.add_argument(
        "--ranking-weight",
        type=float,
        default=0.2,
        help="Weight for the pairwise ranking term when loss-type contains ranking.",
    )
    train_parser.add_argument(
        "--ranking-margin",
        type=float,
        default=0.2,
        help="Margin used inside the softplus pairwise ranking loss.",
    )
    train_parser.add_argument(
        "--neighbor-sampler",
        choices=("uniform", "recent", "hybrid", "risk_recent", "consistency_recent", "risk_consistency_recent"),
        default=None,
        help="Temporal neighbor sampler. Defaults to hybrid for m6_temporal_gat and uniform otherwise.",
    )
    train_parser.add_argument(
        "--recent-window",
        type=int,
        default=50,
        help="Candidate recent-history window size for recent/hybrid temporal sampling.",
    )
    train_parser.add_argument(
        "--recent-ratio",
        type=float,
        default=0.8,
        help="Fraction of fanout drawn from recent-history pool under hybrid sampling.",
    )
    train_parser.add_argument(
        "--consistency-temperature",
        type=float,
        default=0.35,
        help="Temperature for consistency-guided temporal sampling; smaller values sharpen neighbor preference.",
    )
    train_parser.add_argument(
        "--time-decay-strength",
        type=float,
        default=0.0,
        help=(
            "If > 0, exponentially downweight stale temporal edges during message passing "
            "using exp(-strength * normalized_relative_time)."
        ),
    )
    train_parser.add_argument(
        "--target-time-weight-half-life-days",
        type=float,
        default=0.0,
        help=(
            "If > 0, exponentially upweight more recent phase1 training target nodes toward "
            "the validation split threshold."
        ),
    )
    train_parser.add_argument(
        "--target-time-weight-floor",
        type=float,
        default=0.25,
        help="Minimum per-target recency weight when --target-time-weight-half-life-days > 0.",
    )
    train_parser.add_argument(
        "--known-label-feature",
        action="store_true",
        help=(
            "Append a 5-dim one-hot feature for known normal / known fraud / known bg2 / "
            "known bg3 / unknown. Phase1 historical 0/1 and all phase1/phase2 bg2/bg3 are "
            "treated as known; target nodes are masked to unknown."
        ),
    )
    train_parser.add_argument(
        "--known-label-feature-dim",
        type=int,
        choices=(3, 5),
        default=5,
        help="Known-label one-hot width. `3` merges background classes, `5` keeps bg2/bg3 separate.",
    )
    train_parser.add_argument(
        "--target-context-fusion",
        choices=("none", "gate", "concat", "logit_residual"),
        default="none",
        help=(
            "Optional target-node context fusion applied after the base graph classifier. "
            "`gate` adds a residual gated context expert; `concat` learns a compact post-head fusion; "
            "`logit_residual` keeps the graph head unchanged and applies a learned context-only logit correction."
        ),
    )
    train_parser.add_argument(
        "--target-context-extra-groups",
        nargs="*",
        default=(),
        help=(
            "Optional extra feature groups loaded only for the target-node context expert. "
            "These groups do not enter graph message passing."
        ),
    )
    train_parser.add_argument(
        "--primary-multiclass-num-classes",
        type=int,
        choices=(0, 3, 4),
        default=0,
        help=(
            "Primary graph objective class count. `0` keeps binary fraud detection, "
            "`3` uses normal/fraud/background, `4` uses normal/fraud/bg2/bg3."
        ),
    )
    train_parser.add_argument(
        "--prototype-multiclass-num-classes",
        type=int,
        choices=(0, 3, 4),
        default=0,
        help=(
            "Prototype-regularization class count. `0` disables the prototype memory loss, "
            "`3` uses normal/fraud/background, `4` uses normal/fraud/bg2/bg3."
        ),
    )
    train_parser.add_argument(
        "--prototype-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the prototype-memory regularization loss on graph embeddings.",
    )
    train_parser.add_argument(
        "--prototype-temperature",
        type=float,
        default=0.2,
        help="Softmax temperature used inside the prototype-memory regularizer.",
    )
    train_parser.add_argument(
        "--prototype-momentum",
        type=float,
        default=0.9,
        help="EMA momentum used to update class prototypes during training.",
    )
    train_parser.add_argument(
        "--prototype-start-epoch",
        type=int,
        default=1,
        help="Delay the prototype-memory regularizer until this epoch to let the backbone stabilize first.",
    )
    train_parser.add_argument(
        "--prototype-loss-ramp-epochs",
        type=int,
        default=0,
        help="If > 1, linearly ramp the prototype-memory loss over this many epochs after it starts.",
    )
    train_parser.add_argument(
        "--prototype-bucket-mode",
        choices=("global", "time_bucket"),
        default="global",
        help=(
            "Prototype memory layout. `global` shares one class prototype bank across all time, "
            "`time_bucket` keeps an additional bank per temporal bucket and falls back to the global bank."
        ),
    )
    train_parser.add_argument(
        "--prototype-neighbor-blend",
        type=float,
        default=0.0,
        help="Blend ratio for adjacent temporal-bucket prototypes when bucket-mode=time_bucket.",
    )
    train_parser.add_argument(
        "--prototype-global-blend",
        type=float,
        default=0.0,
        help=(
            "When using --prototype-bucket-mode time_bucket, blend this amount of the global "
            "prototype logits into the bucket-specific logits."
        ),
    )
    train_parser.add_argument(
        "--prototype-consistency-weight",
        type=float,
        default=0.0,
        help="Extra weight on same-class temporal/global prototype anchor consistency inside the prototype loss.",
    )
    train_parser.add_argument(
        "--include-historical-background-negatives",
        action="store_true",
        help=(
            "Sample leakage-safe phase1 historical background nodes (labels 2/3) as extra "
            "binary negatives during graph-model training."
        ),
    )
    train_parser.add_argument(
        "--historical-background-negative-ratio",
        type=float,
        default=0.25,
        help=(
            "Maximum number of sampled historical background negatives per split-train positive "
            "inside each temporal partition when --include-historical-background-negatives is enabled."
        ),
    )
    train_parser.add_argument(
        "--historical-background-negative-warmup-epochs",
        type=int,
        default=0,
        help=(
            "If > 1, linearly warm the historical background negative ratio from 0 to "
            "--historical-background-negative-ratio over the first N epochs."
        ),
    )
    train_parser.add_argument(
        "--historical-background-aux-only",
        action="store_true",
        help=(
            "When historical background nodes are sampled, exclude them from the primary binary loss "
            "and use them only for the auxiliary multiclass head."
        ),
    )
    train_parser.add_argument(
        "--aux-multiclass-num-classes",
        type=int,
        choices=(3, 4),
        default=4,
        help="Auxiliary head class count. `3` merges bg2/bg3, `4` keeps them separate.",
    )
    train_parser.add_argument(
        "--aux-multiclass-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the auxiliary multiclass loss on train nodes and sampled historical backgrounds.",
    )
    train_parser.add_argument(
        "--aux-inference-blend",
        type=float,
        default=0.0,
        help=(
            "Blend ratio in [0,1] between the primary fraud score and the auxiliary fraud score "
            "derived from the multiclass head."
        ),
    )

    blend_parser = subparsers.add_parser(
        "blend",
        help="Blend multiple model runs using validation probabilities.",
    )
    blend_parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Model run directories that contain averaged prediction files.",
    )
    blend_parser.add_argument(
        "--outdir",
        type=Path,
        default=BLEND_OUTPUT_ROOT,
        help="Output directory for blending artifacts.",
    )
    blend_parser.add_argument(
        "--name",
        default="m6_blend",
        help="Name for the blending output directory.",
    )
    blend_parser.add_argument(
        "--method",
        choices=("logistic", "mean", "rank_mean"),
        default="logistic",
        help="Blending method. rank_mean avoids fitting on the validation set.",
    )
    return parser.parse_args()


def _model_run_dir(outdir: Path, model_name: str, run_name: str) -> Path:
    return ensure_dir(outdir / model_name / run_name)


def _load_labels_for_splits(split) -> tuple[np.ndarray, np.ndarray]:
    phase1 = load_phase_arrays("phase1", keys=("y",))
    phase2 = load_phase_arrays("phase2", keys=("y",))
    return (
        np.asarray(phase1["y"], dtype=np.int8),
        np.asarray(phase2["y"], dtype=np.int8),
    )


def _prepare_split_ids(args: argparse.Namespace):
    split = load_experiment_split()
    train_ids = slice_node_ids(split.train_ids, args.max_train_nodes, seed=11)
    val_ids = slice_node_ids(split.val_ids, args.max_val_nodes, seed=17)
    external_ids = slice_node_ids(split.external_ids, args.max_external_nodes, seed=29)
    return split, train_ids, val_ids, external_ids


def _build_graph_model_config(args: argparse.Namespace) -> GraphModelConfig:
    neighbor_sampler = (
        str(args.neighbor_sampler)
        if args.neighbor_sampler is not None
        else ("hybrid" if args.model == "m6_temporal_gat" else "uniform")
    )
    return GraphModelConfig(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        feature_norm=str(args.feature_norm),
        norm=str(args.norm),
        residual=bool(args.residual),
        ffn=bool(args.ffn),
        jk=str(args.jk),
        edge_encoder=str(args.edge_encoder),
        subgraph_head=str(args.subgraph_head),
        grad_clip=float(args.grad_clip),
        scheduler=str(args.scheduler),
        early_stop_patience=int(args.early_stop_patience),
        train_negative_ratio=float(args.train_negative_ratio),
        negative_sampler=str(args.negative_sampler),
        hard_negative_mix=float(args.hard_negative_mix),
        hard_negative_warmup_epochs=int(args.hard_negative_warmup_epochs),
        hard_negative_refresh=int(args.hard_negative_refresh),
        hard_negative_candidate_cap=int(args.hard_negative_candidate_cap),
        hard_negative_candidate_multiplier=float(args.hard_negative_candidate_multiplier),
        hard_negative_pool_multiplier=float(args.hard_negative_pool_multiplier),
        loss_type=str(args.loss_type),
        focal_gamma=float(args.focal_gamma),
        focal_alpha=float(args.focal_alpha),
        ranking_weight=float(args.ranking_weight),
        ranking_margin=float(args.ranking_margin),
        neighbor_sampler=neighbor_sampler,
        recent_window=max(int(args.recent_window), 1),
        recent_ratio=float(args.recent_ratio),
        consistency_temperature=float(args.consistency_temperature),
        time_decay_strength=float(args.time_decay_strength),
        target_time_weight_half_life_days=float(args.target_time_weight_half_life_days),
        target_time_weight_floor=float(args.target_time_weight_floor),
        known_label_feature=bool(args.known_label_feature),
        known_label_feature_dim=int(args.known_label_feature_dim),
        target_context_fusion=str(args.target_context_fusion),
        target_context_input_dim=0,
        primary_multiclass_num_classes=int(args.primary_multiclass_num_classes),
        prototype_multiclass_num_classes=int(args.prototype_multiclass_num_classes),
        prototype_loss_weight=float(args.prototype_loss_weight),
        prototype_temperature=float(args.prototype_temperature),
        prototype_momentum=float(args.prototype_momentum),
        prototype_start_epoch=int(args.prototype_start_epoch),
        prototype_loss_ramp_epochs=int(args.prototype_loss_ramp_epochs),
        prototype_bucket_mode=str(args.prototype_bucket_mode),
        prototype_neighbor_blend=float(args.prototype_neighbor_blend),
        prototype_global_blend=float(args.prototype_global_blend),
        prototype_consistency_weight=float(args.prototype_consistency_weight),
        include_historical_background_negatives=bool(args.include_historical_background_negatives),
        historical_background_negative_ratio=float(args.historical_background_negative_ratio),
        historical_background_negative_warmup_epochs=int(args.historical_background_negative_warmup_epochs),
        historical_background_aux_only=bool(args.historical_background_aux_only),
        aux_multiclass_num_classes=int(args.aux_multiclass_num_classes),
        aux_multiclass_loss_weight=float(args.aux_multiclass_loss_weight),
        aux_inference_blend=float(args.aux_inference_blend),
    )


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


def _benchmark_summary_path(outdir: Path, run_name: str) -> Path:
    return outdir / "m2_hybrid" / run_name / "summary.json"


def _build_promotion_decision(
    model_name: str,
    run_name: str,
    outdir: Path,
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    if model_name in {"m1_tabular", "m2_hybrid"}:
        return {
            "benchmark_reference": None,
            "promoted": model_name == "m2_hybrid",
            "reason": "m2_hybrid defines the internal benchmark" if model_name == "m2_hybrid" else "tabular baseline only",
        }

    benchmark_path = _benchmark_summary_path(outdir, run_name)
    if not benchmark_path.exists():
        return {
            "benchmark_reference": _path_repr(benchmark_path),
            "promoted": False,
            "reason": "benchmark summary not found",
        }
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    val_delta = float(summary_payload["phase1_val_auc_mean"] - benchmark["phase1_val_auc_mean"])
    external_delta = float(summary_payload["phase2_external_auc_mean"] - benchmark["phase2_external_auc_mean"])
    promoted = val_delta >= PROMOTION_DELTA_MIN and external_delta >= -PROMOTION_EXTERNAL_DROP_MAX
    return {
        "benchmark_reference": _path_repr(benchmark_path),
        "val_delta_vs_m2": val_delta,
        "external_delta_vs_m2": external_delta,
        "promoted": promoted,
        "rule": {
            "min_val_delta": PROMOTION_DELTA_MIN,
            "max_external_drop": PROMOTION_EXTERNAL_DROP_MAX,
        },
    }


def run_build_features(args: argparse.Namespace) -> None:
    phases = ["phase1", "phase2"] if args.phase == "both" else [args.phase]
    summary = build_feature_artifacts(
        phases=phases,
        outdir=args.outdir,
        build_neighbor=not args.skip_neighbor,
    )
    write_json(args.outdir / "build_summary.json", summary)
    print(f"Feature build finished: {args.outdir}")


def run_train_lightgbm(args: argparse.Namespace) -> None:
    with tqdm(
        total=3,
        desc=f"prepare:{args.model}",
        unit="step",
        dynamic_ncols=True,
    ) as prep_pbar:
        split, train_ids, val_ids, external_ids = _prepare_split_ids(args)
        prep_pbar.update(1)
        phase1_y, phase2_y = _load_labels_for_splits(split)
        feature_groups = resolve_feature_groups(args.model, list(args.extra_groups))
        prep_pbar.update(1)
        phase1_store = FeatureStore("phase1", feature_groups, outdir=args.feature_dir)
        phase2_store = FeatureStore("phase2", feature_groups, outdir=args.feature_dir)
        prep_pbar.update(1)

    run_dir = _model_run_dir(args.outdir, args.model, args.run_name)
    val_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    with tqdm(
        args.seeds,
        desc=f"train:{args.model}:seeds",
        unit="seed",
        dynamic_ncols=True,
    ) as seed_pbar:
        for seed in seed_pbar:
            set_global_seed(seed)
            seed_dir = ensure_dir(run_dir / f"seed_{seed}")
            model = LightGBMExperiment(
                model_name=args.model,
                seed=seed,
                feature_groups=feature_groups,
            )
            fit_metrics = model.fit(
                train_store=phase1_store,
                train_ids=train_ids,
                train_labels=phase1_y[train_ids],
                val_ids=val_ids,
                val_labels=phase1_y[val_ids],
            )
            val_prob = model.predict_proba(phase1_store, val_ids)
            external_prob = model.predict_proba(phase2_store, external_ids)
            val_metrics = compute_binary_classification_metrics(phase1_y[val_ids], val_prob)
            external_metrics = compute_binary_classification_metrics(phase2_y[external_ids], external_prob)
            model.save(seed_dir, feature_names=phase1_store.feature_names)
            save_prediction_npz(seed_dir / "phase1_val_predictions.npz", val_ids, phase1_y[val_ids], val_prob)
            save_prediction_npz(
                seed_dir / "phase2_external_predictions.npz",
                external_ids,
                phase2_y[external_ids],
                external_prob,
            )
            val_predictions.append(val_prob)
            external_predictions.append(external_prob)
            metrics.append(
                {
                    "seed": seed,
                    "val_auc": val_metrics["auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_ap": val_metrics["ap"],
                    "external_auc": external_metrics["auc"],
                    "external_pr_auc": external_metrics["pr_auc"],
                    "external_ap": external_metrics["ap"],
                    "best_iteration": fit_metrics["best_iteration"],
                }
            )
            seed_pbar.set_postfix(
                val_auc=f"{val_metrics['auc']:.4f}",
                val_ap=f"{val_metrics['ap']:.4f}",
                external_auc=f"{external_metrics['auc']:.4f}",
                refresh=False,
            )
            tqdm.write(
                f"[{args.model}] seed={seed} "
                f"phase1_val_auc={val_metrics['auc']:.6f} "
                f"phase1_val_pr_auc={val_metrics['pr_auc']:.6f} "
                f"phase1_val_ap={val_metrics['ap']:.6f} "
                f"phase2_external_auc={external_metrics['auc']:.6f} "
                f"phase2_external_pr_auc={external_metrics['pr_auc']:.6f} "
                f"phase2_external_ap={external_metrics['ap']:.6f}"
            )

    val_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase1_val",
        node_ids=val_ids,
        labels=phase1_y[val_ids],
        predictions=val_predictions,
    )
    external_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase2_external",
        node_ids=external_ids,
        labels=phase2_y[external_ids],
        predictions=external_predictions,
    )

    summary = {
        "model_name": args.model,
        "run_name": args.run_name,
        "feature_groups": feature_groups,
        "seeds": list(args.seeds),
        "phase1_train_size": int(train_ids.size),
        "phase1_val_size": int(val_ids.size),
        "phase2_external_size": int(external_ids.size),
        "phase1_val_auc_mean": float(np.mean([row["val_auc"] for row in metrics])),
        "phase1_val_auc_std": float(np.std([row["val_auc"] for row in metrics])),
        "phase1_val_pr_auc_mean": float(np.mean([row["val_pr_auc"] for row in metrics])),
        "phase1_val_pr_auc_std": float(np.std([row["val_pr_auc"] for row in metrics])),
        "phase1_val_ap_mean": float(np.mean([row["val_ap"] for row in metrics])),
        "phase1_val_ap_std": float(np.std([row["val_ap"] for row in metrics])),
        "phase2_external_auc_mean": float(np.mean([row["external_auc"] for row in metrics])),
        "phase2_external_auc_std": float(np.std([row["external_auc"] for row in metrics])),
        "phase2_external_pr_auc_mean": float(np.mean([row["external_pr_auc"] for row in metrics])),
        "phase2_external_pr_auc_std": float(np.std([row["external_pr_auc"] for row in metrics])),
        "phase2_external_ap_mean": float(np.mean([row["external_ap"] for row in metrics])),
        "phase2_external_ap_std": float(np.std([row["external_ap"] for row in metrics])),
        "seed_metrics": metrics,
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "phase2_external_avg_predictions": _path_repr(external_avg_path),
    }
    summary["promotion_decision"] = _build_promotion_decision(
        model_name=args.model,
        run_name=args.run_name,
        outdir=args.outdir,
        summary_payload=summary,
    )
    write_json(run_dir / "summary.json", summary)
    print(f"Training finished: {run_dir}")


def run_train_graph(args: argparse.Namespace) -> None:
    with tqdm(
        total=3,
        desc=f"prepare:{args.model}",
        unit="step",
        dynamic_ncols=True,
    ) as prep_pbar:
        split, split_train_ids, val_ids, external_ids = _prepare_split_ids(args)
        feature_groups = resolve_feature_groups(args.model, list(args.extra_groups))
        target_context_groups = list(args.target_context_extra_groups)
        graph_config = _build_graph_model_config(args)
        train_ids = np.asarray(split_train_ids, dtype=np.int32)
        label_artifacts = build_graph_label_artifacts(
            feature_dir=args.feature_dir,
            split_train_ids=train_ids,
            threshold_day=int(split.threshold_day),
            known_label_feature=graph_config.known_label_feature,
            include_historical_background_negatives=graph_config.include_historical_background_negatives,
        )
        historical_background_ids = np.asarray(
            label_artifacts["phase1_historical_background_ids"],
            dtype=np.int32,
        )
        prep_pbar.update(1)
        feature_normalizer_state = None
        if graph_config.feature_norm == "hybrid":
            feature_normalizer_state = build_hybrid_feature_normalizer(
                phase="phase1",
                selected_groups=feature_groups,
                train_ids=train_ids,
                outdir=args.feature_dir,
            )
        target_context_normalizer_state = None
        if graph_config.feature_norm == "hybrid" and target_context_groups:
            target_context_normalizer_state = build_hybrid_feature_normalizer(
                phase="phase1",
                selected_groups=target_context_groups,
                train_ids=train_ids,
                outdir=args.feature_dir,
            )
        prep_pbar.update(1)
        phase1_context, phase2_context = make_graph_contexts(
            feature_dir=args.feature_dir,
            model_name=args.model,
            extra_groups=list(args.extra_groups),
            feature_normalizer_state=feature_normalizer_state,
            target_context_groups=target_context_groups,
            target_context_normalizer_state=target_context_normalizer_state,
            phase1_known_label_codes=label_artifacts["phase1_known_label_codes"],
            phase2_known_label_codes=label_artifacts["phase2_known_label_codes"],
            phase1_reference_day=int(split.threshold_day),
            phase2_reference_day=None,
            phase1_historical_background_ids=historical_background_ids,
            build_sampling_profile=graph_config.neighbor_sampler in {"consistency_recent", "risk_consistency_recent"},
        )
        prep_pbar.update(1)
    run_dir = _model_run_dir(args.outdir, args.model, args.run_name)
    label_feature_dim = graph_config.known_label_feature_dim if graph_config.known_label_feature else 0
    input_dim = phase1_context.feature_store.input_dim + int(label_feature_dim)
    target_context_input_dim = (
        int(phase1_context.target_context_store.input_dim)
        if phase1_context.target_context_store is not None
        else int(input_dim)
    )
    graph_config = replace(graph_config, target_context_input_dim=int(target_context_input_dim))
    num_relations = phase1_context.graph_cache.num_relations
    global_max_day = max(phase1_context.graph_cache.max_day, phase2_context.graph_cache.max_day)
    val_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    experiment_cls = resolve_graph_experiment_class(args.model)

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
                feature_groups=phase1_context.feature_store.selected_groups,
                hidden_dim=args.hidden_dim,
                num_layers=len(args.fanouts),
                rel_dim=args.rel_dim,
                fanouts=list(args.fanouts),
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=args.device,
                graph_config=graph_config,
                feature_normalizer_state=feature_normalizer_state,
                target_context_input_dim=target_context_input_dim,
                target_context_feature_groups=target_context_groups,
                target_context_normalizer_state=target_context_normalizer_state,
            )
            fit_metrics = experiment.fit(
                context=phase1_context,
                train_ids=train_ids,
                val_ids=val_ids,
                artifact_dir=seed_dir,
            )
            val_prob = experiment.predict_proba(
                phase1_context,
                val_ids,
                batch_seed=seed + 1000,
                progress_desc=f"{args.model}:seed{seed}:phase1_val",
            )
            external_prob = experiment.predict_proba(
                phase2_context,
                external_ids,
                batch_seed=seed + 2000,
                progress_desc=f"{args.model}:seed{seed}:phase2_external",
            )
            val_metrics = compute_binary_classification_metrics(phase1_context.labels[val_ids], val_prob)
            external_metrics = compute_binary_classification_metrics(
                phase2_context.labels[external_ids],
                external_prob,
            )
            experiment.save(seed_dir)
            save_prediction_npz(
                seed_dir / "phase1_val_predictions.npz",
                val_ids,
                phase1_context.labels[val_ids],
                val_prob,
            )
            save_prediction_npz(
                seed_dir / "phase2_external_predictions.npz",
                external_ids,
                phase2_context.labels[external_ids],
                external_prob,
            )
            val_predictions.append(val_prob)
            external_predictions.append(external_prob)
            metrics.append(
                {
                    "seed": seed,
                    "val_auc": val_metrics["auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_ap": val_metrics["ap"],
                    "external_auc": external_metrics["auc"],
                    "external_pr_auc": external_metrics["pr_auc"],
                    "external_ap": external_metrics["ap"],
                    "best_epoch": fit_metrics["best_epoch"],
                    "loss_pos_weight": fit_metrics["loss_pos_weight"],
                    "train_log_path": _path_repr(seed_dir / "train.log"),
                    "epoch_metrics_path": _path_repr(seed_dir / "epoch_metrics.csv"),
                    "curve_path": _path_repr(seed_dir / "training_curves.png"),
                }
            )
            seed_pbar.set_postfix(
                val_auc=f"{val_metrics['auc']:.4f}",
                val_ap=f"{val_metrics['ap']:.4f}",
                external_auc=f"{external_metrics['auc']:.4f}",
                refresh=False,
            )
            tqdm.write(
                f"[{args.model}] seed={seed} "
                f"phase1_val_auc={val_metrics['auc']:.6f} "
                f"phase1_val_pr_auc={val_metrics['pr_auc']:.6f} "
                f"phase1_val_ap={val_metrics['ap']:.6f} "
                f"phase2_external_auc={external_metrics['auc']:.6f} "
                f"phase2_external_pr_auc={external_metrics['pr_auc']:.6f} "
                f"phase2_external_ap={external_metrics['ap']:.6f}"
            )

    val_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase1_val",
        node_ids=val_ids,
        labels=phase1_context.labels[val_ids],
        predictions=val_predictions,
    )
    external_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase2_external",
        node_ids=external_ids,
        labels=phase2_context.labels[external_ids],
        predictions=external_predictions,
    )
    summary = {
        "model_name": args.model,
        "run_name": args.run_name,
        "feature_groups": phase1_context.feature_store.selected_groups,
        "target_context_feature_groups": (
            []
            if phase1_context.target_context_store is None
            else phase1_context.target_context_store.selected_groups
        ),
        "seeds": list(args.seeds),
        "phase1_train_size": int(np.asarray(split_train_ids, dtype=np.int32).size),
        "phase1_split_train_size": int(np.asarray(split_train_ids, dtype=np.int32).size),
        "phase1_historical_background_train_size": int(historical_background_ids.size),
        "phase1_val_size": int(val_ids.size),
        "phase2_external_size": int(external_ids.size),
        "phase1_val_auc_mean": float(np.mean([row["val_auc"] for row in metrics])),
        "phase1_val_auc_std": float(np.std([row["val_auc"] for row in metrics])),
        "phase1_val_pr_auc_mean": float(np.mean([row["val_pr_auc"] for row in metrics])),
        "phase1_val_pr_auc_std": float(np.std([row["val_pr_auc"] for row in metrics])),
        "phase1_val_ap_mean": float(np.mean([row["val_ap"] for row in metrics])),
        "phase1_val_ap_std": float(np.std([row["val_ap"] for row in metrics])),
        "phase2_external_auc_mean": float(np.mean([row["external_auc"] for row in metrics])),
        "phase2_external_auc_std": float(np.std([row["external_auc"] for row in metrics])),
        "phase2_external_pr_auc_mean": float(np.mean([row["external_pr_auc"] for row in metrics])),
        "phase2_external_pr_auc_std": float(np.std([row["external_pr_auc"] for row in metrics])),
        "phase2_external_ap_mean": float(np.mean([row["external_ap"] for row in metrics])),
        "phase2_external_ap_std": float(np.std([row["external_ap"] for row in metrics])),
        "seed_metrics": metrics,
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "phase2_external_avg_predictions": _path_repr(external_avg_path),
        "graph_config": {
            "input_dim": input_dim,
            "num_relations": num_relations,
            "hidden_dim": args.hidden_dim,
            "rel_dim": args.rel_dim,
            "fanouts": list(args.fanouts),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": args.device,
            "aggregator_type": "attention" if args.model == "m6_temporal_gat" else "sage",
            **graph_config.to_dict(),
        },
    }
    summary["promotion_decision"] = _build_promotion_decision(
        model_name=args.model,
        run_name=args.run_name,
        outdir=args.outdir,
        summary_payload=summary,
    )
    write_json(run_dir / "summary.json", summary)
    print(f"Training finished: {run_dir}")

def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def run_blend(args: argparse.Namespace) -> None:
    run_dir = ensure_dir(args.outdir / args.name)
    val_bundles = []
    external_bundles = []
    model_names = []
    for model_run_dir in args.run_dirs:
        val_bundles.append(load_prediction_npz(resolve_prediction_path(model_run_dir, "phase1_val")))
        external_bundles.append(load_prediction_npz(resolve_prediction_path(model_run_dir, "phase2_external")))
        model_names.append(model_run_dir.parent.name)

    base_val = val_bundles[0]
    base_external = external_bundles[0]
    for idx in range(1, len(val_bundles)):
        if not np.array_equal(val_bundles[idx]["node_ids"], base_val["node_ids"]):
            val_bundles[idx] = align_prediction_bundle(val_bundles[idx], base_val["node_ids"])
        if not np.array_equal(external_bundles[idx]["node_ids"], base_external["node_ids"]):
            external_bundles[idx] = align_prediction_bundle(external_bundles[idx], base_external["node_ids"])
        if not np.array_equal(val_bundles[idx]["y_true"], base_val["y_true"]):
            raise AssertionError("Validation labels are not aligned across model runs.")
        if not np.array_equal(external_bundles[idx]["y_true"], base_external["y_true"]):
            raise AssertionError("External labels are not aligned across model runs.")

    coefficients: list[float] | None = None
    intercept: float | None = None
    if args.method == "logistic":
        val_matrix = np.column_stack([_logit(bundle["probability"]) for bundle in val_bundles])
        external_matrix = np.column_stack([_logit(bundle["probability"]) for bundle in external_bundles])
        logistic = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=0,
        )
        logistic.fit(val_matrix, base_val["y_true"])
        val_prob = logistic.predict_proba(val_matrix)[:, 1].astype(np.float32, copy=False)
        external_prob = logistic.predict_proba(external_matrix)[:, 1].astype(np.float32, copy=False)
        coefficients = logistic.coef_.reshape(-1).astype(float).tolist()
        intercept = float(logistic.intercept_[0])
    elif args.method == "mean":
        val_prob = np.mean(
            [bundle["probability"] for bundle in val_bundles],
            axis=0,
            dtype=np.float32,
        ).astype(np.float32, copy=False)
        external_prob = np.mean(
            [bundle["probability"] for bundle in external_bundles],
            axis=0,
            dtype=np.float32,
        ).astype(np.float32, copy=False)
    else:
        val_rank = np.column_stack(
            [np.argsort(np.argsort(bundle["probability"])).astype(np.float32) for bundle in val_bundles]
        )
        external_rank = np.column_stack(
            [np.argsort(np.argsort(bundle["probability"])).astype(np.float32) for bundle in external_bundles]
        )
        val_prob = np.mean(val_rank, axis=1, dtype=np.float32).astype(np.float32, copy=False)
        external_prob = np.mean(external_rank, axis=1, dtype=np.float32).astype(np.float32, copy=False)

    val_metrics = compute_binary_classification_metrics(base_val["y_true"], val_prob)
    external_metrics = compute_binary_classification_metrics(base_external["y_true"], external_prob)
    save_prediction_npz(
        run_dir / "phase1_val_blend_predictions.npz",
        base_val["node_ids"],
        base_val["y_true"],
        val_prob,
    )
    save_prediction_npz(
        run_dir / "phase2_external_blend_predictions.npz",
        base_external["node_ids"],
        base_external["y_true"],
        external_prob,
    )
    summary = {
        "blend_name": args.name,
        "method": args.method,
        "model_runs": [_path_repr(path) for path in args.run_dirs],
        "model_names": model_names,
        "phase1_val_auc": val_metrics["auc"],
        "phase1_val_pr_auc": val_metrics["pr_auc"],
        "phase1_val_ap": val_metrics["ap"],
        "phase2_external_auc": external_metrics["auc"],
        "phase2_external_pr_auc": external_metrics["pr_auc"],
        "phase2_external_ap": external_metrics["ap"],
    }
    if coefficients is not None and intercept is not None:
        summary["coefficients"] = coefficients
        summary["intercept"] = intercept
    write_json(run_dir / "summary.json", summary)
    print(f"Blend finished: {run_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "build_features":
        run_build_features(args)
        return
    if args.command == "train":
        if args.model in {"m1_tabular", "m2_hybrid", "m3_neighbor"}:
            run_train_lightgbm(args)
            return
        run_train_graph(args)
        return
    if args.command == "blend":
        run_blend(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
