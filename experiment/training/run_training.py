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

from experiment.datasets.registry import get_active_dataset_spec
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
    load_graph_cache,
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
from experiment.training.xgb_utils import binary_score_from_softprob


DEFAULT_SEEDS = [42, 52, 62]
PROMOTION_DELTA_MIN = 0.003
PROMOTION_EXTERNAL_DROP_MAX = 0.001
ACTIVE_DATASET_SPEC = get_active_dataset_spec()


def _path_repr(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    dataset_spec = get_active_dataset_spec()
    build_phase_choices = (*dataset_spec.phase_filenames.keys(), "both")
    default_build_phase = "both" if len(dataset_spec.default_artifacts) > 1 else dataset_spec.default_artifacts[0]
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
        "--skip-neighbor",
        action="store_true",
        help="Skip 1-hop neighbor aggregation features for a faster build.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train one model family and produce the official train/val/test_pool predictions.",
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
            "m7_utpm",
        ),
        help="Model family to train. `m7_utpm` is deprecated here and must be run via `run_thesis_mainline.py`.",
    )
    train_parser.add_argument(
        "--run-name",
        default="default",
        help="Subdirectory name for this experiment run.",
    )
    train_parser.add_argument(
        "--feature-profile",
        choices=("legacy", "utpm_unified"),
        default="legacy",
        help="Feature-group contract consumed by the model. `utpm_unified` is the thesis mainline.",
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
        "--dataset-adaptive-profile",
        choices=("none", "relation_scarcity_auto", "relation_scarcity_bridge_v1"),
        default="none",
        help=(
            "Apply a unified dataset-adaptive recipe. `relation_scarcity_auto` simplifies "
            "low-relation graphs while keeping the richer semantic recipe on multi-relation graphs. "
            "`relation_scarcity_bridge_v1` also switches effective feature groups and "
            "dataset-safe teacher/context inputs before graph contexts are built."
        ),
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
        help="Optional legacy phase2 external node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--max-test-pool-nodes",
        type=int,
        default=None,
        help="Optional test_pool node cap for smoke tests.",
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
        "--hard-negative-teacher-blend",
        type=float,
        default=0.0,
        help=(
            "Blend teacher fraud scores into hard-negative mining. "
            "This value controls both the fraction of candidates preselected by teacher risk "
            "and the final blend between student and teacher scores when ranking hard negatives."
        ),
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
        "--message-risk-strength",
        type=float,
        default=0.0,
        help=(
            "If > 0 and temporal_counterparty_concentration is present, use concentration-derived "
            "node risk to slightly amplify or dampen message passing."
        ),
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
        choices=(
            "none",
            "gate",
            "concat",
            "logit_residual",
            "atm_residual",
            "drift_residual",
            "drift_mix",
            "drift_uncertainty_mix",
            "risk_drift_residual",
        ),
        default="none",
        help=(
            "Optional target-node context fusion applied after the base graph classifier. "
            "`gate` adds a residual gated context expert; `concat` learns a compact post-head fusion; "
            "`logit_residual` keeps the graph head unchanged and applies a learned context-only logit correction; "
            "`atm_residual` adds an ATM-GAD-style adaptive temporal-window residual branch; "
            "`drift_residual` adds a temporal-drift-calibrated residual branch that suppresses "
            "unstable context corrections; `drift_mix` blends a learned drift delta into the base "
            "logit; `drift_uncertainty_mix` additionally down-weights overconfident corrections; "
            "`risk_drift_residual` adds an extra risk gate that can "
            "amplify or dampen the drift branch based on target-context anomaly patterns."
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
        "--target-context-prediction-dir",
        type=Path,
        default=None,
        help=(
            "Optional run directory providing leakage-safe phase1_train/phase1_val/phase2_external "
            "prediction bundles appended to the target-context branch."
        ),
    )
    train_parser.add_argument(
        "--target-context-prediction-transform",
        choices=("raw", "logit"),
        default="raw",
        help="Transform applied to appended target-context prediction features.",
    )
    train_parser.add_argument(
        "--teacher-distill-prediction-dir",
        type=Path,
        default=None,
        help="Optional run directory providing leakage-safe phase1_train predictions for teacher distillation.",
    )
    train_parser.add_argument(
        "--teacher-distill-weight",
        type=float,
        default=0.0,
        help="Weight for teacher distillation on covered phase1-train nodes.",
    )
    train_parser.add_argument(
        "--teacher-distill-loss",
        choices=("bce", "mse", "rank"),
        default="bce",
        help="Teacher distillation loss form.",
    )
    train_parser.add_argument(
        "--teacher-distill-start-epoch",
        type=int,
        default=1,
        help="Epoch index at which teacher distillation becomes active.",
    )
    train_parser.add_argument(
        "--teacher-distill-ramp-epochs",
        type=int,
        default=0,
        help="Optional linear warmup length for teacher distillation weight.",
    )
    train_parser.add_argument(
        "--teacher-distill-agreement-floor",
        type=float,
        default=0.0,
        help="Minimum teacher probability assigned to the true label before distillation is applied.",
    )
    train_parser.add_argument(
        "--teacher-distill-rank-gap",
        type=float,
        default=0.0,
        help="Minimum teacher-score gap used to form pairwise ranking-distillation constraints.",
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
        choices=(0, 2, 3, 4),
        default=0,
        help=(
            "Prototype-regularization class count. `0` disables the prototype memory loss, "
            "`2` uses normal/fraud, `3` uses normal/fraud/background, `4` uses normal/fraud/bg2/bg3."
        ),
    )
    train_parser.add_argument(
        "--prototype-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the prototype-memory regularization loss on graph embeddings.",
    )
    train_parser.add_argument(
        "--prototype-loss-weight-schedule",
        choices=("none", "adaptive_quality"),
        default="none",
        help=(
            "Optional schedule for prototype-memory loss trust. "
            "'adaptive_quality' downweights prototype loss when recent prototype "
            "classification/margin signals are weak."
        ),
    )
    train_parser.add_argument(
        "--prototype-loss-min-weight",
        type=float,
        default=0.0,
        help="Minimum effective prototype-memory loss weight under the selected schedule.",
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
        "--prototype-separation-weight",
        type=float,
        default=0.0,
        help=(
            "Extra weight for margin-based separation between a sample's target prototype and the "
            "hardest negative prototype."
        ),
    )
    train_parser.add_argument(
        "--prototype-separation-margin",
        type=float,
        default=0.1,
        help="Margin used by --prototype-separation-weight for dynamic prototype discrimination.",
    )
    train_parser.add_argument(
        "--normal-bucket-align-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for time-bucket normal-class alignment. Encourages embeddings of normal nodes "
            "from different temporal buckets to stay consistent under time drift."
        ),
    )
    train_parser.add_argument(
        "--normal-bucket-shift-strength",
        type=float,
        default=0.0,
        help=(
            "Strength of explicit normal-anchor temporal shift compensation applied to target "
            "embeddings using the learned normal bucket memory."
        ),
    )
    train_parser.add_argument(
        "--target-time-adapter-strength",
        type=float,
        default=0.0,
        help=(
            "Strength of target-time conditional embedding modulation before the final classifier. "
            "Useful for learning dataset-specific temporal drift correction."
        ),
    )
    train_parser.add_argument(
        "--target-time-adapter-type",
        choices=("affine", "drift_expert"),
        default="affine",
        help=(
            "Target-time adapter form. `affine` keeps the original single expert time modulation; "
            "`drift_expert` uses drift-aware temporal expert routing."
        ),
    )
    train_parser.add_argument(
        "--target-time-adapter-num-experts",
        type=int,
        default=4,
        help="Number of temporal experts used when --target-time-adapter-type=drift_expert.",
    )
    train_parser.add_argument(
        "--target-time-expert-entropy-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum normalized routing entropy encouraged for --target-time-adapter-type=drift_expert. "
            "Values in [0,1]. Set > 0 to discourage late expert collapse."
        ),
    )
    train_parser.add_argument(
        "--target-time-expert-entropy-weight",
        type=float,
        default=0.0,
        help=(
            "Loss weight for the drift_expert entropy-floor regularizer. "
            "Applied only when --target-time-adapter-type=drift_expert."
        ),
    )
    train_parser.add_argument(
        "--atm-gate-strength",
        type=float,
        default=1.0,
        help=(
            "Residual strength for the time-aware context branch when "
            "--target-context-fusion=atm_residual, drift_residual, or risk_drift_residual."
        ),
    )
    train_parser.add_argument(
        "--context-residual-scale",
        type=float,
        default=1.0,
        help="Multiplicative strength on the post-head target-context residual logits.",
    )
    train_parser.add_argument(
        "--context-residual-clip",
        type=float,
        default=0.0,
        help=(
            "If > 0, apply smooth tanh clipping to target-context residual logits using this "
            "absolute cap."
        ),
    )
    train_parser.add_argument(
        "--context-residual-budget",
        type=float,
        default=0.0,
        help=(
            "If > 0, residual magnitudes beyond this absolute value incur a training penalty "
            "before the final classifier."
        ),
    )
    train_parser.add_argument(
        "--context-residual-budget-weight",
        type=float,
        default=0.0,
        help="Weight for the target-context residual budget penalty.",
    )
    train_parser.add_argument(
        "--context-residual-budget-schedule",
        choices=("none", "prototype_release", "prototype_adaptive"),
        default="none",
        help=(
            "Optional curriculum for the target-context residual budget penalty. "
            "'prototype_release' keeps the full penalty until prototype warmup/ramp stabilizes, "
            "then cosine-decays it; 'prototype_adaptive' further gates the release using "
            "prototype-coverage/margin diagnostics and context-residual risk."
        ),
    )
    train_parser.add_argument(
        "--context-residual-budget-min-weight",
        type=float,
        default=0.0,
        help=(
            "Minimum residual budget penalty weight after schedule decay. "
            "Ignored when --context-residual-budget-schedule=none."
        ),
    )
    train_parser.add_argument(
        "--context-residual-budget-release-epochs",
        type=int,
        default=0,
        help=(
            "How many epochs the residual budget penalty takes to decay under the selected "
            "schedule. <= 0 means decay until the configured epoch budget is exhausted."
        ),
    )
    train_parser.add_argument(
        "--context-residual-budget-release-delay-epochs",
        type=int,
        default=0,
        help=(
            "Extra consolidation epochs to wait after prototype warmup/ramp before starting "
            "the residual budget release schedule."
        ),
    )
    train_parser.add_argument(
        "--normal-bucket-adv-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for adversarial time-bucket confusion on normal-node embeddings. "
            "Uses gradient reversal to suppress temporal leakage in normal representations."
        ),
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
    train_parser.add_argument(
        "--pseudo-contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for the pseudo-contrastive auxiliary regularizer on test_pool pseudo labels.",
    )
    train_parser.add_argument(
        "--pseudo-contrastive-temperature",
        type=float,
        default=0.2,
        help="Temperature used by the pseudo-contrastive auxiliary regularizer.",
    )
    train_parser.add_argument(
        "--pseudo-contrastive-sample-size",
        type=int,
        default=128,
        help="Number of test_pool nodes sampled per pseudo-contrastive step.",
    )
    train_parser.add_argument(
        "--pseudo-contrastive-low-quantile",
        type=float,
        default=0.1,
        help="Lower score quantile used to form pseudo-normal anchors from test_pool predictions.",
    )
    train_parser.add_argument(
        "--pseudo-contrastive-high-quantile",
        type=float,
        default=0.9,
        help="Upper score quantile used to form pseudo-anomaly anchors from test_pool predictions.",
    )
    train_parser.add_argument(
        "--pseudo-contrastive-interval",
        type=int,
        default=4,
        help="Run one pseudo-contrastive step every N train batches.",
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


def _load_labels_for_splits(split) -> dict[str, np.ndarray]:
    phase_names: list[str] = [split.train_phase, split.val_phase]
    if split.external_phase:
        phase_names.append(str(split.external_phase))
    labels_by_phase: dict[str, np.ndarray] = {}
    for phase in dict.fromkeys(phase_names):
        labels_by_phase[str(phase)] = np.asarray(load_phase_arrays(str(phase), keys=("y",))["y"], dtype=np.int8)
    return labels_by_phase


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


def _resolve_effective_feature_profile(args: argparse.Namespace) -> str:
    return str(getattr(args, "feature_profile", "legacy") or "legacy").strip().lower()


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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _drop_groups(values: list[str], dropped: set[str]) -> list[str]:
    return [value for value in values if str(value) not in dropped]


def _coerce_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _existing_optional_path(value: Path | str | None) -> Path | None:
    path = _coerce_optional_path(value)
    if path is None or not path.exists():
        return None
    return path


def _transform_target_context_prediction_features(
    score: np.ndarray,
    transform: str,
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if str(transform) == "raw":
        return values.astype(np.float32, copy=False)
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    if clipped.shape[1] == 1:
        return np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)
    return np.log(clipped).astype(np.float32, copy=False)


def _load_target_context_prediction_features(
    *,
    prediction_dir: Path,
    prediction_transform: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    phase1_num_nodes: int,
    phase2_num_nodes: int,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    run_dir = Path(prediction_dir)
    train_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase1_train"))
    val_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase1_val"))
    train_features = _transform_target_context_prediction_features(train_bundle["probability"], prediction_transform)
    val_features = _transform_target_context_prediction_features(val_bundle["probability"], prediction_transform)
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError(
            f"{run_dir}: train/val prediction feature dims mismatch "
            f"{train_features.shape[1]} vs {val_features.shape[1]}."
        )

    def _scatter_known_scores(
        target_matrix: np.ndarray,
        *,
        target_ids: np.ndarray,
        bundle: dict[str, np.ndarray],
        feature_values: np.ndarray,
    ) -> int:
        position = {int(node_id): idx for idx, node_id in enumerate(np.asarray(bundle["node_ids"], dtype=np.int32).tolist())}
        matched_target: list[int] = []
        matched_source: list[int] = []
        for node_id in np.asarray(target_ids, dtype=np.int32).tolist():
            source_idx = position.get(int(node_id))
            if source_idx is None:
                continue
            matched_target.append(int(node_id))
            matched_source.append(int(source_idx))
        if not matched_target:
            return 0
        target_matrix[np.asarray(matched_target, dtype=np.int32)] = feature_values[
            np.asarray(matched_source, dtype=np.int64)
        ]
        return len(matched_target)

    phase1_features = np.zeros((int(phase1_num_nodes), int(train_features.shape[1])), dtype=np.float32)
    _scatter_known_scores(
        phase1_features,
        target_ids=train_ids,
        bundle=train_bundle,
        feature_values=train_features,
    )
    _scatter_known_scores(
        phase1_features,
        target_ids=val_ids,
        bundle=val_bundle,
        feature_values=val_features,
    )

    phase2_features = None
    if external_ids.size:
        external_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase2_external"))
        external_features = _transform_target_context_prediction_features(
            external_bundle["probability"],
            prediction_transform,
        )
        if external_features.shape[1] != train_features.shape[1]:
            raise ValueError(
                f"{run_dir}: train/external prediction feature dims mismatch "
                f"{train_features.shape[1]} vs {external_features.shape[1]}."
            )
        phase2_features = np.zeros((int(phase2_num_nodes), int(train_features.shape[1])), dtype=np.float32)
        _scatter_known_scores(
            phase2_features,
            target_ids=external_ids,
            bundle=external_bundle,
            feature_values=external_features,
        )

    feature_names = [f"teacher_pred_{idx}" for idx in range(int(train_features.shape[1]))]
    return phase1_features, phase2_features, feature_names


def _coerce_teacher_binary_score(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    if values.ndim == 1:
        return values.astype(np.float32, copy=False)
    if values.ndim == 2 and values.shape[1] == 1:
        return values.reshape(-1).astype(np.float32, copy=False)
    if values.ndim == 2 and values.shape[1] == 4:
        return binary_score_from_softprob(values)
    if values.ndim == 2 and values.shape[1] == 2:
        denom = np.clip(values[:, 0] + values[:, 1], 1e-6, None)
        return (values[:, 1] / denom).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported teacher prediction shape: {values.shape}")


def _load_teacher_distill_targets(
    *,
    prediction_dir: Path,
    train_ids: np.ndarray,
    phase1_num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    run_dir = Path(prediction_dir)
    train_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase1_train"))
    score = _coerce_teacher_binary_score(train_bundle["probability"])
    targets = np.zeros((int(phase1_num_nodes),), dtype=np.float32)
    mask = np.zeros((int(phase1_num_nodes),), dtype=bool)
    position = {
        int(node_id): idx
        for idx, node_id in enumerate(np.asarray(train_bundle["node_ids"], dtype=np.int32).tolist())
    }
    matched_target: list[int] = []
    matched_source: list[int] = []
    for node_id in np.asarray(train_ids, dtype=np.int32).tolist():
        source_idx = position.get(int(node_id))
        if source_idx is None:
            continue
        matched_target.append(int(node_id))
        matched_source.append(int(source_idx))
    if matched_target:
        matched_target_idx = np.asarray(matched_target, dtype=np.int32)
        matched_source_idx = np.asarray(matched_source, dtype=np.int64)
        targets[matched_target_idx] = score[matched_source_idx]
        mask[matched_target_idx] = True
    return targets, mask


def _prepare_split_ids(args: argparse.Namespace):
    split = load_experiment_split()
    train_ids = slice_node_ids(split.train_ids, args.max_train_nodes, seed=11)
    val_ids = slice_node_ids(split.val_ids, args.max_val_nodes, seed=17)
    test_pool_limit = args.max_test_pool_nodes
    if test_pool_limit is None:
        test_pool_limit = args.max_external_nodes
    test_pool_ids = slice_node_ids(split.test_pool_ids, test_pool_limit, seed=23)
    external_ids = slice_node_ids(split.external_ids, args.max_external_nodes, seed=29)
    return split, train_ids, val_ids, test_pool_ids, external_ids


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
        hard_negative_teacher_blend=float(args.hard_negative_teacher_blend),
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
        message_risk_strength=float(args.message_risk_strength),
        known_label_feature=bool(args.known_label_feature),
        known_label_feature_dim=int(args.known_label_feature_dim),
        target_context_fusion=str(args.target_context_fusion),
        target_context_input_dim=0,
        primary_multiclass_num_classes=int(args.primary_multiclass_num_classes),
        prototype_multiclass_num_classes=int(args.prototype_multiclass_num_classes),
        prototype_loss_weight=float(args.prototype_loss_weight),
        prototype_loss_weight_schedule=str(args.prototype_loss_weight_schedule),
        prototype_loss_min_weight=float(args.prototype_loss_min_weight),
        prototype_temperature=float(args.prototype_temperature),
        prototype_momentum=float(args.prototype_momentum),
        prototype_start_epoch=int(args.prototype_start_epoch),
        prototype_loss_ramp_epochs=int(args.prototype_loss_ramp_epochs),
        prototype_bucket_mode=str(args.prototype_bucket_mode),
        prototype_neighbor_blend=float(args.prototype_neighbor_blend),
        prototype_global_blend=float(args.prototype_global_blend),
        prototype_consistency_weight=float(args.prototype_consistency_weight),
        prototype_separation_weight=float(args.prototype_separation_weight),
        prototype_separation_margin=float(args.prototype_separation_margin),
        normal_bucket_align_weight=float(args.normal_bucket_align_weight),
        normal_bucket_shift_strength=float(args.normal_bucket_shift_strength),
        target_time_adapter_strength=float(args.target_time_adapter_strength),
        target_time_adapter_type=str(args.target_time_adapter_type),
        target_time_adapter_num_experts=int(args.target_time_adapter_num_experts),
        target_time_expert_entropy_floor=float(args.target_time_expert_entropy_floor),
        target_time_expert_entropy_weight=float(args.target_time_expert_entropy_weight),
        teacher_distill_weight=float(args.teacher_distill_weight),
        teacher_distill_loss=str(args.teacher_distill_loss),
        teacher_distill_start_epoch=int(args.teacher_distill_start_epoch),
        teacher_distill_ramp_epochs=int(args.teacher_distill_ramp_epochs),
        teacher_distill_agreement_floor=float(args.teacher_distill_agreement_floor),
        teacher_distill_rank_gap=float(args.teacher_distill_rank_gap),
        atm_gate_strength=float(args.atm_gate_strength),
        context_residual_scale=float(args.context_residual_scale),
        context_residual_clip=float(args.context_residual_clip),
        context_residual_budget=float(args.context_residual_budget),
        context_residual_budget_weight=float(args.context_residual_budget_weight),
        context_residual_budget_schedule=str(args.context_residual_budget_schedule),
        context_residual_budget_min_weight=float(args.context_residual_budget_min_weight),
        context_residual_budget_release_epochs=int(args.context_residual_budget_release_epochs),
        context_residual_budget_release_delay_epochs=int(args.context_residual_budget_release_delay_epochs),
        normal_bucket_adv_weight=float(args.normal_bucket_adv_weight),
        include_historical_background_negatives=bool(args.include_historical_background_negatives),
        historical_background_negative_ratio=float(args.historical_background_negative_ratio),
        historical_background_negative_warmup_epochs=int(args.historical_background_negative_warmup_epochs),
        historical_background_aux_only=bool(args.historical_background_aux_only),
        aux_multiclass_num_classes=int(args.aux_multiclass_num_classes),
        aux_multiclass_loss_weight=float(args.aux_multiclass_loss_weight),
        aux_inference_blend=float(args.aux_inference_blend),
        pseudo_contrastive_weight=float(args.pseudo_contrastive_weight),
        pseudo_contrastive_temperature=float(args.pseudo_contrastive_temperature),
        pseudo_contrastive_sample_size=int(args.pseudo_contrastive_sample_size),
        pseudo_contrastive_low_quantile=float(args.pseudo_contrastive_low_quantile),
        pseudo_contrastive_high_quantile=float(args.pseudo_contrastive_high_quantile),
        pseudo_contrastive_interval=int(args.pseudo_contrastive_interval),
    )


def _apply_dataset_adaptive_profile(
    *,
    args: argparse.Namespace,
    feature_groups: list[str],
    target_context_groups: list[str],
    graph_config: GraphModelConfig,
    num_relations: int,
) -> tuple[list[str], list[str], Path | None, str, Path | None, GraphModelConfig, dict[str, Any]]:
    profile = str(getattr(args, "dataset_adaptive_profile", "none") or "none").strip().lower()
    dataset_name = str(ACTIVE_DATASET_SPEC.name).strip().lower()
    feature_groups = _dedupe_preserve_order(list(feature_groups))
    target_context_groups = _dedupe_preserve_order(list(target_context_groups))
    target_context_prediction_dir = _coerce_optional_path(getattr(args, "target_context_prediction_dir", None))
    target_context_prediction_transform = str(getattr(args, "target_context_prediction_transform", "raw"))
    teacher_distill_prediction_dir = _coerce_optional_path(getattr(args, "teacher_distill_prediction_dir", None))
    diagnostics: dict[str, Any] = {
        "dataset_adaptive_profile": profile,
        "dataset_adaptive_num_relations": int(num_relations),
        "dataset_adaptive_branch": "none",
    }
    if profile not in {"relation_scarcity_auto", "relation_scarcity_bridge_v1"}:
        return (
            feature_groups,
            target_context_groups,
            target_context_prediction_dir,
            target_context_prediction_transform,
            teacher_distill_prediction_dir,
            graph_config,
            diagnostics,
        )

    if profile == "relation_scarcity_auto":
        if int(num_relations) <= 2:
            diagnostics["dataset_adaptive_branch"] = "low_relation_simplified"
            return (
                feature_groups,
                target_context_groups,
                target_context_prediction_dir,
                target_context_prediction_transform,
                teacher_distill_prediction_dir,
                replace(
                    graph_config,
                    neighbor_sampler="hybrid",
                    recent_window=min(int(graph_config.recent_window), 15),
                    target_context_fusion="none",
                    teacher_distill_weight=0.0,
                    target_time_adapter_strength=0.0,
                    target_time_expert_entropy_floor=0.0,
                    target_time_expert_entropy_weight=0.0,
                    prototype_multiclass_num_classes=0,
                    prototype_loss_weight=0.0,
                    prototype_loss_min_weight=0.0,
                    prototype_consistency_weight=0.0,
                    prototype_separation_weight=0.0,
                    normal_bucket_align_weight=0.0,
                    normal_bucket_adv_weight=0.0,
                ),
                diagnostics,
            )
        return (
            feature_groups,
            target_context_groups,
            target_context_prediction_dir,
            target_context_prediction_transform,
            teacher_distill_prediction_dir,
            replace(
                graph_config,
                target_time_adapter_type="affine",
                target_time_expert_entropy_floor=0.0,
                target_time_expert_entropy_weight=0.0,
            ),
            diagnostics,
        )

    xinye_teacher_dir = _existing_optional_path(
        "experiment/outputs/training/blends/plan53_hier2_oof_ctxadaptwin_logit_v1"
    )
    ellipticpp_teacher_dir = _existing_optional_path(
        "experiment/outputs/ellipticpp_transactions/training/blends/teacher_epplus_oof_calib_v3_clean"
    )
    if int(num_relations) <= 2:
        diagnostics["dataset_adaptive_branch"] = "low_relation_proto_bridge"
        feature_groups = _drop_groups(
            feature_groups,
            {"temporal_counterparty_concentration", "temporal_counterparty_concentration_recent"},
        )
        target_context_prediction_dir = None
        teacher_distill_prediction_dir = None
        teacher_distill_weight = 0.0
        if "ellipticpp" in dataset_name and ellipticpp_teacher_dir is not None:
            diagnostics["dataset_adaptive_branch"] = "low_relation_proto_teacher"
            target_context_prediction_dir = ellipticpp_teacher_dir
            target_context_prediction_transform = "logit"
            teacher_distill_prediction_dir = ellipticpp_teacher_dir
            teacher_distill_weight = max(float(graph_config.teacher_distill_weight), 0.05)
        return (
            feature_groups,
            target_context_groups,
            target_context_prediction_dir,
            target_context_prediction_transform,
            teacher_distill_prediction_dir,
            replace(
                graph_config,
                neighbor_sampler="consistency_recent",
                recent_window=max(int(graph_config.recent_window), 50),
                target_context_fusion="logit_residual",
                target_time_adapter_strength=0.0,
                target_time_expert_entropy_floor=0.0,
                target_time_expert_entropy_weight=0.0,
                prototype_multiclass_num_classes=max(int(graph_config.prototype_multiclass_num_classes), 2),
                prototype_loss_weight=max(float(graph_config.prototype_loss_weight), 0.05),
                prototype_loss_min_weight=max(float(graph_config.prototype_loss_min_weight), 0.01),
                prototype_consistency_weight=max(float(graph_config.prototype_consistency_weight), 0.05),
                prototype_separation_weight=max(float(graph_config.prototype_separation_weight), 0.05),
                normal_bucket_align_weight=max(float(graph_config.normal_bucket_align_weight), 0.05),
                normal_bucket_adv_weight=max(float(graph_config.normal_bucket_adv_weight), 0.05),
                teacher_distill_weight=float(teacher_distill_weight),
                teacher_distill_loss="rank" if teacher_distill_weight > 0.0 else str(graph_config.teacher_distill_loss),
                teacher_distill_start_epoch=2 if teacher_distill_weight > 0.0 else int(graph_config.teacher_distill_start_epoch),
                teacher_distill_ramp_epochs=6 if teacher_distill_weight > 0.0 else int(graph_config.teacher_distill_ramp_epochs),
                teacher_distill_agreement_floor=(
                    max(float(graph_config.teacher_distill_agreement_floor), 0.65)
                    if teacher_distill_weight > 0.0
                    else float(graph_config.teacher_distill_agreement_floor)
                ),
                teacher_distill_rank_gap=(
                    max(float(graph_config.teacher_distill_rank_gap), 0.2)
                    if teacher_distill_weight > 0.0
                    else float(graph_config.teacher_distill_rank_gap)
                ),
            ),
            diagnostics,
        )

    diagnostics["dataset_adaptive_branch"] = "rich_relation_teacher_bridge"
    if xinye_teacher_dir is not None:
        if target_context_prediction_dir is None:
            target_context_prediction_dir = xinye_teacher_dir
        if teacher_distill_prediction_dir is None:
            teacher_distill_prediction_dir = xinye_teacher_dir
        target_context_prediction_transform = "logit"
    return (
        feature_groups,
        target_context_groups,
        target_context_prediction_dir,
        target_context_prediction_transform,
        teacher_distill_prediction_dir,
        graph_config,
        diagnostics,
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
    current_val = summary_payload.get("val_auc_mean", summary_payload.get("phase1_val_auc_mean"))
    benchmark_val = benchmark.get("val_auc_mean", benchmark.get("phase1_val_auc_mean"))
    val_delta = float(current_val - benchmark_val)
    promoted = val_delta >= PROMOTION_DELTA_MIN
    decision: dict[str, Any] = {
        "benchmark_reference": _path_repr(benchmark_path),
        "val_delta_vs_m2": val_delta,
        "promoted": promoted,
        "rule": {"min_val_delta": PROMOTION_DELTA_MIN},
    }
    current_test = summary_payload.get("test_auc_mean")
    benchmark_test = benchmark.get("test_auc_mean")
    if current_test is not None and benchmark_test is not None:
        decision["test_delta_vs_m2"] = float(current_test - benchmark_test)
    decision["promoted"] = promoted
    return decision


def run_build_features(args: argparse.Namespace) -> None:
    phases = list(ACTIVE_DATASET_SPEC.default_artifacts) if args.phase == "both" else [args.phase]
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
        split, train_ids, val_ids, _test_pool_ids, external_ids = _prepare_split_ids(args)
        prep_pbar.update(1)
        if split.train_phase != split.val_phase:
            raise NotImplementedError("LightGBM training currently requires train/val to come from the same graph.")
        labels_by_phase = _load_labels_for_splits(split)
        feature_groups = resolve_feature_groups(
            args.model,
            list(args.extra_groups),
            feature_profile=_resolve_effective_feature_profile(args),
        )
        prep_pbar.update(1)
        phase1_store = FeatureStore(split.train_phase, feature_groups, outdir=args.feature_dir)
        phase2_store = (
            FeatureStore(str(split.external_phase), feature_groups, outdir=args.feature_dir)
            if external_ids.size and split.external_phase is not None
            else None
        )
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
                train_labels=labels_by_phase[split.train_phase][train_ids],
                val_ids=val_ids,
                val_labels=labels_by_phase[split.val_phase][val_ids],
            )
            val_prob = model.predict_proba(phase1_store, val_ids)
            val_metrics = compute_binary_classification_metrics(labels_by_phase[split.val_phase][val_ids], val_prob)
            external_prob = None
            external_metrics = None
            if external_ids.size and phase2_store is not None and split.external_phase is not None:
                external_prob = model.predict_proba(phase2_store, external_ids)
                external_metrics = compute_binary_classification_metrics(
                    labels_by_phase[str(split.external_phase)][external_ids],
                    external_prob,
                )
            model.save(seed_dir, feature_names=phase1_store.feature_names)
            save_prediction_npz(
                seed_dir / "phase1_val_predictions.npz",
                val_ids,
                labels_by_phase[split.val_phase][val_ids],
                val_prob,
            )
            if external_prob is not None and external_metrics is not None and split.external_phase is not None:
                save_prediction_npz(
                    seed_dir / "phase2_external_predictions.npz",
                    external_ids,
                    labels_by_phase[str(split.external_phase)][external_ids],
                    external_prob,
                )
            val_predictions.append(val_prob)
            if external_prob is not None:
                external_predictions.append(external_prob)
            metrics.append(
                {
                    "seed": seed,
                    "val_auc": val_metrics["auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_ap": val_metrics["ap"],
                    "external_auc": None if external_metrics is None else external_metrics["auc"],
                    "external_pr_auc": None if external_metrics is None else external_metrics["pr_auc"],
                    "external_ap": None if external_metrics is None else external_metrics["ap"],
                    "best_iteration": fit_metrics["best_iteration"],
                }
            )
            seed_pbar.set_postfix(
                val_auc=f"{val_metrics['auc']:.4f}",
                val_ap=f"{val_metrics['ap']:.4f}",
                external_auc=_format_metric(None if external_metrics is None else external_metrics["auc"]),
                refresh=False,
            )
            tqdm.write(
                f"[{args.model}] seed={seed} "
                f"phase1_val_auc={val_metrics['auc']:.6f} "
                f"phase1_val_pr_auc={val_metrics['pr_auc']:.6f} "
                f"phase1_val_ap={val_metrics['ap']:.6f} "
                f"phase2_external_auc={_format_metric(None if external_metrics is None else external_metrics['auc'])} "
                f"phase2_external_pr_auc={_format_metric(None if external_metrics is None else external_metrics['pr_auc'])} "
                f"phase2_external_ap={_format_metric(None if external_metrics is None else external_metrics['ap'])}"
            )

    val_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase1_val",
        node_ids=val_ids,
        labels=labels_by_phase[split.val_phase][val_ids],
        predictions=val_predictions,
    )
    external_avg_path = (
        _save_average_predictions(
            run_dir=run_dir,
            split_name="phase2_external",
            node_ids=external_ids,
            labels=labels_by_phase[str(split.external_phase)][external_ids],
            predictions=external_predictions,
        )
        if external_predictions and split.external_phase is not None
        else None
    )

    summary = {
        "model_name": args.model,
        "run_name": args.run_name,
        "feature_groups": feature_groups,
        "seeds": list(args.seeds),
        "split_style": split.split_style,
        "train_phase": split.train_phase,
        "val_phase": split.val_phase,
        "external_phase": split.external_phase,
        "train_size": int(train_ids.size),
        "val_size": int(val_ids.size),
        "external_size": int(external_ids.size),
        "val_auc_mean": _metric_mean(metrics, "val_auc"),
        "val_auc_std": _metric_std(metrics, "val_auc"),
        "val_pr_auc_mean": _metric_mean(metrics, "val_pr_auc"),
        "val_pr_auc_std": _metric_std(metrics, "val_pr_auc"),
        "val_ap_mean": _metric_mean(metrics, "val_ap"),
        "val_ap_std": _metric_std(metrics, "val_ap"),
        "external_auc_mean": _metric_mean(metrics, "external_auc"),
        "external_auc_std": _metric_std(metrics, "external_auc"),
        "external_pr_auc_mean": _metric_mean(metrics, "external_pr_auc"),
        "external_pr_auc_std": _metric_std(metrics, "external_pr_auc"),
        "external_ap_mean": _metric_mean(metrics, "external_ap"),
        "external_ap_std": _metric_std(metrics, "external_ap"),
        "phase1_train_size": int(train_ids.size),
        "phase1_val_size": int(val_ids.size),
        "phase2_external_size": int(external_ids.size),
        "phase1_val_auc_mean": _metric_mean(metrics, "val_auc"),
        "phase1_val_auc_std": _metric_std(metrics, "val_auc"),
        "phase1_val_pr_auc_mean": _metric_mean(metrics, "val_pr_auc"),
        "phase1_val_pr_auc_std": _metric_std(metrics, "val_pr_auc"),
        "phase1_val_ap_mean": _metric_mean(metrics, "val_ap"),
        "phase1_val_ap_std": _metric_std(metrics, "val_ap"),
        "phase2_external_auc_mean": _metric_mean(metrics, "external_auc"),
        "phase2_external_auc_std": _metric_std(metrics, "external_auc"),
        "phase2_external_pr_auc_mean": _metric_mean(metrics, "external_pr_auc"),
        "phase2_external_pr_auc_std": _metric_std(metrics, "external_pr_auc"),
        "phase2_external_ap_mean": _metric_mean(metrics, "external_ap"),
        "phase2_external_ap_std": _metric_std(metrics, "external_ap"),
        "seed_metrics": metrics,
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "phase2_external_avg_predictions": None if external_avg_path is None else _path_repr(external_avg_path),
        "val_avg_predictions": _path_repr(val_avg_path),
        "external_avg_predictions": None if external_avg_path is None else _path_repr(external_avg_path),
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
    effective_feature_profile = _resolve_effective_feature_profile(args)
    with tqdm(
        total=3,
        desc=f"prepare:{args.model}",
        unit="step",
        dynamic_ncols=True,
    ) as prep_pbar:
        split, split_train_ids, val_ids, test_pool_ids, external_ids = _prepare_split_ids(args)
        if split.train_phase != split.val_phase:
            raise NotImplementedError("Graph training currently requires train/val to come from the same graph.")
        graph_config = _build_graph_model_config(args)
        phase1_graph_meta = load_graph_cache(split.train_phase, outdir=args.feature_dir)
        feature_groups = resolve_feature_groups(
            args.model,
            list(args.extra_groups),
            feature_profile=effective_feature_profile,
        )
        target_context_groups = list(args.target_context_extra_groups)
        target_context_prediction_dir = _coerce_optional_path(args.target_context_prediction_dir)
        target_context_prediction_transform = str(args.target_context_prediction_transform)
        teacher_distill_prediction_dir = _coerce_optional_path(args.teacher_distill_prediction_dir)
        (
            feature_groups,
            target_context_groups,
            target_context_prediction_dir,
            target_context_prediction_transform,
            teacher_distill_prediction_dir,
            graph_config,
            dataset_adaptive_diagnostics,
        ) = _apply_dataset_adaptive_profile(
            args=args,
            feature_groups=feature_groups,
            target_context_groups=target_context_groups,
            graph_config=graph_config,
            num_relations=int(phase1_graph_meta.num_relations),
        )
        train_ids = np.asarray(split_train_ids, dtype=np.int32)
        eval_phase = str(split.external_phase or split.val_phase)
        label_artifacts = build_graph_label_artifacts(
            feature_dir=args.feature_dir,
            split_train_ids=train_ids,
            threshold_day=int(split.threshold_day),
            known_label_feature=graph_config.known_label_feature,
            include_historical_background_negatives=graph_config.include_historical_background_negatives,
            train_phase=split.train_phase,
            eval_phase=eval_phase,
        )
        historical_background_ids = np.asarray(
            label_artifacts["phase1_historical_background_ids"],
            dtype=np.int32,
        )
        prep_pbar.update(1)
        feature_normalizer_state = None
        if graph_config.feature_norm == "hybrid":
            feature_normalizer_state = build_hybrid_feature_normalizer(
                phase=split.train_phase,
                selected_groups=feature_groups,
                train_ids=train_ids,
                outdir=args.feature_dir,
            )
        target_context_normalizer_state = None
        if graph_config.feature_norm == "hybrid" and target_context_groups:
            target_context_normalizer_state = build_hybrid_feature_normalizer(
                phase=split.train_phase,
                selected_groups=target_context_groups,
                train_ids=train_ids,
                outdir=args.feature_dir,
            )
        prep_pbar.update(1)
        phase1_context, phase2_context = make_graph_contexts(
            feature_dir=args.feature_dir,
            model_name=args.model,
            train_phase=split.train_phase,
            eval_phase=eval_phase,
            selected_groups=feature_groups,
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
        if target_context_prediction_dir is not None:
            (
                phase1_prediction_features,
                phase2_prediction_features,
                target_context_prediction_feature_names,
            ) = _load_target_context_prediction_features(
                prediction_dir=target_context_prediction_dir,
                prediction_transform=target_context_prediction_transform,
                train_ids=train_ids,
                val_ids=val_ids,
                external_ids=external_ids,
                phase1_num_nodes=phase1_context.labels.shape[0],
                phase2_num_nodes=phase2_context.labels.shape[0],
            )
            phase1_context = replace(
                phase1_context,
                target_aux_features=phase1_prediction_features,
                target_aux_feature_names=tuple(target_context_prediction_feature_names),
            )
            phase2_context = replace(
                phase2_context,
                target_aux_features=phase2_prediction_features,
                target_aux_feature_names=tuple(target_context_prediction_feature_names),
            )
        if teacher_distill_prediction_dir is not None and float(graph_config.teacher_distill_weight) > 0.0:
            phase1_distill_targets, phase1_distill_mask = _load_teacher_distill_targets(
                prediction_dir=teacher_distill_prediction_dir,
                train_ids=train_ids,
                phase1_num_nodes=phase1_context.labels.shape[0],
            )
            phase1_context = replace(
                phase1_context,
                distill_targets=phase1_distill_targets,
                distill_target_mask=phase1_distill_mask,
            )
        prep_pbar.update(1)
    print(
        "[train_graph] "
        f"dataset={ACTIVE_DATASET_SPEC.name} "
        f"feature_profile={effective_feature_profile} "
        "thesis_mainline=False "
        f"adaptive_profile={dataset_adaptive_diagnostics['dataset_adaptive_profile']} "
        f"adaptive_branch={dataset_adaptive_diagnostics['dataset_adaptive_branch']} "
        f"num_relations={dataset_adaptive_diagnostics['dataset_adaptive_num_relations']}"
    )
    run_dir = _model_run_dir(args.outdir, args.model, args.run_name)
    label_feature_dim = graph_config.known_label_feature_dim if graph_config.known_label_feature else 0
    input_dim = phase1_context.feature_store.input_dim + int(label_feature_dim)
    target_context_input_dim = 0
    if phase1_context.target_context_store is not None:
        target_context_input_dim += int(phase1_context.target_context_store.input_dim)
    if phase1_context.target_aux_features is not None:
        target_context_input_dim += (
            int(phase1_context.target_aux_features.shape[1])
            if phase1_context.target_aux_features.ndim == 2
            else 1
        )
    if target_context_input_dim <= 0:
        target_context_input_dim = int(input_dim)
    graph_config = replace(graph_config, target_context_input_dim=int(target_context_input_dim))
    num_relations = phase1_context.graph_cache.num_relations
    global_max_day = max(phase1_context.graph_cache.max_day, phase2_context.graph_cache.max_day)
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
            train_metrics = compute_binary_classification_metrics(train_labels, train_prob)
            val_metrics = compute_binary_classification_metrics(val_labels, val_prob)
            test_pool_prob = None
            test_metrics = None
            if test_pool_ids.size:
                test_pool_prob = experiment.predict_proba(
                    phase1_context,
                    test_pool_ids,
                    batch_seed=seed + 1500,
                    progress_desc=f"{args.model}:seed{seed}:test_pool",
                )
                test_metrics = _maybe_compute_binary_metrics(test_pool_labels, test_pool_prob)
            external_prob = None
            external_metrics = None
            if external_ids.size:
                external_prob = experiment.predict_proba(
                    phase2_context,
                    external_ids,
                    batch_seed=seed + 2000,
                    progress_desc=f"{args.model}:seed{seed}:phase2_external",
                )
                external_metrics = _maybe_compute_binary_metrics(external_labels, external_prob)
            experiment.save(seed_dir)
            save_prediction_npz(
                seed_dir / "phase1_train_predictions.npz",
                train_ids,
                train_labels,
                train_prob,
            )
            save_prediction_npz(
                seed_dir / "phase1_val_predictions.npz",
                val_ids,
                val_labels,
                val_prob,
            )
            if test_pool_prob is not None:
                save_prediction_npz(
                    seed_dir / "test_pool_predictions.npz",
                    test_pool_ids,
                    test_pool_labels,
                    test_pool_prob,
                )
            if external_prob is not None and external_metrics is not None:
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
                f"train_pr_auc={train_metrics['pr_auc']:.6f} "
                f"train_ap={train_metrics['ap']:.6f} "
                f"val_auc={val_metrics['auc']:.6f} "
                f"val_pr_auc={val_metrics['pr_auc']:.6f} "
                f"val_ap={val_metrics['ap']:.6f} "
                f"test_auc={_format_metric(None if test_metrics is None else test_metrics['auc'])} "
                f"test_pr_auc={_format_metric(None if test_metrics is None else test_metrics['pr_auc'])} "
                f"test_ap={_format_metric(None if test_metrics is None else test_metrics['ap'])} "
                f"legacy_external_auc={_format_metric(None if external_metrics is None else external_metrics['auc'])}"
            )

    train_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase1_train",
        node_ids=train_ids,
        labels=train_labels,
        predictions=train_predictions,
    )
    val_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase1_val",
        node_ids=val_ids,
        labels=val_labels,
        predictions=val_predictions,
    )
    test_avg_path = (
        _save_average_predictions(
            run_dir=run_dir,
            split_name="test_pool",
            node_ids=test_pool_ids,
            labels=test_pool_labels,
            predictions=test_pool_predictions,
        )
        if test_pool_predictions
        else None
    )
    external_avg_path = (
        _save_average_predictions(
            run_dir=run_dir,
            split_name="phase2_external",
            node_ids=external_ids,
            labels=external_labels,
            predictions=external_predictions,
        )
        if external_predictions
        else None
    )
    summary = {
        "model_name": args.model,
        "run_name": args.run_name,
        "feature_profile": effective_feature_profile,
        "official_eval_contract": ["train", "val", "test_pool"],
        "thesis_mainline_enabled": thesis_metadata["thesis_mainline_enabled"],
        "main_innovation": thesis_metadata["main_innovation"],
        "aux_regularizer": thesis_metadata["aux_regularizer"],
        "dataset_adaptive_profile": dataset_adaptive_diagnostics["dataset_adaptive_profile"],
        "dataset_adaptive_branch": dataset_adaptive_diagnostics["dataset_adaptive_branch"],
        "dataset_adaptive_num_relations": dataset_adaptive_diagnostics["dataset_adaptive_num_relations"],
        "feature_groups": phase1_context.feature_store.selected_groups,
        "target_context_feature_groups": (
            []
            if phase1_context.target_context_store is None
            else phase1_context.target_context_store.selected_groups
        ),
        "target_context_prediction_dir": (
            None
            if target_context_prediction_dir is None
            else _path_repr(target_context_prediction_dir)
        ),
        "target_context_prediction_transform": str(target_context_prediction_transform),
        "target_context_prediction_feature_names": list(phase1_context.target_aux_feature_names or ()),
        "teacher_distill_prediction_dir": (
            None
            if teacher_distill_prediction_dir is None
            else _path_repr(teacher_distill_prediction_dir)
        ),
        "teacher_distill_weight": float(graph_config.teacher_distill_weight),
        "teacher_distill_loss": str(graph_config.teacher_distill_loss),
        "teacher_distill_start_epoch": int(graph_config.teacher_distill_start_epoch),
        "teacher_distill_ramp_epochs": int(graph_config.teacher_distill_ramp_epochs),
        "teacher_distill_agreement_floor": float(graph_config.teacher_distill_agreement_floor),
        "teacher_distill_rank_gap": float(graph_config.teacher_distill_rank_gap),
        "seeds": list(args.seeds),
        "split_style": split.split_style,
        "train_phase": split.train_phase,
        "val_phase": split.val_phase,
        "external_phase": split.external_phase,
        "test_pool_phase": split.train_phase,
        "train_size": int(np.asarray(split_train_ids, dtype=np.int32).size),
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
        "legacy_external_prediction_path": None if external_avg_path is None else _path_repr(external_avg_path),
        "phase1_train_size": int(np.asarray(split_train_ids, dtype=np.int32).size),
        "phase1_split_train_size": int(np.asarray(split_train_ids, dtype=np.int32).size),
        "phase1_historical_background_train_size": int(historical_background_ids.size),
        "phase1_val_size": int(val_ids.size),
        "phase1_train_auc_mean": _metric_mean(metrics, "train_auc"),
        "phase1_train_auc_std": _metric_std(metrics, "train_auc"),
        "phase1_train_pr_auc_mean": _metric_mean(metrics, "train_pr_auc"),
        "phase1_train_pr_auc_std": _metric_std(metrics, "train_pr_auc"),
        "phase1_train_ap_mean": _metric_mean(metrics, "train_ap"),
        "phase1_train_ap_std": _metric_std(metrics, "train_ap"),
        "phase2_external_size": int(external_ids.size),
        "phase1_val_auc_mean": _metric_mean(metrics, "val_auc"),
        "phase1_val_auc_std": _metric_std(metrics, "val_auc"),
        "phase1_val_pr_auc_mean": _metric_mean(metrics, "val_pr_auc"),
        "phase1_val_pr_auc_std": _metric_std(metrics, "val_pr_auc"),
        "phase1_val_ap_mean": _metric_mean(metrics, "val_ap"),
        "phase1_val_ap_std": _metric_std(metrics, "val_ap"),
        "test_pool_auc_mean": _metric_mean(metrics, "test_auc"),
        "test_pool_auc_std": _metric_std(metrics, "test_auc"),
        "test_pool_pr_auc_mean": _metric_mean(metrics, "test_pr_auc"),
        "test_pool_pr_auc_std": _metric_std(metrics, "test_pr_auc"),
        "test_pool_ap_mean": _metric_mean(metrics, "test_ap"),
        "test_pool_ap_std": _metric_std(metrics, "test_ap"),
        "phase2_external_auc_mean": _metric_mean(metrics, "external_auc"),
        "phase2_external_auc_std": _metric_std(metrics, "external_auc"),
        "phase2_external_pr_auc_mean": _metric_mean(metrics, "external_pr_auc"),
        "phase2_external_pr_auc_std": _metric_std(metrics, "external_pr_auc"),
        "phase2_external_ap_mean": _metric_mean(metrics, "external_ap"),
        "phase2_external_ap_std": _metric_std(metrics, "external_ap"),
        "seed_metrics": metrics,
        "phase1_train_avg_predictions": _path_repr(train_avg_path),
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "test_pool_avg_predictions": None if test_avg_path is None else _path_repr(test_avg_path),
        "phase2_external_avg_predictions": None if external_avg_path is None else _path_repr(external_avg_path),
        "train_avg_predictions": _path_repr(train_avg_path),
        "val_avg_predictions": _path_repr(val_avg_path),
        "test_avg_predictions": None if test_avg_path is None else _path_repr(test_avg_path),
        "external_avg_predictions": None if external_avg_path is None else _path_repr(external_avg_path),
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
        if args.model == "m7_utpm":
            raise ValueError(
                "`run_training.py` no longer accepts `m7_utpm`. "
                "Use `experiment/training/run_thesis_mainline.py` for the official unified backbone "
                "or `experiment/training/run_thesis_hybrid_suite.py` for the official final thesis suite."
            )
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
