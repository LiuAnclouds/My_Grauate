from __future__ import annotations

OFFICIAL_BASELINE_MODEL = "m5_temporal_graphsage"
OFFICIAL_BASELINE_PRESET = "unified_baseline"

# Legacy stable GraphSAGE thesis backbone kept for shared-module ablations.
OFFICIAL_BACKBONE_MODEL = "m7_utpm"
OFFICIAL_BACKBONE_PRESET = "utpm_temporal_shift_v4"
OFFICIAL_BACKBONE_FEATURE_PROFILE = "utpm_unified"
DYRIFT_GNN_MODEL = "dyrift_gnn"
TRANSFORMER_BACKBONE_MODEL = DYRIFT_GNN_MODEL
TRANSFORMER_BACKBONE_PRESET = "dyrift_tgat_base_v1"
TRANSFORMER_BACKBONE_DEPLOY_PRESET = "dyrift_tgat_deploy_v1"
LEGACY_TRANSFORMER_BACKBONE_PRESET = "dyrift_trgt_base_v1"
LEGACY_TRANSFORMER_BACKBONE_DEPLOY_PRESET = "dyrift_trgt_deploy_v1"
TGAT_BACKBONE_DISPLAY_NAME = "Temporal Graph Attention Network"
TGAT_BACKBONE_SHORT_NAME = "TGAT"
TRGT_BACKBONE_DISPLAY_NAME = TGAT_BACKBONE_DISPLAY_NAME
TRGT_BACKBONE_SHORT_NAME = TGAT_BACKBONE_SHORT_NAME
DYRIFT_MODEL_DISPLAY_NAME = "Dynamic Risk-Informed Temporal Graph Attention Network"
DYRIFT_MODEL_SHORT_NAME = "DyRIFT-TGAT"
OFFICIAL_TARGET_CONTEXT_GROUPS = (
    "graph_time_detrend",
    "neighbor_similarity",
    "activation_early",
)

OFFICIAL_DATASETS = (
    "xinye_dgraph",
    "elliptic_transactions",
    "ellipticpp_transactions",
)

OFFICIAL_MAINLINE_BATCH_SIZE = 512
OFFICIAL_MAINLINE_HIDDEN_DIM = 128
OFFICIAL_MAINLINE_REL_DIM = 32
OFFICIAL_MAINLINE_FANOUTS = (15, 10)

OFFICIAL_TRAIN_EPOCHS = 30
OFFICIAL_TRAIN_SEEDS = (42,)

# Primary thesis result: deployable pure DyRIFT-TGAT with dataset-local tuning.
OFFICIAL_FULL_EXPERIMENT_NAME = "full_dyrift_gnn"
