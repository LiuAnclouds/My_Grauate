from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiment.datasets.registry import get_active_dataset_spec
from experiment.training.core.engine import (
    GraphPhaseContext,
    RelationGraphSAGEExperiment,
    TemporalRelationGATExperiment,
    TemporalRelationGraphSAGEExperiment,
)
from experiment.training.core.spec import DYRIFT_GNN_MODEL
from experiment.training.data.features import (
    FeatureStore,
    HybridFeatureNormalizerState,
    load_graph_cache,
    resolve_feature_groups,
)
from experiment.training.modules.trainer import DyRIFTTrainer
from experiment.training.utils.common import load_phase_arrays
from experiment.training.utils.sampling import load_or_build_raw_consistency_profile


GRAPH_EXPERIMENTS = {
    "m4_graphsage": RelationGraphSAGEExperiment,
    "m5_temporal_graphsage": TemporalRelationGraphSAGEExperiment,
    "m6_temporal_gat": TemporalRelationGATExperiment,
    "m7_utpm": TemporalRelationGraphSAGEExperiment,
    DYRIFT_GNN_MODEL: DyRIFTTrainer,
}


def get_experiment_cls(model_name: str):
    try:
        return GRAPH_EXPERIMENTS[model_name]
    except KeyError as exc:
        raise KeyError(f"Unsupported graph model: {model_name}") from exc


def build_label_artifacts(
    *,
    feature_dir: Path,
    split_train_ids: np.ndarray,
    threshold_day: int,
    known_label_feature: bool,
    include_historical_background_negatives: bool,
    train_phase: str = "phase1",
    eval_phase: str = "phase2",
) -> dict[str, Any]:
    dataset_spec = get_active_dataset_spec()
    background_labels = tuple(int(label) for label in dataset_spec.background_labels)
    phase1_y: np.ndarray | None = None
    phase2_y: np.ndarray | None = None
    historical_background_ids = np.empty(0, dtype=np.int32)

    if known_label_feature or include_historical_background_negatives:
        phase1_y = np.asarray(load_phase_arrays(train_phase, keys=("y",))["y"], dtype=np.int8)
    if known_label_feature:
        phase2_y = np.asarray(load_phase_arrays(eval_phase, keys=("y",))["y"], dtype=np.int8)
    if include_historical_background_negatives:
        phase1_graph = load_graph_cache(train_phase, outdir=feature_dir)
        first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)
        background_mask = (first_active <= int(threshold_day)) & np.isin(phase1_y, background_labels)
        historical_background_ids = np.flatnonzero(background_mask).astype(np.int32, copy=False)

    phase1_known_label_codes: np.ndarray | None = None
    phase2_known_label_codes: np.ndarray | None = None
    if known_label_feature:
        train_ids = np.asarray(split_train_ids, dtype=np.int32)
        phase1_known_label_codes = np.full(phase1_y.shape[0], 4, dtype=np.int8)
        phase2_known_label_codes = np.full(phase2_y.shape[0], 4, dtype=np.int8)
        for offset, label in enumerate(background_labels[:2], start=2):
            phase1_known_label_codes[phase1_y == label] = offset
            phase2_known_label_codes[phase2_y == label] = offset
        phase1_known_label_codes[train_ids] = phase1_y[train_ids]

    return {
        "phase1_known_label_codes": phase1_known_label_codes,
        "phase2_known_label_codes": phase2_known_label_codes,
        "phase1_historical_background_ids": historical_background_ids,
    }


def build_contexts(
    *,
    feature_dir: Path,
    model_name: str,
    train_phase: str = "phase1",
    eval_phase: str = "phase2",
    selected_groups: list[str] | None = None,
    extra_groups: list[str] | None = None,
    feature_profile: str = "legacy",
    feature_normalizer_state=None,
    target_context_groups: list[str] | None = None,
    target_context_normalizer_state=None,
    phase1_known_label_codes: np.ndarray | None = None,
    phase2_known_label_codes: np.ndarray | None = None,
    phase1_reference_day: int | None = None,
    phase2_reference_day: int | None = None,
    phase1_historical_background_ids: np.ndarray | None = None,
    build_sampling_profile: bool = False,
) -> tuple[GraphPhaseContext, GraphPhaseContext]:
    if isinstance(feature_normalizer_state, dict):
        feature_normalizer_state = HybridFeatureNormalizerState.from_dict(feature_normalizer_state)
    if isinstance(target_context_normalizer_state, dict):
        target_context_normalizer_state = HybridFeatureNormalizerState.from_dict(
            target_context_normalizer_state
        )
    feature_groups = (
        list(selected_groups)
        if selected_groups is not None
        else resolve_feature_groups(model_name, extra_groups, feature_profile=feature_profile)
    )
    target_context_groups = list(target_context_groups or [])
    phase1_store = FeatureStore(
        train_phase,
        feature_groups,
        outdir=feature_dir,
        normalizer_state=feature_normalizer_state,
    )
    phase2_store = FeatureStore(
        eval_phase,
        feature_groups,
        outdir=feature_dir,
        normalizer_state=feature_normalizer_state,
    )
    phase1_target_context_store = (
        FeatureStore(
            train_phase,
            target_context_groups,
            outdir=feature_dir,
            normalizer_state=target_context_normalizer_state,
        )
        if target_context_groups
        else None
    )
    phase2_target_context_store = (
        FeatureStore(
            eval_phase,
            target_context_groups,
            outdir=feature_dir,
            normalizer_state=target_context_normalizer_state,
        )
        if target_context_groups
        else None
    )
    phase1_graph = load_graph_cache(train_phase, outdir=feature_dir)
    phase2_graph = load_graph_cache(eval_phase, outdir=feature_dir)
    phase1_y = np.asarray(load_phase_arrays(train_phase, keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays(eval_phase, keys=("y",))["y"], dtype=np.int8)
    phase1_sampling_profile = None
    phase2_sampling_profile = None
    if build_sampling_profile:
        phase1_sampling_profile = load_or_build_raw_consistency_profile(phase1_store)
        phase2_sampling_profile = load_or_build_raw_consistency_profile(phase2_store)
    return (
        GraphPhaseContext(
            phase=train_phase,
            feature_store=phase1_store,
            target_context_store=phase1_target_context_store,
            graph_cache=phase1_graph,
            labels=phase1_y,
            known_label_codes=phase1_known_label_codes,
            reference_day=phase1_reference_day,
            historical_background_ids=phase1_historical_background_ids,
            sampling_profile=phase1_sampling_profile,
        ),
        GraphPhaseContext(
            phase=eval_phase,
            feature_store=phase2_store,
            target_context_store=phase2_target_context_store,
            graph_cache=phase2_graph,
            labels=phase2_y,
            known_label_codes=phase2_known_label_codes,
            reference_day=phase2_reference_day,
            historical_background_ids=None,
            sampling_profile=phase2_sampling_profile,
        ),
    )
