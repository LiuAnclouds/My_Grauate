from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from experiment.training.common import ExperimentSplit
from experiment.training.features import build_hybrid_feature_normalizer, resolve_feature_groups
from experiment.training.gnn_models import GraphModelConfig, GraphPhaseContext
from experiment.training.graph_runtime import build_graph_label_artifacts, make_graph_contexts
from experiment.training.prediction_signal_utils import (
    load_target_context_prediction_features,
    load_teacher_distill_targets,
)
from experiment.training.thesis_contract import OFFICIAL_TARGET_CONTEXT_GROUPS


@dataclass(frozen=True)
class ThesisPreparedRuntime:
    graph_config: GraphModelConfig
    feature_groups: list[str]
    feature_normalizer_state: object | None
    target_context_feature_groups: list[str]
    target_context_normalizer_state: object | None
    phase1_context: GraphPhaseContext
    phase2_context: GraphPhaseContext
    input_dim: int
    target_context_input_dim: int
    num_relations: int
    global_max_day: int


def prepare_thesis_runtime(
    *,
    feature_dir: Path,
    model_name: str,
    split: ExperimentSplit,
    train_ids: np.ndarray,
    graph_config: GraphModelConfig,
    feature_profile: str = "utpm_unified",
    target_context_groups: list[str] | tuple[str, ...] | None = None,
    target_context_prediction_dir: Path | None = None,
    target_context_prediction_transform: str = "raw",
    teacher_distill_prediction_dir: Path | None = None,
) -> ThesisPreparedRuntime:
    feature_groups = resolve_feature_groups(model_name, feature_profile=feature_profile)
    resolved_target_context_groups = list(
        OFFICIAL_TARGET_CONTEXT_GROUPS if target_context_groups is None else target_context_groups
    )
    eval_phase = str(split.external_phase or split.val_phase)
    label_artifacts = build_graph_label_artifacts(
        feature_dir=feature_dir,
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

    feature_normalizer_state = None
    if graph_config.feature_norm == "hybrid":
        feature_normalizer_state = build_hybrid_feature_normalizer(
            phase=split.train_phase,
            selected_groups=feature_groups,
            train_ids=train_ids,
            outdir=feature_dir,
        )
    target_context_normalizer_state = None
    if graph_config.feature_norm == "hybrid" and resolved_target_context_groups:
        target_context_normalizer_state = build_hybrid_feature_normalizer(
            phase=split.train_phase,
            selected_groups=resolved_target_context_groups,
            train_ids=train_ids,
            outdir=feature_dir,
        )

    phase1_context, phase2_context = make_graph_contexts(
        feature_dir=feature_dir,
        model_name=model_name,
        train_phase=split.train_phase,
        eval_phase=eval_phase,
        selected_groups=feature_groups,
        feature_normalizer_state=feature_normalizer_state,
        target_context_groups=resolved_target_context_groups,
        target_context_normalizer_state=target_context_normalizer_state,
        phase1_known_label_codes=label_artifacts["phase1_known_label_codes"],
        phase2_known_label_codes=label_artifacts["phase2_known_label_codes"],
        phase1_reference_day=int(split.threshold_day),
        phase2_reference_day=None,
        phase1_historical_background_ids=historical_background_ids,
        build_sampling_profile=graph_config.neighbor_sampler
        in {"consistency_recent", "risk_consistency_recent"},
    )
    if target_context_prediction_dir is not None:
        (
            phase1_prediction_features,
            phase2_prediction_features,
            target_context_prediction_feature_names,
        ) = load_target_context_prediction_features(
            prediction_dir=target_context_prediction_dir,
            prediction_transform=target_context_prediction_transform,
            train_ids=train_ids,
            val_ids=np.asarray(split.val_ids, dtype=np.int32),
            external_ids=np.asarray(split.external_ids, dtype=np.int32),
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
        phase1_distill_targets, phase1_distill_mask = load_teacher_distill_targets(
            prediction_dir=teacher_distill_prediction_dir,
            train_ids=train_ids,
            phase1_num_nodes=phase1_context.labels.shape[0],
        )
        phase1_context = replace(
            phase1_context,
            distill_targets=phase1_distill_targets,
            distill_target_mask=phase1_distill_mask,
        )
    label_feature_dim = graph_config.known_label_feature_dim if graph_config.known_label_feature else 0
    input_dim = int(phase1_context.feature_store.input_dim) + int(label_feature_dim)
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

    return ThesisPreparedRuntime(
        graph_config=graph_config,
        feature_groups=feature_groups,
        feature_normalizer_state=feature_normalizer_state,
        target_context_feature_groups=resolved_target_context_groups,
        target_context_normalizer_state=target_context_normalizer_state,
        phase1_context=phase1_context,
        phase2_context=phase2_context,
        input_dim=input_dim,
        target_context_input_dim=int(target_context_input_dim),
        num_relations=int(phase1_context.graph_cache.num_relations),
        global_max_day=max(
            int(phase1_context.graph_cache.max_day),
            int(phase2_context.graph_cache.max_day),
        ),
    )
