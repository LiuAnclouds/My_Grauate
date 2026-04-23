from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from experiment.features.features import build_hybrid_feature_normalizer, resolve_feature_groups
from experiment.models.engine import GraphModelConfig, GraphPhaseContext
from experiment.models.graph import build_contexts, build_label_artifacts
from experiment.models.spec import OFFICIAL_TARGET_CONTEXT_GROUPS
from experiment.utils.common import ExperimentSplit


@dataclass(frozen=True)
class RuntimeBundle:
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


def build_runtime(
    *,
    feature_dir: Path,
    model_name: str,
    split: ExperimentSplit,
    train_ids: np.ndarray,
    graph_config: GraphModelConfig,
    feature_profile: str = "utpm_unified",
    target_context_groups: list[str] | tuple[str, ...] | None = None,
) -> RuntimeBundle:
    feature_groups = resolve_feature_groups(model_name, feature_profile=feature_profile)
    resolved_target_context_groups = list(
        OFFICIAL_TARGET_CONTEXT_GROUPS if target_context_groups is None else target_context_groups
    )
    eval_phase = str(split.external_phase or split.val_phase)
    label_artifacts = build_label_artifacts(
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

    phase1_context, phase2_context = build_contexts(
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
    label_feature_dim = graph_config.known_label_feature_dim if graph_config.known_label_feature else 0
    input_dim = int(phase1_context.feature_store.input_dim) + int(label_feature_dim)
    target_context_input_dim = 0
    if phase1_context.target_context_store is not None:
        target_context_input_dim += int(phase1_context.target_context_store.input_dim)
    if target_context_input_dim <= 0:
        target_context_input_dim = int(input_dim)
    graph_config = replace(graph_config, target_context_input_dim=int(target_context_input_dim))

    return RuntimeBundle(
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
