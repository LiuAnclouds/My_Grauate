from __future__ import annotations

from dataclasses import replace
from typing import Any

from dyrift.models.engine import GraphModelConfig
from dyrift.models.spec import (
    DYRIFT_GNN_MODEL,
    LEGACY_TRANSFORMER_BACKBONE_DEPLOY_PRESET,
    LEGACY_TRANSFORMER_BACKBONE_PRESET,
    OFFICIAL_BACKBONE_PRESET,
    TRANSFORMER_BACKBONE_MODEL,
    TRANSFORMER_BACKBONE_PRESET,
    TRANSFORMER_BACKBONE_DEPLOY_PRESET,
)


_BASE_GRAPH_CONFIG = GraphModelConfig(
    learning_rate=3e-4,
    weight_decay=1e-4,
    dropout=0.2,
    feature_norm="hybrid",
    norm="layer",
    residual=True,
    ffn=True,
    jk="sum",
    edge_encoder="gated",
    subgraph_head="meanmax",
    grad_clip=1.0,
    scheduler="plateau",
    min_early_stop_epoch=0,
    train_negative_ratio=3.0,
    negative_sampler="mixed",
    neighbor_sampler="consistency_recent",
    recent_window=50,
    recent_ratio=0.8,
    consistency_temperature=0.35,
    time_decay_strength=4.0,
    known_label_feature=False,
    target_context_fusion="none",
    include_historical_background_negatives=False,
    aux_multiclass_loss_weight=0.0,
    aux_inference_blend=0.0,
)

_DYRIFT_TGAT_SHARED: dict[str, Any] = {
    "early_stop_patience": 5,
    "min_early_stop_epoch": 0,
    "loss_type": "bce",
    "ranking_weight": 0.0,
    "ranking_margin": 0.2,
    "recent_window": 50,
    "dropout": 0.15,
    "attention_num_heads": 4,
    "attention_logit_scale": 1.0,
    "prototype_multiclass_num_classes": 2,
    "prototype_loss_weight": 0.04,
    "prototype_loss_weight_schedule": "adaptive_quality",
    "prototype_loss_min_weight": 0.01,
    "prototype_temperature": 0.2,
    "prototype_momentum": 0.9,
    "prototype_start_epoch": 2,
    "prototype_loss_ramp_epochs": 4,
    "prototype_bucket_mode": "time_bucket",
    "prototype_neighbor_blend": 0.10,
    "prototype_global_blend": 0.20,
    "prototype_consistency_weight": 0.05,
    "prototype_separation_weight": 0.05,
    "prototype_separation_margin": 0.15,
    "pseudo_contrastive_weight": 0.02,
    "pseudo_contrastive_temperature": 0.20,
    "pseudo_contrastive_sample_size": 192,
    "pseudo_contrastive_low_quantile": 0.12,
    "pseudo_contrastive_high_quantile": 0.88,
    "pseudo_contrastive_interval": 6,
    "pseudo_contrastive_start_epoch": 3,
    "pseudo_contrastive_time_balanced": True,
    "pseudo_contrastive_min_confidence_gap": 0.08,
    "normal_bucket_align_weight": 0.05,
    "context_residual_clip": 0.18,
    "context_residual_budget": 0.08,
    "context_residual_budget_weight": 0.10,
    "context_residual_budget_schedule": "prototype_adaptive",
    "context_residual_budget_min_weight": 0.03,
    "context_residual_budget_release_epochs": 10,
    "context_residual_budget_release_delay_epochs": 3,
}

_DYRIFT_TGAT_CONTEXT_BRIDGE: dict[str, Any] = {
    "target_context_fusion": "drift_residual",
    "target_time_adapter_strength": 0.15,
    "target_time_adapter_type": "drift_expert",
    "target_time_expert_entropy_floor": 0.9,
    "target_time_expert_entropy_weight": 0.05,
}

_DYRIFT_TGAT_BASE_PRESET: dict[str, Any] = {
    **_DYRIFT_TGAT_SHARED,
    "target_context_fusion": "none",
    "target_time_adapter_strength": 0.0,
    "internal_risk_fusion": "residual",
    "internal_risk_short_time_scale": 0.12,
    "internal_risk_long_time_scale": 0.45,
}
_DYRIFT_TGAT_DEPLOY_PRESET: dict[str, Any] = {
    **_DYRIFT_TGAT_SHARED,
    **_DYRIFT_TGAT_CONTEXT_BRIDGE,
    "dropout": 0.12,
    "attention_num_heads": 8,
    "message_risk_strength": 0.15,
    "internal_risk_fusion": "residual",
    "internal_risk_residual_scale": 0.35,
    "internal_risk_short_time_scale": 0.12,
    "internal_risk_long_time_scale": 0.45,
}

_DYRIFT_PUBLIC_PRESETS: dict[str, dict[str, Any]] = {
    TRANSFORMER_BACKBONE_PRESET: _DYRIFT_TGAT_BASE_PRESET,
    TRANSFORMER_BACKBONE_DEPLOY_PRESET: _DYRIFT_TGAT_DEPLOY_PRESET,
    LEGACY_TRANSFORMER_BACKBONE_PRESET: _DYRIFT_TGAT_BASE_PRESET,
    LEGACY_TRANSFORMER_BACKBONE_DEPLOY_PRESET: _DYRIFT_TGAT_DEPLOY_PRESET,
}

_PRESET_UPDATES: dict[str, dict[str, dict[str, Any]]] = {
    "m5_temporal_graphsage": {
        "unified_baseline": {
            "early_stop_patience": 5,
            "min_early_stop_epoch": 0,
            "loss_type": "bce",
        },
    },
    "m7_utpm": {
        "utpm_temporal_shift_v4": {
            "early_stop_patience": 5,
            "min_early_stop_epoch": 0,
            "loss_type": "bce",
            "ranking_weight": 0.0,
            "ranking_margin": 0.2,
            "recent_window": 50,
            "prototype_multiclass_num_classes": 2,
            "prototype_loss_weight": 0.05,
            "prototype_loss_weight_schedule": "adaptive_quality",
            "prototype_loss_min_weight": 0.01,
            "prototype_temperature": 0.2,
            "prototype_momentum": 0.9,
            "prototype_start_epoch": 2,
            "prototype_loss_ramp_epochs": 4,
            "prototype_bucket_mode": "time_bucket",
            "prototype_neighbor_blend": 0.10,
            "prototype_global_blend": 0.20,
            "prototype_consistency_weight": 0.05,
            "prototype_separation_weight": 0.05,
            "prototype_separation_margin": 0.15,
            "pseudo_contrastive_weight": 0.015,
            "pseudo_contrastive_temperature": 0.20,
            "pseudo_contrastive_sample_size": 192,
            "pseudo_contrastive_low_quantile": 0.12,
            "pseudo_contrastive_high_quantile": 0.88,
            "pseudo_contrastive_interval": 6,
            "pseudo_contrastive_start_epoch": 3,
            "pseudo_contrastive_time_balanced": True,
            "pseudo_contrastive_min_confidence_gap": 0.08,
            "target_context_fusion": "drift_residual",
            "target_time_adapter_strength": 0.15,
            "target_time_adapter_type": "drift_expert",
            "target_time_expert_entropy_floor": 0.9,
            "target_time_expert_entropy_weight": 0.05,
            "normal_bucket_align_weight": 0.05,
            "context_residual_clip": 0.18,
            "context_residual_budget": 0.08,
            "context_residual_budget_weight": 0.10,
            "context_residual_budget_schedule": "prototype_adaptive",
            "context_residual_budget_min_weight": 0.03,
            "context_residual_budget_release_epochs": 10,
            "context_residual_budget_release_delay_epochs": 3,
        },
    },
    DYRIFT_GNN_MODEL: _DYRIFT_PUBLIC_PRESETS,
}


_DEFAULT_PRESET_BY_MODEL = {
    "m5_temporal_graphsage": "unified_baseline",
    "m7_utpm": OFFICIAL_BACKBONE_PRESET,
    TRANSFORMER_BACKBONE_MODEL: TRANSFORMER_BACKBONE_PRESET,
}


def list_presets(model_name: str) -> tuple[str, ...]:
    preset_names = _PRESET_UPDATES.get(str(model_name))
    if preset_names is None:
        raise ValueError(f"Unsupported thesis model for presets: {model_name}")
    return tuple(preset_names.keys())


def default_preset(model_name: str) -> str:
    try:
        return _DEFAULT_PRESET_BY_MODEL[str(model_name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported thesis model for presets: {model_name}") from exc


def build_graph_cfg(model_name: str, preset_name: str | None = None) -> GraphModelConfig:
    resolved_preset = preset_name or default_preset(model_name)
    preset_updates = _PRESET_UPDATES.get(str(model_name), {}).get(str(resolved_preset))
    if preset_updates is None:
        supported = ", ".join(list_presets(model_name))
        raise ValueError(
            f"Unsupported preset `{resolved_preset}` for `{model_name}`. Supported presets: {supported}"
        )
    return replace(_BASE_GRAPH_CONFIG, **preset_updates)


def _coerce_bool(raw_value: str) -> bool:
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got `{raw_value}`.")


def _coerce_like(template_value: Any, raw_value: str) -> Any:
    if isinstance(template_value, bool):
        return _coerce_bool(raw_value)
    if isinstance(template_value, int) and not isinstance(template_value, bool):
        return int(raw_value)
    if isinstance(template_value, float):
        return float(raw_value)
    return str(raw_value)


def apply_cfg_overrides(
    config: GraphModelConfig,
    overrides: list[str] | tuple[str, ...] | None,
) -> tuple[GraphModelConfig, dict[str, Any]]:
    if not overrides:
        return config, {}

    payload = config.to_dict()
    applied: dict[str, Any] = {}
    for raw_override in overrides:
        key, separator, raw_value = str(raw_override).partition("=")
        override_key = key.strip()
        if not separator or not override_key:
            raise ValueError(
                f"Invalid graph-config override `{raw_override}`. Expected `key=value`."
            )
        if override_key not in payload:
            supported = ", ".join(sorted(payload.keys()))
            raise ValueError(
                f"Unknown graph-config override key `{override_key}`. Supported keys: {supported}"
            )
        coerced_value = _coerce_like(payload[override_key], raw_value.strip())
        payload[override_key] = coerced_value
        applied[override_key] = coerced_value
    return GraphModelConfig.from_dict(payload), applied
