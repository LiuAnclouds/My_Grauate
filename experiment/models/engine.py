from __future__ import annotations

import csv
import copy
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from experiment.features.features import (
    FeatureStore,
    GraphCache,
    HybridFeatureNormalizerState,
    default_feature_groups,
)
from experiment.models.modules.backbone import (
    TRGTInternalRiskEncoder,
    TRGTMeanRelationBlock,
    TRGTTemporalRelationAttentionBlock,
)
from experiment.models.modules.bridge import TargetContextFusionHead
from experiment.models.modules.memory import (
    PrototypeMemoryBank,
    PrototypeMemoryConfig,
    TemporalNormalAlignmentBank,
    TemporalNormalAlignmentConfig,
)
from experiment.utils.common import (
    compute_binary_classification_metrics,
    ensure_dir,
    resolve_device,
    set_global_seed,
    write_json,
)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def _grad_reverse(input_tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(input_tensor, float(scale))


@dataclass(frozen=True)
class GraphPhaseContext:
    phase: str
    feature_store: FeatureStore
    graph_cache: GraphCache
    labels: np.ndarray
    target_context_store: FeatureStore | None = None
    known_label_codes: np.ndarray | None = None
    reference_day: int | None = None
    historical_background_ids: np.ndarray | None = None
    sampling_profile: np.ndarray | None = None


@dataclass(frozen=True)
class GraphModelConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.2
    feature_norm: str = "none"
    norm: str = "none"
    residual: bool = False
    ffn: bool = False
    jk: str = "last"
    edge_encoder: str = "basic"
    subgraph_head: str = "none"
    grad_clip: float = 0.0
    scheduler: str = "none"
    early_stop_patience: int = 0
    min_early_stop_epoch: int = 0
    train_negative_ratio: float = 0.0
    negative_sampler: str = "random"
    hard_negative_mix: float = 0.5
    hard_negative_warmup_epochs: int = 1
    hard_negative_refresh: int = 2
    hard_negative_candidate_cap: int = 100000
    hard_negative_candidate_multiplier: float = 4.0
    hard_negative_pool_multiplier: float = 2.0
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = -1.0
    ranking_weight: float = 0.2
    ranking_margin: float = 0.2
    neighbor_sampler: str = "uniform"
    recent_window: int = 50
    recent_ratio: float = 0.8
    consistency_temperature: float = 0.35
    time_decay_strength: float = 0.0
    target_time_weight_half_life_days: float = 0.0
    target_time_weight_floor: float = 0.25
    message_risk_strength: float = 0.0
    attention_num_heads: int = 1
    attention_logit_scale: float = 1.0
    known_label_feature: bool = False
    known_label_feature_dim: int = 5
    target_context_fusion: str = "none"
    target_context_input_dim: int = 0
    target_context_feature_gate_strength: float = 0.0
    cold_start_residual_strength: float = 0.0
    primary_multiclass_num_classes: int = 0
    prototype_multiclass_num_classes: int = 0
    prototype_loss_weight: float = 0.0
    prototype_loss_weight_schedule: str = "none"
    prototype_loss_min_weight: float = 0.0
    prototype_temperature: float = 0.2
    prototype_momentum: float = 0.9
    prototype_start_epoch: int = 1
    prototype_loss_ramp_epochs: int = 0
    prototype_bucket_mode: str = "global"
    prototype_neighbor_blend: float = 0.0
    prototype_global_blend: float = 0.0
    prototype_consistency_weight: float = 0.0
    prototype_separation_weight: float = 0.0
    prototype_separation_margin: float = 0.1
    pseudo_contrastive_weight: float = 0.0
    pseudo_contrastive_temperature: float = 0.2
    pseudo_contrastive_sample_size: int = 128
    pseudo_contrastive_low_quantile: float = 0.1
    pseudo_contrastive_high_quantile: float = 0.9
    pseudo_contrastive_interval: int = 4
    pseudo_contrastive_start_epoch: int = 1
    pseudo_contrastive_time_balanced: bool = False
    pseudo_contrastive_min_confidence_gap: float = 0.0
    normal_bucket_align_weight: float = 0.0
    normal_bucket_shift_strength: float = 0.0
    target_time_adapter_strength: float = 0.0
    target_time_adapter_type: str = "affine"
    target_time_adapter_num_experts: int = 4
    target_time_expert_entropy_floor: float = 0.0
    target_time_expert_entropy_weight: float = 0.0
    atm_gate_strength: float = 1.0
    context_residual_scale: float = 1.0
    context_residual_clip: float = 0.0
    context_residual_budget: float = 0.0
    context_residual_budget_weight: float = 0.0
    context_residual_budget_schedule: str = "none"
    context_residual_budget_min_weight: float = 0.0
    context_residual_budget_release_epochs: int = 0
    context_residual_budget_release_delay_epochs: int = 0
    normal_bucket_adv_weight: float = 0.0
    include_historical_background_negatives: bool = False
    historical_background_negative_ratio: float = 0.25
    historical_background_negative_warmup_epochs: int = 0
    historical_background_aux_only: bool = False
    aux_multiclass_num_classes: int = 4
    aux_multiclass_loss_weight: float = 0.0
    aux_inference_blend: float = 0.0
    internal_risk_fusion: str = "none"
    internal_risk_residual_scale: float = 1.0
    internal_risk_short_time_scale: float = 0.12
    internal_risk_long_time_scale: float = 0.45

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "dropout": float(self.dropout),
            "feature_norm": self.feature_norm,
            "norm": self.norm,
            "residual": bool(self.residual),
            "ffn": bool(self.ffn),
            "jk": self.jk,
            "edge_encoder": self.edge_encoder,
            "subgraph_head": self.subgraph_head,
            "grad_clip": float(self.grad_clip),
            "scheduler": self.scheduler,
            "early_stop_patience": int(self.early_stop_patience),
            "min_early_stop_epoch": int(self.min_early_stop_epoch),
            "train_negative_ratio": float(self.train_negative_ratio),
            "negative_sampler": self.negative_sampler,
            "hard_negative_mix": float(self.hard_negative_mix),
            "hard_negative_warmup_epochs": int(self.hard_negative_warmup_epochs),
            "hard_negative_refresh": int(self.hard_negative_refresh),
            "hard_negative_candidate_cap": int(self.hard_negative_candidate_cap),
            "hard_negative_candidate_multiplier": float(self.hard_negative_candidate_multiplier),
            "hard_negative_pool_multiplier": float(self.hard_negative_pool_multiplier),
            "loss_type": self.loss_type,
            "focal_gamma": float(self.focal_gamma),
            "focal_alpha": float(self.focal_alpha),
            "ranking_weight": float(self.ranking_weight),
            "ranking_margin": float(self.ranking_margin),
            "neighbor_sampler": self.neighbor_sampler,
            "recent_window": int(self.recent_window),
            "recent_ratio": float(self.recent_ratio),
            "consistency_temperature": float(self.consistency_temperature),
            "time_decay_strength": float(self.time_decay_strength),
            "target_time_weight_half_life_days": float(self.target_time_weight_half_life_days),
            "target_time_weight_floor": float(self.target_time_weight_floor),
            "message_risk_strength": float(self.message_risk_strength),
            "attention_num_heads": int(self.attention_num_heads),
            "attention_logit_scale": float(self.attention_logit_scale),
            "known_label_feature": bool(self.known_label_feature),
            "known_label_feature_dim": int(self.known_label_feature_dim),
            "target_context_fusion": self.target_context_fusion,
            "target_context_input_dim": int(self.target_context_input_dim),
            "target_context_feature_gate_strength": float(self.target_context_feature_gate_strength),
            "cold_start_residual_strength": float(self.cold_start_residual_strength),
            "primary_multiclass_num_classes": int(self.primary_multiclass_num_classes),
            "prototype_multiclass_num_classes": int(self.prototype_multiclass_num_classes),
            "prototype_loss_weight": float(self.prototype_loss_weight),
            "prototype_loss_weight_schedule": self.prototype_loss_weight_schedule,
            "prototype_loss_min_weight": float(self.prototype_loss_min_weight),
            "prototype_temperature": float(self.prototype_temperature),
            "prototype_momentum": float(self.prototype_momentum),
            "prototype_start_epoch": int(self.prototype_start_epoch),
            "prototype_loss_ramp_epochs": int(self.prototype_loss_ramp_epochs),
            "prototype_bucket_mode": self.prototype_bucket_mode,
            "prototype_neighbor_blend": float(self.prototype_neighbor_blend),
            "prototype_global_blend": float(self.prototype_global_blend),
            "prototype_consistency_weight": float(self.prototype_consistency_weight),
            "prototype_separation_weight": float(self.prototype_separation_weight),
            "prototype_separation_margin": float(self.prototype_separation_margin),
            "pseudo_contrastive_weight": float(self.pseudo_contrastive_weight),
            "pseudo_contrastive_temperature": float(self.pseudo_contrastive_temperature),
            "pseudo_contrastive_sample_size": int(self.pseudo_contrastive_sample_size),
            "pseudo_contrastive_low_quantile": float(self.pseudo_contrastive_low_quantile),
            "pseudo_contrastive_high_quantile": float(self.pseudo_contrastive_high_quantile),
            "pseudo_contrastive_interval": int(self.pseudo_contrastive_interval),
            "pseudo_contrastive_start_epoch": int(self.pseudo_contrastive_start_epoch),
            "pseudo_contrastive_time_balanced": bool(self.pseudo_contrastive_time_balanced),
            "pseudo_contrastive_min_confidence_gap": float(
                self.pseudo_contrastive_min_confidence_gap
            ),
            "normal_bucket_align_weight": float(self.normal_bucket_align_weight),
            "normal_bucket_shift_strength": float(self.normal_bucket_shift_strength),
            "target_time_adapter_strength": float(self.target_time_adapter_strength),
            "target_time_adapter_type": self.target_time_adapter_type,
            "target_time_adapter_num_experts": int(self.target_time_adapter_num_experts),
            "target_time_expert_entropy_floor": float(self.target_time_expert_entropy_floor),
            "target_time_expert_entropy_weight": float(self.target_time_expert_entropy_weight),
            "atm_gate_strength": float(self.atm_gate_strength),
            "context_residual_scale": float(self.context_residual_scale),
            "context_residual_clip": float(self.context_residual_clip),
            "context_residual_budget": float(self.context_residual_budget),
            "context_residual_budget_weight": float(self.context_residual_budget_weight),
            "context_residual_budget_schedule": self.context_residual_budget_schedule,
            "context_residual_budget_min_weight": float(self.context_residual_budget_min_weight),
            "context_residual_budget_release_epochs": int(self.context_residual_budget_release_epochs),
            "context_residual_budget_release_delay_epochs": int(
                self.context_residual_budget_release_delay_epochs
            ),
            "normal_bucket_adv_weight": float(self.normal_bucket_adv_weight),
            "include_historical_background_negatives": bool(self.include_historical_background_negatives),
            "historical_background_negative_ratio": float(self.historical_background_negative_ratio),
            "historical_background_negative_warmup_epochs": int(
                self.historical_background_negative_warmup_epochs
            ),
            "historical_background_aux_only": bool(self.historical_background_aux_only),
            "aux_multiclass_num_classes": int(self.aux_multiclass_num_classes),
            "aux_multiclass_loss_weight": float(self.aux_multiclass_loss_weight),
            "aux_inference_blend": float(self.aux_inference_blend),
            "internal_risk_fusion": self.internal_risk_fusion,
            "internal_risk_residual_scale": float(self.internal_risk_residual_scale),
            "internal_risk_short_time_scale": float(self.internal_risk_short_time_scale),
            "internal_risk_long_time_scale": float(self.internal_risk_long_time_scale),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GraphModelConfig":
        if not payload:
            return cls()
        return cls(
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-5)),
            dropout=float(payload.get("dropout", 0.2)),
            feature_norm=str(payload.get("feature_norm", "none")),
            norm=str(payload.get("norm", "none")),
            residual=bool(payload.get("residual", False)),
            ffn=bool(payload.get("ffn", False)),
            jk=str(payload.get("jk", "last")),
            edge_encoder=str(payload.get("edge_encoder", "basic")),
            subgraph_head=str(payload.get("subgraph_head", "none")),
            grad_clip=float(payload.get("grad_clip", 0.0)),
            scheduler=str(payload.get("scheduler", "none")),
            early_stop_patience=int(payload.get("early_stop_patience", 0)),
            min_early_stop_epoch=int(payload.get("min_early_stop_epoch", 0)),
            train_negative_ratio=float(payload.get("train_negative_ratio", 0.0)),
            negative_sampler=str(payload.get("negative_sampler", "random")),
            hard_negative_mix=float(payload.get("hard_negative_mix", 0.5)),
            hard_negative_warmup_epochs=int(payload.get("hard_negative_warmup_epochs", 1)),
            hard_negative_refresh=int(payload.get("hard_negative_refresh", 2)),
            hard_negative_candidate_cap=int(payload.get("hard_negative_candidate_cap", 100000)),
            hard_negative_candidate_multiplier=float(payload.get("hard_negative_candidate_multiplier", 4.0)),
            hard_negative_pool_multiplier=float(payload.get("hard_negative_pool_multiplier", 2.0)),
            loss_type=str(payload.get("loss_type", "bce")),
            focal_gamma=float(payload.get("focal_gamma", 2.0)),
            focal_alpha=float(payload.get("focal_alpha", -1.0)),
            ranking_weight=float(payload.get("ranking_weight", 0.2)),
            ranking_margin=float(payload.get("ranking_margin", 0.2)),
            neighbor_sampler=str(payload.get("neighbor_sampler", "uniform")),
            recent_window=int(payload.get("recent_window", 50)),
            recent_ratio=float(payload.get("recent_ratio", 0.8)),
            consistency_temperature=float(payload.get("consistency_temperature", 0.35)),
            time_decay_strength=float(payload.get("time_decay_strength", 0.0)),
            target_time_weight_half_life_days=float(payload.get("target_time_weight_half_life_days", 0.0)),
            target_time_weight_floor=float(payload.get("target_time_weight_floor", 0.25)),
            message_risk_strength=float(payload.get("message_risk_strength", 0.0)),
            attention_num_heads=int(payload.get("attention_num_heads", 1)),
            attention_logit_scale=float(payload.get("attention_logit_scale", 1.0)),
            known_label_feature=bool(payload.get("known_label_feature", False)),
            known_label_feature_dim=int(payload.get("known_label_feature_dim", 5)),
            target_context_fusion=str(payload.get("target_context_fusion", "none")),
            target_context_input_dim=int(payload.get("target_context_input_dim", 0)),
            target_context_feature_gate_strength=float(
                payload.get("target_context_feature_gate_strength", 0.0)
            ),
            cold_start_residual_strength=float(payload.get("cold_start_residual_strength", 0.0)),
            primary_multiclass_num_classes=int(payload.get("primary_multiclass_num_classes", 0)),
            prototype_multiclass_num_classes=int(payload.get("prototype_multiclass_num_classes", 0)),
            prototype_loss_weight=float(payload.get("prototype_loss_weight", 0.0)),
            prototype_loss_weight_schedule=str(payload.get("prototype_loss_weight_schedule", "none")),
            prototype_loss_min_weight=float(payload.get("prototype_loss_min_weight", 0.0)),
            prototype_temperature=float(payload.get("prototype_temperature", 0.2)),
            prototype_momentum=float(payload.get("prototype_momentum", 0.9)),
            prototype_start_epoch=int(payload.get("prototype_start_epoch", 1)),
            prototype_loss_ramp_epochs=int(payload.get("prototype_loss_ramp_epochs", 0)),
            prototype_bucket_mode=str(payload.get("prototype_bucket_mode", "global")),
            prototype_neighbor_blend=float(payload.get("prototype_neighbor_blend", 0.0)),
            prototype_global_blend=float(payload.get("prototype_global_blend", 0.0)),
            prototype_consistency_weight=float(payload.get("prototype_consistency_weight", 0.0)),
            prototype_separation_weight=float(payload.get("prototype_separation_weight", 0.0)),
            prototype_separation_margin=float(payload.get("prototype_separation_margin", 0.1)),
            pseudo_contrastive_weight=float(payload.get("pseudo_contrastive_weight", 0.0)),
            pseudo_contrastive_temperature=float(payload.get("pseudo_contrastive_temperature", 0.2)),
            pseudo_contrastive_sample_size=int(payload.get("pseudo_contrastive_sample_size", 128)),
            pseudo_contrastive_low_quantile=float(payload.get("pseudo_contrastive_low_quantile", 0.1)),
            pseudo_contrastive_high_quantile=float(payload.get("pseudo_contrastive_high_quantile", 0.9)),
            pseudo_contrastive_interval=int(payload.get("pseudo_contrastive_interval", 4)),
            pseudo_contrastive_start_epoch=int(payload.get("pseudo_contrastive_start_epoch", 1)),
            pseudo_contrastive_time_balanced=bool(
                payload.get("pseudo_contrastive_time_balanced", False)
            ),
            pseudo_contrastive_min_confidence_gap=float(
                payload.get("pseudo_contrastive_min_confidence_gap", 0.0)
            ),
            normal_bucket_align_weight=float(payload.get("normal_bucket_align_weight", 0.0)),
            normal_bucket_shift_strength=float(payload.get("normal_bucket_shift_strength", 0.0)),
            target_time_adapter_strength=float(payload.get("target_time_adapter_strength", 0.0)),
            target_time_adapter_type=str(payload.get("target_time_adapter_type", "affine")),
            target_time_adapter_num_experts=int(payload.get("target_time_adapter_num_experts", 4)),
            target_time_expert_entropy_floor=float(
                payload.get("target_time_expert_entropy_floor", 0.0)
            ),
            target_time_expert_entropy_weight=float(
                payload.get("target_time_expert_entropy_weight", 0.0)
            ),
            atm_gate_strength=float(payload.get("atm_gate_strength", 1.0)),
            context_residual_scale=float(payload.get("context_residual_scale", 1.0)),
            context_residual_clip=float(payload.get("context_residual_clip", 0.0)),
            context_residual_budget=float(payload.get("context_residual_budget", 0.0)),
            context_residual_budget_weight=float(payload.get("context_residual_budget_weight", 0.0)),
            context_residual_budget_schedule=str(payload.get("context_residual_budget_schedule", "none")),
            context_residual_budget_min_weight=float(payload.get("context_residual_budget_min_weight", 0.0)),
            context_residual_budget_release_epochs=int(
                payload.get("context_residual_budget_release_epochs", 0)
            ),
            context_residual_budget_release_delay_epochs=int(
                payload.get("context_residual_budget_release_delay_epochs", 0)
            ),
            normal_bucket_adv_weight=float(payload.get("normal_bucket_adv_weight", 0.0)),
            include_historical_background_negatives=bool(
                payload.get("include_historical_background_negatives", False)
            ),
            historical_background_negative_ratio=float(
                payload.get("historical_background_negative_ratio", 0.25)
            ),
            historical_background_negative_warmup_epochs=int(
                payload.get("historical_background_negative_warmup_epochs", 0)
            ),
            historical_background_aux_only=bool(payload.get("historical_background_aux_only", False)),
            aux_multiclass_num_classes=int(payload.get("aux_multiclass_num_classes", 4)),
            aux_multiclass_loss_weight=float(payload.get("aux_multiclass_loss_weight", 0.0)),
            aux_inference_blend=float(payload.get("aux_inference_blend", 0.0)),
            internal_risk_fusion=str(payload.get("internal_risk_fusion", "none")),
            internal_risk_residual_scale=float(payload.get("internal_risk_residual_scale", 1.0)),
            internal_risk_short_time_scale=float(payload.get("internal_risk_short_time_scale", 0.12)),
            internal_risk_long_time_scale=float(payload.get("internal_risk_long_time_scale", 0.45)),
        )

    def use_legacy_path(self) -> bool:
        return (
            self.norm == "none"
            and not self.residual
            and not self.ffn
            and self.jk == "last"
            and self.edge_encoder == "basic"
            and self.subgraph_head == "none"
        )


@dataclass(frozen=True)
class GraphForwardOutput:
    logits: torch.Tensor
    diagnostics: dict[str, float] | None = None
    embedding: torch.Tensor | None = None
    aux_logits: torch.Tensor | None = None
    context_regularization_loss: torch.Tensor | None = None
    adapter_regularization_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class SampledSubgraph:
    node_ids: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    rel_ids: np.ndarray
    edge_timestamp: np.ndarray
    target_local_idx: np.ndarray
    node_subgraph_id: np.ndarray | None = None
    edge_subgraph_id: np.ndarray | None = None
    node_hop_depth: np.ndarray | None = None
    edge_hop_depth: np.ndarray | None = None


@dataclass(frozen=True)
class TrainBatchStats:
    target_count: int
    positive_count: int
    negative_count: int
    hard_negative_count: int = 0
    background_negative_count: int = 0

    @property
    def positive_rate(self) -> float:
        return 0.0 if self.target_count == 0 else float(self.positive_count / self.target_count)


def _append_text_line(path: Path, line: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line.rstrip("\n"))
        fp.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False))
        fp.write("\n")


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_training_curves(path: Path, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return "no training rows collected"
    try:
        mpl_cache_dir = path.parent / ".mplconfig"
        ensure_dir(mpl_cache_dir)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_auc = [float(row["val_auc"]) for row in rows]
    val_pr_auc = [float(row["val_pr_auc"]) for row in rows]
    val_ap = [float(row["val_ap"]) for row in rows]
    best_epoch = int(max(rows, key=lambda row: float(row["val_auc"]))["epoch"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_loss, ax_auc, ax_ap, ax_pr = axes.flat

    ax_loss.plot(epochs, train_loss, marker="o", linewidth=2.0, color="#1f77b4")
    ax_loss.set_title("Train Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)

    ax_auc.plot(epochs, val_auc, marker="o", linewidth=2.0, color="#d62728")
    ax_auc.axvline(best_epoch, linestyle="--", linewidth=1.2, color="#444444")
    ax_auc.set_title("Validation ROC-AUC")
    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("AUC")
    ax_auc.grid(alpha=0.25)

    ax_ap.plot(epochs, val_ap, marker="o", linewidth=2.0, color="#2ca02c")
    ax_ap.axvline(best_epoch, linestyle="--", linewidth=1.2, color="#444444")
    ax_ap.set_title("Validation Average Precision")
    ax_ap.set_xlabel("Epoch")
    ax_ap.set_ylabel("AP")
    ax_ap.grid(alpha=0.25)

    ax_pr.plot(epochs, val_pr_auc, marker="o", linewidth=2.0, color="#9467bd")
    ax_pr.axvline(best_epoch, linestyle="--", linewidth=1.2, color="#444444")
    ax_pr.set_title("Validation PR-AUC")
    ax_pr.set_xlabel("Epoch")
    ax_pr.set_ylabel("PR-AUC")
    ax_pr.grid(alpha=0.25)

    ensure_dir(path.parent)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return None


def _sampler_uses_relation_risk(sampler: str) -> bool:
    return str(sampler) in {"risk_recent", "risk_consistency_recent"}


def _sampler_uses_consistency_profile(sampler: str) -> bool:
    return str(sampler) in {"consistency_recent", "risk_consistency_recent"}


def _consistency_score(
    *,
    center_node: int | None,
    neighbor_ids: np.ndarray | None,
    node_profile: np.ndarray | None,
    temperature: float,
) -> np.ndarray | None:
    if center_node is None or neighbor_ids is None or node_profile is None:
        return None
    if neighbor_ids.size == 0:
        return np.empty(0, dtype=np.float64)

    center_idx = int(center_node)
    neighbor_idx = np.asarray(neighbor_ids, dtype=np.int32)
    center_profile = np.asarray(node_profile[center_idx], dtype=np.float32)
    neighbor_profile = np.asarray(node_profile[neighbor_idx], dtype=np.float32)
    if center_profile.ndim != 1 or neighbor_profile.ndim != 2:
        return None
    if not np.all(np.isfinite(center_profile)):
        center_profile = np.nan_to_num(center_profile, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(neighbor_profile)):
        neighbor_profile = np.nan_to_num(neighbor_profile, nan=0.0, posinf=0.0, neginf=0.0)

    similarity = np.matmul(neighbor_profile, center_profile.astype(np.float32, copy=False)).astype(
        np.float64,
        copy=False,
    )
    similarity = np.clip(similarity, -1.0, 1.0)
    if temperature <= 0.0:
        return np.clip((similarity + 1.0) * 0.5, 1e-6, None)
    shifted = similarity - float(np.max(similarity))
    score = np.exp(shifted / max(float(temperature), 1e-3))
    return np.clip(score, 1e-6, None).astype(np.float64, copy=False)


def _recent_sampler_score(
    *,
    sampler: str,
    recent_pool: np.ndarray,
    edge_relation_ids: np.ndarray | None,
    relation_weight: np.ndarray | None,
    center_node: int | None,
    edge_neighbor_ids: np.ndarray | None,
    node_profile: np.ndarray | None,
    consistency_temperature: float,
) -> np.ndarray | None:
    score: np.ndarray | None = None
    if _sampler_uses_relation_risk(sampler) and edge_relation_ids is not None and relation_weight is not None:
        recent_rel = np.asarray(edge_relation_ids[recent_pool], dtype=np.int64)
        score = np.asarray(relation_weight[recent_rel], dtype=np.float64)
    if _sampler_uses_consistency_profile(sampler) and edge_neighbor_ids is not None and node_profile is not None:
        consistency = _consistency_score(
            center_node=center_node,
            neighbor_ids=np.asarray(edge_neighbor_ids[recent_pool], dtype=np.int32),
            node_profile=node_profile,
            temperature=consistency_temperature,
        )
        if consistency is not None:
            score = consistency if score is None else score * consistency
    if score is None:
        return None
    return np.clip(score, 1e-6, None).astype(np.float64, copy=False)


def _sample_edge_indices(
    edge_timestamp: np.ndarray,
    fanout: int,
    rng: np.random.Generator,
    snapshot_end: int | None,
    edge_relation_ids: np.ndarray | None = None,
    relation_weight: np.ndarray | None = None,
    center_node: int | None = None,
    edge_neighbor_ids: np.ndarray | None = None,
    node_profile: np.ndarray | None = None,
    sampler: str = "uniform",
    recent_window: int = 50,
    recent_ratio: float = 0.8,
    consistency_temperature: float = 0.35,
    training: bool = True,
) -> np.ndarray:
    if edge_timestamp.size == 0:
        return np.empty(0, dtype=np.int32)
    if snapshot_end is not None:
        valid = np.flatnonzero(edge_timestamp <= snapshot_end)
    else:
        valid = np.arange(edge_timestamp.size, dtype=np.int32)
    if valid.size <= fanout:
        return valid.astype(np.int32, copy=False)

    sampler = str(sampler)
    if sampler == "uniform":
        choice = rng.choice(valid, size=fanout, replace=False)
        return np.sort(choice.astype(np.int32, copy=False))

    window = max(int(recent_window), fanout, 1)
    valid_timestamps = edge_timestamp[valid]
    if valid.size > window:
        recent_order = np.argpartition(valid_timestamps, -window)[-window:]
        recent_pool = valid[recent_order].astype(np.int32, copy=False)
    else:
        recent_pool = valid.astype(np.int32, copy=False)

    recent_score = _recent_sampler_score(
        sampler=sampler,
        recent_pool=recent_pool,
        edge_relation_ids=edge_relation_ids,
        relation_weight=relation_weight,
        center_node=center_node,
        edge_neighbor_ids=edge_neighbor_ids,
        node_profile=node_profile,
        consistency_temperature=consistency_temperature,
    )

    if not training:
        if recent_score is not None:
            top_score = np.argpartition(recent_score, -fanout)[-fanout:]
            choice = recent_pool[top_score]
            return np.sort(choice.astype(np.int32, copy=False))
        recent_pool_ts = edge_timestamp[recent_pool]
        top_recent = np.argpartition(recent_pool_ts, -fanout)[-fanout:]
        choice = recent_pool[top_recent]
        return np.sort(choice.astype(np.int32, copy=False))

    if sampler in {"recent", "consistency_recent", "risk_recent", "risk_consistency_recent"}:
        if recent_pool.size <= fanout:
            return np.sort(recent_pool.astype(np.int32, copy=False))
        if recent_score is None:
            choice = rng.choice(recent_pool, size=fanout, replace=False)
            return np.sort(choice.astype(np.int32, copy=False))
        weight = np.asarray(recent_score, dtype=np.float64)
        weight = np.where(np.isfinite(weight) & (weight > 0.0), weight, 0.0)
        total_weight = float(np.sum(weight, dtype=np.float64))
        if total_weight <= 0.0:
            choice = rng.choice(recent_pool, size=fanout, replace=False)
            return np.sort(choice.astype(np.int32, copy=False))
        weight = weight / total_weight
        choice = rng.choice(recent_pool, size=fanout, replace=False, p=weight)
        return np.sort(choice.astype(np.int32, copy=False))

    if sampler != "hybrid":
        raise ValueError(f"Unsupported neighbor sampler: {sampler}")

    old_pool = np.setdiff1d(valid, recent_pool, assume_unique=False).astype(np.int32, copy=False)
    requested_recent = int(round(fanout * float(recent_ratio)))
    recent_take = max(1, min(fanout, requested_recent))
    recent_take = min(recent_take, recent_pool.size)
    old_take = min(fanout - recent_take, old_pool.size)
    taken = []

    if recent_take > 0:
        recent_choice = rng.choice(recent_pool, size=recent_take, replace=False)
        taken.append(recent_choice.astype(np.int32, copy=False))
    if old_take > 0:
        old_choice = rng.choice(old_pool, size=old_take, replace=False)
        taken.append(old_choice.astype(np.int32, copy=False))

    chosen = np.concatenate(taken).astype(np.int32, copy=False) if taken else np.empty(0, dtype=np.int32)
    if chosen.size < fanout:
        remaining_pool = np.setdiff1d(valid, chosen, assume_unique=False).astype(np.int32, copy=False)
        fill = rng.choice(remaining_pool, size=fanout - chosen.size, replace=False)
        chosen = np.concatenate([chosen, fill.astype(np.int32, copy=False)]).astype(np.int32, copy=False)
    return np.sort(chosen.astype(np.int32, copy=False))


def sample_relation_subgraph(
    graph: GraphCache,
    seed_nodes: np.ndarray,
    fanouts: list[int],
    rng: np.random.Generator,
    snapshot_end: int | None = None,
    relation_weight: np.ndarray | None = None,
    node_profile: np.ndarray | None = None,
    sampler: str = "uniform",
    recent_window: int = 50,
    recent_ratio: float = 0.8,
    consistency_temperature: float = 0.35,
    training: bool = True,
) -> SampledSubgraph:
    seeds = np.asarray(seed_nodes, dtype=np.int32)
    ordered_nodes: list[int] = []
    seen_nodes: set[int] = set()

    def add_nodes(nodes: np.ndarray) -> None:
        for node in nodes.tolist():
            if node not in seen_nodes:
                seen_nodes.add(node)
                ordered_nodes.append(int(node))

    add_nodes(seeds)
    frontier = seeds
    edge_records: list[tuple[int, int, int, int]] = []

    for fanout in fanouts:
        if frontier.size == 0:
            break
        next_frontier: list[np.ndarray] = []
        for center in frontier.tolist():
            in_start = int(graph.in_ptr[center])
            in_end = int(graph.in_ptr[center + 1])
            in_rel_ids = np.asarray(graph.in_edge_type[in_start:in_end], dtype=np.int64) - 1
            in_choice = _sample_edge_indices(
                edge_timestamp=np.asarray(graph.in_edge_timestamp[in_start:in_end]),
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
                edge_relation_ids=in_rel_ids,
                relation_weight=relation_weight,
                center_node=int(center),
                edge_neighbor_ids=np.asarray(graph.in_neighbors[in_start:in_end], dtype=np.int32),
                node_profile=node_profile,
                sampler=sampler,
                recent_window=recent_window,
                recent_ratio=recent_ratio,
                consistency_temperature=consistency_temperature,
                training=training,
            )
            if in_choice.size:
                in_neighbors = np.asarray(graph.in_neighbors[in_start:in_end])[in_choice]
                in_type = np.asarray(graph.in_edge_type[in_start:in_end])[in_choice]
                in_time = np.asarray(graph.in_edge_timestamp[in_start:in_end])[in_choice]
                next_frontier.append(in_neighbors.astype(np.int32, copy=False))
                add_nodes(in_neighbors.astype(np.int32, copy=False))
                edge_records.extend(
                    (
                        int(src),
                        int(center),
                        int(edge_type - 1),
                        int(edge_time),
                    )
                    for src, edge_type, edge_time in zip(
                        in_neighbors.tolist(),
                        in_type.tolist(),
                        in_time.tolist(),
                        strict=True,
                    )
                )

            out_start = int(graph.out_ptr[center])
            out_end = int(graph.out_ptr[center + 1])
            out_rel_ids = (
                np.asarray(graph.out_edge_type[out_start:out_end], dtype=np.int64) - 1 + graph.num_edge_types
            )
            out_choice = _sample_edge_indices(
                edge_timestamp=np.asarray(graph.out_edge_timestamp[out_start:out_end]),
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
                edge_relation_ids=out_rel_ids,
                relation_weight=relation_weight,
                center_node=int(center),
                edge_neighbor_ids=np.asarray(graph.out_neighbors[out_start:out_end], dtype=np.int32),
                node_profile=node_profile,
                sampler=sampler,
                recent_window=recent_window,
                recent_ratio=recent_ratio,
                consistency_temperature=consistency_temperature,
                training=training,
            )
            if out_choice.size:
                out_neighbors = np.asarray(graph.out_neighbors[out_start:out_end])[out_choice]
                out_type = np.asarray(graph.out_edge_type[out_start:out_end])[out_choice]
                out_time = np.asarray(graph.out_edge_timestamp[out_start:out_end])[out_choice]
                next_frontier.append(out_neighbors.astype(np.int32, copy=False))
                add_nodes(out_neighbors.astype(np.int32, copy=False))
                edge_records.extend(
                    (
                        int(src),
                        int(center),
                        int(edge_type - 1 + graph.num_edge_types),
                        int(edge_time),
                    )
                    for src, edge_type, edge_time in zip(
                        out_neighbors.tolist(),
                        out_type.tolist(),
                        out_time.tolist(),
                        strict=True,
                    )
                )

        if next_frontier:
            frontier = np.unique(np.concatenate(next_frontier)).astype(np.int32, copy=False)
        else:
            frontier = np.empty(0, dtype=np.int32)

    node_ids = np.asarray(ordered_nodes, dtype=np.int32)
    global_to_local = {int(node): idx for idx, node in enumerate(node_ids.tolist())}
    target_local_idx = np.asarray(
        [global_to_local[int(node)] for node in seeds.tolist()],
        dtype=np.int64,
    )

    if edge_records:
        edge_src = np.asarray(
            [global_to_local[src] for src, _, _, _ in edge_records],
            dtype=np.int64,
        )
        edge_dst = np.asarray(
            [global_to_local[dst] for _, dst, _, _ in edge_records],
            dtype=np.int64,
        )
        rel_ids = np.asarray([rel for _, _, rel, _ in edge_records], dtype=np.int64)
        edge_timestamp = np.asarray([ts for _, _, _, ts in edge_records], dtype=np.int64)
    else:
        edge_src = np.empty(0, dtype=np.int64)
        edge_dst = np.empty(0, dtype=np.int64)
        rel_ids = np.empty(0, dtype=np.int64)
        edge_timestamp = np.empty(0, dtype=np.int64)

    return SampledSubgraph(
        node_ids=node_ids,
        edge_src=edge_src,
        edge_dst=edge_dst,
        rel_ids=rel_ids,
        edge_timestamp=edge_timestamp,
        target_local_idx=target_local_idx,
    )


def _sample_single_seed_subgraph(
    graph: GraphCache,
    seed: int,
    fanouts: list[int],
    rng: np.random.Generator,
    snapshot_end: int | None = None,
    relation_weight: np.ndarray | None = None,
    node_profile: np.ndarray | None = None,
    sampler: str = "uniform",
    recent_window: int = 50,
    recent_ratio: float = 0.8,
    consistency_temperature: float = 0.35,
    training: bool = True,
) -> SampledSubgraph:
    seed = int(seed)
    ordered_nodes = [seed]
    ordered_node_hops = [0]
    global_to_local = {seed: 0}
    frontier = np.asarray([seed], dtype=np.int32)
    edge_src: list[int] = []
    edge_dst: list[int] = []
    rel_ids: list[int] = []
    edge_timestamp: list[int] = []
    edge_hop_depth: list[int] = []

    in_ptr = graph.in_ptr
    in_neighbors = graph.in_neighbors
    in_edge_type = graph.in_edge_type
    in_edge_timestamp = graph.in_edge_timestamp
    out_ptr = graph.out_ptr
    out_neighbors = graph.out_neighbors
    out_edge_type = graph.out_edge_type
    out_edge_timestamp = graph.out_edge_timestamp
    num_edge_types = graph.num_edge_types

    for hop_depth, fanout in enumerate(fanouts, start=1):
        if frontier.size == 0:
            break
        next_nodes: list[int] = []
        for center in frontier.tolist():
            center = int(center)
            center_local = global_to_local[center]

            in_start = int(in_ptr[center])
            in_end = int(in_ptr[center + 1])
            in_time_slice = np.asarray(in_edge_timestamp[in_start:in_end])
            in_rel_ids = np.asarray(in_edge_type[in_start:in_end], dtype=np.int64) - 1
            in_choice = _sample_edge_indices(
                edge_timestamp=in_time_slice,
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
                edge_relation_ids=in_rel_ids,
                relation_weight=relation_weight,
                center_node=center,
                edge_neighbor_ids=np.asarray(in_neighbors[in_start:in_end], dtype=np.int32),
                node_profile=node_profile,
                sampler=sampler,
                recent_window=recent_window,
                recent_ratio=recent_ratio,
                consistency_temperature=consistency_temperature,
                training=training,
            )
            if in_choice.size:
                selected_neighbors = np.asarray(in_neighbors[in_start:in_end])[in_choice]
                selected_types = np.asarray(in_edge_type[in_start:in_end])[in_choice]
                selected_times = in_time_slice[in_choice]
                for src, edge_type, edge_time in zip(
                    selected_neighbors.tolist(),
                    selected_types.tolist(),
                    selected_times.tolist(),
                    strict=True,
                ):
                    src_local = global_to_local.get(src)
                    if src_local is None:
                        src_local = len(ordered_nodes)
                        global_to_local[src] = src_local
                        ordered_nodes.append(src)
                        ordered_node_hops.append(int(hop_depth))
                    next_nodes.append(src)
                    edge_src.append(src_local)
                    edge_dst.append(center_local)
                    rel_ids.append(int(edge_type - 1))
                    edge_timestamp.append(int(edge_time))
                    edge_hop_depth.append(int(hop_depth))

            out_start = int(out_ptr[center])
            out_end = int(out_ptr[center + 1])
            out_time_slice = np.asarray(out_edge_timestamp[out_start:out_end])
            out_rel_ids = np.asarray(out_edge_type[out_start:out_end], dtype=np.int64) - 1 + num_edge_types
            out_choice = _sample_edge_indices(
                edge_timestamp=out_time_slice,
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
                edge_relation_ids=out_rel_ids,
                relation_weight=relation_weight,
                center_node=center,
                edge_neighbor_ids=np.asarray(out_neighbors[out_start:out_end], dtype=np.int32),
                node_profile=node_profile,
                sampler=sampler,
                recent_window=recent_window,
                recent_ratio=recent_ratio,
                consistency_temperature=consistency_temperature,
                training=training,
            )
            if out_choice.size:
                selected_neighbors = np.asarray(out_neighbors[out_start:out_end])[out_choice]
                selected_types = np.asarray(out_edge_type[out_start:out_end])[out_choice]
                selected_times = out_time_slice[out_choice]
                for src, edge_type, edge_time in zip(
                    selected_neighbors.tolist(),
                    selected_types.tolist(),
                    selected_times.tolist(),
                    strict=True,
                ):
                    src_local = global_to_local.get(src)
                    if src_local is None:
                        src_local = len(ordered_nodes)
                        global_to_local[src] = src_local
                        ordered_nodes.append(src)
                        ordered_node_hops.append(int(hop_depth))
                    next_nodes.append(src)
                    edge_src.append(src_local)
                    edge_dst.append(center_local)
                    rel_ids.append(int(edge_type - 1 + num_edge_types))
                    edge_timestamp.append(int(edge_time))
                    edge_hop_depth.append(int(hop_depth))

        if next_nodes:
            frontier = np.unique(np.asarray(next_nodes, dtype=np.int32))
        else:
            frontier = np.empty(0, dtype=np.int32)

    return SampledSubgraph(
        node_ids=np.asarray(ordered_nodes, dtype=np.int32),
        edge_src=np.asarray(edge_src, dtype=np.int64),
        edge_dst=np.asarray(edge_dst, dtype=np.int64),
        rel_ids=np.asarray(rel_ids, dtype=np.int64),
        edge_timestamp=np.asarray(edge_timestamp, dtype=np.int64),
        target_local_idx=np.asarray([0], dtype=np.int64),
        node_hop_depth=np.asarray(ordered_node_hops, dtype=np.int64),
        edge_hop_depth=np.asarray(edge_hop_depth, dtype=np.int64),
    )


def sample_batched_relation_subgraphs(
    graph: GraphCache,
    seed_nodes: np.ndarray,
    fanouts: list[int],
    rng: np.random.Generator,
    snapshot_end: int | None = None,
    relation_weight: np.ndarray | None = None,
    node_profile: np.ndarray | None = None,
    sampler: str = "uniform",
    recent_window: int = 50,
    recent_ratio: float = 0.8,
    consistency_temperature: float = 0.35,
    training: bool = True,
) -> SampledSubgraph:
    seeds = np.asarray(seed_nodes, dtype=np.int32)
    if seeds.size == 0:
        return SampledSubgraph(
            node_ids=np.empty(0, dtype=np.int32),
            edge_src=np.empty(0, dtype=np.int64),
            edge_dst=np.empty(0, dtype=np.int64),
            rel_ids=np.empty(0, dtype=np.int64),
            edge_timestamp=np.empty(0, dtype=np.int64),
            target_local_idx=np.empty(0, dtype=np.int64),
            node_subgraph_id=np.empty(0, dtype=np.int64),
            edge_subgraph_id=np.empty(0, dtype=np.int64),
            node_hop_depth=np.empty(0, dtype=np.int64),
            edge_hop_depth=np.empty(0, dtype=np.int64),
        )

    node_parts: list[np.ndarray] = []
    edge_src_parts: list[np.ndarray] = []
    edge_dst_parts: list[np.ndarray] = []
    rel_parts: list[np.ndarray] = []
    edge_time_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    node_group_parts: list[np.ndarray] = []
    edge_group_parts: list[np.ndarray] = []
    node_hop_parts: list[np.ndarray] = []
    edge_hop_parts: list[np.ndarray] = []
    node_offset = 0

    for subgraph_id, seed in enumerate(seeds.tolist()):
        subgraph = _sample_single_seed_subgraph(
            graph=graph,
            seed=seed,
            fanouts=fanouts,
            rng=rng,
            snapshot_end=snapshot_end,
            relation_weight=relation_weight,
            node_profile=node_profile,
            sampler=sampler,
            recent_window=recent_window,
            recent_ratio=recent_ratio,
            consistency_temperature=consistency_temperature,
            training=training,
        )
        node_parts.append(subgraph.node_ids)
        edge_src_parts.append(subgraph.edge_src + node_offset)
        edge_dst_parts.append(subgraph.edge_dst + node_offset)
        rel_parts.append(subgraph.rel_ids)
        edge_time_parts.append(subgraph.edge_timestamp)
        target_parts.append(subgraph.target_local_idx + node_offset)
        node_group_parts.append(
            np.full(subgraph.node_ids.shape[0], subgraph_id, dtype=np.int64)
        )
        edge_group_parts.append(
            np.full(subgraph.edge_src.shape[0], subgraph_id, dtype=np.int64)
        )
        node_hop_parts.append(
            np.asarray(subgraph.node_hop_depth, dtype=np.int64)
            if subgraph.node_hop_depth is not None
            else np.zeros(subgraph.node_ids.shape[0], dtype=np.int64)
        )
        edge_hop_parts.append(
            np.asarray(subgraph.edge_hop_depth, dtype=np.int64)
            if subgraph.edge_hop_depth is not None
            else np.zeros(subgraph.edge_src.shape[0], dtype=np.int64)
        )
        node_offset += int(subgraph.node_ids.shape[0])

    return SampledSubgraph(
        node_ids=np.concatenate(node_parts).astype(np.int32, copy=False),
        edge_src=np.concatenate(edge_src_parts).astype(np.int64, copy=False),
        edge_dst=np.concatenate(edge_dst_parts).astype(np.int64, copy=False),
        rel_ids=np.concatenate(rel_parts).astype(np.int64, copy=False),
        edge_timestamp=np.concatenate(edge_time_parts).astype(np.int64, copy=False),
        target_local_idx=np.concatenate(target_parts).astype(np.int64, copy=False),
        node_subgraph_id=np.concatenate(node_group_parts).astype(np.int64, copy=False),
        edge_subgraph_id=np.concatenate(edge_group_parts).astype(np.int64, copy=False),
        node_hop_depth=np.concatenate(node_hop_parts).astype(np.int64, copy=False),
        edge_hop_depth=np.concatenate(edge_hop_parts).astype(np.int64, copy=False),
    )


class TimeEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, relative_time: torch.Tensor) -> torch.Tensor:
        return self.net(relative_time)


class TargetTimeDriftExpertAdapter(nn.Module):
    """Drift-aware temporal expert router for target-node embedding calibration."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        context_input_dim: int,
        dropout: float,
        num_experts: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_experts = max(int(num_experts), 2)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        router_input_dim = embedding_dim * 4
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(router_input_dim, embedding_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim, embedding_dim * 2),
                )
                for _ in range(self.num_experts)
            ]
        )
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(
        self,
        *,
        base_embedding: torch.Tensor,
        target_time_position: torch.Tensor,
        target_features: torch.Tensor,
        strength: float,
        entropy_floor: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        time_embedding = self.time_encoder(target_time_position)
        context_embedding = self.context_encoder(target_features)
        drift_signal = torch.abs(base_embedding - context_embedding)
        router_input = torch.cat(
            [base_embedding, time_embedding, context_embedding, drift_signal],
            dim=-1,
        )
        route_weight = torch.softmax(self.router(router_input), dim=-1)
        expert_params = torch.stack(
            [expert(router_input) for expert in self.experts],
            dim=1,
        )
        scale_raw, bias_raw = torch.chunk(expert_params, 2, dim=-1)
        scale = torch.sum(route_weight.unsqueeze(-1) * torch.tanh(scale_raw), dim=1)
        bias = torch.sum(route_weight.unsqueeze(-1) * torch.tanh(bias_raw), dim=1)
        route_entropy = -torch.sum(
            route_weight * torch.log(route_weight.clamp_min(1e-8)),
            dim=-1,
        )
        normalized_entropy = (
            route_entropy / max(math.log(float(self.num_experts)), 1e-6)
        ).clamp_(0.0, 1.0)
        entropy_gate = normalized_entropy
        delta = torch.tanh(scale) * base_embedding + bias
        adapted_embedding = base_embedding + float(strength) * entropy_gate.unsqueeze(-1) * delta
        entropy_penalty = F.relu(float(entropy_floor) - normalized_entropy).mean()
        diagnostics = {
            "time_expert_entropy": float(route_entropy.mean().detach().item()),
            "time_expert_entropy_norm": float(normalized_entropy.mean().detach().item()),
            "time_expert_entropy_gate": float(entropy_gate.mean().detach().item()),
            "time_expert_entropy_penalty": float(entropy_penalty.detach().item()),
            "time_expert_max_weight": float(route_weight.max(dim=-1).values.mean().detach().item()),
            "time_expert_delta_norm": float(
                (adapted_embedding - base_embedding).norm(dim=-1).mean().detach().item()
            ),
            "time_expert_context_drift": float(drift_signal.mean().detach().item()),
            "time_expert_time_emb_norm": float(time_embedding.norm(dim=-1).mean().detach().item()),
        }
        return adapted_embedding, diagnostics, entropy_penalty


def _compute_grad_norm(parameters: Any) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().norm(2).item())
        total_sq += grad_norm * grad_norm
    return math.sqrt(total_sq)


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None,
    gamma: float,
    alpha: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal_factor = (1.0 - pt).pow(max(float(gamma), 0.0))
    loss = bce * focal_factor
    if alpha >= 0.0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = loss * alpha_t
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def _pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    pos_logits = logits[targets > 0.5]
    neg_logits = logits[targets <= 0.5]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return logits.new_tensor(0.0)
    margin_term = float(margin) - (pos_logits[:, None] - neg_logits[None, :])
    return F.softplus(margin_term).mean()


def _dirichlet_energy(
    x: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
) -> torch.Tensor:
    if edge_src.numel() == 0:
        return x.new_tensor(0.0)
    delta = x[edge_src] - x[edge_dst]
    return (delta.pow(2).sum(dim=-1)).mean()


def _pool_mean_max(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.dim() != 2:
        raise ValueError("Expected a 2D tensor for pooling.")
    pooled_mean = values.new_zeros((num_groups, values.shape[1]))
    pooled_max = values.new_zeros((num_groups, values.shape[1]))
    if num_groups == 0 or values.shape[0] == 0:
        return pooled_mean, pooled_max

    counts = values.new_zeros((num_groups, 1))
    pooled_mean.index_add_(0, group_ids, values)
    counts.index_add_(
        0,
        group_ids,
        torch.ones((group_ids.shape[0], 1), device=values.device, dtype=values.dtype),
    )
    pooled_mean = pooled_mean / counts.clamp_min(1.0)

    if hasattr(pooled_max, "scatter_reduce_"):
        pooled_max.fill_(float("-inf"))
        pooled_max.scatter_reduce_(
            0,
            group_ids.view(-1, 1).expand(-1, values.shape[1]),
            values,
            reduce="amax",
            include_self=True,
        )
        pooled_max = torch.where(
            torch.isfinite(pooled_max),
            pooled_max,
            torch.zeros_like(pooled_max),
        )
    else:
        for group_idx in range(num_groups):
            mask = group_ids == group_idx
            if torch.any(mask):
                pooled_max[group_idx] = values[mask].max(dim=0).values
    return pooled_mean, pooled_max


class RelationSAGELayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        rel_dim: int,
        time_dim: int = 0,
    ) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, rel_dim)
        msg_in_dim = in_dim + rel_dim + time_dim
        self.msg_linear = nn.Linear(msg_in_dim, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(out_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        time_feature: torch.Tensor | None = None,
        time_weight: torch.Tensor | None = None,
        message_node_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_src.numel() == 0:
            return F.relu(self.self_linear(x))
        relation = self.relation_embedding(rel_ids)
        msg_parts = [x[edge_src], relation]
        if time_feature is not None:
            msg_parts.append(time_feature)
        msg = self.msg_linear(torch.cat(msg_parts, dim=-1))
        if time_weight is not None:
            msg = msg * time_weight
        if message_node_scale is not None:
            edge_scale = 0.5 * (message_node_scale[edge_src] + message_node_scale[edge_dst])
            msg = msg * edge_scale
        agg = x.new_zeros((x.shape[0], msg.shape[1]))
        agg.index_add_(0, edge_dst, msg)
        deg = x.new_zeros((x.shape[0], 1))
        deg.index_add_(
            0,
            edge_dst,
            torch.ones((edge_dst.shape[0], 1), device=x.device, dtype=x.dtype),
        )
        agg = agg / deg.clamp_min(1.0)
        out = self.self_linear(x) + self.neigh_linear(agg)
        return F.relu(out)


class RelationGraphSAGENetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_relations: int,
        rel_dim: int,
        dropout: float,
        temporal: bool,
        model_config: GraphModelConfig,
        aggregator_type: str = "sage",
    ) -> None:
        super().__init__()
        self.temporal = temporal
        self.model_config = model_config
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_edge_types = max(num_relations // 2, 1)
        self.aggregator_type = aggregator_type
        self.use_legacy = aggregator_type == "sage" and model_config.use_legacy_path()
        self.target_context_fusion = str(model_config.target_context_fusion)
        self.target_time_adapter_type = str(model_config.target_time_adapter_type)
        self.internal_risk_fusion = str(model_config.internal_risk_fusion)
        self.internal_risk_residual_scale = float(model_config.internal_risk_residual_scale)
        self.message_risk_strength = float(model_config.message_risk_strength)
        self.message_risk_feature_slice: tuple[int, int] | None = None
        self.target_context_input_dim = (
            int(model_config.target_context_input_dim)
            if int(model_config.target_context_input_dim) > 0
            else int(input_dim)
        )
        self.target_context_feature_gate_strength = float(model_config.target_context_feature_gate_strength)
        self.cold_start_residual_strength = float(model_config.cold_start_residual_strength)
        target_context_gate_hidden_dim = int(
            min(hidden_dim, max(32, self.target_context_input_dim // 2))
        )
        self.target_context_feature_gate = (
            nn.Sequential(
                nn.Linear(self.target_context_input_dim + 1, target_context_gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(target_context_gate_hidden_dim, self.target_context_input_dim),
            )
            if self.target_context_feature_gate_strength > 0.0 and self.target_context_input_dim > 0
            else None
        )
        self.primary_num_classes = max(int(model_config.primary_multiclass_num_classes), 0)
        self.aux_num_classes = max(int(model_config.aux_multiclass_num_classes), 0)
        primary_output_dim = self.primary_num_classes if self.primary_num_classes >= 3 else 1
        cold_start_context_dim = max(int(self.target_context_input_dim), 0)
        cold_start_feature_input_dim = int(input_dim + cold_start_context_dim)
        cold_start_gate_hidden_dim = int(min(hidden_dim, 64))
        self.cold_start_gate = (
            nn.Sequential(
                nn.Linear(3, cold_start_gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cold_start_gate_hidden_dim, 1),
            )
            if self.cold_start_residual_strength > 0.0
            else None
        )
        self.cold_start_feature_head = (
            nn.Sequential(
                nn.LayerNorm(cold_start_feature_input_dim),
                nn.Linear(cold_start_feature_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, primary_output_dim),
            )
            if self.cold_start_residual_strength > 0.0
            else None
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.time_encoder = TimeEncoder(rel_dim) if temporal else None
        self.internal_risk_encoder = (
            TRGTInternalRiskEncoder(
                hidden_dim=hidden_dim,
                num_edge_types=self.num_edge_types,
                dropout=dropout,
                short_time_scale=float(model_config.internal_risk_short_time_scale),
                long_time_scale=float(model_config.internal_risk_long_time_scale),
            )
            if self.internal_risk_fusion == "residual"
            else None
        )
        self.internal_risk_gate = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            if self.internal_risk_encoder is not None
            else None
        )

        if self.use_legacy:
            time_dim = rel_dim if temporal else 0
            self.layers = nn.ModuleList(
                [
                    RelationSAGELayer(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        num_relations=num_relations,
                        rel_dim=rel_dim,
                        time_dim=time_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, primary_output_dim),
            )
            self.aux_classifier = (
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.aux_num_classes),
                )
                if float(model_config.aux_multiclass_loss_weight) > 0.0 and self.aux_num_classes >= 3
                else None
            )
            self.rel_embedding = None
            self.stats_proj = None
            self.subgraph_classifier = None
            self.context_fusion_head = (
                TargetContextFusionHead(
                    input_dim=self.target_context_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=primary_output_dim,
                    dropout=dropout,
                    mode=self.target_context_fusion,
                    adaptive_window_strength=float(model_config.atm_gate_strength),
                    context_residual_scale=float(model_config.context_residual_scale),
                    context_residual_clip=float(model_config.context_residual_clip),
                    context_residual_budget=float(model_config.context_residual_budget),
                )
                if self.target_context_fusion in {
                    "gate",
                    "concat",
                    "logit_residual",
                    "atm_residual",
                    "drift_residual",
                    "drift_mix",
                    "drift_uncertainty_mix",
                    "risk_drift_residual",
                }
                else None
            )
            self.target_time_adapter = (
                TargetTimeDriftExpertAdapter(
                    embedding_dim=hidden_dim,
                    context_input_dim=self.target_context_input_dim,
                    dropout=dropout,
                    num_experts=int(model_config.target_time_adapter_num_experts),
                )
                if (
                    float(model_config.target_time_adapter_strength) > 0.0
                    and self.target_time_adapter_type == "drift_expert"
                )
                else (
                    nn.Sequential(
                        nn.Linear(1, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim * 2),
                    )
                    if float(model_config.target_time_adapter_strength) > 0.0
                    else None
                )
            )
            return

        self.rel_embedding = nn.Embedding(num_relations, rel_dim)
        edge_dim = rel_dim + (rel_dim if temporal else 0)
        block_cls = TRGTMeanRelationBlock if aggregator_type == "sage" else TRGTTemporalRelationAttentionBlock
        block_kwargs = dict(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            dropout=dropout,
            norm=model_config.norm,
            residual=model_config.residual,
            ffn=model_config.ffn,
            edge_encoder=model_config.edge_encoder,
        )
        if aggregator_type != "sage":
            block_kwargs["num_heads"] = int(model_config.attention_num_heads)
            block_kwargs["attention_logit_scale"] = float(model_config.attention_logit_scale)
        self.layers = nn.ModuleList([block_cls(**block_kwargs) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, primary_output_dim),
        )
        aux_input_dim = hidden_dim
        if model_config.subgraph_head == "meanmax":
            self.stats_proj = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.subgraph_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 6, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, primary_output_dim),
            )
            aux_input_dim = hidden_dim * 6
        else:
            self.stats_proj = None
            self.subgraph_classifier = None
        self.aux_classifier = (
            nn.Sequential(
                nn.Linear(aux_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.aux_num_classes),
            )
            if float(model_config.aux_multiclass_loss_weight) > 0.0 and self.aux_num_classes >= 3
            else None
        )
        self.context_fusion_head = (
            TargetContextFusionHead(
                input_dim=self.target_context_input_dim,
                hidden_dim=hidden_dim,
                output_dim=primary_output_dim,
                dropout=dropout,
                mode=self.target_context_fusion,
                adaptive_window_strength=float(model_config.atm_gate_strength),
                context_residual_scale=float(model_config.context_residual_scale),
                context_residual_clip=float(model_config.context_residual_clip),
                context_residual_budget=float(model_config.context_residual_budget),
            )
            if self.target_context_fusion in {
                "gate",
                "concat",
                "logit_residual",
                "atm_residual",
                "drift_residual",
                "drift_mix",
                "drift_uncertainty_mix",
                "risk_drift_residual",
            }
            else None
        )
        self.target_time_adapter = (
            TargetTimeDriftExpertAdapter(
                embedding_dim=hidden_dim,
                context_input_dim=self.target_context_input_dim,
                dropout=dropout,
                num_experts=int(model_config.target_time_adapter_num_experts),
            )
            if (
                float(model_config.target_time_adapter_strength) > 0.0
                and self.target_time_adapter_type == "drift_expert"
            )
            else (
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                )
                if float(model_config.target_time_adapter_strength) > 0.0
                else None
            )
        )

    def _build_edge_embedding(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        time_feature = None
        if self.temporal and edge_relative_time is not None and edge_relative_time.numel():
            time_feature = self.time_encoder(edge_relative_time)

        if self.use_legacy:
            return None, time_feature

        if rel_ids.numel() == 0:
            relation = x.new_zeros((0, self.rel_embedding.embedding_dim))
        else:
            relation = self.rel_embedding(rel_ids)

        if self.temporal:
            if time_feature is None:
                time_feature = relation.new_zeros((relation.shape[0], relation.shape[1]))
            edge_emb = torch.cat([relation, time_feature], dim=-1)
        else:
            edge_emb = relation
        return edge_emb, time_feature

    def set_message_risk_feature_slice(
        self,
        feature_slice: tuple[int, int] | None,
    ) -> None:
        self.message_risk_feature_slice = feature_slice

    def _compute_message_node_scale(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float] | None]:
        if self.message_risk_strength <= 0.0 or self.message_risk_feature_slice is None:
            return None, None
        start, end = self.message_risk_feature_slice
        if end <= start or x.shape[1] < end:
            return None, None

        concentration_block = x[:, start:end]
        if concentration_block.shape[1] == 0:
            return None, None

        if concentration_block.shape[1] == 16:
            bg_ratio_signal = concentration_block[:, 3:6].mean(dim=-1)
            top_share_signal = concentration_block[:, 7:10].mean(dim=-1)
            entropy_signal = concentration_block[:, 10:13].mean(dim=-1)
            activity_peak_signal = concentration_block[:, 15]
            risk_signal = (
                0.35 * bg_ratio_signal
                + 0.45 * top_share_signal
                - 0.35 * entropy_signal
                + 0.15 * activity_peak_signal
            )
        elif concentration_block.shape[1] >= 6 and concentration_block.shape[1] % 6 == 0:
            concentration_view = concentration_block.reshape(concentration_block.shape[0], -1, 6)
            top_share_signal = concentration_view[:, :, :3].mean(dim=(1, 2))
            entropy_signal = concentration_view[:, :, 3:].mean(dim=(1, 2))
            risk_signal = top_share_signal - entropy_signal
        else:
            risk_signal = concentration_block.mean(dim=-1)

        risk_score = torch.sigmoid(risk_signal)
        node_scale = (1.0 + self.message_risk_strength * (risk_score - 0.5)).unsqueeze(-1)
        diagnostics = {
            "message_risk_score_mean": float(risk_score.mean().detach().item()),
            "message_risk_scale_mean": float(node_scale.mean().detach().item()),
        }
        return node_scale, diagnostics

    def _subgraph_stats(
        self,
        node_repr: torch.Tensor,
        target_local_idx: torch.Tensor,
        node_subgraph_id: torch.Tensor,
        edge_subgraph_id: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
    ) -> torch.Tensor:
        num_subgraphs = int(target_local_idx.shape[0])
        stats = node_repr.new_zeros((num_subgraphs, 6))
        stats[:, 0].index_add_(
            0,
            node_subgraph_id,
            torch.ones_like(node_subgraph_id, dtype=node_repr.dtype),
        )
        stats[:, 1].index_add_(
            0,
            edge_subgraph_id,
            torch.ones_like(edge_subgraph_id, dtype=node_repr.dtype),
        )
        if edge_relative_time is not None and edge_relative_time.numel():
            stats[:, 3].index_add_(0, edge_subgraph_id, edge_relative_time.view(-1))
            stats[:, 3] = stats[:, 3] / stats[:, 1].clamp_min(1.0)

        if rel_ids.numel():
            unique_rel_pairs = torch.unique(edge_subgraph_id * self.num_relations + rel_ids)
            unique_rel_subgraph_ids = torch.floor_divide(unique_rel_pairs, self.num_relations)
            stats[:, 2].index_add_(
                0,
                unique_rel_subgraph_ids,
                torch.ones_like(unique_rel_subgraph_ids, dtype=node_repr.dtype),
            )

            target_edge_mask = edge_dst == target_local_idx[edge_subgraph_id]
            if torch.any(target_edge_mask):
                target_subgraph_ids = edge_subgraph_id[target_edge_mask]
                target_rel_ids = rel_ids[target_edge_mask]
                inbound_mask = target_rel_ids < self.num_edge_types
                outbound_mask = ~inbound_mask
                if torch.any(inbound_mask):
                    stats[:, 4].index_add_(
                        0,
                        target_subgraph_ids[inbound_mask],
                        torch.ones_like(target_subgraph_ids[inbound_mask], dtype=node_repr.dtype),
                    )
                if torch.any(outbound_mask):
                    stats[:, 5].index_add_(
                        0,
                        target_subgraph_ids[outbound_mask],
                        torch.ones_like(target_subgraph_ids[outbound_mask], dtype=node_repr.dtype),
                    )

        stats[:, 0] = torch.log1p(stats[:, 0])
        stats[:, 1] = torch.log1p(stats[:, 1])
        stats[:, 2] = torch.log1p(stats[:, 2])
        stats[:, 4] = torch.log1p(stats[:, 4])
        stats[:, 5] = torch.log1p(stats[:, 5])
        return stats

    def _compose_subgraph_target_repr(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        target_local_idx: torch.Tensor,
        node_subgraph_id: torch.Tensor | None,
        edge_subgraph_id: torch.Tensor | None,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
    ) -> torch.Tensor:
        if node_subgraph_id is None or edge_subgraph_id is None:
            raise ValueError("subgraph_head=meanmax requires subgraph ids.")

        num_subgraphs = int(target_local_idx.shape[0])
        target_repr = node_repr[target_local_idx]
        target_mask = torch.zeros(
            (node_repr.shape[0],),
            device=node_repr.device,
            dtype=torch.bool,
        )
        target_mask[target_local_idx] = True
        context_mask = ~target_mask
        ctx_mean, ctx_max = _pool_mean_max(
            node_repr[context_mask],
            node_subgraph_id[context_mask],
            num_subgraphs,
        )
        edge_mean, edge_max = _pool_mean_max(
            edge_repr,
            edge_subgraph_id,
            num_subgraphs,
        )
        stats = self._subgraph_stats(
            node_repr=node_repr,
            target_local_idx=target_local_idx,
            node_subgraph_id=node_subgraph_id,
            edge_subgraph_id=edge_subgraph_id,
            edge_dst=edge_dst,
            rel_ids=rel_ids,
            edge_relative_time=edge_relative_time,
        )
        fused = torch.cat(
            [
                target_repr,
                ctx_mean,
                ctx_max,
                edge_mean,
                edge_max,
                self.stats_proj(stats),
            ],
            dim=-1,
        )
        return fused

    def _apply_classifier_with_embedding(
        self,
        representation: torch.Tensor,
        classifier: nn.Sequential,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = classifier[0](representation)
        hidden = classifier[1](hidden)
        hidden = classifier[2](hidden)
        logits = classifier[3](hidden)
        if logits.dim() == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits, hidden

    def _apply_aux_classifier(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.aux_classifier is None:
            return None
        return self.aux_classifier(representation)

    def _gate_target_context_features(
        self,
        target_features: torch.Tensor | None,
        target_time_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, float] | None]:
        if (
            self.target_context_feature_gate is None
            or target_features is None
            or not target_features.numel()
        ):
            return target_features, None
        if target_time_position is None or target_time_position.shape[0] != target_features.shape[0]:
            target_time_position = target_features.new_zeros((target_features.shape[0], 1))
        gate_input = torch.cat([target_features, target_time_position], dim=-1)
        gate = torch.sigmoid(self.target_context_feature_gate(gate_input))
        strength = float(np.clip(self.target_context_feature_gate_strength, 0.0, 1.0))
        gated_features = target_features * ((1.0 - strength) + strength * gate)
        diagnostics = {
            "target_context_feature_gate_mean": float(gate.mean().detach().item()),
            "target_context_feature_gate_min": float(gate.min().detach().item()),
            "target_context_feature_gate_max": float(gate.max().detach().item()),
        }
        return gated_features, diagnostics

    def _build_target_support_count(
        self,
        target_local_idx: torch.Tensor,
        edge_subgraph_id: torch.Tensor | None,
        edge_src: torch.Tensor,
    ) -> torch.Tensor:
        num_targets = int(target_local_idx.shape[0])
        support_count = edge_src.new_zeros((num_targets,), dtype=torch.float32)
        if num_targets == 0 or edge_src.numel() == 0:
            return support_count
        if edge_subgraph_id is not None and edge_subgraph_id.numel() == edge_src.numel():
            support_count.index_add_(
                0,
                edge_subgraph_id,
                torch.ones_like(edge_subgraph_id, dtype=torch.float32),
            )
            return support_count
        support_count.fill_(float(edge_src.numel()))
        return support_count

    def _apply_cold_start_residual(
        self,
        *,
        logits: torch.Tensor,
        input_features: torch.Tensor,
        target_context_features: torch.Tensor | None,
        target_local_idx: torch.Tensor,
        edge_src: torch.Tensor,
        edge_subgraph_id: torch.Tensor | None,
        target_time_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        if (
            self.cold_start_feature_head is None
            or self.cold_start_gate is None
            or self.cold_start_residual_strength <= 0.0
            or target_local_idx.numel() == 0
        ):
            return logits, None

        target_features = input_features[target_local_idx]
        if (
            target_context_features is not None
            and target_context_features.numel()
            and target_context_features.shape[0] == target_features.shape[0]
        ):
            cold_context_features = target_context_features
        else:
            cold_context_features = target_features.new_zeros(
                (target_features.shape[0], max(int(self.target_context_input_dim), 0))
            )
        cold_input = torch.cat([target_features, cold_context_features], dim=-1)
        cold_logits = self.cold_start_feature_head(cold_input)
        if cold_logits.dim() == 2 and cold_logits.shape[-1] == 1:
            cold_logits = cold_logits.squeeze(-1)

        support_count = self._build_target_support_count(
            target_local_idx=target_local_idx,
            edge_subgraph_id=edge_subgraph_id,
            edge_src=edge_src,
        )
        support_prior = (1.0 / (1.0 + support_count)).unsqueeze(-1)
        if target_time_position is None or target_time_position.shape[0] != target_features.shape[0]:
            target_time_position = target_features.new_zeros((target_features.shape[0], 1))
        context_signal = torch.log1p(cold_context_features.abs().mean(dim=-1, keepdim=True))
        gate_input = torch.cat(
            [torch.log1p(support_count).unsqueeze(-1), target_time_position, context_signal],
            dim=-1,
        )
        learned_gate = torch.sigmoid(self.cold_start_gate(gate_input))
        blend_gate = float(np.clip(self.cold_start_residual_strength, 0.0, 1.0)) * support_prior * learned_gate
        if logits.dim() == 1:
            logits = logits * (1.0 - blend_gate.squeeze(-1)) + cold_logits * blend_gate.squeeze(-1)
        else:
            logits = logits * (1.0 - blend_gate) + cold_logits * blend_gate

        diagnostics = {
            "cold_start_support_count_mean": float(support_count.mean().detach().item()),
            "cold_start_zero_support_ratio": float((support_count <= 0.0).float().mean().detach().item()),
            "cold_start_gate_mean": float(blend_gate.mean().detach().item()),
            "cold_start_gate_max": float(blend_gate.max().detach().item()),
            "cold_start_gate_min": float(blend_gate.min().detach().item()),
            "cold_start_context_signal_mean": float(context_signal.mean().detach().item()),
        }
        return logits, diagnostics

    def forward_output(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
        target_local_idx: torch.Tensor,
        target_context_features: torch.Tensor | None = None,
        target_embedding_shift: torch.Tensor | None = None,
        target_time_position: torch.Tensor | None = None,
        node_subgraph_id: torch.Tensor | None = None,
        edge_subgraph_id: torch.Tensor | None = None,
        node_hop_depth: torch.Tensor | None = None,
        include_diagnostics: bool = False,
        include_embedding: bool = False,
        include_aux: bool = False,
    ) -> GraphForwardOutput:
        h = self.input_proj(x)
        h = F.relu(h) if self.use_legacy else F.gelu(h)
        h = self.input_dropout(h)

        edge_emb, time_feature = self._build_edge_embedding(x, rel_ids, edge_relative_time)
        time_weight = None
        if (
            self.temporal
            and edge_relative_time is not None
            and edge_relative_time.numel()
            and float(self.model_config.time_decay_strength) > 0.0
        ):
            time_weight = torch.exp(
                -float(self.model_config.time_decay_strength) * edge_relative_time
            )
        message_node_scale, message_risk_diagnostics = self._compute_message_node_scale(x)
        last_edge_repr = h.new_zeros((0, self.hidden_dim))
        layer_outputs: list[torch.Tensor] = []

        if self.use_legacy:
            for layer in self.layers:
                h = layer(
                    h,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    rel_ids=rel_ids,
                    time_feature=time_feature,
                    time_weight=time_weight,
                    message_node_scale=message_node_scale,
                )
                h = self.input_dropout(h)
                layer_outputs.append(h)
            node_repr = layer_outputs[-1] if layer_outputs else h
            target_representation = node_repr[target_local_idx]
            active_classifier = self.classifier
            logits, embedding = self._apply_classifier_with_embedding(
                target_representation,
                active_classifier,
            )
        else:
            for layer in self.layers:
                h, last_edge_repr = layer(
                    h,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_emb=edge_emb,
                    time_weight=time_weight,
                    message_node_scale=message_node_scale,
                )
                layer_outputs.append(h)
            node_repr = (
                torch.stack(layer_outputs, dim=0).sum(dim=0)
                if self.model_config.jk == "sum" and layer_outputs
                else (layer_outputs[-1] if layer_outputs else h)
            )
            if self.model_config.subgraph_head == "meanmax":
                target_representation = self._compose_subgraph_target_repr(
                    node_repr=node_repr,
                    edge_repr=last_edge_repr,
                    target_local_idx=target_local_idx,
                    node_subgraph_id=node_subgraph_id,
                    edge_subgraph_id=edge_subgraph_id,
                    edge_dst=edge_dst,
                    rel_ids=rel_ids,
                    edge_relative_time=edge_relative_time,
                )
                active_classifier = self.subgraph_classifier
                logits, embedding = self._apply_classifier_with_embedding(
                    target_representation,
                    active_classifier,
                )
            else:
                target_representation = node_repr[target_local_idx]
                active_classifier = self.classifier
                logits, embedding = self._apply_classifier_with_embedding(
                    target_representation,
                    active_classifier,
                )

        fusion_diagnostics = None
        feature_gate_diagnostics = None
        adapter_diagnostics = None
        internal_risk_diagnostics = None
        cold_start_diagnostics = None
        context_regularization_loss = embedding.new_tensor(0.0)
        adapter_regularization_loss = embedding.new_tensor(0.0)
        gated_target_context_features, feature_gate_diagnostics = self._gate_target_context_features(
            target_context_features,
            target_time_position,
        )
        if self.internal_risk_encoder is not None:
            if self.internal_risk_gate is None:
                raise RuntimeError("internal risk fusion requires an initialized gate.")
            risk_embedding, internal_risk_diagnostics = self.internal_risk_encoder(
                node_repr=node_repr,
                edge_repr=last_edge_repr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                rel_ids=rel_ids,
                edge_relative_time=edge_relative_time,
                target_local_idx=target_local_idx,
                node_subgraph_id=node_subgraph_id,
                edge_subgraph_id=edge_subgraph_id,
                node_hop_depth=node_hop_depth,
            )
            risk_gate = self.internal_risk_gate(torch.cat([embedding, risk_embedding], dim=-1))
            risk_delta = self.internal_risk_residual_scale * risk_gate * risk_embedding
            embedding = embedding + risk_delta
            logits = active_classifier[3](embedding)
            if logits.dim() == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            if internal_risk_diagnostics is not None:
                internal_risk_diagnostics = dict(internal_risk_diagnostics)
                internal_risk_diagnostics["internal_risk_gate_mean"] = float(
                    risk_gate.mean().detach().item()
                )
                internal_risk_diagnostics["internal_risk_delta_norm"] = float(
                    risk_delta.norm(dim=-1).mean().detach().item()
                )
        if target_embedding_shift is not None and target_embedding_shift.numel():
            embedding_norm = embedding.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            embedding = embedding + embedding_norm * target_embedding_shift
            logits = active_classifier[3](embedding)
            if logits.dim() == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
        if self.target_time_adapter is not None and target_time_position is not None:
            if self.target_time_adapter_type == "drift_expert":
                adapter_features = (
                    gated_target_context_features
                    if gated_target_context_features is not None
                    else x[target_local_idx]
                )
                embedding, adapter_diagnostics, adapter_regularization_loss = self.target_time_adapter(
                    base_embedding=embedding,
                    target_time_position=target_time_position,
                    target_features=adapter_features,
                    strength=float(self.model_config.target_time_adapter_strength),
                    entropy_floor=float(self.model_config.target_time_expert_entropy_floor),
                )
            else:
                time_params = self.target_time_adapter(target_time_position)
                time_scale, time_bias = torch.chunk(time_params, 2, dim=-1)
                strength = float(self.model_config.target_time_adapter_strength)
                embedding = embedding * (1.0 + strength * torch.tanh(time_scale)) + strength * time_bias
            logits = active_classifier[3](embedding)
            if logits.dim() == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
        if self.context_fusion_head is not None:
            fusion_features = (
                gated_target_context_features
                if gated_target_context_features is not None
                else x[target_local_idx]
            )
            logits, embedding, fusion_diagnostics, context_regularization_loss = self.context_fusion_head(
                base_embedding=embedding,
                base_logits=logits,
                target_features=fusion_features,
                target_time_position=target_time_position,
            )
        logits, cold_start_diagnostics = self._apply_cold_start_residual(
            logits=logits,
            input_features=x,
            target_context_features=gated_target_context_features,
            target_local_idx=target_local_idx,
            edge_src=edge_src,
            edge_subgraph_id=edge_subgraph_id,
            target_time_position=target_time_position,
        )

        diagnostics = None
        if include_diagnostics:
            diagnostics = {
                "emb_norm": float(node_repr.norm(dim=-1).mean().detach().item()),
                "dirichlet_energy": float(
                    _dirichlet_energy(node_repr, edge_src=edge_src, edge_dst=edge_dst)
                    .detach()
                    .item()
                ),
            }
            if fusion_diagnostics is not None:
                diagnostics.update(fusion_diagnostics)
            if target_embedding_shift is not None and target_embedding_shift.numel():
                diagnostics["normal_shift_norm"] = float(
                    target_embedding_shift.norm(dim=-1).mean().detach().item()
                )
            if self.target_time_adapter is not None and target_time_position is not None:
                diagnostics["target_time_position_mean"] = float(
                    target_time_position.mean().detach().item()
                )
            if feature_gate_diagnostics is not None:
                diagnostics.update(feature_gate_diagnostics)
            if adapter_diagnostics is not None:
                diagnostics.update(adapter_diagnostics)
            if internal_risk_diagnostics is not None:
                diagnostics.update(internal_risk_diagnostics)
            if message_risk_diagnostics is not None:
                diagnostics.update(message_risk_diagnostics)
            if cold_start_diagnostics is not None:
                diagnostics.update(cold_start_diagnostics)
        aux_logits = self._apply_aux_classifier(target_representation) if include_aux else None
        return GraphForwardOutput(
            logits=logits,
            diagnostics=diagnostics,
            embedding=embedding if include_embedding else None,
            aux_logits=aux_logits,
            context_regularization_loss=context_regularization_loss,
            adapter_regularization_loss=adapter_regularization_loss,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
        target_local_idx: torch.Tensor,
        target_context_features: torch.Tensor | None = None,
        node_subgraph_id: torch.Tensor | None = None,
        edge_subgraph_id: torch.Tensor | None = None,
        node_hop_depth: torch.Tensor | None = None,
        return_details: bool = False,
        return_embedding: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, dict[str, float]]
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, float], torch.Tensor]
    ):
        output = self.forward_output(
            x=x,
            edge_src=edge_src,
            edge_dst=edge_dst,
            rel_ids=rel_ids,
            edge_relative_time=edge_relative_time,
            target_local_idx=target_local_idx,
            target_context_features=target_context_features,
            node_subgraph_id=node_subgraph_id,
            edge_subgraph_id=edge_subgraph_id,
            node_hop_depth=node_hop_depth,
            include_diagnostics=return_details,
            include_embedding=return_embedding,
            include_aux=False,
        )
        if return_details and return_embedding:
            return output.logits, output.diagnostics or {}, output.embedding
        if return_details:
            return output.logits, output.diagnostics or {}
        if return_embedding:
            return output.logits, output.embedding
        return output.logits


class BaseGraphSAGEExperiment:
    def __init__(
        self,
        model_name: str,
        seed: int,
        input_dim: int,
        num_relations: int,
        max_day: int,
        feature_groups: list[str] | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rel_dim: int = 32,
        fanouts: list[int] | None = None,
        batch_size: int = 1024,
        epochs: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.2,
        device: str | None = None,
        temporal: bool = False,
        aggregator_type: str = "sage",
        graph_config: GraphModelConfig | None = None,
        feature_normalizer_state: HybridFeatureNormalizerState | None = None,
        target_context_input_dim: int | None = None,
        target_context_feature_groups: list[str] | None = None,
        target_context_normalizer_state: HybridFeatureNormalizerState | None = None,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.feature_groups = feature_groups or default_feature_groups(model_name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rel_dim = rel_dim
        self.fanouts = fanouts or [15, 10]
        self.batch_size = batch_size
        self.eval_batch_size = max(batch_size, 2048)
        self.epochs = epochs
        self.max_day = max_day
        self.temporal = temporal
        self.aggregator_type = aggregator_type
        self.device = torch.device(resolve_device(device))
        self.feature_normalizer_state = feature_normalizer_state
        self.target_context_feature_groups = list(target_context_feature_groups or [])
        self.target_context_normalizer_state = target_context_normalizer_state
        self.graph_config = graph_config or GraphModelConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
        )
        if target_context_input_dim is not None:
            self.graph_config = replace(
                self.graph_config,
                target_context_input_dim=int(target_context_input_dim),
            )

        self.network = self._build_network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_relations=num_relations,
            rel_dim=rel_dim,
            dropout=self.graph_config.dropout,
            temporal=temporal,
            model_config=self.graph_config,
            aggregator_type=aggregator_type,
        ).to(self.device)
        self._hard_negative_pools: dict[int, np.ndarray] = {}
        self._hard_negative_pool_stats: dict[int, dict[str, float | int]] = {}
        self._relation_sampling_weight: np.ndarray | None = None
        self._normal_alignment_regularizer: TemporalNormalAlignmentBank | None = None
        self._normal_bucket_discriminator: nn.Module | None = None
        self._context_budget_schedule_state: dict[str, float] = {}
        self._prototype_weight_schedule_state: dict[str, float] = {}

    def _build_network(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_relations: int,
        rel_dim: int,
        dropout: float,
        temporal: bool,
        model_config: GraphModelConfig,
        aggregator_type: str,
    ) -> nn.Module:
        return RelationGraphSAGENetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_relations=num_relations,
            rel_dim=rel_dim,
            dropout=dropout,
            temporal=temporal,
            model_config=model_config,
            aggregator_type=aggregator_type,
        )

    def _resolve_local_feature_group_slice(
        self,
        context: GraphPhaseContext,
        group_name: str,
    ) -> tuple[int, int] | None:
        offset = 0
        for spec in getattr(context.feature_store, "_group_specs", []):
            width = int(spec["end"] - spec["start"])
            if str(spec.get("group_name", "")) == str(group_name):
                return offset, offset + width
            offset += width
        return None

    def _refresh_message_risk_feature_slice(
        self,
        context: GraphPhaseContext,
    ) -> None:
        feature_slice = None
        if float(self.graph_config.message_risk_strength) > 0.0:
            for group_name in ("temporal_counterparty_concentration", "utpm_temporal_risk"):
                feature_slice = self._resolve_local_feature_group_slice(
                    context=context,
                    group_name=group_name,
                )
                if feature_slice is not None:
                    break
        self.network.set_message_risk_feature_slice(feature_slice)

    def _binary_train_labels(self, labels: np.ndarray) -> np.ndarray:
        labels_arr = np.asarray(labels, dtype=np.int8)
        if not self.graph_config.include_historical_background_negatives:
            return labels_arr.astype(np.int8, copy=False)
        return (labels_arr == 1).astype(np.int8, copy=False)

    def _primary_multiclass_enabled(self) -> bool:
        return int(self.graph_config.primary_multiclass_num_classes) >= 3

    def _aux_multiclass_enabled(self) -> bool:
        return (
            float(self.graph_config.aux_multiclass_loss_weight) > 0.0
            and int(self.graph_config.aux_multiclass_num_classes) >= 3
        )

    def _prototype_enabled(self) -> bool:
        return (
            float(self.graph_config.prototype_loss_weight) > 0.0
            and int(self.graph_config.prototype_multiclass_num_classes) >= 2
        )

    def _pseudo_contrastive_enabled(self) -> bool:
        return float(self.graph_config.pseudo_contrastive_weight) > 0.0

    def _background_aux_only_enabled(self) -> bool:
        return (
            not self._primary_multiclass_enabled()
            and
            bool(self.graph_config.historical_background_aux_only)
            and bool(self.graph_config.include_historical_background_negatives)
        )

    def _map_multiclass_targets(
        self,
        labels: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        labels_arr = np.asarray(labels, dtype=np.int8)
        if num_classes < 2:
            return np.full(labels_arr.shape[0], -1, dtype=np.int64)
        if num_classes == 2:
            mapped = np.full(labels_arr.shape[0], -1, dtype=np.int64)
            mapped[labels_arr == 0] = 0
            mapped[labels_arr == 1] = 1
            return mapped
        if num_classes == 3:
            mapped = np.full(labels_arr.shape[0], -1, dtype=np.int64)
            mapped[labels_arr == 0] = 0
            mapped[labels_arr == 1] = 1
            mapped[np.isin(labels_arr, (2, 3))] = 2
            return mapped
        mapped = np.full(labels_arr.shape[0], -1, dtype=np.int64)
        valid_mask = np.isin(labels_arr, (0, 1, 2, 3))
        mapped[valid_mask] = labels_arr[valid_mask].astype(np.int64, copy=False)
        return mapped

    def _map_primary_targets(self, labels: np.ndarray) -> np.ndarray:
        return self._map_multiclass_targets(
            labels=labels,
            num_classes=int(self.graph_config.primary_multiclass_num_classes),
        )

    def _map_aux_targets(self, labels: np.ndarray) -> np.ndarray:
        return self._map_multiclass_targets(
            labels=labels,
            num_classes=int(self.graph_config.aux_multiclass_num_classes),
        )

    def _map_prototype_targets(self, labels: np.ndarray) -> np.ndarray:
        return self._map_multiclass_targets(
            labels=labels,
            num_classes=int(self.graph_config.prototype_multiclass_num_classes),
        )

    def _compute_multiclass_class_weight(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
        num_classes: int,
    ) -> torch.Tensor | None:
        if num_classes < 2:
            return None
        label_blocks = [np.asarray(context.labels[np.asarray(train_ids, dtype=np.int32)], dtype=np.int8)]
        if (
            self.graph_config.include_historical_background_negatives
            and context.historical_background_ids is not None
            and np.asarray(context.historical_background_ids).size
        ):
            label_blocks.append(
                np.asarray(
                    context.labels[np.asarray(context.historical_background_ids, dtype=np.int32)],
                    dtype=np.int8,
                )
            )
        mapped_targets = self._map_multiclass_targets(
            np.concatenate(label_blocks, axis=0),
            num_classes=num_classes,
        )
        valid_targets = mapped_targets[mapped_targets >= 0]
        if valid_targets.size == 0:
            return None
        counts = np.bincount(valid_targets, minlength=num_classes).astype(np.float32, copy=False)
        present = counts > 0
        if not np.any(present):
            return None
        weights = np.ones(num_classes, dtype=np.float32)
        weights[present] = 1.0 / np.sqrt(counts[present])
        weights[present] = weights[present] / float(np.mean(weights[present], dtype=np.float64))
        return torch.as_tensor(weights, dtype=torch.float32, device=self.device)

    def _compute_primary_class_weight(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
    ) -> torch.Tensor | None:
        if not self._primary_multiclass_enabled():
            return None
        return self._compute_multiclass_class_weight(
            context=context,
            train_ids=train_ids,
            num_classes=int(self.graph_config.primary_multiclass_num_classes),
        )

    def _compute_aux_class_weight(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
    ) -> torch.Tensor | None:
        if not self._aux_multiclass_enabled():
            return None
        return self._compute_multiclass_class_weight(
            context=context,
            train_ids=train_ids,
            num_classes=int(self.graph_config.aux_multiclass_num_classes),
        )

    def _compute_prototype_class_weight(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
    ) -> torch.Tensor | None:
        if not self._prototype_enabled():
            return None
        return self._compute_multiclass_class_weight(
            context=context,
            train_ids=train_ids,
            num_classes=int(self.graph_config.prototype_multiclass_num_classes),
        )

    def _build_prototype_regularizer(
        self,
        context: GraphPhaseContext,
    ) -> PrototypeMemoryBank | None:
        if not self._prototype_enabled():
            return None
        num_buckets = max(len(context.graph_cache.time_windows), 1)
        return PrototypeMemoryBank(
            config=PrototypeMemoryConfig(
                num_classes=int(self.graph_config.prototype_multiclass_num_classes),
                embedding_dim=int(self.hidden_dim),
                temperature=float(self.graph_config.prototype_temperature),
                momentum=float(self.graph_config.prototype_momentum),
                start_epoch=int(self.graph_config.prototype_start_epoch),
                loss_ramp_epochs=int(self.graph_config.prototype_loss_ramp_epochs),
                bucket_mode=str(self.graph_config.prototype_bucket_mode),
                num_buckets=int(num_buckets),
                neighbor_blend=float(self.graph_config.prototype_neighbor_blend),
                global_blend=float(self.graph_config.prototype_global_blend),
                consistency_weight=float(self.graph_config.prototype_consistency_weight),
                separation_weight=float(self.graph_config.prototype_separation_weight),
                separation_margin=float(self.graph_config.prototype_separation_margin),
            ),
            device=self.device,
        )

    def _prototype_time_bucketed(self) -> bool:
        return self._prototype_enabled() and str(self.graph_config.prototype_bucket_mode) == "time_bucket"

    def _rebalance_pseudo_sample_by_time_bucket(
        self,
        *,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        candidates = np.asarray(node_ids, dtype=np.int32)
        if candidates.size <= sample_size:
            return candidates

        bucket_ids = np.asarray(context.graph_cache.node_time_bucket[candidates], dtype=np.int32)
        unique_buckets = np.unique(bucket_ids)
        if unique_buckets.size <= 1:
            choice = rng.choice(candidates.size, size=sample_size, replace=False)
            return np.asarray(candidates[choice], dtype=np.int32, copy=False)

        base_take = max(sample_size // unique_buckets.size, 1)
        selected_parts: list[np.ndarray] = []
        leftover_parts: list[np.ndarray] = []
        for bucket_id in unique_buckets:
            bucket_mask = bucket_ids == int(bucket_id)
            bucket_nodes = np.asarray(candidates[bucket_mask], dtype=np.int32, copy=False)
            if bucket_nodes.size == 0:
                continue
            permutation = rng.permutation(bucket_nodes.size)
            take = min(bucket_nodes.size, base_take)
            if take > 0:
                selected_parts.append(bucket_nodes[permutation[:take]])
            if bucket_nodes.size > take:
                leftover_parts.append(bucket_nodes[permutation[take:]])

        selected = (
            np.concatenate(selected_parts).astype(np.int32, copy=False)
            if selected_parts
            else np.empty(0, dtype=np.int32)
        )
        if selected.size < sample_size and leftover_parts:
            leftover = np.concatenate(leftover_parts).astype(np.int32, copy=False)
            need = sample_size - selected.size
            if leftover.size > need:
                fill_choice = rng.choice(leftover.size, size=need, replace=False)
                leftover = np.asarray(leftover[fill_choice], dtype=np.int32, copy=False)
            selected = np.concatenate([selected, leftover]).astype(np.int32, copy=False)
        if selected.size > sample_size:
            trim_choice = rng.choice(selected.size, size=sample_size, replace=False)
            selected = np.asarray(selected[trim_choice], dtype=np.int32, copy=False)
        if selected.size < sample_size:
            remaining_mask = ~np.isin(candidates, selected, assume_unique=False)
            remaining = np.asarray(candidates[remaining_mask], dtype=np.int32, copy=False)
            if remaining.size:
                need = min(sample_size - selected.size, remaining.size)
                fill_choice = rng.choice(remaining.size, size=need, replace=False)
                selected = np.concatenate([selected, remaining[fill_choice]]).astype(np.int32, copy=False)
        return selected

    def _compute_pseudo_contrastive_loss(
        self,
        *,
        context: GraphPhaseContext,
        test_pool_ids: np.ndarray,
        snapshot_end: int | None,
        epoch: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        disabled = (
            not self._pseudo_contrastive_enabled()
            or test_pool_ids.size == 0
            or self._primary_multiclass_enabled()
            or epoch < max(int(self.graph_config.pseudo_contrastive_start_epoch), 1)
        )
        if disabled:
            return self.network.classifier[-1].weight.new_tensor(0.0), {}

        pool = np.asarray(test_pool_ids, dtype=np.int32)
        if snapshot_end is not None:
            first_active = np.asarray(context.graph_cache.first_active[pool], dtype=np.int32)
            pool = pool[first_active <= int(snapshot_end)].astype(np.int32, copy=False)
        sample_size = min(
            int(pool.size),
            max(int(self.graph_config.pseudo_contrastive_sample_size), 0),
        )
        if sample_size < 8:
            return self.network.classifier[-1].weight.new_tensor(0.0), {}
        if pool.size > sample_size:
            if bool(self.graph_config.pseudo_contrastive_time_balanced):
                candidate_size = min(pool.size, max(sample_size * 4, sample_size))
                choice = rng.choice(pool.size, size=candidate_size, replace=False)
                candidates = np.asarray(pool[choice], dtype=np.int32, copy=False)
                sampled_nodes = self._rebalance_pseudo_sample_by_time_bucket(
                    context=context,
                    node_ids=candidates,
                    sample_size=sample_size,
                    rng=rng,
                )
            else:
                choice = rng.choice(pool.size, size=sample_size, replace=False)
                sampled_nodes = np.asarray(pool[choice], dtype=np.int32, copy=False)
        else:
            sampled_nodes = pool

        subgraph = self._sample_batch_subgraph(
            graph=context.graph_cache,
            context=context,
            batch_nodes=sampled_nodes,
            rng=rng,
            snapshot_end=snapshot_end,
            training=False,
        )
        (
            x,
            edge_src,
            edge_dst,
            rel_ids,
            edge_relative_time,
            target_idx,
            node_subgraph_id,
            edge_subgraph_id,
            node_hop_depth,
            target_context_x,
        ) = self._tensorize_subgraph(
            context=context,
            subgraph=subgraph,
            snapshot_end=snapshot_end,
        )
        target_time_position = self._build_target_time_position(
            context=context,
            batch_nodes=sampled_nodes,
        )
        forward_output = self.network.forward_output(
            x=x,
            edge_src=edge_src,
            edge_dst=edge_dst,
            rel_ids=rel_ids,
            edge_relative_time=edge_relative_time,
            target_local_idx=target_idx,
            target_context_features=target_context_x,
            target_embedding_shift=None,
            target_time_position=target_time_position,
            node_subgraph_id=node_subgraph_id,
            edge_subgraph_id=edge_subgraph_id,
            node_hop_depth=node_hop_depth,
            include_diagnostics=False,
            include_embedding=True,
            include_aux=False,
        )
        if forward_output.embedding is None or forward_output.logits.ndim != 1:
            return self.network.classifier[-1].weight.new_tensor(0.0), {}

        detached_score = torch.sigmoid(forward_output.logits.detach())
        low_q = float(np.clip(self.graph_config.pseudo_contrastive_low_quantile, 0.0, 0.5))
        high_q = float(np.clip(self.graph_config.pseudo_contrastive_high_quantile, 0.5, 1.0))
        if high_q <= low_q:
            return self.network.classifier[-1].weight.new_tensor(0.0), {}
        low_threshold = torch.quantile(detached_score, low_q)
        high_threshold = torch.quantile(detached_score, high_q)
        min_confidence_gap = max(float(self.graph_config.pseudo_contrastive_min_confidence_gap), 0.0)
        normal_threshold = low_threshold
        anomaly_threshold = high_threshold
        if min_confidence_gap > 0.0:
            normal_threshold = torch.minimum(
                low_threshold,
                low_threshold.new_tensor(max(0.5 - min_confidence_gap, 0.0)),
            )
            anomaly_threshold = torch.maximum(
                high_threshold,
                high_threshold.new_tensor(min(0.5 + min_confidence_gap, 1.0)),
            )
        pseudo_normal = detached_score <= normal_threshold
        pseudo_anomaly = detached_score >= anomaly_threshold
        selected_mask = pseudo_normal | pseudo_anomaly
        if int(selected_mask.sum().item()) < 4:
            return self.network.classifier[-1].weight.new_tensor(0.0), {}

        embedding = F.normalize(forward_output.embedding[selected_mask], dim=-1)
        pseudo_labels = pseudo_anomaly[selected_mask].to(dtype=torch.long)
        bucket_ids = torch.as_tensor(
            np.asarray(context.graph_cache.node_time_bucket[sampled_nodes], dtype=np.int64)[
                selected_mask.detach().cpu().numpy()
            ],
            dtype=torch.long,
            device=self.device,
        )
        if int(torch.sum(pseudo_labels == 0).item()) < 2 or int(torch.sum(pseudo_labels == 1).item()) < 2:
            return self.network.classifier[-1].weight.new_tensor(0.0), {}

        global_centroids: list[torch.Tensor | None] = []
        for label_value in (0, 1):
            mask = pseudo_labels == label_value
            if torch.any(mask):
                centroid = F.normalize(torch.mean(embedding[mask], dim=0, keepdim=True), dim=-1).squeeze(0)
                global_centroids.append(centroid)
            else:
                global_centroids.append(None)

        bucket_centroids: dict[tuple[int, int], torch.Tensor] = {}
        unique_bucket_ids = torch.unique(bucket_ids).tolist()
        for bucket_id in unique_bucket_ids:
            for label_value in (0, 1):
                mask = (bucket_ids == int(bucket_id)) & (pseudo_labels == label_value)
                if torch.any(mask):
                    bucket_centroids[(int(bucket_id), label_value)] = F.normalize(
                        torch.mean(embedding[mask], dim=0, keepdim=True),
                        dim=-1,
                    ).squeeze(0)

        pos_terms: list[torch.Tensor] = []
        neg_terms: list[torch.Tensor] = []
        for idx in range(embedding.shape[0]):
            bucket_id = int(bucket_ids[idx].item())
            label_value = int(pseudo_labels[idx].item())
            pos_centroid = bucket_centroids.get((bucket_id, label_value), global_centroids[label_value])
            neg_centroid = bucket_centroids.get((bucket_id, 1 - label_value), global_centroids[1 - label_value])
            if pos_centroid is None or neg_centroid is None:
                continue
            pos_terms.append(1.0 - torch.sum(embedding[idx] * pos_centroid))
            neg_terms.append(F.relu(torch.sum(embedding[idx] * neg_centroid) - 0.15))

        if not pos_terms:
            return self.network.classifier[-1].weight.new_tensor(0.0), {}
        temperature = max(float(self.graph_config.pseudo_contrastive_temperature), 1e-3)
        loss = (torch.stack(pos_terms).mean() + torch.stack(neg_terms).mean()) / temperature
        diagnostics = {
            "pseudo_contrastive_selected_ratio": float(selected_mask.to(dtype=torch.float32).mean().detach().item()),
            "pseudo_contrastive_normal_count": float(torch.sum(pseudo_labels == 0).detach().item()),
            "pseudo_contrastive_anomaly_count": float(torch.sum(pseudo_labels == 1).detach().item()),
            "pseudo_contrastive_bucket_count": float(torch.unique(bucket_ids).numel()),
        }
        return loss, diagnostics

    def _normal_bucket_alignment_enabled(self) -> bool:
        return float(self.graph_config.normal_bucket_align_weight) > 0.0

    def _normal_bucket_shift_enabled(self) -> bool:
        return float(self.graph_config.normal_bucket_shift_strength) > 0.0

    def _normal_alignment_bank_enabled(self) -> bool:
        return self._normal_bucket_alignment_enabled() or self._normal_bucket_shift_enabled()

    def _normal_bucket_adv_enabled(self) -> bool:
        return float(self.graph_config.normal_bucket_adv_weight) > 0.0 and self.temporal

    def _build_normal_alignment_regularizer(
        self,
        *,
        context: GraphPhaseContext,
    ) -> TemporalNormalAlignmentBank | None:
        if not self._normal_alignment_bank_enabled():
            return None
        num_buckets = max(len(context.graph_cache.time_windows), 1)
        return TemporalNormalAlignmentBank(
            config=TemporalNormalAlignmentConfig(
                embedding_dim=int(self.hidden_dim),
                num_buckets=int(num_buckets),
                momentum=float(self.graph_config.prototype_momentum),
                start_epoch=int(self.graph_config.prototype_start_epoch),
                neighbor_blend=float(self.graph_config.prototype_neighbor_blend),
                global_blend=float(self.graph_config.prototype_global_blend),
            ),
            device=self.device,
        )

    def _build_normal_bucket_shift(
        self,
        *,
        bucket_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            not self._normal_bucket_shift_enabled()
            or bucket_ids is None
            or self._normal_alignment_regularizer is None
        ):
            return None
        return self._normal_alignment_regularizer.build_shift(
            bucket_ids=bucket_ids,
            strength=float(self.graph_config.normal_bucket_shift_strength),
        )

    def _build_target_time_position(
        self,
        *,
        context: GraphPhaseContext,
        batch_nodes: np.ndarray,
    ) -> torch.Tensor | None:
        needs_time_position = (
            float(self.graph_config.target_time_adapter_strength) > 0.0
            or str(self.graph_config.target_context_fusion)
            in {
                "atm_residual",
                "drift_residual",
                "drift_mix",
                "drift_uncertainty_mix",
                "risk_drift_residual",
            }
        )
        if not needs_time_position:
            return None
        bucket_ids = np.asarray(context.graph_cache.node_time_bucket[np.asarray(batch_nodes, dtype=np.int32)], dtype=np.float32)
        denominator = max(len(context.graph_cache.time_windows) - 1, 1)
        bucket_position = (bucket_ids / float(denominator)).reshape(-1, 1).astype(np.float32, copy=False)
        return torch.as_tensor(bucket_position, dtype=torch.float32, device=self.device)

    def _build_normal_bucket_discriminator(
        self,
        *,
        context: GraphPhaseContext,
    ) -> nn.Module | None:
        if not self._normal_bucket_adv_enabled():
            return None
        num_buckets = max(len(context.graph_cache.time_windows), 1)
        if num_buckets < 2:
            return None
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.graph_config.dropout),
            nn.Linear(self.hidden_dim, num_buckets),
        ).to(self.device)

    def _compute_normal_bucket_adv_loss(
        self,
        *,
        embedding: torch.Tensor | None,
        raw_labels: torch.Tensor | None,
        bucket_ids: torch.Tensor | None,
        discriminator: nn.Module | None,
        memory_embedding: torch.Tensor | None = None,
        memory_bucket_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            not self._normal_bucket_adv_enabled()
            or embedding is None
            or raw_labels is None
            or bucket_ids is None
            or discriminator is None
        ):
            return self.network.input_proj.weight.new_tensor(0.0)
        normal_mask = raw_labels == 0
        if int(torch.sum(normal_mask).item()) < 4:
            return embedding.new_tensor(0.0)
        normal_bucket_ids = bucket_ids[normal_mask]
        normal_embedding = F.normalize(embedding[normal_mask], dim=-1)
        bucket_inputs = [_grad_reverse(normal_embedding)]
        bucket_targets = [normal_bucket_ids]
        if memory_embedding is not None and memory_bucket_ids is not None and memory_embedding.numel():
            bucket_inputs.append(memory_embedding.detach())
            bucket_targets.append(memory_bucket_ids.detach())
        merged_targets = torch.cat(bucket_targets, dim=0)
        if int(torch.unique(merged_targets).numel()) < 2:
            return embedding.new_tensor(0.0)
        merged_inputs = torch.cat(bucket_inputs, dim=0)
        bucket_logits = discriminator(merged_inputs)
        return F.cross_entropy(bucket_logits, merged_targets)

    def _append_normal_bucket_adv_memory(
        self,
        *,
        embedding: torch.Tensor | None,
        raw_labels: torch.Tensor | None,
        bucket_ids: torch.Tensor | None,
        memory_embedding: torch.Tensor | None,
        memory_bucket_ids: torch.Tensor | None,
        memory_limit: int = 4096,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if embedding is None or raw_labels is None or bucket_ids is None:
            return memory_embedding, memory_bucket_ids
        normal_mask = raw_labels == 0
        if not torch.any(normal_mask):
            return memory_embedding, memory_bucket_ids
        current_embedding = F.normalize(embedding[normal_mask].detach(), dim=-1)
        current_bucket_ids = bucket_ids[normal_mask].detach()
        if memory_embedding is None or memory_bucket_ids is None:
            merged_embedding = current_embedding
            merged_bucket_ids = current_bucket_ids
        else:
            merged_embedding = torch.cat([memory_embedding, current_embedding], dim=0)
            merged_bucket_ids = torch.cat([memory_bucket_ids, current_bucket_ids], dim=0)
        if int(merged_embedding.shape[0]) > int(memory_limit):
            merged_embedding = merged_embedding[-int(memory_limit) :]
            merged_bucket_ids = merged_bucket_ids[-int(memory_limit) :]
        return merged_embedding, merged_bucket_ids

    def _fraud_probability_from_multiclass_logits(self, logits: torch.Tensor) -> torch.Tensor:
        multiclass_prob = torch.softmax(logits, dim=-1)
        foreground = multiclass_prob[:, 0] + multiclass_prob[:, 1]
        return multiclass_prob[:, 1] / foreground.clamp_min(1e-6)

    def _fraud_score_from_aux_logits(self, aux_logits: torch.Tensor) -> torch.Tensor:
        aux_prob = torch.softmax(aux_logits, dim=-1)
        foreground = aux_prob[:, 0] + aux_prob[:, 1]
        return aux_prob[:, 1] / foreground.clamp_min(1e-6)

    def _primary_probability_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self._primary_multiclass_enabled():
            return self._fraud_probability_from_multiclass_logits(logits)
        return torch.sigmoid(logits)

    def _blend_primary_and_aux_probability(
        self,
        logits: torch.Tensor,
        aux_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        primary_prob = self._primary_probability_from_logits(logits)
        blend = float(np.clip(self.graph_config.aux_inference_blend, 0.0, 1.0))
        if aux_logits is None or blend <= 0.0:
            return primary_prob
        aux_score = self._fraud_score_from_aux_logits(aux_logits)
        return (1.0 - blend) * primary_prob + blend * aux_score

    def _hard_negative_pool_key(self, snapshot_end: int | None) -> int:
        return -1 if snapshot_end is None else int(snapshot_end)

    def _hard_negative_enabled(self) -> bool:
        return (
            self.graph_config.train_negative_ratio > 0.0
            and self.graph_config.negative_sampler in {"hard", "mixed"}
        )

    def _select_hard_negative_candidates(
        self,
        neg_nodes: np.ndarray,
        candidate_count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        neg_nodes = np.asarray(neg_nodes, dtype=np.int32)
        if candidate_count >= neg_nodes.size:
            return neg_nodes.astype(np.int32, copy=False)
        choice = rng.choice(neg_nodes.size, size=candidate_count, replace=False)
        return neg_nodes[choice].astype(np.int32, copy=False)

    def _relation_risk_sampler_enabled(self) -> bool:
        return _sampler_uses_relation_risk(self.graph_config.neighbor_sampler)

    def _consistency_sampler_enabled(self) -> bool:
        return _sampler_uses_consistency_profile(self.graph_config.neighbor_sampler)

    def _build_relation_sampling_weight(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
    ) -> np.ndarray | None:
        if not self._relation_risk_sampler_enabled():
            return None

        graph = context.graph_cache
        labels = np.asarray(context.labels[np.asarray(train_ids, dtype=np.int32)], dtype=np.int8)
        pos_counts = np.ones(graph.num_relations, dtype=np.float64)
        neg_counts = np.ones(graph.num_relations, dtype=np.float64)
        snapshot_end = context.reference_day
        train_nodes = np.asarray(train_ids, dtype=np.int32)

        for node, label in zip(train_nodes.tolist(), labels.tolist(), strict=True):
            if label not in (0, 1):
                continue
            target_counts = pos_counts if int(label) == 1 else neg_counts

            in_start = int(graph.in_ptr[node])
            in_end = int(graph.in_ptr[node + 1])
            in_ts = np.asarray(graph.in_edge_timestamp[in_start:in_end], dtype=np.int32)
            in_rel = np.asarray(graph.in_edge_type[in_start:in_end], dtype=np.int64) - 1
            if snapshot_end is not None:
                in_rel = in_rel[in_ts <= int(snapshot_end)]
            if in_rel.size:
                np.add.at(target_counts, in_rel, 1.0)

            out_start = int(graph.out_ptr[node])
            out_end = int(graph.out_ptr[node + 1])
            out_ts = np.asarray(graph.out_edge_timestamp[out_start:out_end], dtype=np.int32)
            out_rel = np.asarray(graph.out_edge_type[out_start:out_end], dtype=np.int64) - 1 + graph.num_edge_types
            if snapshot_end is not None:
                out_rel = out_rel[out_ts <= int(snapshot_end)]
            if out_rel.size:
                np.add.at(target_counts, out_rel, 1.0)

        pos_rate = pos_counts / float(np.sum(pos_counts, dtype=np.float64))
        neg_rate = neg_counts / float(np.sum(neg_counts, dtype=np.float64))
        log_ratio = np.log(pos_rate) - np.log(neg_rate)
        log_ratio = np.clip(log_ratio, -2.0, 2.0)
        weight = np.exp(log_ratio).astype(np.float32, copy=False)
        mean_weight = float(np.mean(weight, dtype=np.float64))
        if mean_weight > 0.0:
            weight = weight / mean_weight
        return np.clip(weight, 0.25, 4.0).astype(np.float32, copy=False)

    def _uses_ranking_loss(self) -> bool:
        return "ranking" in str(self.graph_config.loss_type)

    def _iter_train_partitions(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
        nodes = np.asarray(node_ids, dtype=np.int32)
        positions = np.arange(nodes.size, dtype=np.int32)
        if not self.temporal:
            return [(nodes, positions, None)]

        buckets = np.asarray(context.graph_cache.node_time_bucket[nodes], dtype=np.int8)
        partitions: list[tuple[np.ndarray, np.ndarray, int | None]] = []
        for bucket_idx, window in enumerate(context.graph_cache.time_windows):
            bucket_mask = buckets == bucket_idx
            bucket_nodes = nodes[bucket_mask]
            bucket_positions = positions[bucket_mask]
            if bucket_nodes.size == 0:
                continue
            partitions.append((bucket_nodes, bucket_positions, int(window["end_day"])))
        return partitions

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        aux_logits: torch.Tensor | None = None,
        aux_targets: torch.Tensor | None = None,
        primary_class_weight: torch.Tensor | None = None,
        aux_class_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_type = str(self.graph_config.loss_type)
        if self._primary_multiclass_enabled():
            primary_ce = F.cross_entropy(
                logits,
                targets,
                reduction="none",
            )
            if primary_class_weight is not None:
                primary_ce = primary_ce * primary_class_weight[targets]
            if sample_weight is not None:
                primary_ce = primary_ce * sample_weight
            base_loss = primary_ce.mean()
        else:
            if "focal" in loss_type:
                base_loss = _focal_bce_with_logits(
                    logits=logits,
                    targets=targets,
                    pos_weight=pos_weight,
                    gamma=float(self.graph_config.focal_gamma),
                    alpha=float(self.graph_config.focal_alpha),
                    sample_weight=sample_weight,
                )
            else:
                bce = F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                if sample_weight is not None:
                    bce = bce * sample_weight
                base_loss = bce.mean()

        ranking_loss = logits.new_tensor(0.0)
        if (
            not self._primary_multiclass_enabled()
            and self._uses_ranking_loss()
            and self.graph_config.ranking_weight > 0.0
        ):
            ranking_loss = _pairwise_ranking_loss(
                logits=logits,
                targets=targets,
                margin=float(self.graph_config.ranking_margin),
            )

        aux_loss = logits.new_tensor(0.0)
        if (
            aux_logits is not None
            and aux_targets is not None
            and float(self.graph_config.aux_multiclass_loss_weight) > 0.0
        ):
            aux_ce = F.cross_entropy(
                aux_logits,
                aux_targets,
                reduction="none",
            )
            if aux_class_weight is not None:
                aux_ce = aux_ce * aux_class_weight[aux_targets]
            if sample_weight is not None:
                aux_ce = aux_ce * sample_weight
            aux_loss = aux_ce.mean()

        total_loss = (
            base_loss
            + float(self.graph_config.ranking_weight) * ranking_loss
            + float(self.graph_config.aux_multiclass_loss_weight) * aux_loss
        )
        return total_loss, {
            "base_loss": float(base_loss.detach().item()),
            "ranking_loss": float(ranking_loss.detach().item()),
            "aux_loss": float(aux_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }

    def _compute_target_time_weight(
        self,
        context: GraphPhaseContext,
        batch_nodes: np.ndarray,
    ) -> torch.Tensor | None:
        half_life = float(self.graph_config.target_time_weight_half_life_days)
        if half_life <= 0.0 or context.reference_day is None:
            return None
        first_active = np.asarray(
            context.graph_cache.first_active[np.asarray(batch_nodes, dtype=np.int32)],
            dtype=np.float32,
        )
        age = np.clip(float(context.reference_day) - first_active, 0.0, None)
        decay = np.power(np.float32(0.5), age / half_life).astype(np.float32, copy=False)
        floor = float(np.clip(self.graph_config.target_time_weight_floor, 0.0, 1.0))
        weight = (floor + (1.0 - floor) * decay).astype(np.float32, copy=False)
        mean_weight = float(np.mean(weight, dtype=np.float64))
        if mean_weight > 0.0:
            weight = weight / mean_weight
        return torch.as_tensor(weight, dtype=torch.float32, device=self.device)

    def _sample_partition_background_negatives(
        self,
        context: GraphPhaseContext,
        partition_nodes: np.ndarray,
        snapshot_end: int | None,
        epoch: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if (
            not self.graph_config.include_historical_background_negatives
            or context.historical_background_ids is None
        ):
            return np.empty(0, dtype=np.int32)
        ratio = self._current_historical_background_negative_ratio(epoch)
        if ratio <= 0.0:
            return np.empty(0, dtype=np.int32)

        partition_labels = self._binary_train_labels(context.labels[partition_nodes])
        pos_count = int(np.sum(partition_labels == 1))
        if pos_count <= 0:
            return np.empty(0, dtype=np.int32)

        pool = np.asarray(context.historical_background_ids, dtype=np.int32)
        if pool.size == 0:
            return np.empty(0, dtype=np.int32)

        if snapshot_end is not None:
            first_active = np.asarray(context.graph_cache.first_active[pool], dtype=np.int32)
            eligible = pool[first_active <= int(snapshot_end)].astype(np.int32, copy=False)
            if eligible.size:
                pool = eligible
        if pool.size == 0:
            return np.empty(0, dtype=np.int32)

        sample_size = min(pool.size, max(1, int(math.ceil(pos_count * ratio))))
        if pool.size <= sample_size:
            return np.asarray(pool, dtype=np.int32, copy=False)
        choice = rng.choice(pool.size, size=sample_size, replace=False)
        return np.asarray(pool[choice], dtype=np.int32, copy=False)

    def _current_historical_background_negative_ratio(self, epoch: int) -> float:
        target_ratio = max(float(self.graph_config.historical_background_negative_ratio), 0.0)
        if target_ratio <= 0.0:
            return 0.0
        warmup_epochs = max(int(self.graph_config.historical_background_negative_warmup_epochs), 0)
        if warmup_epochs <= 1:
            return target_ratio
        progress = min(max(int(epoch), 1), warmup_epochs) / float(warmup_epochs)
        return float(target_ratio * progress)

    def _prototype_release_anchor_epoch(self) -> int:
        prototype_anchor_epoch = max(
            1,
            int(self.graph_config.prototype_start_epoch)
            + max(int(self.graph_config.prototype_loss_ramp_epochs), 1)
            - 1,
        )
        return prototype_anchor_epoch + max(
            int(self.graph_config.context_residual_budget_release_delay_epochs),
            0,
        )

    def _history_metric_values(
        self,
        history: list[dict[str, Any]],
        key: str,
    ) -> list[float]:
        values: list[float] = []
        for row in history:
            value = row.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return values

    def _adaptive_prototype_readiness(
        self,
        history: list[dict[str, Any]],
    ) -> float:
        if not self._prototype_enabled():
            return 1.0
        if not history:
            return 0.0

        memory_init = self._history_metric_values(history, "prototype_memory_initialized_ratio")
        availability = self._history_metric_values(history, "prototype_target_available_ratio")
        classification = self._history_metric_values(history, "prototype_classification_ratio")
        margins = self._history_metric_values(history, "prototype_margin")
        temporal_stability_values = self._history_metric_values(history, "prototype_temporal_stability")
        prototype_losses = [
            value
            for value in self._history_metric_values(history, "train_prototype_loss")
            if value > 0.0
        ]

        margin_scale = max(float(self.graph_config.prototype_separation_margin), 0.05)
        margin_ready = 0.0
        if margins:
            margin_mean = float(np.mean(margins, dtype=np.float64))
            margin_ready = float(np.clip(0.5 + 0.5 * (margin_mean / margin_scale), 0.0, 1.0))

        stability = 0.0
        if len(prototype_losses) >= 2:
            loss_arr = np.asarray(prototype_losses, dtype=np.float64)
            volatility = float(
                np.mean(np.abs(np.diff(loss_arr)), dtype=np.float64)
                / max(float(np.mean(np.abs(loss_arr), dtype=np.float64)), 1e-6)
            )
            stability = float(np.clip(1.0 - volatility / 0.35, 0.0, 1.0))

        temporal_stability = 1.0
        if self._prototype_time_bucketed():
            temporal_stability = (
                float(np.mean(temporal_stability_values, dtype=np.float64))
                if temporal_stability_values
                else 0.0
            )

        readiness = (
            0.18 * (float(np.mean(memory_init, dtype=np.float64)) if memory_init else 0.0)
            + 0.24 * (float(np.mean(availability, dtype=np.float64)) if availability else 0.0)
            + 0.12 * (float(np.mean(classification, dtype=np.float64)) if classification else 0.0)
            + 0.16 * margin_ready
            + 0.10 * stability
            + 0.20 * temporal_stability
        )
        return float(np.clip(readiness, 0.0, 1.0))

    def _adaptive_context_guard_strength(
        self,
        history: list[dict[str, Any]],
    ) -> float:
        if not history:
            return 0.0

        last_row = history[-1]
        clip_fraction = max(float(last_row.get("context_residual_clip_fraction", 0.0)), 0.0)
        penalty = max(
            float(
                last_row.get(
                    "train_context_regularization_loss",
                    last_row.get("context_residual_penalty", 0.0),
                )
            ),
            0.0,
        )
        logit_abs = max(float(last_row.get("context_logit_abs_mean", 0.0)), 0.0)

        residual_budget = max(float(self.graph_config.context_residual_budget), 1e-6)
        clip_target = 0.01
        penalty_target = max(residual_budget * 0.02, 5e-4)
        logit_target = max(residual_budget * 0.75, 1e-3)
        clip_risk = clip_fraction / clip_target
        penalty_risk = penalty / penalty_target
        logit_risk = logit_abs / logit_target
        risk = max(clip_risk, penalty_risk, logit_risk)

        if len(history) >= 2:
            prev_row = history[-2]
            prev_clip = max(float(prev_row.get("context_residual_clip_fraction", clip_fraction)), 0.0)
            prev_penalty = max(
                float(
                    prev_row.get(
                        "train_context_regularization_loss",
                        prev_row.get("context_residual_penalty", penalty),
                    )
                ),
                0.0,
            )
            prev_logit = max(float(prev_row.get("context_logit_abs_mean", logit_abs)), 0.0)
            growth_risk = max(
                max(clip_fraction - prev_clip, 0.0) / max(0.5 * clip_target, 1e-6)
                if clip_fraction > 0.5 * clip_target
                else 0.0,
                max(penalty - prev_penalty, 0.0) / max(0.5 * penalty_target, 1e-6)
                if penalty > 0.5 * penalty_target
                else 0.0,
                max(logit_abs - prev_logit, 0.0) / max(0.5 * logit_target, 1e-6)
                if logit_abs > 0.5 * logit_target
                else 0.0,
            )
            if growth_risk > 0.0:
                risk = max(risk, 1.0 + growth_risk)

        val_auc_history = self._history_metric_values(history, "val_auc")
        if val_auc_history:
            last_val_auc = val_auc_history[-1]
            best_val_auc = max(val_auc_history)
            val_gap = max(best_val_auc - last_val_auc, 0.0)
            if val_gap > 0.0:
                risk = max(risk, 1.0 + min(val_gap / 0.0015, 1.0))

        return float(np.clip((risk - 1.0) / 0.8, 0.0, 1.0))

    def _adaptive_prototype_trust_score(
        self,
        history: list[dict[str, Any]],
    ) -> float:
        if not history:
            return 1.0

        availability = self._history_metric_values(history, "prototype_target_available_ratio")
        classification_losses = self._history_metric_values(history, "prototype_classification_loss")
        consistency_losses = self._history_metric_values(history, "prototype_consistency_loss")
        margins = self._history_metric_values(history, "prototype_margin")
        temporal_stability_values = self._history_metric_values(history, "prototype_temporal_stability")

        availability_score = float(np.mean(availability, dtype=np.float64)) if availability else 0.0
        classification_score = 1.0
        if classification_losses:
            classification_score = float(
                np.clip(1.0 - float(np.mean(classification_losses, dtype=np.float64)) / 0.35, 0.0, 1.0)
            )
        consistency_score = 1.0
        if consistency_losses:
            consistency_score = float(
                np.clip(1.0 - float(np.mean(consistency_losses, dtype=np.float64)) / 0.25, 0.0, 1.0)
            )
        margin_score = 0.0
        if margins:
            margin_score = float(
                np.clip(float(np.mean(margins, dtype=np.float64)) / 0.45, 0.0, 1.0)
            )
        temporal_stability_score = 1.0
        if self._prototype_time_bucketed():
            temporal_stability_score = (
                float(np.mean(temporal_stability_values, dtype=np.float64))
                if temporal_stability_values
                else 0.0
            )

        trust_score = (
            0.18 * availability_score
            + 0.30 * classification_score
            + 0.14 * margin_score
            + 0.10 * consistency_score
            + 0.28 * temporal_stability_score
        )
        return float(np.clip(trust_score, 0.0, 1.0))

    def _current_prototype_loss_weight(self, epoch: int) -> float:
        base_weight = max(float(self.graph_config.prototype_loss_weight), 0.0)
        if base_weight <= 0.0 or not self._prototype_enabled():
            self._prototype_weight_schedule_state = {
                "trust_score": 0.0,
            }
            return 0.0

        schedule = str(self.graph_config.prototype_loss_weight_schedule).strip().lower()
        min_weight = min(
            max(float(self.graph_config.prototype_loss_min_weight), 0.0),
            base_weight,
        )
        if not schedule or schedule == "none":
            self._prototype_weight_schedule_state = {
                "trust_score": 1.0,
            }
            return base_weight
        if schedule != "adaptive_quality":
            raise ValueError(f"Unsupported prototype loss weight schedule: {schedule}")

        history = list(getattr(self, "training_history", []))
        history_window = history[-3:]
        if len(history_window) < 2:
            self._prototype_weight_schedule_state = {
                "trust_score": 1.0,
            }
            return base_weight

        trust_score = self._adaptive_prototype_trust_score(history_window)
        effective_weight = max(min_weight, base_weight * trust_score)
        self._prototype_weight_schedule_state = {
            "trust_score": float(trust_score),
        }
        return float(effective_weight)

    def _current_context_residual_budget_weight(self, epoch: int) -> float:
        base_weight = max(float(self.graph_config.context_residual_budget_weight), 0.0)
        if base_weight <= 0.0:
            self._context_budget_schedule_state = {
                "time_progress": 0.0,
                "prototype_readiness": 0.0,
                "release_progress": 0.0,
                "guard_strength": 0.0,
            }
            return 0.0
        schedule = str(self.graph_config.context_residual_budget_schedule).strip().lower()
        if not schedule or schedule == "none":
            self._context_budget_schedule_state = {
                "time_progress": 0.0,
                "prototype_readiness": 0.0,
                "release_progress": 0.0,
                "guard_strength": 0.0,
            }
            return base_weight

        min_weight = min(
            max(float(self.graph_config.context_residual_budget_min_weight), 0.0),
            base_weight,
        )
        if schedule not in {"prototype_release", "prototype_adaptive"}:
            raise ValueError(f"Unsupported context residual budget schedule: {schedule}")

        release_epochs = int(self.graph_config.context_residual_budget_release_epochs)
        prototype_anchor_epoch = self._prototype_release_anchor_epoch()
        if release_epochs <= 0:
            release_epochs = max(self.epochs - prototype_anchor_epoch, 1)
        if epoch <= prototype_anchor_epoch:
            self._context_budget_schedule_state = {
                "time_progress": 0.0,
                "prototype_readiness": 0.0,
                "release_progress": 0.0,
                "guard_strength": 1.0 if schedule == "prototype_adaptive" else 0.0,
            }
            return base_weight

        time_progress = min(max(epoch - prototype_anchor_epoch, 0), release_epochs) / float(release_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * time_progress))
        scheduled_weight = float(min_weight + (base_weight - min_weight) * cosine_decay)
        if schedule == "prototype_release":
            self._context_budget_schedule_state = {
                "time_progress": float(time_progress),
                "prototype_readiness": 1.0,
                "release_progress": float(time_progress),
                "guard_strength": 0.0,
            }
            return scheduled_weight

        history = list(getattr(self, "training_history", []))
        history_window = history[-3:]
        if len(history_window) < 2:
            self._context_budget_schedule_state = {
                "time_progress": float(time_progress),
                "prototype_readiness": 0.0,
                "release_progress": 0.0,
                "guard_strength": 1.0,
            }
            return base_weight

        prototype_readiness = self._adaptive_prototype_readiness(history_window)
        release_progress = float(time_progress * prototype_readiness)
        adaptive_weight = float(min_weight + (base_weight - min_weight) * (1.0 - release_progress))
        guard_strength = self._adaptive_context_guard_strength(history_window)
        guarded_floor = float(min_weight + (base_weight - min_weight) * guard_strength)
        effective_weight = max(adaptive_weight, guarded_floor)
        self._context_budget_schedule_state = {
            "time_progress": float(time_progress),
            "prototype_readiness": float(prototype_readiness),
            "release_progress": float(release_progress),
            "guard_strength": float(guard_strength),
        }
        return float(effective_weight)

    def _current_hard_negative_pool_size(self) -> int:
        return int(sum(pool.shape[0] for pool in self._hard_negative_pools.values()))

    def _current_hard_negative_candidate_count(self) -> int:
        return int(
            sum(int(stats.get("candidate_count", 0)) for stats in self._hard_negative_pool_stats.values())
        )

    def _maybe_refresh_hard_negative_pools(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
        epoch: int,
        rng: np.random.Generator,
    ) -> dict[str, int | bool | float]:
        if not self._hard_negative_enabled():
            self._hard_negative_pools = {}
            self._hard_negative_pool_stats = {}
            return {
                "refreshed": False,
                "pool_size": 0,
                "candidate_count": 0,
                "partition_count": 0,
            }

        warmup_epochs = max(int(self.graph_config.hard_negative_warmup_epochs), 0)
        if epoch <= warmup_epochs:
            self._hard_negative_pools = {}
            self._hard_negative_pool_stats = {}
            return {
                "refreshed": False,
                "pool_size": 0,
                "candidate_count": 0,
                "partition_count": 0,
            }

        refresh_every = max(int(self.graph_config.hard_negative_refresh), 1)
        refresh_due = not self._hard_negative_pools or (
            (epoch - warmup_epochs - 1) % refresh_every == 0
        )
        if not refresh_due:
            return {
                "refreshed": False,
                "pool_size": self._current_hard_negative_pool_size(),
                "candidate_count": self._current_hard_negative_candidate_count(),
                "partition_count": len(self._hard_negative_pools),
            }

        negative_ratio = max(float(self.graph_config.train_negative_ratio), 0.0)
        candidate_cap = max(int(self.graph_config.hard_negative_candidate_cap), 1)
        candidate_multiplier = max(float(self.graph_config.hard_negative_candidate_multiplier), 1.0)
        pool_multiplier = max(float(self.graph_config.hard_negative_pool_multiplier), 1.0)
        hard_pools: dict[int, np.ndarray] = {}
        hard_pool_stats: dict[int, dict[str, int]] = {}

        for partition_idx, (nodes, _, snapshot_end) in enumerate(
            self._iter_train_partitions(context=context, node_ids=train_ids)
        ):
            labels = self._binary_train_labels(context.labels[nodes])
            pos_nodes = nodes[labels == 1]
            neg_nodes = nodes[labels == 0]
            if pos_nodes.size == 0 or neg_nodes.size == 0:
                continue

            sampled_negatives = min(
                neg_nodes.size,
                max(1, int(math.ceil(pos_nodes.size * negative_ratio))),
            )
            requested_candidates = max(
                sampled_negatives,
                int(math.ceil(sampled_negatives * candidate_multiplier)),
            )
            candidate_count = min(neg_nodes.size, candidate_cap, requested_candidates)
            if candidate_count <= 0:
                continue

            candidate_nodes = self._select_hard_negative_candidates(
                neg_nodes=neg_nodes,
                candidate_count=candidate_count,
                rng=rng,
            )

            candidate_scores = self.predict_proba(
                context=context,
                node_ids=candidate_nodes,
                batch_seed=self.seed + epoch * 1009 + partition_idx * 53 + 100000,
                progress_desc=None,
                show_progress=False,
            )
            if candidate_scores.size == 0:
                continue
            pool_count = min(
                candidate_nodes.size,
                max(1, int(math.ceil(sampled_negatives * pool_multiplier))),
            )
            if pool_count >= candidate_nodes.size:
                top_idx = np.argsort(-candidate_scores, kind="stable")
            else:
                top_idx = np.argpartition(candidate_scores, -pool_count)[-pool_count:]
                top_idx = top_idx[np.argsort(-candidate_scores[top_idx], kind="stable")]
            pool_nodes = candidate_nodes[top_idx].astype(np.int32, copy=False)
            pool_key = self._hard_negative_pool_key(snapshot_end)
            hard_pools[pool_key] = pool_nodes
            hard_pool_stats[pool_key] = {
                "candidate_count": int(candidate_nodes.size),
                "pool_count": int(pool_nodes.size),
                "sampled_negatives": int(sampled_negatives),
            }

        self._hard_negative_pools = hard_pools
        self._hard_negative_pool_stats = hard_pool_stats
        return {
            "refreshed": True,
            "pool_size": self._current_hard_negative_pool_size(),
            "candidate_count": self._current_hard_negative_candidate_count(),
            "partition_count": len(self._hard_negative_pools),
        }

    def _sample_negative_partition(
        self,
        neg_nodes: np.ndarray,
        neg_positions: np.ndarray,
        neg_background_mask: np.ndarray | None,
        sampled_negatives: int,
        snapshot_end: int | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        if sampled_negatives <= 0 or neg_nodes.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                0,
                0,
            )

        sampler = str(self.graph_config.negative_sampler)
        if sampler not in {"random", "hard", "mixed"}:
            raise ValueError(f"Unsupported negative sampler: {sampler}")
        if sampler == "random":
            choice = rng.choice(neg_nodes.size, size=sampled_negatives, replace=False)
            return (
                neg_nodes[choice].astype(np.int32, copy=False),
                neg_positions[choice].astype(np.int32, copy=False),
                0,
                int(np.sum(neg_background_mask[choice])) if neg_background_mask is not None else 0,
            )

        hard_pool = self._hard_negative_pools.get(self._hard_negative_pool_key(snapshot_end))
        if hard_pool is None or hard_pool.size == 0:
            choice = rng.choice(neg_nodes.size, size=sampled_negatives, replace=False)
            return (
                neg_nodes[choice].astype(np.int32, copy=False),
                neg_positions[choice].astype(np.int32, copy=False),
                0,
                int(np.sum(neg_background_mask[choice])) if neg_background_mask is not None else 0,
            )

        hard_mask = np.isin(neg_nodes, hard_pool, assume_unique=False)
        hard_candidate_idx = np.flatnonzero(hard_mask)
        if sampler == "hard":
            requested_hard = sampled_negatives
        else:
            requested_hard = int(round(sampled_negatives * float(self.graph_config.hard_negative_mix)))
            requested_hard = max(0, min(sampled_negatives, requested_hard))

        selected_hard_idx = np.empty(0, dtype=np.int32)
        if hard_candidate_idx.size and requested_hard > 0:
            if hard_candidate_idx.size <= requested_hard:
                selected_hard_idx = hard_candidate_idx.astype(np.int32, copy=False)
            else:
                hard_choice = rng.choice(hard_candidate_idx.size, size=requested_hard, replace=False)
                selected_hard_idx = hard_candidate_idx[hard_choice].astype(np.int32, copy=False)

        remaining_take = sampled_negatives - selected_hard_idx.size
        selected_random_idx = np.empty(0, dtype=np.int32)
        if remaining_take > 0:
            available_mask = np.ones(neg_nodes.size, dtype=bool)
            if selected_hard_idx.size:
                available_mask[selected_hard_idx] = False

            non_hard_idx = np.flatnonzero(available_mask & ~hard_mask)
            if non_hard_idx.size:
                random_take = min(remaining_take, non_hard_idx.size)
                if non_hard_idx.size <= random_take:
                    selected_random_idx = non_hard_idx.astype(np.int32, copy=False)
                else:
                    random_choice = rng.choice(non_hard_idx.size, size=random_take, replace=False)
                    selected_random_idx = non_hard_idx[random_choice].astype(np.int32, copy=False)
                available_mask[selected_random_idx] = False
                remaining_take -= selected_random_idx.size

            if remaining_take > 0:
                fallback_idx = np.flatnonzero(available_mask)
                if fallback_idx.size:
                    fallback_take = min(remaining_take, fallback_idx.size)
                    if fallback_idx.size <= fallback_take:
                        extra_idx = fallback_idx.astype(np.int32, copy=False)
                    else:
                        extra_choice = rng.choice(fallback_idx.size, size=fallback_take, replace=False)
                        extra_idx = fallback_idx[extra_choice].astype(np.int32, copy=False)
                    selected_random_idx = (
                        extra_idx
                        if selected_random_idx.size == 0
                        else np.concatenate([selected_random_idx, extra_idx]).astype(np.int32, copy=False)
                    )

        selected_idx = (
            selected_hard_idx
            if selected_random_idx.size == 0
            else (
                selected_random_idx
                if selected_hard_idx.size == 0
                else np.concatenate([selected_hard_idx, selected_random_idx]).astype(np.int32, copy=False)
            )
        )
        if selected_idx.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                0,
                0,
            )

        if selected_idx.size < sampled_negatives:
            refill_take = min(sampled_negatives - selected_idx.size, neg_nodes.size - selected_idx.size)
            if refill_take > 0:
                refill_mask = np.ones(neg_nodes.size, dtype=bool)
                refill_mask[selected_idx] = False
                refill_idx = np.flatnonzero(refill_mask)
                if refill_idx.size:
                    if refill_idx.size <= refill_take:
                        extra_idx = refill_idx.astype(np.int32, copy=False)
                    else:
                        extra_choice = rng.choice(refill_idx.size, size=refill_take, replace=False)
                        extra_idx = refill_idx[extra_choice].astype(np.int32, copy=False)
                    selected_idx = np.concatenate([selected_idx, extra_idx]).astype(np.int32, copy=False)

        order = rng.permutation(selected_idx.size)
        selected_idx = selected_idx[order].astype(np.int32, copy=False)
        hard_selected_count = int(np.sum(hard_mask[selected_idx]))
        background_selected_count = (
            int(np.sum(neg_background_mask[selected_idx])) if neg_background_mask is not None else 0
        )
        return (
            neg_nodes[selected_idx].astype(np.int32, copy=False),
            neg_positions[selected_idx].astype(np.int32, copy=False),
            hard_selected_count,
            background_selected_count,
        )

    def _iter_batches(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        training: bool,
        rng: np.random.Generator,
    ) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
        nodes = np.asarray(node_ids, dtype=np.int32)
        positions = np.arange(nodes.size, dtype=np.int32)
        effective_batch_size = self.batch_size if training else self.eval_batch_size
        if self.temporal:
            buckets = np.asarray(context.graph_cache.node_time_bucket[nodes], dtype=np.int8)
            batches: list[tuple[np.ndarray, np.ndarray, int | None]] = []
            for bucket_idx, window in enumerate(context.graph_cache.time_windows):
                bucket_mask = buckets == bucket_idx
                bucket_nodes = nodes[bucket_mask]
                bucket_positions = positions[bucket_mask]
                if bucket_nodes.size == 0:
                    continue
                if training:
                    order = rng.permutation(bucket_nodes.size)
                    bucket_nodes = bucket_nodes[order]
                    bucket_positions = bucket_positions[order]
                for start in range(0, bucket_nodes.size, effective_batch_size):
                    batches.append(
                        (
                            bucket_nodes[start : start + effective_batch_size],
                            bucket_positions[start : start + effective_batch_size],
                            int(window["end_day"]),
                        )
                    )
            return batches

        if training:
            order = rng.permutation(nodes.size)
            nodes = nodes[order]
            positions = positions[order]
        return [
            (
                nodes[start : start + effective_batch_size],
                positions[start : start + effective_batch_size],
                None,
            )
            for start in range(0, nodes.size, effective_batch_size)
        ]

    def _chunk_batch_partition(
        self,
        nodes: np.ndarray,
        positions: np.ndarray,
        snapshot_end: int | None,
        effective_batch_size: int,
    ) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
        return [
            (
                nodes[start : start + effective_batch_size],
                positions[start : start + effective_batch_size],
                snapshot_end,
            )
            for start in range(0, nodes.size, effective_batch_size)
        ]

    def _build_balanced_partition_batches(
        self,
        nodes: np.ndarray,
        positions: np.ndarray,
        labels: np.ndarray,
        background_negative_mask: np.ndarray | None,
        snapshot_end: int | None,
        rng: np.random.Generator,
        effective_batch_size: int,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, int | None]], TrainBatchStats]:
        background_mask = (
            np.asarray(background_negative_mask, dtype=bool)
            if background_negative_mask is not None
            else np.zeros(nodes.shape[0], dtype=bool)
        )
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_nodes = nodes[pos_mask]
        pos_positions = positions[pos_mask]
        neg_nodes = nodes[neg_mask]
        neg_positions = positions[neg_mask]
        neg_background_mask = background_mask[neg_mask]

        if pos_nodes.size == 0 or neg_nodes.size == 0:
            order = rng.permutation(nodes.size)
            shuffled_nodes = nodes[order]
            shuffled_positions = positions[order]
            return (
                self._chunk_batch_partition(
                    nodes=shuffled_nodes,
                    positions=shuffled_positions,
                    snapshot_end=snapshot_end,
                    effective_batch_size=effective_batch_size,
                ),
                TrainBatchStats(
                    target_count=int(nodes.size),
                    positive_count=int(pos_nodes.size),
                    negative_count=int(neg_nodes.size),
                    background_negative_count=int(np.sum(neg_background_mask)),
                ),
            )

        negative_ratio = max(float(self.graph_config.train_negative_ratio), 0.0)
        if negative_ratio <= 0.0:
            negative_ratio = float(neg_nodes.size / max(pos_nodes.size, 1))
        sampled_negatives = min(
            neg_nodes.size,
            max(1, int(math.ceil(pos_nodes.size * negative_ratio))),
        )
        neg_nodes, neg_positions, hard_negative_count, background_negative_count = self._sample_negative_partition(
            neg_nodes=neg_nodes,
            neg_positions=neg_positions,
            neg_background_mask=neg_background_mask,
            sampled_negatives=sampled_negatives,
            snapshot_end=snapshot_end,
            rng=rng,
        )

        pos_order = rng.permutation(pos_nodes.size)
        neg_order = rng.permutation(neg_nodes.size)
        pos_nodes = pos_nodes[pos_order]
        pos_positions = pos_positions[pos_order]
        neg_nodes = neg_nodes[neg_order]
        neg_positions = neg_positions[neg_order]

        pos_per_batch = max(1, int(math.floor(effective_batch_size / (1.0 + negative_ratio))))
        neg_per_batch = max(1, effective_batch_size - pos_per_batch)
        batches: list[tuple[np.ndarray, np.ndarray, int | None]] = []
        pos_ptr = 0
        neg_ptr = 0
        while pos_ptr < pos_nodes.size or neg_ptr < neg_nodes.size:
            batch_node_parts: list[np.ndarray] = []
            batch_position_parts: list[np.ndarray] = []

            if pos_ptr < pos_nodes.size:
                next_pos_ptr = min(pos_ptr + pos_per_batch, pos_nodes.size)
                batch_node_parts.append(pos_nodes[pos_ptr:next_pos_ptr])
                batch_position_parts.append(pos_positions[pos_ptr:next_pos_ptr])
                pos_ptr = next_pos_ptr

            if neg_ptr < neg_nodes.size:
                next_neg_ptr = min(neg_ptr + neg_per_batch, neg_nodes.size)
                batch_node_parts.append(neg_nodes[neg_ptr:next_neg_ptr])
                batch_position_parts.append(neg_positions[neg_ptr:next_neg_ptr])
                neg_ptr = next_neg_ptr

            batch_nodes = np.concatenate(batch_node_parts, axis=0)
            batch_positions = np.concatenate(batch_position_parts, axis=0)
            order = rng.permutation(batch_nodes.size)
            batches.append(
                (
                    batch_nodes[order],
                    batch_positions[order],
                    snapshot_end,
                )
            )

        return (
            batches,
            TrainBatchStats(
                target_count=int(pos_nodes.size + neg_nodes.size),
                positive_count=int(pos_nodes.size),
                negative_count=int(neg_nodes.size),
                hard_negative_count=int(hard_negative_count),
                background_negative_count=int(background_negative_count),
            ),
        )

    def _build_train_batches(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        epoch: int,
        rng: np.random.Generator,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, int | None]], TrainBatchStats]:
        nodes = np.asarray(node_ids, dtype=np.int32)
        effective_batch_size = self.batch_size
        if (
            self.graph_config.train_negative_ratio <= 0.0
            and not self.graph_config.include_historical_background_negatives
        ):
            batches = self._iter_batches(
                context=context,
                node_ids=nodes,
                training=True,
                rng=rng,
            )
            labels = self._binary_train_labels(context.labels[nodes])
            pos_count = int(np.sum(labels == 1))
            neg_count = int(np.sum(labels == 0))
            return (
                batches,
                TrainBatchStats(
                    target_count=int(nodes.size),
                    positive_count=pos_count,
                    negative_count=neg_count,
                    background_negative_count=0,
                ),
            )

        batches: list[tuple[np.ndarray, np.ndarray, int | None]] = []
        total_targets = 0
        total_pos = 0
        total_neg = 0
        total_hard_neg = 0
        total_background_neg = 0
        for partition_nodes, partition_positions, snapshot_end in self._iter_train_partitions(
            context=context,
            node_ids=nodes,
        ):
            partition_labels = self._binary_train_labels(context.labels[partition_nodes])
            background_negative_mask = np.zeros(partition_nodes.shape[0], dtype=bool)
            background_nodes = self._sample_partition_background_negatives(
                context=context,
                partition_nodes=partition_nodes,
                snapshot_end=snapshot_end,
                epoch=epoch,
                rng=rng,
            )
            if background_nodes.size:
                partition_nodes = np.concatenate([partition_nodes, background_nodes]).astype(np.int32, copy=False)
                partition_positions = np.concatenate(
                    [partition_positions, np.full(background_nodes.shape[0], -1, dtype=np.int32)]
                ).astype(np.int32, copy=False)
                partition_labels = np.concatenate(
                    [partition_labels, np.zeros(background_nodes.shape[0], dtype=np.int8)]
                ).astype(np.int8, copy=False)
                background_negative_mask = np.concatenate(
                    [background_negative_mask, np.ones(background_nodes.shape[0], dtype=bool)]
                )
            partition_batches, partition_stats = self._build_balanced_partition_batches(
                nodes=partition_nodes,
                positions=partition_positions,
                labels=partition_labels,
                background_negative_mask=background_negative_mask,
                snapshot_end=snapshot_end,
                rng=rng,
                effective_batch_size=effective_batch_size,
            )
            batches.extend(partition_batches)
            total_targets += partition_stats.target_count
            total_pos += partition_stats.positive_count
            total_neg += partition_stats.negative_count
            total_hard_neg += partition_stats.hard_negative_count
            total_background_neg += partition_stats.background_negative_count
        return (
            batches,
            TrainBatchStats(
                target_count=total_targets,
                positive_count=total_pos,
                negative_count=total_neg,
                hard_negative_count=total_hard_neg,
                background_negative_count=total_background_neg,
            ),
        )

    def _sample_batch_subgraph(
        self,
        graph: GraphCache,
        context: GraphPhaseContext,
        batch_nodes: np.ndarray,
        rng: np.random.Generator,
        snapshot_end: int | None,
        training: bool,
    ) -> SampledSubgraph:
        sampler = (
            sample_batched_relation_subgraphs
            if self.graph_config.subgraph_head == "meanmax"
            else sample_relation_subgraph
        )
        neighbor_sampler = self.graph_config.neighbor_sampler if self.temporal else "uniform"
        return sampler(
            graph=graph,
            seed_nodes=batch_nodes,
            fanouts=self.fanouts,
            rng=rng,
            snapshot_end=snapshot_end,
            relation_weight=self._relation_sampling_weight,
            node_profile=context.sampling_profile,
            sampler=neighbor_sampler,
            recent_window=self.graph_config.recent_window,
            recent_ratio=self.graph_config.recent_ratio,
            consistency_temperature=self.graph_config.consistency_temperature,
            training=training,
        )

    def _tensorize_subgraph(
        self,
        context: GraphPhaseContext,
        subgraph: SampledSubgraph,
        snapshot_end: int | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        x_np = context.feature_store.take_rows(subgraph.node_ids)
        if self.graph_config.known_label_feature and context.known_label_codes is not None:
            label_codes = np.asarray(context.known_label_codes[subgraph.node_ids], dtype=np.int64).copy()
            label_dim = max(int(self.graph_config.known_label_feature_dim), 1)
            unknown_index = label_dim - 1
            if subgraph.target_local_idx.size:
                label_codes[np.asarray(subgraph.target_local_idx, dtype=np.int64)] = 4
            if label_dim == 5:
                label_index = np.clip(label_codes, 0, 4)
            elif label_dim == 3:
                label_index = np.full(label_codes.shape[0], unknown_index, dtype=np.int64)
                label_index[np.isin(label_codes, (0, 2, 3))] = 0
                label_index[label_codes == 1] = 1
                label_index[label_codes == 4] = unknown_index
            else:
                label_index = np.clip(label_codes, 0, unknown_index)
                label_index[label_codes == 4] = unknown_index
            label_features = np.zeros((label_codes.shape[0], label_dim), dtype=np.float32)
            label_features[np.arange(label_codes.shape[0]), label_index] = 1.0
            x_np = np.concatenate([x_np, label_features], axis=1).astype(np.float32, copy=False)
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        edge_src = torch.as_tensor(subgraph.edge_src, dtype=torch.long, device=self.device)
        edge_dst = torch.as_tensor(subgraph.edge_dst, dtype=torch.long, device=self.device)
        rel_ids = torch.as_tensor(subgraph.rel_ids, dtype=torch.long, device=self.device)
        target_idx = torch.as_tensor(
            subgraph.target_local_idx,
            dtype=torch.long,
            device=self.device,
        )

        edge_relative_time = None
        if self.temporal and subgraph.edge_timestamp.size:
            snapshot = snapshot_end if snapshot_end is not None else self.max_day
            relative_time = (
                snapshot - subgraph.edge_timestamp.astype(np.float32, copy=False)
            ) / max(float(self.max_day), 1.0)
            relative_time = np.clip(relative_time, 0.0, 1.0)
            edge_relative_time = torch.as_tensor(
                relative_time.reshape(-1, 1),
                dtype=torch.float32,
                device=self.device,
            )

        node_subgraph_id = None
        if subgraph.node_subgraph_id is not None:
            node_subgraph_id = torch.as_tensor(
                subgraph.node_subgraph_id,
                dtype=torch.long,
                device=self.device,
            )

        edge_subgraph_id = None
        if subgraph.edge_subgraph_id is not None:
            edge_subgraph_id = torch.as_tensor(
                subgraph.edge_subgraph_id,
                dtype=torch.long,
                device=self.device,
            )

        node_hop_depth = None
        if subgraph.node_hop_depth is not None:
            node_hop_depth = torch.as_tensor(
                subgraph.node_hop_depth,
                dtype=torch.long,
                device=self.device,
            )

        target_context_x = None
        if subgraph.target_local_idx.size:
            target_global_ids = subgraph.node_ids[np.asarray(subgraph.target_local_idx, dtype=np.int64)]
            target_context_blocks: list[np.ndarray] = []
            if context.target_context_store is not None:
                target_context_blocks.append(context.target_context_store.take_rows(target_global_ids))
            if target_context_blocks:
                target_context_np = (
                    target_context_blocks[0]
                    if len(target_context_blocks) == 1
                    else np.concatenate(target_context_blocks, axis=1).astype(np.float32, copy=False)
                )
                target_context_x = torch.as_tensor(
                    target_context_np,
                    dtype=torch.float32,
                    device=self.device,
                )

        return (
            x,
            edge_src,
            edge_dst,
            rel_ids,
            edge_relative_time,
            target_idx,
            node_subgraph_id,
            edge_subgraph_id,
            node_hop_depth,
            target_context_x,
        )

    def fit(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
        test_pool_ids: np.ndarray | None = None,
        artifact_dir: Path | None = None,
    ) -> dict[str, float]:
        set_global_seed(self.seed)
        train_ids = np.asarray(train_ids, dtype=np.int32)
        val_ids = np.asarray(val_ids, dtype=np.int32)
        test_pool_ids = (
            np.empty(0, dtype=np.int32)
            if test_pool_ids is None
            else np.asarray(test_pool_ids, dtype=np.int32)
        )
        train_labels = self._binary_train_labels(context.labels[train_ids]).astype(np.float32, copy=False)
        val_labels = context.labels[val_ids].astype(np.int8, copy=False)
        self._refresh_message_risk_feature_slice(context)
        primary_class_weight = self._compute_primary_class_weight(context=context, train_ids=train_ids)
        self.training_history: list[dict[str, Any]] = []
        self._context_budget_schedule_state = {}
        self._prototype_weight_schedule_state = {}
        aux_class_weight = self._compute_aux_class_weight(context=context, train_ids=train_ids)
        prototype_class_weight = self._compute_prototype_class_weight(context=context, train_ids=train_ids)
        self._relation_sampling_weight = self._build_relation_sampling_weight(
            context=context,
            train_ids=train_ids,
        )
        prototype_regularizer = self._build_prototype_regularizer(context=context)
        self._normal_alignment_regularizer = self._build_normal_alignment_regularizer(context=context)
        normal_alignment_regularizer = self._normal_alignment_regularizer
        self._normal_bucket_discriminator = self._build_normal_bucket_discriminator(context=context)
        normal_bucket_discriminator = self._normal_bucket_discriminator

        log_path = None if artifact_dir is None else artifact_dir / "train.log"
        history_jsonl_path = None if artifact_dir is None else artifact_dir / "epoch_metrics.jsonl"
        history_csv_path = None if artifact_dir is None else artifact_dir / "epoch_metrics.csv"
        curve_path = None if artifact_dir is None else artifact_dir / "training_curves.png"
        fit_summary_path = None if artifact_dir is None else artifact_dir / "fit_summary.json"
        if artifact_dir is not None:
            ensure_dir(artifact_dir)
            for path in (log_path, history_jsonl_path):
                if path is not None:
                    path.write_text("", encoding="utf-8")

        optimizer_params: list[torch.nn.Parameter] = list(self.network.parameters())
        if normal_bucket_discriminator is not None:
            optimizer_params.extend(list(normal_bucket_discriminator.parameters()))
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.graph_config.learning_rate,
            weight_decay=self.graph_config.weight_decay,
        )
        scheduler = None
        if self.graph_config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
            )

        if self._primary_multiclass_enabled():
            pos_weight = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        else:
            pos_count = float(np.sum(train_labels == 1))
            neg_count = float(np.sum(train_labels == 0))
            effective_neg_count = neg_count
            if self.graph_config.train_negative_ratio > 0.0:
                effective_neg_count = min(
                    neg_count,
                    float(math.ceil(pos_count * self.graph_config.train_negative_ratio)),
                )
            pos_weight = torch.tensor(
                [effective_neg_count / max(pos_count, 1.0)],
                dtype=torch.float32,
                device=self.device,
            )
        if log_path is not None:
            _append_text_line(
                log_path,
                (
                    f"[{self.model_name}] seed={self.seed} phase={context.phase} "
                    f"train_size={train_ids.size} val_size={val_ids.size} "
                    f"batch_size={self.batch_size} eval_batch_size={self.eval_batch_size} "
                    f"aggregator_type={self.aggregator_type} "
                    f"neighbor_sampler={self.graph_config.neighbor_sampler} "
                    f"relation_risk_sampler={self._relation_risk_sampler_enabled()} "
                    f"consistency_sampler={self._consistency_sampler_enabled()} "
                    f"recent_window={self.graph_config.recent_window} "
                    f"recent_ratio={self.graph_config.recent_ratio:.4f} "
                    f"consistency_temperature={self.graph_config.consistency_temperature:.4f} "
                    f"message_risk_strength={self.graph_config.message_risk_strength:.4f} "
                    f"train_negative_ratio={self.graph_config.train_negative_ratio:.4f} "
                    f"historical_background_negative_ratio="
                    f"{self.graph_config.historical_background_negative_ratio:.4f} "
                    f"historical_background_negative_warmup_epochs="
                    f"{self.graph_config.historical_background_negative_warmup_epochs} "
                    f"historical_background_aux_only={self.graph_config.historical_background_aux_only} "
                    f"negative_sampler={self.graph_config.negative_sampler} "
                    f"early_stop_patience={self.graph_config.early_stop_patience} "
                    f"min_early_stop_epoch={self.graph_config.min_early_stop_epoch} "
                    f"loss_type={self.graph_config.loss_type} "
                    f"ranking_weight={self.graph_config.ranking_weight:.4f} "
                    f"ranking_margin={self.graph_config.ranking_margin:.4f} "
                    f"focal_gamma={self.graph_config.focal_gamma:.4f} "
                    f"focal_alpha={self.graph_config.focal_alpha:.4f} "
                    f"primary_multiclass_num_classes={self.graph_config.primary_multiclass_num_classes} "
                    f"prototype_multiclass_num_classes={self.graph_config.prototype_multiclass_num_classes} "
                    f"prototype_loss_weight={self.graph_config.prototype_loss_weight:.4f} "
                    f"prototype_loss_weight_schedule={self.graph_config.prototype_loss_weight_schedule} "
                    f"prototype_loss_min_weight={self.graph_config.prototype_loss_min_weight:.4f} "
                    f"prototype_temperature={self.graph_config.prototype_temperature:.4f} "
                    f"prototype_momentum={self.graph_config.prototype_momentum:.4f} "
                    f"prototype_start_epoch={self.graph_config.prototype_start_epoch} "
                    f"prototype_loss_ramp_epochs={self.graph_config.prototype_loss_ramp_epochs} "
                    f"prototype_bucket_mode={self.graph_config.prototype_bucket_mode} "
                    f"prototype_neighbor_blend={self.graph_config.prototype_neighbor_blend:.4f} "
                    f"prototype_global_blend={self.graph_config.prototype_global_blend:.4f} "
                    f"prototype_consistency_weight={self.graph_config.prototype_consistency_weight:.4f} "
                    f"prototype_separation_weight={self.graph_config.prototype_separation_weight:.4f} "
                    f"prototype_separation_margin={self.graph_config.prototype_separation_margin:.4f} "
                    f"target_context_fusion={self.graph_config.target_context_fusion} "
                    f"atm_gate_strength={self.graph_config.atm_gate_strength:.4f} "
                    f"context_residual_scale={self.graph_config.context_residual_scale:.4f} "
                    f"context_residual_clip={self.graph_config.context_residual_clip:.4f} "
                    f"context_residual_budget={self.graph_config.context_residual_budget:.4f} "
                    f"context_residual_budget_weight={self.graph_config.context_residual_budget_weight:.4f} "
                    f"context_residual_budget_schedule={self.graph_config.context_residual_budget_schedule} "
                    f"context_residual_budget_min_weight="
                    f"{self.graph_config.context_residual_budget_min_weight:.4f} "
                    f"context_residual_budget_release_epochs="
                    f"{self.graph_config.context_residual_budget_release_epochs} "
                    f"context_residual_budget_release_delay_epochs="
                    f"{self.graph_config.context_residual_budget_release_delay_epochs} "
                    f"normal_bucket_align_weight={self.graph_config.normal_bucket_align_weight:.4f} "
                    f"normal_bucket_shift_strength={self.graph_config.normal_bucket_shift_strength:.4f} "
                    f"target_time_adapter_strength={self.graph_config.target_time_adapter_strength:.4f} "
                    f"target_time_adapter_type={self.graph_config.target_time_adapter_type} "
                    f"target_time_adapter_num_experts={self.graph_config.target_time_adapter_num_experts} "
                    f"target_time_expert_entropy_floor="
                    f"{self.graph_config.target_time_expert_entropy_floor:.4f} "
                    f"target_time_expert_entropy_weight="
                    f"{self.graph_config.target_time_expert_entropy_weight:.4f} "
                    f"normal_bucket_adv_weight={self.graph_config.normal_bucket_adv_weight:.4f} "
                    f"aux_multiclass_num_classes={self.graph_config.aux_multiclass_num_classes} "
                    f"aux_multiclass_loss_weight={self.graph_config.aux_multiclass_loss_weight:.4f} "
                    f"aux_inference_blend={self.graph_config.aux_inference_blend:.4f} "
                    f"pseudo_contrastive_weight={self.graph_config.pseudo_contrastive_weight:.4f} "
                    f"pseudo_contrastive_temperature={self.graph_config.pseudo_contrastive_temperature:.4f} "
                    f"pseudo_contrastive_sample_size={self.graph_config.pseudo_contrastive_sample_size} "
                    f"pseudo_contrastive_low_quantile={self.graph_config.pseudo_contrastive_low_quantile:.4f} "
                    f"pseudo_contrastive_high_quantile={self.graph_config.pseudo_contrastive_high_quantile:.4f} "
                    f"pseudo_contrastive_interval={self.graph_config.pseudo_contrastive_interval} "
                    f"pseudo_contrastive_start_epoch={self.graph_config.pseudo_contrastive_start_epoch} "
                    f"pseudo_contrastive_time_balanced={self.graph_config.pseudo_contrastive_time_balanced} "
                    f"pseudo_contrastive_min_confidence_gap="
                    f"{self.graph_config.pseudo_contrastive_min_confidence_gap:.4f} "
                    f"historical_background_pool="
                    f"{0 if context.historical_background_ids is None else int(context.historical_background_ids.size)} "
                    f"test_pool_size={int(test_pool_ids.size)} "
                    f"loss_pos_weight={float(pos_weight.item()):.4f}"
                ),
            )

        best_state = None
        best_val_auc = -math.inf
        best_epoch = -1
        epochs_without_improvement = 0
        epoch_rng = np.random.default_rng(self.seed)

        with tqdm(
            range(1, self.epochs + 1),
            desc=f"{self.model_name}:seed{self.seed}:epochs",
            unit="epoch",
            dynamic_ncols=True,
        ) as epoch_pbar:
            for epoch in epoch_pbar:
                self.network.train()
                normal_bucket_adv_memory_embedding = None
                normal_bucket_adv_memory_bucket_ids = None
                context_budget_weight = self._current_context_residual_budget_weight(epoch)
                prototype_loss_weight = self._current_prototype_loss_weight(epoch)
                batch_losses: list[float] = []
                batch_base_losses: list[float] = []
                batch_ranking_losses: list[float] = []
                batch_prototype_losses: list[float] = []
                batch_normal_align_losses: list[float] = []
                batch_normal_bucket_adv_losses: list[float] = []
                batch_aux_losses: list[float] = []
                batch_context_regularization_losses: list[float] = []
                batch_adapter_regularization_losses: list[float] = []
                batch_pseudo_contrastive_losses: list[float] = []
                batch_subgraph_nodes: list[float] = []
                batch_subgraph_edges: list[float] = []
                batch_emb_norm: list[float] = []
                batch_dirichlet: list[float] = []
                batch_grad_norm: list[float] = []
                batch_extra_diagnostics: dict[str, list[float]] = {}
                hard_negative_refresh = self._maybe_refresh_hard_negative_pools(
                    context=context,
                    train_ids=train_ids,
                    epoch=epoch,
                    rng=epoch_rng,
                )
                if hard_negative_refresh["refreshed"]:
                    refresh_line = (
                        f"[{self.model_name}] hard_negative_refresh epoch={epoch} "
                        f"partitions={hard_negative_refresh['partition_count']} "
                        f"candidates={hard_negative_refresh['candidate_count']} "
                        f"pool_size={hard_negative_refresh['pool_size']}"
                    )
                    tqdm.write(refresh_line)
                    if log_path is not None:
                        _append_text_line(log_path, refresh_line)

                train_batches, train_batch_stats = self._build_train_batches(
                    context=context,
                    node_ids=train_ids,
                    epoch=epoch,
                    rng=epoch_rng,
                )
                with tqdm(
                    train_batches,
                    desc=f"{self.model_name}:seed{self.seed}:train:{epoch}/{self.epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False,
                ) as batch_pbar:
                    for batch_idx, (batch_nodes, batch_positions, snapshot_end) in enumerate(batch_pbar, start=1):
                        subgraph = self._sample_batch_subgraph(
                            graph=context.graph_cache,
                            context=context,
                            batch_nodes=batch_nodes,
                            rng=epoch_rng,
                            snapshot_end=snapshot_end,
                            training=True,
                        )
                        (
                            x,
                            edge_src,
                            edge_dst,
                            rel_ids,
                            edge_relative_time,
                            target_idx,
                            node_subgraph_id,
                            edge_subgraph_id,
                            node_hop_depth,
                            target_context_x,
                        ) = self._tensorize_subgraph(
                            context=context,
                            subgraph=subgraph,
                            snapshot_end=snapshot_end,
                        )

                        if self._primary_multiclass_enabled():
                            y_batch = torch.as_tensor(
                                self._map_primary_targets(context.labels[batch_nodes]),
                                dtype=torch.long,
                                device=self.device,
                            )
                        else:
                            y_batch = torch.as_tensor(
                                self._binary_train_labels(context.labels[batch_nodes]),
                                dtype=torch.float32,
                                device=self.device,
                            )
                        sample_weight = self._compute_target_time_weight(
                            context=context,
                            batch_nodes=batch_nodes,
                        )
                        prototype_sample_weight = sample_weight.clone() if sample_weight is not None else None
                        aux_targets = None
                        if self._aux_multiclass_enabled():
                            aux_targets = torch.as_tensor(
                                self._map_aux_targets(context.labels[batch_nodes]),
                                dtype=torch.long,
                                device=self.device,
                            )
                        prototype_targets = None
                        if self._prototype_enabled():
                            prototype_targets = torch.as_tensor(
                                self._map_prototype_targets(context.labels[batch_nodes]),
                                dtype=torch.long,
                                device=self.device,
                            )
                        raw_batch_labels = None
                        if self._normal_bucket_alignment_enabled() or self._normal_bucket_adv_enabled():
                            raw_batch_labels = torch.as_tensor(
                                np.asarray(context.labels[batch_nodes], dtype=np.int64),
                                dtype=torch.long,
                                device=self.device,
                            )
                        prototype_bucket_ids = None
                        if self._prototype_time_bucketed():
                            prototype_bucket_ids = torch.as_tensor(
                                np.asarray(context.graph_cache.node_time_bucket[batch_nodes], dtype=np.int64),
                                dtype=torch.long,
                                device=self.device,
                            )
                        normal_align_bucket_ids = None
                        if (
                            self._normal_bucket_alignment_enabled()
                            or self._normal_bucket_shift_enabled()
                            or self._normal_bucket_adv_enabled()
                        ):
                            normal_align_bucket_ids = torch.as_tensor(
                                np.asarray(context.graph_cache.node_time_bucket[batch_nodes], dtype=np.int64),
                                dtype=torch.long,
                                device=self.device,
                            )
                        if self._background_aux_only_enabled():
                            background_mask = torch.as_tensor(
                                np.asarray(batch_positions, dtype=np.int32) < 0,
                                dtype=torch.bool,
                                device=self.device,
                            )
                            if sample_weight is None:
                                sample_weight = torch.ones_like(y_batch, dtype=torch.float32, device=self.device)
                            sample_weight = sample_weight.clone()
                            sample_weight[background_mask] = 0.0
                            nonzero = sample_weight > 0
                            if torch.any(nonzero):
                                sample_weight[nonzero] = sample_weight[nonzero] / sample_weight[nonzero].mean().clamp_min(
                                    1e-6
                                )

                        optimizer.zero_grad(set_to_none=True)
                        target_embedding_shift = self._build_normal_bucket_shift(
                            bucket_ids=normal_align_bucket_ids,
                        )
                        target_time_position = self._build_target_time_position(
                            context=context,
                            batch_nodes=batch_nodes,
                        )
                        forward_output = self.network.forward_output(
                            x=x,
                            edge_src=edge_src,
                            edge_dst=edge_dst,
                            rel_ids=rel_ids,
                            edge_relative_time=edge_relative_time,
                            target_local_idx=target_idx,
                            target_context_features=target_context_x,
                            target_embedding_shift=target_embedding_shift,
                            target_time_position=target_time_position,
                            node_subgraph_id=node_subgraph_id,
                            edge_subgraph_id=edge_subgraph_id,
                            node_hop_depth=node_hop_depth,
                            include_diagnostics=True,
                            include_embedding=(
                                self._prototype_enabled()
                                or self._normal_bucket_alignment_enabled()
                                or self._normal_bucket_adv_enabled()
                            ),
                            include_aux=self._aux_multiclass_enabled(),
                        )
                        loss, loss_parts = self._compute_loss(
                            logits=forward_output.logits,
                            targets=y_batch,
                            pos_weight=pos_weight,
                            sample_weight=sample_weight,
                            aux_logits=forward_output.aux_logits,
                            aux_targets=aux_targets,
                            primary_class_weight=primary_class_weight,
                            aux_class_weight=aux_class_weight,
                        )
                        context_regularization_loss = (
                            forward_output.context_regularization_loss
                            if forward_output.context_regularization_loss is not None
                            else forward_output.logits.new_tensor(0.0)
                        )
                        if context_budget_weight > 0.0:
                            loss = (
                                loss
                                + context_budget_weight * context_regularization_loss
                            )
                        adapter_regularization_loss = (
                            forward_output.adapter_regularization_loss
                            if forward_output.adapter_regularization_loss is not None
                            else forward_output.logits.new_tensor(0.0)
                        )
                        adapter_entropy_weight = max(
                            float(self.graph_config.target_time_expert_entropy_weight),
                            0.0,
                        )
                        if adapter_entropy_weight > 0.0:
                            loss = loss + adapter_entropy_weight * adapter_regularization_loss
                        prototype_loss = forward_output.logits.new_tensor(0.0)
                        if self._prototype_enabled():
                            if (
                                forward_output.embedding is None
                                or prototype_targets is None
                                or prototype_regularizer is None
                            ):
                                raise RuntimeError("Prototype regularizer expected embeddings and initialized memory.")
                            prototype_loss = prototype_regularizer.compute_loss(
                                embedding=forward_output.embedding,
                                targets=prototype_targets,
                                epoch=epoch,
                                sample_weight=prototype_sample_weight,
                                class_weight=prototype_class_weight,
                                bucket_ids=prototype_bucket_ids,
                            )
                            loss = loss + prototype_loss_weight * prototype_loss
                            for diag_name, diag_value in prototype_regularizer.last_metrics.items():
                                batch_extra_diagnostics.setdefault(str(diag_name), []).append(float(diag_value))
                        normal_align_loss = forward_output.logits.new_tensor(0.0)
                        if (
                            normal_alignment_regularizer is not None
                            and forward_output.embedding is not None
                            and raw_batch_labels is not None
                            and normal_align_bucket_ids is not None
                        ):
                            normal_align_loss = normal_alignment_regularizer.compute_loss(
                                embedding=forward_output.embedding,
                                raw_labels=raw_batch_labels,
                                bucket_ids=normal_align_bucket_ids,
                                epoch=epoch,
                            )
                        if self._normal_bucket_alignment_enabled():
                            loss = loss + float(self.graph_config.normal_bucket_align_weight) * normal_align_loss
                        normal_bucket_adv_loss = self._compute_normal_bucket_adv_loss(
                            embedding=forward_output.embedding,
                            raw_labels=raw_batch_labels,
                            bucket_ids=normal_align_bucket_ids,
                            discriminator=normal_bucket_discriminator,
                            memory_embedding=normal_bucket_adv_memory_embedding,
                            memory_bucket_ids=normal_bucket_adv_memory_bucket_ids,
                        )
                        if self._normal_bucket_adv_enabled():
                            loss = loss + float(self.graph_config.normal_bucket_adv_weight) * normal_bucket_adv_loss
                        (
                            normal_bucket_adv_memory_embedding,
                            normal_bucket_adv_memory_bucket_ids,
                        ) = self._append_normal_bucket_adv_memory(
                            embedding=forward_output.embedding,
                            raw_labels=raw_batch_labels,
                            bucket_ids=normal_align_bucket_ids,
                            memory_embedding=normal_bucket_adv_memory_embedding,
                            memory_bucket_ids=normal_bucket_adv_memory_bucket_ids,
                        )
                        pseudo_contrastive_loss = forward_output.logits.new_tensor(0.0)
                        pseudo_interval = max(int(self.graph_config.pseudo_contrastive_interval), 1)
                        should_run_pseudo = (
                            float(self.graph_config.pseudo_contrastive_weight) > 0.0
                            and test_pool_ids.size > 0
                            and (batch_idx % pseudo_interval == 0)
                        )
                        if should_run_pseudo:
                            pseudo_contrastive_loss, pseudo_diagnostics = self._compute_pseudo_contrastive_loss(
                                context=context,
                                test_pool_ids=test_pool_ids,
                                snapshot_end=snapshot_end,
                                epoch=epoch,
                                rng=epoch_rng,
                            )
                            if pseudo_diagnostics:
                                for diag_name, diag_value in pseudo_diagnostics.items():
                                    batch_extra_diagnostics.setdefault(str(diag_name), []).append(float(diag_value))
                            loss = (
                                loss
                                + float(self.graph_config.pseudo_contrastive_weight) * pseudo_contrastive_loss
                            )
                        loss_parts["prototype_loss"] = float(prototype_loss.detach().item())
                        loss_parts["normal_align_loss"] = float(normal_align_loss.detach().item())
                        loss_parts["normal_bucket_adv_loss"] = float(normal_bucket_adv_loss.detach().item())
                        loss_parts["context_regularization_loss"] = float(
                            context_regularization_loss.detach().item()
                        )
                        loss_parts["adapter_regularization_loss"] = float(
                            adapter_regularization_loss.detach().item()
                        )
                        loss_parts["pseudo_contrastive_loss"] = float(
                            pseudo_contrastive_loss.detach().item()
                        )
                        loss_parts["total_loss"] = float(loss.detach().item())
                        loss.backward()

                        if self.graph_config.grad_clip > 0:
                            grad_norm = float(
                                torch.nn.utils.clip_grad_norm_(
                                    self.network.parameters(),
                                    self.graph_config.grad_clip,
                                ).item()
                            )
                        else:
                            grad_norm = _compute_grad_norm(self.network.parameters())

                        optimizer.step()
                        batch_losses.append(float(loss_parts["total_loss"]))
                        batch_base_losses.append(float(loss_parts["base_loss"]))
                        batch_ranking_losses.append(float(loss_parts["ranking_loss"]))
                        batch_prototype_losses.append(float(loss_parts["prototype_loss"]))
                        batch_normal_align_losses.append(float(loss_parts["normal_align_loss"]))
                        batch_normal_bucket_adv_losses.append(float(loss_parts["normal_bucket_adv_loss"]))
                        batch_aux_losses.append(float(loss_parts["aux_loss"]))
                        batch_context_regularization_losses.append(
                            float(loss_parts["context_regularization_loss"])
                        )
                        batch_adapter_regularization_losses.append(
                            float(loss_parts["adapter_regularization_loss"])
                        )
                        batch_pseudo_contrastive_losses.append(
                            float(loss_parts["pseudo_contrastive_loss"])
                        )
                        batch_subgraph_nodes.append(float(subgraph.node_ids.shape[0]))
                        batch_subgraph_edges.append(float(subgraph.edge_src.shape[0]))
                        forward_diagnostics = forward_output.diagnostics or {}
                        batch_emb_norm.append(float(forward_diagnostics["emb_norm"]))
                        batch_dirichlet.append(float(forward_diagnostics["dirichlet_energy"]))
                        for diag_name, diag_value in forward_diagnostics.items():
                            if diag_name in {"emb_norm", "dirichlet_energy"}:
                                continue
                            batch_extra_diagnostics.setdefault(str(diag_name), []).append(float(diag_value))
                        batch_grad_norm.append(float(grad_norm))
                        batch_pbar.set_postfix(
                            loss=f"{batch_losses[-1]:.4f}",
                            nodes=int(subgraph.node_ids.shape[0]),
                            edges=int(subgraph.edge_src.shape[0]),
                            refresh=False,
                        )

                val_prob = self.predict_proba(
                    context=context,
                    node_ids=val_ids,
                    batch_seed=self.seed + 1000,
                    progress_desc=f"{self.model_name}:seed{self.seed}:val:{epoch}/{self.epochs}",
                )
                val_metrics = compute_binary_classification_metrics(val_labels, val_prob)
                val_auc = val_metrics["auc"]
                if scheduler is not None:
                    scheduler.step(1.0 - val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.network.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                current_lr = float(optimizer.param_groups[0]["lr"])
                epoch_pbar.set_postfix(
                    val_auc=f"{val_auc:.4f}",
                    best_epoch=best_epoch,
                    lr=f"{current_lr:.2e}",
                    refresh=False,
                )
                epoch_row = {
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(batch_losses)),
                    "train_base_loss": float(np.mean(batch_base_losses)),
                    "train_ranking_loss": float(np.mean(batch_ranking_losses)),
                    "train_prototype_loss": float(np.mean(batch_prototype_losses)),
                    "train_normal_align_loss": float(np.mean(batch_normal_align_losses)),
                    "train_normal_bucket_adv_loss": float(np.mean(batch_normal_bucket_adv_losses)),
                    "train_aux_loss": float(np.mean(batch_aux_losses)),
                    "train_context_regularization_loss": float(np.mean(batch_context_regularization_losses)),
                    "train_adapter_regularization_loss": float(
                        np.mean(batch_adapter_regularization_losses)
                    ),
                    "train_pseudo_contrastive_loss": float(
                        np.mean(batch_pseudo_contrastive_losses)
                    ),
                    "prototype_loss_weight_effective": float(prototype_loss_weight),
                    "prototype_trust_score": float(
                        self._prototype_weight_schedule_state.get("trust_score", 1.0)
                    ),
                    "context_residual_budget_weight_effective": float(context_budget_weight),
                    "context_residual_budget_release_progress": float(
                        self._context_budget_schedule_state.get("release_progress", 0.0)
                    ),
                    "context_residual_budget_prototype_readiness": float(
                        self._context_budget_schedule_state.get("prototype_readiness", 0.0)
                    ),
                    "context_residual_budget_guard_strength": float(
                        self._context_budget_schedule_state.get("guard_strength", 0.0)
                    ),
                    "val_auc": float(val_auc),
                    "val_pr_auc": float(val_metrics["pr_auc"]),
                    "val_ap": float(val_metrics["ap"]),
                    "train_targets": int(train_batch_stats.target_count),
                    "train_pos": int(train_batch_stats.positive_count),
                    "train_neg": int(train_batch_stats.negative_count),
                    "train_hard_neg": int(train_batch_stats.hard_negative_count),
                    "train_bg_neg": int(train_batch_stats.background_negative_count),
                    "train_pos_rate": float(train_batch_stats.positive_rate),
                    "hard_negative_pool_size": int(self._current_hard_negative_pool_size()),
                    "hard_negative_candidate_count": int(self._current_hard_negative_candidate_count()),
                    "avg_subgraph_nodes": float(np.mean(batch_subgraph_nodes)),
                    "avg_subgraph_edges": float(np.mean(batch_subgraph_edges)),
                    "emb_norm": float(np.mean(batch_emb_norm)),
                    "dirichlet": float(np.mean(batch_dirichlet)),
                    "grad_norm": float(np.mean(batch_grad_norm)),
                    "best_epoch": int(best_epoch),
                    "lr": float(current_lr),
                    "loss_pos_weight": float(pos_weight.item()),
                }
                for diag_name, diag_values in sorted(batch_extra_diagnostics.items()):
                    if diag_values:
                        epoch_row[diag_name] = float(np.mean(diag_values))
                self.training_history.append(epoch_row)
                extra_diag_log = " ".join(
                    f"{diag_name}={epoch_row[diag_name]:.6f}"
                    for diag_name in sorted(batch_extra_diagnostics)
                    if diag_name in epoch_row
                )
                epoch_log_line = (
                    f"[{self.model_name}] epoch={epoch} "
                    f"train_loss={epoch_row['train_loss']:.6f} "
                    f"train_base_loss={epoch_row['train_base_loss']:.6f} "
                    f"train_ranking_loss={epoch_row['train_ranking_loss']:.6f} "
                    f"train_prototype_loss={epoch_row['train_prototype_loss']:.6f} "
                    f"train_normal_align_loss={epoch_row['train_normal_align_loss']:.6f} "
                    f"train_normal_bucket_adv_loss={epoch_row['train_normal_bucket_adv_loss']:.6f} "
                    f"train_aux_loss={epoch_row['train_aux_loss']:.6f} "
                    f"train_context_regularization_loss={epoch_row['train_context_regularization_loss']:.6f} "
                    f"train_adapter_regularization_loss="
                    f"{epoch_row['train_adapter_regularization_loss']:.6f} "
                    f"train_pseudo_contrastive_loss={epoch_row['train_pseudo_contrastive_loss']:.6f} "
                    f"prototype_loss_weight_effective={epoch_row['prototype_loss_weight_effective']:.6f} "
                    f"prototype_trust_score={epoch_row['prototype_trust_score']:.6f} "
                    f"context_residual_budget_weight_effective="
                    f"{epoch_row['context_residual_budget_weight_effective']:.6f} "
                    f"context_residual_budget_release_progress="
                    f"{epoch_row['context_residual_budget_release_progress']:.6f} "
                    f"context_residual_budget_prototype_readiness="
                    f"{epoch_row['context_residual_budget_prototype_readiness']:.6f} "
                    f"context_residual_budget_guard_strength="
                    f"{epoch_row['context_residual_budget_guard_strength']:.6f} "
                    f"val_auc={val_auc:.6f} "
                    f"val_pr_auc={val_metrics['pr_auc']:.6f} "
                    f"val_ap={val_metrics['ap']:.6f} "
                    f"train_targets={train_batch_stats.target_count} "
                    f"train_pos={train_batch_stats.positive_count} "
                    f"train_neg={train_batch_stats.negative_count} "
                    f"train_hard_neg={train_batch_stats.hard_negative_count} "
                    f"train_bg_neg={train_batch_stats.background_negative_count} "
                    f"train_pos_rate={train_batch_stats.positive_rate:.4f} "
                    f"hard_negative_pool_size={self._current_hard_negative_pool_size()} "
                    f"avg_subgraph_nodes={np.mean(batch_subgraph_nodes):.2f} "
                    f"avg_subgraph_edges={np.mean(batch_subgraph_edges):.2f} "
                    f"emb_norm={np.mean(batch_emb_norm):.6f} "
                    f"dirichlet={np.mean(batch_dirichlet):.6f} "
                    f"grad_norm={np.mean(batch_grad_norm):.6f} "
                    f"best_epoch={best_epoch} "
                    f"lr={current_lr:.6g} "
                    f"loss_pos_weight={float(pos_weight.item()):.4f}"
                )
                if extra_diag_log:
                    epoch_log_line = f"{epoch_log_line} {extra_diag_log}"
                tqdm.write(epoch_log_line)
                if log_path is not None:
                    _append_text_line(log_path, epoch_log_line)
                if history_jsonl_path is not None:
                    _append_jsonl(history_jsonl_path, epoch_row)

                if (
                    self.graph_config.early_stop_patience > 0
                    and epoch >= max(int(self.graph_config.min_early_stop_epoch), 1)
                    and epochs_without_improvement >= self.graph_config.early_stop_patience
                ):
                    early_stop_line = (
                        f"[{self.model_name}] early_stop epoch={epoch} "
                        f"patience={self.graph_config.early_stop_patience} "
                        f"min_early_stop_epoch={self.graph_config.min_early_stop_epoch}"
                    )
                    tqdm.write(early_stop_line)
                    if log_path is not None:
                        _append_text_line(log_path, early_stop_line)
                    break

        if best_state is None:
            raise RuntimeError(f"{self.model_name}: failed to capture a best checkpoint.")
        self.network.load_state_dict(best_state)
        fit_summary = {
            "val_auc": float(best_val_auc),
            "best_epoch": float(best_epoch),
            "trained_epochs": int(len(self.training_history)),
            "loss_pos_weight": float(pos_weight.item()),
            "loss_type": str(self.graph_config.loss_type),
            "negative_sampler": str(self.graph_config.negative_sampler),
            "min_early_stop_epoch": int(self.graph_config.min_early_stop_epoch),
        }
        self.fit_summary = fit_summary
        if history_csv_path is not None:
            _write_history_csv(history_csv_path, self.training_history)
        if fit_summary_path is not None:
            write_json(fit_summary_path, fit_summary)
        if curve_path is not None:
            plot_error = _plot_training_curves(curve_path, self.training_history)
            if plot_error is not None and log_path is not None:
                _append_text_line(log_path, f"[{self.model_name}] plot_warning={plot_error}")
        return fit_summary

    @torch.no_grad()
    def _predict_outputs(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        batch_seed: int | None = None,
        progress_desc: str | None = None,
        show_progress: bool = True,
        return_embeddings: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self.network.eval()
        self._refresh_message_risk_feature_slice(context)
        node_ids = np.asarray(node_ids, dtype=np.int32)
        rng = np.random.default_rng(self.seed if batch_seed is None else batch_seed)
        probabilities = np.zeros(node_ids.shape[0], dtype=np.float32)
        embeddings = (
            np.zeros((node_ids.shape[0], self.hidden_dim), dtype=np.float32)
            if return_embeddings
            else None
        )
        batches = self._iter_batches(
            context=context,
            node_ids=node_ids,
            training=False,
            rng=rng,
        )
        desc = progress_desc or f"{self.model_name}:seed{self.seed}:{context.phase}:predict"
        processed = 0
        with tqdm(
            batches,
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        ) as batch_pbar:
            for batch_nodes, batch_positions, snapshot_end in batch_pbar:
                subgraph = self._sample_batch_subgraph(
                    graph=context.graph_cache,
                    context=context,
                    batch_nodes=batch_nodes,
                    rng=rng,
                    snapshot_end=snapshot_end,
                    training=False,
                )
                (
                    x,
                    edge_src,
                    edge_dst,
                    rel_ids,
                    edge_relative_time,
                    target_idx,
                    node_subgraph_id,
                    edge_subgraph_id,
                    node_hop_depth,
                    target_context_x,
                ) = self._tensorize_subgraph(
                    context=context,
                    subgraph=subgraph,
                    snapshot_end=snapshot_end,
                )
                target_bucket_ids = None
                if self._normal_bucket_shift_enabled():
                    target_bucket_ids = torch.as_tensor(
                        np.asarray(context.graph_cache.node_time_bucket[batch_nodes], dtype=np.int64),
                        dtype=torch.long,
                        device=self.device,
                    )
                target_embedding_shift = self._build_normal_bucket_shift(bucket_ids=target_bucket_ids)
                target_time_position = self._build_target_time_position(
                    context=context,
                    batch_nodes=batch_nodes,
                )
                forward_output = self.network.forward_output(
                    x=x,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    rel_ids=rel_ids,
                    edge_relative_time=edge_relative_time,
                    target_local_idx=target_idx,
                    target_context_features=target_context_x,
                    target_embedding_shift=target_embedding_shift,
                    target_time_position=target_time_position,
                    node_subgraph_id=node_subgraph_id,
                    edge_subgraph_id=edge_subgraph_id,
                    node_hop_depth=node_hop_depth,
                    include_embedding=return_embeddings,
                    include_aux=float(self.graph_config.aux_inference_blend) > 0.0,
                )
                batch_prob = (
                    self._blend_primary_and_aux_probability(
                        logits=forward_output.logits,
                        aux_logits=forward_output.aux_logits,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                probabilities[batch_positions] = batch_prob
                if embeddings is not None:
                    embeddings[batch_positions] = (
                        forward_output.embedding.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                processed += int(batch_positions.size)
                batch_pbar.set_postfix(done=f"{processed}/{node_ids.size}", refresh=False)
        return probabilities, embeddings

    @torch.no_grad()
    def predict_proba(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        batch_seed: int | None = None,
        progress_desc: str | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        probabilities, _ = self._predict_outputs(
            context=context,
            node_ids=node_ids,
            batch_seed=batch_seed,
            progress_desc=progress_desc,
            show_progress=show_progress,
            return_embeddings=False,
        )
        return probabilities

    @torch.no_grad()
    def predict_proba_and_embeddings(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        batch_seed: int | None = None,
        progress_desc: str | None = None,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities, embeddings = self._predict_outputs(
            context=context,
            node_ids=node_ids,
            batch_seed=batch_seed,
            progress_desc=progress_desc,
            show_progress=show_progress,
            return_embeddings=True,
        )
        if embeddings is None:
            raise RuntimeError("Expected embeddings to be returned.")
        return probabilities, embeddings

    @torch.no_grad()
    def predict_embeddings(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        batch_seed: int | None = None,
        progress_desc: str | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        _, embeddings = self._predict_outputs(
            context=context,
            node_ids=node_ids,
            batch_seed=batch_seed,
            progress_desc=progress_desc,
            show_progress=show_progress,
            return_embeddings=True,
        )
        if embeddings is None:
            raise RuntimeError("Expected embeddings to be returned.")
        return embeddings

    def save(self, run_dir: Path) -> None:
        ensure_dir(run_dir)
        torch.save(self.network.state_dict(), run_dir / "model.pt")
        metadata = {
            "model_name": self.model_name,
            "seed": self.seed,
            "feature_groups": self.feature_groups,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "rel_dim": self.rel_dim,
            "fanouts": self.fanouts,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "epochs": self.epochs,
            "learning_rate": self.graph_config.learning_rate,
            "weight_decay": self.graph_config.weight_decay,
            "dropout": self.graph_config.dropout,
            "max_day": self.max_day,
            "temporal": self.temporal,
            "aggregator_type": self.aggregator_type,
            "graph_model_config": self.graph_config.to_dict(),
            "feature_normalizer_state": (
                None
                if self.feature_normalizer_state is None
                else self.feature_normalizer_state.to_dict()
            ),
            "target_context_feature_groups": self.target_context_feature_groups,
            "target_context_normalizer_state": (
                None
                if self.target_context_normalizer_state is None
                else self.target_context_normalizer_state.to_dict()
            ),
        }
        write_json(run_dir / "model_meta.json", metadata)

    @classmethod
    def load(
        cls,
        run_dir: Path,
        input_dim: int,
        num_relations: int,
        device: str | None = None,
    ) -> "BaseGraphSAGEExperiment":
        meta = json.loads((run_dir / "model_meta.json").read_text(encoding="utf-8"))
        graph_config_payload = meta.get("graph_model_config")
        if graph_config_payload is not None:
            graph_config = GraphModelConfig.from_dict(graph_config_payload)
        else:
            graph_config = GraphModelConfig(
                learning_rate=float(meta.get("learning_rate", 1e-3)),
                weight_decay=float(meta.get("weight_decay", 1e-5)),
                dropout=float(meta.get("dropout", 0.2)),
            )

        instance = cls(
            model_name=meta["model_name"],
            seed=int(meta["seed"]),
            input_dim=input_dim,
            num_relations=num_relations,
            max_day=int(meta["max_day"]),
            feature_groups=list(meta["feature_groups"]),
            hidden_dim=int(meta["hidden_dim"]),
            num_layers=int(meta["num_layers"]),
            rel_dim=int(meta["rel_dim"]),
            fanouts=list(meta["fanouts"]),
            batch_size=int(meta["batch_size"]),
            epochs=int(meta["epochs"]),
            device=device,
            temporal=bool(meta.get("temporal", False)),
            aggregator_type=str(meta.get("aggregator_type", "sage")),
            graph_config=graph_config,
            feature_normalizer_state=HybridFeatureNormalizerState.from_dict(
                meta.get("feature_normalizer_state")
            ),
            target_context_input_dim=int(graph_config.target_context_input_dim or input_dim),
            target_context_feature_groups=list(meta.get("target_context_feature_groups") or []),
            target_context_normalizer_state=HybridFeatureNormalizerState.from_dict(
                meta.get("target_context_normalizer_state")
            ),
        )
        instance.eval_batch_size = int(meta.get("eval_batch_size", max(instance.batch_size, 2048)))
        state_dict = torch.load(
            run_dir / "model.pt",
            map_location=instance.device,
            weights_only=True,
        )
        instance.network.load_state_dict(state_dict, strict=False)
        return instance


class RelationGraphSAGEExperiment(BaseGraphSAGEExperiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["temporal"] = False
        kwargs.setdefault("aggregator_type", "sage")
        super().__init__(*args, **kwargs)


class TemporalRelationGraphSAGEExperiment(BaseGraphSAGEExperiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["temporal"] = True
        kwargs.setdefault("aggregator_type", "sage")
        super().__init__(*args, **kwargs)


class TemporalRelationGATExperiment(BaseGraphSAGEExperiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["temporal"] = True
        kwargs.setdefault("aggregator_type", "attention")
        super().__init__(*args, **kwargs)


class TRGTExperiment(BaseGraphSAGEExperiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["temporal"] = True
        kwargs.setdefault("aggregator_type", "attention")
        super().__init__(*args, **kwargs)


# Backward-compatible alias for historical imports and saved metadata readers.
TemporalRelationGraphTransformerExperiment = TRGTExperiment
