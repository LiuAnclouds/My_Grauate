from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(kind: str, dim: int) -> nn.Module:
    normalized = str(kind).lower()
    if normalized == "layer":
        return nn.LayerNorm(dim)
    if normalized == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


def segment_weighted_mean(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError("Expected a 2D tensor for segment pooling.")
    pooled = values.new_zeros((num_groups, values.shape[1]))
    if num_groups == 0 or values.shape[0] == 0:
        return pooled
    if weights is None:
        weight_view = torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device)
    else:
        weight_view = weights.reshape(-1, 1).to(dtype=values.dtype)
    pooled.index_add_(0, group_ids, values * weight_view)
    denom = values.new_zeros((num_groups, 1))
    denom.index_add_(0, group_ids, weight_view)
    return pooled / denom.clamp_min(1e-6)


def segment_softmax(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    max_values = scores.new_full((num_groups,), -torch.inf)
    max_values.scatter_reduce_(0, group_ids, scores, reduce="amax", include_self=True)
    stabilized = scores - max_values[group_ids]
    exp_scores = torch.exp(stabilized)
    denom = scores.new_zeros((num_groups,))
    denom.index_add_(0, group_ids, exp_scores)
    return exp_scores / denom[group_ids].clamp_min(1e-12)


class TRGTMeanRelationBlock(nn.Module):
    """Relation-aware residual message block used by non-attention ablations."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float,
        norm: str,
        residual: bool,
        ffn: bool,
        edge_encoder: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.gated = edge_encoder == "gated"
        self.norm1 = _make_norm(norm, hidden_dim)
        self.norm2 = _make_norm(norm, hidden_dim) if ffn else nn.Identity()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.gated
            else None
        )
        self.agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            if ffn
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_emb: torch.Tensor,
        time_weight: torch.Tensor | None = None,
        message_node_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_in = x
        h = self.norm1(x)
        if edge_src.numel() == 0:
            edge_repr = h.new_zeros((0, self.hidden_dim))
            agg = h.new_zeros((h.shape[0], self.hidden_dim))
        else:
            msg = self.msg_mlp(torch.cat([h[edge_src], edge_emb], dim=-1))
            if self.gate_mlp is not None:
                gate = torch.sigmoid(
                    self.gate_mlp(torch.cat([h[edge_dst], h[edge_src], edge_emb], dim=-1))
                )
                edge_repr = gate * msg
            else:
                edge_repr = msg
            if time_weight is not None:
                edge_repr = edge_repr * time_weight
            if message_node_scale is not None:
                edge_scale = 0.5 * (message_node_scale[edge_src] + message_node_scale[edge_dst])
                edge_repr = edge_repr * edge_scale
            agg = h.new_zeros((h.shape[0], self.hidden_dim))
            agg.index_add_(0, edge_dst, edge_repr)
            deg = h.new_zeros((h.shape[0], 1))
            deg.index_add_(
                0,
                edge_dst,
                torch.ones((edge_dst.shape[0], 1), device=h.device, dtype=h.dtype),
            )
            agg = agg / deg.clamp_min(1.0)

        update = self.self_linear(h) + self.agg_proj(agg)
        update = self.dropout(update)
        out = h_in + update if self.residual else update

        if self.ffn is not None:
            ffn_update = self.ffn(self.norm2(out))
            ffn_update = self.dropout(ffn_update)
            out = out + ffn_update if self.residual else ffn_update
        return out, edge_repr


class TRGTTemporalRelationAttentionBlock(nn.Module):
    """TGAT-style temporal relation multi-head attention on sampled subgraphs."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float,
        norm: str,
        residual: bool,
        ffn: bool,
        edge_encoder: str,
        num_heads: int = 1,
        attention_logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.gated = edge_encoder == "gated"
        self.num_heads = max(int(num_heads), 1)
        if hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by attention heads={self.num_heads}."
            )
        self.head_dim = hidden_dim // self.num_heads
        self.attention_logit_scale = max(float(attention_logit_scale), 1e-6)
        self.norm1 = _make_norm(norm, hidden_dim)
        self.norm2 = _make_norm(norm, hidden_dim) if ffn else nn.Identity()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_heads),
        )
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_heads),
            )
            if self.gated
            else None
        )
        self.agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            if ffn
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_emb: torch.Tensor,
        time_weight: torch.Tensor | None = None,
        message_node_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_in = x
        h = self.norm1(x)
        if edge_src.numel() == 0:
            edge_repr = h.new_zeros((0, self.hidden_dim))
            agg = h.new_zeros((h.shape[0], self.hidden_dim))
        else:
            edge_context = torch.cat([h[edge_dst], h[edge_src], edge_emb], dim=-1)
            msg = self.msg_mlp(torch.cat([h[edge_src], edge_emb], dim=-1)).view(
                -1, self.num_heads, self.head_dim
            )
            if self.gate_mlp is not None:
                msg = msg * torch.sigmoid(self.gate_mlp(edge_context)).unsqueeze(-1)
            if time_weight is not None:
                msg = msg * time_weight.unsqueeze(-1)
            if message_node_scale is not None:
                edge_scale = 0.5 * (message_node_scale[edge_src] + message_node_scale[edge_dst])
                msg = msg * edge_scale.unsqueeze(-1)
            attn_score = self.attn_mlp(edge_context) * (
                self.attention_logit_scale / math.sqrt(float(self.head_dim))
            )
            head_offsets = torch.arange(self.num_heads, device=h.device, dtype=edge_dst.dtype).unsqueeze(0)
            attn_group_ids = edge_dst.unsqueeze(1) * self.num_heads + head_offsets
            attn_weight = segment_softmax(
                attn_score.reshape(-1),
                attn_group_ids.reshape(-1),
                h.shape[0] * self.num_heads,
            ).view(-1, self.num_heads, 1)
            edge_repr = self.attn_dropout(attn_weight) * msg
            agg = h.new_zeros((h.shape[0], self.hidden_dim))
            agg.index_add_(0, edge_dst, edge_repr.reshape(-1, self.hidden_dim))
            edge_repr = edge_repr.reshape(-1, self.hidden_dim)

        update = self.self_linear(h) + self.agg_proj(agg)
        update = self.dropout(update)
        out = h_in + update if self.residual else update

        if self.ffn is not None:
            ffn_update = self.ffn(self.norm2(out))
            ffn_update = self.dropout(ffn_update)
            out = out + ffn_update if self.residual else ffn_update
        return out, edge_repr


class TRGTInternalRiskEncoder(nn.Module):
    """Model multi-scale target risk deltas inside the pure-GNN path."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_edge_types: int,
        dropout: float,
        short_time_scale: float,
        long_time_scale: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_edge_types = max(int(num_edge_types), 1)
        self.short_time_scale = max(float(short_time_scale), 1e-3)
        self.long_time_scale = max(float(long_time_scale), self.short_time_scale + 1e-3)
        scalar_dim = max(self.hidden_dim // 4, 8)
        self.scalar_proj = nn.Sequential(
            nn.Linear(8, scalar_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * 8 + scalar_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def _time_weight(
        self,
        edge_relative_time: torch.Tensor | None,
        scale: float,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if edge_relative_time is None or edge_relative_time.numel() == 0:
            return torch.ones((size,), dtype=dtype, device=device)
        relative_time = edge_relative_time.view(-1).to(dtype=dtype)
        return torch.exp(-relative_time / max(float(scale), 1e-3))

    def _count_by_group(
        self,
        mask: torch.Tensor,
        group_ids: torch.Tensor,
        num_groups: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        counts = torch.zeros((num_groups,), dtype=dtype, device=device)
        if torch.any(mask):
            counts.index_add_(
                0,
                group_ids[mask],
                torch.ones(int(mask.sum().item()), dtype=dtype, device=device),
            )
        return counts

    def forward(
        self,
        *,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
        target_local_idx: torch.Tensor,
        node_subgraph_id: torch.Tensor | None,
        edge_subgraph_id: torch.Tensor | None,
        node_hop_depth: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        num_targets = int(target_local_idx.shape[0])
        zero = node_repr.new_zeros((num_targets, self.hidden_dim))
        if (
            num_targets == 0
            or edge_subgraph_id is None
            or node_subgraph_id is None
            or node_hop_depth is None
            or edge_repr.shape[0] == 0
        ):
            return zero, {
                "internal_risk_gate_mean": 0.0,
                "internal_risk_hop_gap_norm": 0.0,
                "internal_risk_short_long_gap_norm": 0.0,
                "internal_risk_direction_gap_norm": 0.0,
            }

        device = node_repr.device
        dtype = node_repr.dtype
        short_weight = self._time_weight(
            edge_relative_time,
            self.short_time_scale,
            edge_repr.shape[0],
            device,
            dtype,
        )
        long_weight = self._time_weight(
            edge_relative_time,
            self.long_time_scale,
            edge_repr.shape[0],
            device,
            dtype,
        )

        target_edge_mask = edge_dst == target_local_idx[edge_subgraph_id]
        inbound_mask = target_edge_mask & (rel_ids < self.num_edge_types)
        outbound_mask = target_edge_mask & (rel_ids >= self.num_edge_types)

        in_short = segment_weighted_mean(
            edge_repr[inbound_mask],
            edge_subgraph_id[inbound_mask],
            num_targets,
            short_weight[inbound_mask],
        )
        out_short = segment_weighted_mean(
            edge_repr[outbound_mask],
            edge_subgraph_id[outbound_mask],
            num_targets,
            short_weight[outbound_mask],
        )
        in_long = segment_weighted_mean(
            edge_repr[inbound_mask],
            edge_subgraph_id[inbound_mask],
            num_targets,
            long_weight[inbound_mask],
        )
        out_long = segment_weighted_mean(
            edge_repr[outbound_mask],
            edge_subgraph_id[outbound_mask],
            num_targets,
            long_weight[outbound_mask],
        )

        hop1_mask = node_hop_depth == 1
        hop2_mask = node_hop_depth >= 2
        hop1_mean = segment_weighted_mean(
            node_repr[hop1_mask],
            node_subgraph_id[hop1_mask],
            num_targets,
        )
        hop2_mean = segment_weighted_mean(
            node_repr[hop2_mask],
            node_subgraph_id[hop2_mask],
            num_targets,
        )

        hop1_edges_short = segment_weighted_mean(
            node_repr[edge_src[target_edge_mask]],
            edge_subgraph_id[target_edge_mask],
            num_targets,
            short_weight[target_edge_mask],
        )
        hop1_edges_long = segment_weighted_mean(
            node_repr[edge_src[target_edge_mask]],
            edge_subgraph_id[target_edge_mask],
            num_targets,
            long_weight[target_edge_mask],
        )

        burst_delta = F.layer_norm((in_short + out_short) - (in_long + out_long), (self.hidden_dim,))
        direction_gap = F.layer_norm(out_long - in_long, (self.hidden_dim,))
        hop_gap = F.layer_norm(hop1_mean - hop2_mean, (self.hidden_dim,))
        short_long_gap = F.layer_norm(hop1_edges_short - hop1_edges_long, (self.hidden_dim,))
        asymmetry = F.layer_norm(torch.abs(direction_gap), (self.hidden_dim,))
        hop1_mean = F.layer_norm(hop1_mean, (self.hidden_dim,))
        hop2_mean = F.layer_norm(hop2_mean, (self.hidden_dim,))
        risk_base = F.layer_norm(in_long + out_long, (self.hidden_dim,))

        edge_time_mean = torch.zeros((num_targets,), dtype=dtype, device=device)
        if edge_relative_time is not None and edge_relative_time.numel() and torch.any(target_edge_mask):
            target_edge_time = edge_relative_time.view(-1)[target_edge_mask].to(dtype=dtype)
            edge_time_mean.index_add_(0, edge_subgraph_id[target_edge_mask], target_edge_time)
            target_edge_count = self._count_by_group(
                target_edge_mask,
                edge_subgraph_id,
                num_targets,
                dtype=dtype,
                device=device,
            )
            edge_time_mean = edge_time_mean / target_edge_count.clamp_min(1e-6)
        else:
            target_edge_count = self._count_by_group(
                target_edge_mask,
                edge_subgraph_id,
                num_targets,
                dtype=dtype,
                device=device,
            )

        inbound_count = self._count_by_group(
            inbound_mask,
            edge_subgraph_id,
            num_targets,
            dtype=dtype,
            device=device,
        )
        outbound_count = self._count_by_group(
            outbound_mask,
            edge_subgraph_id,
            num_targets,
            dtype=dtype,
            device=device,
        )
        hop1_count = self._count_by_group(
            hop1_mask,
            node_subgraph_id,
            num_targets,
            dtype=dtype,
            device=device,
        )
        hop2_count = self._count_by_group(
            hop2_mask,
            node_subgraph_id,
            num_targets,
            dtype=dtype,
            device=device,
        )
        short_mass = torch.zeros((num_targets,), dtype=dtype, device=device)
        long_mass = torch.zeros((num_targets,), dtype=dtype, device=device)
        if torch.any(target_edge_mask):
            short_mass.index_add_(0, edge_subgraph_id[target_edge_mask], short_weight[target_edge_mask])
            long_mass.index_add_(0, edge_subgraph_id[target_edge_mask], long_weight[target_edge_mask])

        scalar_features = torch.stack(
            [
                torch.log1p(inbound_count),
                torch.log1p(outbound_count),
                torch.log1p(hop1_count),
                torch.log1p(hop2_count),
                edge_time_mean,
                short_mass,
                long_mass,
                short_mass - long_mass,
            ],
            dim=-1,
        )
        scalar_embedding = self.scalar_proj(scalar_features)
        fused_input = torch.cat(
            [
                burst_delta,
                direction_gap,
                hop_gap,
                short_long_gap,
                asymmetry,
                hop1_mean,
                hop2_mean,
                risk_base,
                scalar_embedding,
            ],
            dim=-1,
        )
        risk_embedding = 0.25 * torch.tanh(self.fuse(fused_input))
        diagnostics = {
            "internal_risk_hop_gap_norm": float(hop_gap.norm(dim=-1).mean().detach().item()),
            "internal_risk_short_long_gap_norm": float(short_long_gap.norm(dim=-1).mean().detach().item()),
            "internal_risk_direction_gap_norm": float(direction_gap.norm(dim=-1).mean().detach().item()),
        }
        return risk_embedding, diagnostics
