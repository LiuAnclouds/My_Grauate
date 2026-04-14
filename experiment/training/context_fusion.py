from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetContextFusionHead(nn.Module):
    """Fuse target-node context features with the graph embedding after the base head."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        mode: str,
        adaptive_window_strength: float = 1.0,
        context_residual_scale: float = 1.0,
        context_residual_clip: float = 0.0,
        context_residual_budget: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"gate", "concat", "logit_residual", "atm_residual", "drift_residual"}:
            raise ValueError(f"Unsupported target-context fusion mode: {mode}")
        self.mode = mode
        self.adaptive_window_strength = float(adaptive_window_strength)
        self.context_residual_scale = float(context_residual_scale)
        self.context_residual_clip = max(float(context_residual_clip), 0.0)
        self.context_residual_budget = max(float(context_residual_budget), 0.0)
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if mode == "gate":
            self.fusion_block = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        elif mode == "concat":
            self.fusion_block = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif mode == "logit_residual":
            self.fusion_block = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid(),
            )
            self.context_output_layer = nn.Linear(hidden_dim, output_dim)
            nn.init.zeros_(self.context_output_layer.weight)
            nn.init.zeros_(self.context_output_layer.bias)
        elif mode == "atm_residual":
            self.time_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.window_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion_block = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid(),
            )
            self.context_output_layer = nn.Linear(hidden_dim, output_dim)
            nn.init.zeros_(self.context_output_layer.weight)
            nn.init.zeros_(self.context_output_layer.bias)
        else:
            self.time_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.stability_gate = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion_block = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid(),
            )
            self.context_output_layer = nn.Linear(hidden_dim, output_dim)
            nn.init.zeros_(self.context_output_layer.weight)
            nn.init.zeros_(self.context_output_layer.bias)
        if mode in {"gate", "concat"}:
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_layer = None

    def _stabilize_residual(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        stabilized = self.context_residual_scale * residual
        clip_fraction = 0.0
        if self.context_residual_clip > 0.0:
            clip_fraction = float(
                (stabilized.abs() > self.context_residual_clip).to(dtype=torch.float32).mean().detach().item()
            )
            stabilized = self.context_residual_clip * torch.tanh(stabilized / self.context_residual_clip)
        regularization_loss = stabilized.new_tensor(0.0)
        if self.context_residual_budget > 0.0:
            regularization_loss = F.relu(stabilized.abs() - self.context_residual_budget).mean()
        return stabilized, regularization_loss, clip_fraction

    def forward(
        self,
        *,
        base_embedding: torch.Tensor,
        base_logits: torch.Tensor | None = None,
        target_features: torch.Tensor,
        target_time_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], torch.Tensor]:
        context_embedding = self.context_encoder(target_features)
        fusion_input = torch.cat([base_embedding, context_embedding], dim=-1)
        regularization_loss = context_embedding.new_tensor(0.0)
        if self.mode == "gate":
            gate = self.fusion_block(fusion_input)
            fused_embedding = base_embedding + gate * context_embedding
            diagnostics = {
                "context_gate_mean": float(gate.mean().detach().item()),
                "context_delta_norm": float((gate * context_embedding).norm(dim=-1).mean().detach().item()),
            }
        elif self.mode == "concat":
            fused_embedding = self.fusion_block(fusion_input)
            diagnostics = {
                "context_gate_mean": 0.0,
                "context_delta_norm": float((fused_embedding - base_embedding).norm(dim=-1).mean().detach().item()),
            }
        elif self.mode == "logit_residual":
            if base_logits is None:
                raise ValueError("base_logits are required for logit_residual target-context fusion.")
            residual_gate = self.fusion_block(fusion_input)
            context_logits = self.context_output_layer(context_embedding)
            if residual_gate.dim() == 2 and residual_gate.shape[-1] == 1:
                residual_gate = residual_gate.squeeze(-1)
            if context_logits.dim() == 2 and context_logits.shape[-1] == 1:
                context_logits = context_logits.squeeze(-1)
            residual, regularization_loss, clip_fraction = self._stabilize_residual(
                residual_gate * context_logits
            )
            logits = base_logits + residual
            if logits.dim() == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            fused_embedding = base_embedding
            diagnostics = {
                "context_gate_mean": float(residual_gate.mean().detach().item()),
                "context_delta_norm": float(residual.abs().mean().detach().item()),
                "context_logit_abs_mean": float(context_logits.abs().mean().detach().item()),
                "context_residual_clip_fraction": clip_fraction,
                "context_residual_penalty": float(regularization_loss.detach().item()),
            }
        else:
            if base_logits is None:
                raise ValueError(f"base_logits are required for {self.mode} target-context fusion.")
            if target_time_position is None:
                target_time_position = torch.zeros(
                    (base_embedding.shape[0], 1),
                    dtype=base_embedding.dtype,
                    device=base_embedding.device,
                )
            time_embedding = self.time_encoder(target_time_position)
            if self.mode == "atm_residual":
                fusion_input = torch.cat([base_embedding, context_embedding, time_embedding], dim=-1)
                adaptive_window_gate = self.window_gate(fusion_input)
                residual_gate = self.fusion_block(fusion_input)
                motif_delta = self.adaptive_window_strength * adaptive_window_gate * (
                    context_embedding + time_embedding
                )
                context_logits = self.context_output_layer(motif_delta)
                if residual_gate.dim() == 2 and residual_gate.shape[-1] == 1:
                    residual_gate = residual_gate.squeeze(-1)
                if context_logits.dim() == 2 and context_logits.shape[-1] == 1:
                    context_logits = context_logits.squeeze(-1)
                residual, regularization_loss, clip_fraction = self._stabilize_residual(
                    residual_gate * context_logits
                )
                logits = base_logits + residual
                if logits.dim() == 2 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                fused_embedding = base_embedding + motif_delta
                diagnostics = {
                    "context_gate_mean": float(residual_gate.mean().detach().item()),
                    "adaptive_window_gate_mean": float(adaptive_window_gate.mean().detach().item()),
                    "context_delta_norm": float(motif_delta.norm(dim=-1).mean().detach().item()),
                    "context_logit_abs_mean": float(context_logits.abs().mean().detach().item()),
                    "context_residual_clip_fraction": clip_fraction,
                    "context_residual_penalty": float(regularization_loss.detach().item()),
                    "time_emb_norm": float(time_embedding.norm(dim=-1).mean().detach().item()),
                }
            else:
                drift_signal = torch.abs(base_embedding - context_embedding)
                fusion_input = torch.cat(
                    [base_embedding, context_embedding, time_embedding, drift_signal],
                    dim=-1,
                )
                stability_gate = self.stability_gate(fusion_input)
                stabilized_context = stability_gate * (
                    context_embedding + self.adaptive_window_strength * time_embedding
                )
                residual_gate = self.fusion_block(fusion_input)
                context_logits = self.context_output_layer(stabilized_context)
                if residual_gate.dim() == 2 and residual_gate.shape[-1] == 1:
                    residual_gate = residual_gate.squeeze(-1)
                if context_logits.dim() == 2 and context_logits.shape[-1] == 1:
                    context_logits = context_logits.squeeze(-1)
                residual, regularization_loss, clip_fraction = self._stabilize_residual(
                    residual_gate * context_logits
                )
                logits = base_logits + residual
                if logits.dim() == 2 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                fused_embedding = base_embedding
                diagnostics = {
                    "context_gate_mean": float(residual_gate.mean().detach().item()),
                    "context_stability_gate_mean": float(stability_gate.mean().detach().item()),
                    "context_drift_abs_mean": float(drift_signal.mean().detach().item()),
                    "context_delta_norm": float(stabilized_context.norm(dim=-1).mean().detach().item()),
                    "context_logit_abs_mean": float(context_logits.abs().mean().detach().item()),
                    "context_residual_clip_fraction": clip_fraction,
                    "context_residual_penalty": float(regularization_loss.detach().item()),
                    "time_emb_norm": float(time_embedding.norm(dim=-1).mean().detach().item()),
                }
        if self.output_layer is not None:
            logits = self.output_layer(fused_embedding)
            if logits.dim() == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
        diagnostics["context_emb_norm"] = float(context_embedding.norm(dim=-1).mean().detach().item())
        return logits, fused_embedding, diagnostics, regularization_loss
