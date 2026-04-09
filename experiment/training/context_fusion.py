from __future__ import annotations

import torch
import torch.nn as nn


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
    ) -> None:
        super().__init__()
        if mode not in {"gate", "concat", "logit_residual"}:
            raise ValueError(f"Unsupported target-context fusion mode: {mode}")
        self.mode = mode
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
        else:
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
        if mode in {"gate", "concat"}:
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_layer = None

    def forward(
        self,
        *,
        base_embedding: torch.Tensor,
        base_logits: torch.Tensor | None = None,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        context_embedding = self.context_encoder(target_features)
        fusion_input = torch.cat([base_embedding, context_embedding], dim=-1)
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
        else:
            if base_logits is None:
                raise ValueError("base_logits are required for logit_residual target-context fusion.")
            residual_gate = self.fusion_block(fusion_input)
            context_logits = self.context_output_layer(context_embedding)
            if residual_gate.dim() == 2 and residual_gate.shape[-1] == 1:
                residual_gate = residual_gate.squeeze(-1)
            if context_logits.dim() == 2 and context_logits.shape[-1] == 1:
                context_logits = context_logits.squeeze(-1)
            residual = residual_gate * context_logits
            logits = base_logits + residual
            if logits.dim() == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            fused_embedding = base_embedding
            diagnostics = {
                "context_gate_mean": float(residual_gate.mean().detach().item()),
                "context_delta_norm": float(residual.abs().mean().detach().item()),
                "context_logit_abs_mean": float(context_logits.abs().mean().detach().item()),
            }
        if self.output_layer is not None:
            logits = self.output_layer(fused_embedding)
            if logits.dim() == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
        diagnostics["context_emb_norm"] = float(context_embedding.norm(dim=-1).mean().detach().item())
        return logits, fused_embedding, diagnostics
