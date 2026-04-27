from __future__ import annotations

import torch.nn as nn

from dyrift.models.engine import DyRIFTTGATExperiment, GraphModelConfig
from dyrift.models.components.model import build_model


class DyRIFTTrainer(DyRIFTTGATExperiment):
    """Training wrapper for the final DyRIFT-TGAT method.

    The legacy experiment base still owns batching, sampling, losses, metrics, and checkpointing.
    This wrapper isolates the final thesis model construction from the shared engine
    runtime while exposing the public `dyrift_gnn` model id.
    """

    method_display_name = "DyRIFT-TGAT"
    backbone_display_name = "TGAT"

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
        return build_model(
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
