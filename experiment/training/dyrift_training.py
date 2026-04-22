from __future__ import annotations

import torch.nn as nn

from experiment.training.dyrift_model import build_dyrift_gnn_model
from experiment.training.gnn_models import GraphModelConfig, TRGTExperiment


class DyRIFTGNNExperiment(TRGTExperiment):
    """Training wrapper for the final DyRIFT-GNN method.

    The legacy experiment base still owns batching, sampling, losses, metrics, and checkpointing.
    This wrapper isolates the final thesis model construction so `m8_utgt` no longer needs to be
    mentally decoded from the legacy `gnn_models.py` file.
    """

    method_display_name = "DyRIFT-GNN"
    backbone_display_name = "TRGT"

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
        return build_dyrift_gnn_model(
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
