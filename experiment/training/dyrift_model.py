from __future__ import annotations

from experiment.training.gnn_models import GraphModelConfig, RelationGraphSAGENetwork


DYRIFT_MODEL_NAME = "DyRIFT-GNN"
DYRIFT_BACKBONE_NAME = "TRGT"


class DyRIFTGNNModel(RelationGraphSAGENetwork):
    """Named facade for the final thesis single-model pure-GNN architecture."""

    model_display_name = DYRIFT_MODEL_NAME
    backbone_display_name = DYRIFT_BACKBONE_NAME


def build_dyrift_gnn_model(
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
) -> DyRIFTGNNModel:
    return DyRIFTGNNModel(
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


# Backward-compatible aliases for older local notebooks/scripts.
DyRIFTGraphModel = DyRIFTGNNModel
build_dyrift_model = build_dyrift_gnn_model
