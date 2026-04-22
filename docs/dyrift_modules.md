# DyRIFT-GNN Modules

This document explains the modules used by the final `DyRIFT-GNN` route.

## 1. Module Map

| Module | Code | Purpose |
| --- | --- | --- |
| TRGT backbone | `TRGTTemporalRelationAttentionBlock` | temporal-relation graph attention |
| Internal risk encoder | `TRGTInternalRiskEncoder` | multi-scale risk fusion inside GNN |
| Target-context bridge | `TargetContextFusionHead` | target-level temporal-normality fusion |
| Drift expert | `TargetTimeDriftExpertAdapter` | temporal drift adaptation |
| Prototype memory | `PrototypeMemoryBank` | class-structure regularization |
| Pseudo-contrastive mining | training loop in `gnn_models.py` | time-balanced hard sample mining |
| Cold-start residual | `RelationGraphSAGENetwork` cold-start path | late cold-start compensation |

## 2. Temporal-Relation Attention

The backbone attention module learns which neighbors should matter for a target node.

It is relation-aware because every edge has a relation embedding.

It is temporal because edge-relative time is encoded and can also affect message weights.

It is target-specific because attention is normalized by destination node and head.

## 3. Temporal-Normality Bridge

The bridge uses target-context feature groups to calibrate graph embeddings.

Typical groups:

- `graph_stats`
- `graph_time_detrend`
- `neighbor_similarity`
- `activation_early`

The bridge is a neural module inside the GNN prediction path.

## 4. Drift-Expert Adaptation

Financial graph behavior changes over time. The drift expert adjusts target embeddings according to target time position and context features.

The model uses this to reduce time-distribution mismatch between train and validation windows.

## 5. Prototype Memory

Prototype memory stores class-level representation prototypes and adds auxiliary regularization during training.

Its role is representation stability, not a second prediction head.

## 6. Pseudo-Contrastive Temporal Mining

Pseudo-contrastive mining selects confident high-risk and low-risk samples in a time-aware way.

The goal is to improve separation between suspicious and normal patterns under temporal drift.

Important configuration fields:

- `pseudo_contrastive_weight`
- `pseudo_contrastive_temperature`
- `pseudo_contrastive_time_balanced`
- `pseudo_contrastive_start_epoch`

## 7. Internal Causal Risk Fusion

The internal risk encoder computes risk deltas from GNN states:

- inbound vs outbound gap
- short-window vs long-window gap
- one-hop vs two-hop gap
- direction asymmetry
- temporal support mass

These signals are learned inside the GNN and fused as a residual representation update.

## 8. Context-Conditioned Cold-Start Residual

Cold-start nodes may have too few informative messages, especially in late time windows.

The cold-start residual uses:

- support count
- target time position
- context signal
- base target features
- target-context features

It produces a gated residual correction inside the GNN logits.

For the final EPP profile, this is enabled by:

```json
"cold_start_residual_strength": 0.35
```

This is a module-level hyperparameter in the same architecture, not a different model.

## 9. Deployment Path

At inference time, the prediction still follows:

`features + graph -> DyRIFT-GNN -> fraud probability`

No module requires a separately trained external model at inference.
