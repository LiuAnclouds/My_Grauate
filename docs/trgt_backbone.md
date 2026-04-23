# TRGT Backbone

`TRGT` stands for `Temporal-Relational Graph Transformer`.

It is the backbone of `DyRIFT-GNN`. The main implementation is:

- [../experiment/training/modules/backbone.py](../experiment/training/modules/backbone.py)

## 1. Backbone Goal

TRGT is designed for dynamic financial graphs where fraud risk depends on:

- when an interaction happens
- what relation type the interaction has
- which direction the interaction follows
- how local neighborhoods evolve over time

It is closer to a temporal relation-aware graph transformer than to a plain GCN, GraphSAGE, or vanilla GAT.

## 2. Core Data Flow

For each target batch, the trainer samples a local dynamic subgraph.

The TRGT block receives:

- `x`: node representations
- `edge_src`: source node indices
- `edge_dst`: destination node indices
- `edge_emb`: relation and time edge embedding
- `time_weight`: optional temporal decay weight
- `message_node_scale`: optional risk-aware node message scale

The output is:

- updated node representations
- edge-level message representations

## 3. Attention Block

Main class:

- `TRGTTemporalRelationAttentionBlock`

The attention input for each edge is:

`concat(dst_repr, src_repr, edge_embedding)`

The edge embedding includes:

- relation embedding
- temporal encoding when temporal mode is enabled

The block computes:

1. source-to-target message projection
2. optional gated message modulation
3. multi-head attention score per destination node
4. segment softmax over incoming edges
5. weighted message aggregation
6. residual update and optional feed-forward network

This makes the target representation sensitive to relation semantics and time context.

## 4. Internal Risk Encoder

Main class:

- `TRGTInternalRiskEncoder`

This encoder computes target-level risk signals from the sampled subgraph:

- short-window inbound and outbound summaries
- long-window inbound and outbound summaries
- one-hop and two-hop representation gaps
- direction asymmetry
- short-long temporal gap
- support counts and time-mass features

It returns a learned risk embedding that is fused inside the GNN path.

## 5. Helper Functions

| Function | Role |
| --- | --- |
| `segment_weighted_mean` | weighted segment pooling for per-target summaries |
| `segment_softmax` | grouped softmax for destination-wise attention |
| `_make_norm` | local normalization factory for TRGT blocks |

## 6. Why This Is A GNN Backbone

TRGT performs message passing over sampled graph neighborhoods. Each layer updates node states using edge-indexed messages and destination-wise aggregation.

It is a GNN because:

- computation depends on graph edges
- neighbor messages update target node states
- relation and temporal edge features affect aggregation
- predictions are produced from graph-updated target representations

It is transformer-style because:

- neighbor aggregation is multi-head attention
- attention scores are learned from source, target, relation, and time context
- residual and feed-forward blocks are used after message aggregation
