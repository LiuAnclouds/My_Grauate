# DyRIFT-GNN Method

## 1. Method Identity

Full model name:

- `Dynamic Risk-Informed Fraud Graph Neural Network`
- short name: `DyRIFT-GNN`

Backbone name:

- `Temporal-Relational Graph Transformer`
- short name: `TRGT`

Runtime id:

- `dyrift_gnn`

The runtime id is used by runners and checkpoints. The thesis-facing method name is `DyRIFT-GNN`.

## 2. End-to-End Route

The deployed route is:

`raw dataset -> dataset-local preprocessing -> UTPM contract -> TRGT -> DyRIFT modules -> fraud probability`

This final route is pure GNN:

- no external classifier
- no teacher branch at inference time
- no second-stage fusion model

## 3. Unified Input Contract

The three datasets have different raw schemas, but all are mapped to one semantic input family:

- node attribute statistics
- graph structural statistics
- temporal activity statistics
- relation-aware interaction statistics
- target-context groups for the bridge branch

Supported UTPM profiles include:

- `utpm_shift_enhanced`
- `utpm_shift_compact`
- `utpm_unified`

The goal is unified input semantics, not identical raw columns.

## 4. Backbone

TRGT performs relation-aware, time-aware message passing over sampled dynamic subgraphs.

Main code:

- [../experiment/training/modules/backbone.py](../experiment/training/modules/backbone.py)
- [../experiment/training/modules/trainer.py](../experiment/training/modules/trainer.py)
- [../experiment/training/core/engine.py](../experiment/training/core/engine.py)

Key backbone class:

- `TRGTTemporalRelationAttentionBlock`

Trainer wrapper:

- `DyRIFTTrainer`

## 5. DyRIFT Modules

DyRIFT-GNN adds several risk-oriented modules on top of the TRGT path.

### 5.1 Temporal-Normality Bridge

The target-context bridge injects target-level context features after graph encoding.

Typical context groups:

- `graph_time_detrend`
- `neighbor_similarity`
- `activation_early`

Implementation:

- [../experiment/training/modules/bridge.py](../experiment/training/modules/bridge.py)

### 5.2 Drift-Expert Adaptation

The drift adapter changes context fusion behavior across time buckets.

Implementation:

- `TargetTimeDriftExpertAdapter` in [../experiment/training/core/engine.py](../experiment/training/core/engine.py)

### 5.3 Prototype Memory

Prototype memory regularizes representation structure and stabilizes class centers.

Implementation:

- [../experiment/training/modules/memory.py](../experiment/training/modules/memory.py)

### 5.4 Pseudo-Contrastive Temporal Mining

Pseudo-contrastive mining selects high-confidence temporal hard cases during training.

Implementation:

- training logic in [../experiment/training/core/engine.py](../experiment/training/core/engine.py)

### 5.5 Internal Causal Risk Fusion

Internal multi-scale risk features are learned directly from the sampled subgraph.

Implementation:

- `TRGTInternalRiskEncoder` in [../experiment/training/modules/backbone.py](../experiment/training/modules/backbone.py)

### 5.6 Context-Conditioned Cold-Start Residual

This branch compensates message sparsity for late cold-start nodes inside the same GNN route.

Implementation:

- cold-start residual logic in [../experiment/training/core/engine.py](../experiment/training/core/engine.py)

## 6. Final Results

Final suite:

- [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json)

Validation AUC:

- XinYe DGraph: `0.790455`
- Elliptic Transactions: `0.821329`
- Elliptic++ Transactions: `0.821953`
- Macro: `0.811246`

## 7. Leakage Rules

The final route enforces:

- datasets are trained separately
- no cross-dataset label or prediction reuse
- validation and test labels never flow back into training
- no external model is required at deployment

Audit outputs:

- [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md)
- [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json)
