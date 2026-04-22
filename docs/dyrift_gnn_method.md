# DyRIFT-GNN Method

## 1. Method Position

`DyRIFT-GNN` is the final single-model dynamic-graph fraud detection method in this repository.

Paper name:

- `Dynamic Risk-Informed Fraud Graph Neural Network`
- short name: `DyRIFT-GNN`

Backbone name:

- `Temporal-Relational Graph Transformer`
- short name: `TRGT`

Code runtime id:

- `dyrift_gnn`

The runtime id is used by runners, checkpoints, and saved output folders. The thesis-facing method name is `DyRIFT-GNN`.

## 2. End-to-End Path

The deployed path is:

`raw dataset -> dataset-local preprocessing -> UTPM contract -> TRGT -> DyRIFT modules -> fraud probability`

The final route is pure GNN:

- no external final classifier
- no extra inference branch

## 3. Unified Input Contract

The three datasets have different raw schemas, but they are mapped to the same semantic feature family:

- node attribute statistics
- graph structural statistics
- temporal activity statistics
- relation-aware interaction statistics
- target-context feature groups

The project uses UTPM feature profiles such as:

- `utpm_shift_enhanced`
- `utpm_shift_compact`

The purpose is not to force identical raw columns, but to force identical semantic roles for the model input.

## 4. Backbone

The backbone is `TRGT`, implemented with relation-aware, time-aware attention over sampled subgraphs.

TRGT learns from:

- source node representation
- destination node representation
- relation embedding
- temporal encoding
- sampled neighborhood context

The main attention block is:

- [TRGTTemporalRelationAttentionBlock](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/trgt_backbone.py)

The trainer-level experiment wrapper is:

- [DyRIFTGNNExperiment](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/dyrift_training.py)

## 5. DyRIFT Modules

DyRIFT-GNN is not only a backbone replacement. It combines several risk-oriented modules on top of the TRGT path.

### 5.1 Temporal-Normality Bridge

The target-context bridge injects target-level context features into the target representation after graph encoding.

Typical context groups:

- `graph_stats`
- `graph_time_detrend`
- `neighbor_similarity`
- `activation_early`

This is implemented by:

- [TargetContextFusionHead](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/context_fusion.py)

### 5.2 Drift-Expert Adaptation

The target-time expert adapter learns how context fusion should change under temporal drift.

This is implemented inside:

- [TargetTimeDriftExpertAdapter](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/gnn_models.py)

### 5.3 Prototype Memory

Prototype memory regularizes class structure and improves representation stability.

Related implementation:

- [prototype_memory.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/prototype_memory.py)

### 5.4 Pseudo-Contrastive Temporal Mining

Pseudo-contrastive mining selects temporal hard cases during training, especially under time-balanced settings.

This is part of the shared training logic in:

- [gnn_models.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/gnn_models.py)

### 5.5 Internal Causal Risk Fusion

Internal multi-scale risk features are learned from the sampled subgraph itself.

This module is:

- [TRGTInternalRiskEncoder](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/trgt_backbone.py)

### 5.6 Context-Conditioned Cold-Start Residual

This is the EPP-oriented cold-start correction branch inside the pure-GNN path. It uses target support and context signal to compensate message sparsity in late cold-start slices.

This logic lives in:

- [RelationGraphSAGENetwork](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/gnn_models.py)

## 6. Final Results

The final suite is:

- [thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json)

Main validation AUC:

- XinYe DGraph: `0.790455`
- Elliptic Transactions: `0.821329`
- Elliptic++ Transactions: `0.821953`
- Macro: `0.811246`

## 7. Safety And Leakage Rules

The final route follows these constraints:

- datasets are trained separately
- no cross-dataset labels or prediction reuse
- validation and test labels do not flow back into training
- no external model is required at deployment

Audit outputs:

- [leakage_audit.md](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md)
- [leakage_audit.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json)
