# Code Reference

This document is a compact map of the final `DyRIFT-GNN / TRGT` code path.

## 1. Main Files

| File | Role |
| --- | --- |
| [experiment/training/trgt_backbone.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/trgt_backbone.py) | TRGT backbone blocks and internal risk encoder |
| [experiment/training/dyrift_model.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/dyrift_model.py) | model facade and factory |
| [experiment/training/dyrift_training.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/dyrift_training.py) | final training wrapper |
| [experiment/training/gnn_models.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/gnn_models.py) | shared network runtime, loss, sampling, evaluation |
| [experiment/training/run_thesis_mainline.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/run_thesis_mainline.py) | build/train single-dataset runner |
| [experiment/training/run_thesis_suite.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/run_thesis_suite.py) | three-dataset suite runner |
| [experiment/training/graph_runtime.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/graph_runtime.py) | model-name to experiment-class resolution |

## 2. Final Public Classes And Functions

| Symbol | File | Meaning |
| --- | --- | --- |
| `DyRIFTGNNModel` | `dyrift_model.py` | final paper-facing model facade |
| `build_dyrift_gnn_model` | `dyrift_model.py` | final model factory |
| `DyRIFTGNNExperiment` | `dyrift_training.py` | final trainer wrapper for `dyrift_gnn` |
| `TRGTExperiment` | `gnn_models.py` | TRGT attention experiment base |
| `TRGTTemporalRelationAttentionBlock` | `trgt_backbone.py` | backbone attention block |
| `TRGTInternalRiskEncoder` | `trgt_backbone.py` | internal risk fusion module |

Backward-compatible aliases remain for older imports:

- `DyRIFTGraphModel`
- `build_dyrift_model`
- `TemporalRelationGraphTransformerExperiment`

## 3. Runtime Resolution

The `dyrift_gnn` runtime id is resolved in:

- [graph_runtime.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/graph_runtime.py)

Current mapping:

`dyrift_gnn -> DyRIFTGNNExperiment`

This keeps the public runner argument aligned with the final DyRIFT-GNN method name.

## 4. Training Call Chain

The final mainline call chain is:

1. `run_thesis_mainline.py`
2. `resolve_graph_experiment_class("dyrift_gnn")`
3. `DyRIFTGNNExperiment`
4. `build_dyrift_gnn_model`
5. `DyRIFTGNNModel`
6. `RelationGraphSAGENetwork`
7. `TRGTTemporalRelationAttentionBlock` and related modules

## 5. Important Shared Runtime Components

These still live in `gnn_models.py` because they are used across the final graph training pipeline:

- `GraphModelConfig`
- `GraphPhaseContext`
- `BaseGraphSAGEExperiment`
- `RelationGraphSAGENetwork`
- subgraph tensorization and sampling pipeline
- training loss logic
- evaluation and epoch logging

## 6. Configuration Files

| File | Role |
| --- | --- |
| [experiment/training/configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json) | suite manifest |
| [experiment/training/configs/dyrift_gnn/xinye_dgraph.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/configs/dyrift_gnn/xinye_dgraph.json) | XinYe profile |
| [experiment/training/configs/dyrift_gnn/elliptic_transactions.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/configs/dyrift_gnn/elliptic_transactions.json) | ET profile |
| [experiment/training/configs/dyrift_gnn/ellipticpp_transactions.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/configs/dyrift_gnn/ellipticpp_transactions.json) | EPP profile |

## 7. Result Files

| File | Role |
| --- | --- |
| [summary.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json) | suite-level result metadata |
| [leakage_audit.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json) | hard-leakage audit |
| [thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv](/home/moonxkj/Desktop/MyWork/Graduation_Project/docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv) | final metrics |
| [thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv](/home/moonxkj/Desktop/MyWork/Graduation_Project/docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv) | epoch-level logs |
