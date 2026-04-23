# Code Reference

This is the compact code map for the final `DyRIFT-GNN / TRGT` route.

## 1. Package Layout

| Path | Role |
| --- | --- |
| [../experiment/training/runners/mainline.py](../experiment/training/runners/mainline.py) | single-dataset build/train entry |
| [../experiment/training/runners/suite.py](../experiment/training/runners/suite.py) | tri-dataset suite runner |
| [../experiment/training/runners/audit.py](../experiment/training/runners/audit.py) | hard-leakage audit |
| [../experiment/training/core/engine.py](../experiment/training/core/engine.py) | graph engine, losses, samplers, evaluation |
| [../experiment/training/core/runtime.py](../experiment/training/core/runtime.py) | runtime bundle assembly |
| [../experiment/training/core/presets.py](../experiment/training/core/presets.py) | official preset registry |
| [../experiment/training/core/hparams.py](../experiment/training/core/hparams.py) | suite manifest and dataset hparam loading |
| [../experiment/training/data/features.py](../experiment/training/data/features.py) | feature cache and normalizer utilities |
| [../experiment/training/data/graph.py](../experiment/training/data/graph.py) | experiment resolution and graph contexts |
| [../experiment/training/modules/backbone.py](../experiment/training/modules/backbone.py) | TRGT blocks and internal risk encoder |
| [../experiment/training/modules/model.py](../experiment/training/modules/model.py) | DyRIFT model facade |
| [../experiment/training/modules/trainer.py](../experiment/training/modules/trainer.py) | DyRIFT trainer wrapper |
| [../experiment/training/modules/bridge.py](../experiment/training/modules/bridge.py) | target-context bridge |
| [../experiment/training/modules/memory.py](../experiment/training/modules/memory.py) | prototype memory and normal-alignment memory |
| [../experiment/training/utils/common.py](../experiment/training/utils/common.py) | IO, metrics, split loading, paths |
| [../experiment/training/utils/sampling.py](../experiment/training/utils/sampling.py) | sampling-profile helpers |

## 2. Public Symbols

| Symbol | File | Meaning |
| --- | --- | --- |
| `DyRIFTModel` | `modules/model.py` | final paper-facing model facade |
| `build_model` | `modules/model.py` | DyRIFT-GNN model factory |
| `DyRIFTTrainer` | `modules/trainer.py` | trainer wrapper bound to `dyrift_gnn` |
| `get_experiment_cls` | `data/graph.py` | runtime model-id to experiment-class resolver |
| `RuntimeBundle` | `core/runtime.py` | prepared runtime assets for train/inference |
| `GraphModelConfig` | `core/engine.py` | central graph-model configuration object |
| `TRGTTemporalRelationAttentionBlock` | `modules/backbone.py` | relation-temporal attention block |
| `TRGTInternalRiskEncoder` | `modules/backbone.py` | internal multi-scale risk encoder |
| `TargetContextFusionHead` | `modules/bridge.py` | temporal-normality bridge |
| `PrototypeMemoryBank` | `modules/memory.py` | prototype regularization bank |

## 3. Runtime Resolution

Runtime id mapping is defined in [../experiment/training/data/graph.py](../experiment/training/data/graph.py):

`dyrift_gnn -> DyRIFTTrainer`

This keeps the CLI, checkpoints, and thesis method name aligned.

## 4. Training Call Chain

The final call chain is:

1. `runners/mainline.py`
2. `build_graph_cfg(...)`
3. `build_runtime(...)`
4. `get_experiment_cls("dyrift_gnn")`
5. `DyRIFTTrainer`
6. `build_model(...)`
7. `DyRIFTModel`
8. `RelationGraphSAGENetwork`
9. `TRGTTemporalRelationAttentionBlock` and related DyRIFT modules

## 5. Shared Engine Responsibilities

[../experiment/training/core/engine.py](../experiment/training/core/engine.py) still owns the shared training runtime:

- graph tensorization and neighborhood sampling
- loss composition
- pseudo-contrastive temporal mining
- drift expert adapter
- cold-start residual logic
- evaluation, logging, checkpointing

## 6. Config Files

| File | Role |
| --- | --- |
| [../experiment/training/configs/dyrift_suite.json](../experiment/training/configs/dyrift_suite.json) | suite manifest |
| [../experiment/training/configs/datasets/xinye_dgraph.json](../experiment/training/configs/datasets/xinye_dgraph.json) | XinYe profile |
| [../experiment/training/configs/datasets/elliptic_transactions.json](../experiment/training/configs/datasets/elliptic_transactions.json) | ET profile |
| [../experiment/training/configs/datasets/ellipticpp_transactions.json](../experiment/training/configs/datasets/ellipticpp_transactions.json) | EPP profile |

## 7. Result Files

| File | Role |
| --- | --- |
| [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json) | suite-level summary |
| [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json) | hard-leakage audit |
| [results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv) | final dataset metrics |
| [results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv) | epoch-level logs |
