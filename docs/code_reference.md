# Code Reference

This is the compact code map for the final `DyRIFT-GNN / TRGT` route.

## 1. Package Layout

| Path | Role |
| --- | --- |
| [../experiment/mainline.py](../experiment/mainline.py) | single-dataset build/train entry |
| [../experiment/suite.py](../experiment/suite.py) | tri-dataset suite runner |
| [../experiment/audit.py](../experiment/audit.py) | hard-leakage audit |
| [../experiment/models/engine.py](../experiment/models/engine.py) | graph engine, losses, samplers, evaluation |
| [../experiment/models/runtime.py](../experiment/models/runtime.py) | runtime bundle assembly |
| [../experiment/models/presets.py](../experiment/models/presets.py) | official preset registry |
| [../experiment/config_loader.py](../experiment/config_loader.py) | suite manifest and dataset hparam loading |
| [../experiment/features/features.py](../experiment/features/features.py) | feature cache and normalizer utilities |
| [../experiment/models/graph.py](../experiment/models/graph.py) | experiment resolution and graph contexts |
| [../experiment/models/modules/backbone.py](../experiment/models/modules/backbone.py) | TRGT blocks and internal risk encoder |
| [../experiment/models/modules/model.py](../experiment/models/modules/model.py) | DyRIFT model facade |
| [../experiment/models/modules/trainer.py](../experiment/models/modules/trainer.py) | DyRIFT trainer wrapper |
| [../experiment/models/modules/bridge.py](../experiment/models/modules/bridge.py) | target-context bridge |
| [../experiment/models/modules/memory.py](../experiment/models/modules/memory.py) | prototype memory and normal-alignment memory |
| [../experiment/utils/common.py](../experiment/utils/common.py) | IO, metrics, split loading, paths |
| [../experiment/utils/sampling.py](../experiment/utils/sampling.py) | sampling-profile helpers |

## 2. Public Symbols

| Symbol | File | Meaning |
| --- | --- | --- |
| `DyRIFTModel` | `modules/model.py` | final paper-facing model facade |
| `build_model` | `modules/model.py` | DyRIFT-GNN model factory |
| `DyRIFTTrainer` | `modules/trainer.py` | trainer wrapper bound to `dyrift_gnn` |
| `get_experiment_cls` | `models/graph.py` | runtime model-id to experiment-class resolver |
| `RuntimeBundle` | `models/runtime.py` | prepared runtime assets for train/inference |
| `GraphModelConfig` | `models/engine.py` | central graph-model configuration object |
| `TRGTTemporalRelationAttentionBlock` | `modules/backbone.py` | relation-temporal attention block |
| `TRGTInternalRiskEncoder` | `modules/backbone.py` | internal multi-scale risk encoder |
| `TargetContextFusionHead` | `modules/bridge.py` | temporal-normality bridge |
| `PrototypeMemoryBank` | `modules/memory.py` | prototype regularization bank |

## 3. Runtime Resolution

Runtime id mapping is defined in [../experiment/models/graph.py](../experiment/models/graph.py):

`dyrift_gnn -> DyRIFTTrainer`

This keeps the CLI, checkpoints, and thesis method name aligned.

## 4. Training Call Chain

The final call chain is:

1. `pipelines/mainline.py`
2. `build_graph_cfg(...)`
3. `build_runtime(...)`
4. `get_experiment_cls("dyrift_gnn")`
5. `DyRIFTTrainer`
6. `build_model(...)`
7. `DyRIFTModel`
8. `RelationGraphSAGENetwork`
9. `TRGTTemporalRelationAttentionBlock` and related DyRIFT modules

## 5. Shared Engine Responsibilities

[../experiment/models/engine.py](../experiment/models/engine.py) still owns the shared training runtime:

- graph tensorization and neighborhood sampling
- loss composition
- pseudo-contrastive temporal mining
- drift expert adapter
- cold-start residual logic
- evaluation, logging, checkpointing

## 6. Config Files

| File | Role |
| --- | --- |
| [../experiment/configs/dyrift_suite.json](../experiment/configs/dyrift_suite.json) | suite manifest |
| [../experiment/configs/xinye_dgraph.json](../experiment/configs/xinye_dgraph.json) | XinYe profile |
| [../experiment/configs/elliptic_transactions.json](../experiment/configs/elliptic_transactions.json) | ET profile |
| [../experiment/configs/ellipticpp_transactions.json](../experiment/configs/ellipticpp_transactions.json) | EPP profile |

## 7. Result Files

| File | Role |
| --- | --- |
| [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json) | suite-level summary |
| [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json) | hard-leakage audit |
| [results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv) | final dataset metrics |
| [results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv) | epoch-level logs |
