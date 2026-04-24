# Code Reference

This is the compact code map for the current `DyRIFT-GNN / TRGT` route.

## 1. Top-Level Entry Points

| Path | Role |
| --- | --- |
| [../experiment/mainline.py](../experiment/mainline.py) | single-dataset feature build and train entry |
| [../experiment/suite.py](../experiment/suite.py) | three-dataset mainline rerun entry |
| [../experiment/audit.py](../experiment/audit.py) | hard-leakage audit |
| [../experiment/config_loader.py](../experiment/config_loader.py) | shared config loading and per-dataset merge |
| [../experiment/sync_results.py](../experiment/sync_results.py) | regenerate tracked result CSV/JSON tables from saved summaries |

## 2. Model Package Layout

| Path | Role |
| --- | --- |
| [../experiment/models/runtime.py](../experiment/models/runtime.py) | runtime bundle assembly |
| [../experiment/models/graph.py](../experiment/models/graph.py) | runtime-id to trainer resolution |
| [../experiment/models/presets.py](../experiment/models/presets.py) | preset registry |
| [../experiment/models/spec.py](../experiment/models/spec.py) | official model and suite constants |
| [../experiment/models/engine.py](../experiment/models/engine.py) | training engine, loss composition, evaluation, logging |
| [../experiment/models/modules/backbone.py](../experiment/models/modules/backbone.py) | TRGT blocks and internal risk encoder |
| [../experiment/models/modules/model.py](../experiment/models/modules/model.py) | DyRIFT model facade |
| [../experiment/models/modules/trainer.py](../experiment/models/modules/trainer.py) | `dyrift_gnn` trainer wrapper |
| [../experiment/models/modules/bridge.py](../experiment/models/modules/bridge.py) | target-context bridge |
| [../experiment/models/modules/memory.py](../experiment/models/modules/memory.py) | prototype memory |

## 3. Feature And Dataset Utilities

| Path | Role |
| --- | --- |
| [../experiment/features/features.py](../experiment/features/features.py) | unified feature cache, normalization, and feature-group helpers |
| [../experiment/datasets/core/](../experiment/datasets/core) | dataset registry, dataset spec, and contracts |
| [../experiment/utils/common.py](../experiment/utils/common.py) | IO, metrics, split loading, output paths |
| [../experiment/utils/sampling.py](../experiment/utils/sampling.py) | sampling-profile helpers |

## 4. Study Workspace

| Path | Role |
| --- | --- |
| [../experiment/studies/common/launcher.py](../experiment/studies/common/launcher.py) | shared study launcher |
| [../experiment/studies/common/graph_runner.py](../experiment/studies/common/graph_runner.py) | GNN-family study runner |
| [../experiment/studies/common/xgboost_runner.py](../experiment/studies/common/xgboost_runner.py) | same-input XGBoost study runner |
| [../experiment/studies/common/xinye_phase12_joint_runner.py](../experiment/studies/common/xinye_phase12_joint_runner.py) | XinYe `phase1+phase2` joint-train supplementary runner |
| [../experiment/studies/comparisons/](../experiment/studies/comparisons) | comparison experiments |
| [../experiment/studies/ablations/](../experiment/studies/ablations) | subtractive ablations |
| [../experiment/studies/progressive/](../experiment/studies/progressive) | progressive method-building studies |
| [../experiment/studies/supplementary/](../experiment/studies/supplementary) | supplementary experiments |

## 5. Public Symbols

| Symbol | File | Meaning |
| --- | --- | --- |
| `DyRIFTModel` | `modules/model.py` | final paper-facing model facade |
| `DyRIFTTrainer` | `modules/trainer.py` | trainer wrapper bound to `dyrift_gnn` |
| `build_model` | `modules/model.py` | DyRIFT model factory |
| `RuntimeBundle` | `models/runtime.py` | prepared runtime assets |
| `GraphModelConfig` | `models/engine.py` | central graph-model config object |
| `TrainParameters` / `Parameter` | `config_loader.py` | explicit JSON/CLI train-parameter container |
| `TRGTTemporalRelationAttentionBlock` | `modules/backbone.py` | temporal-relation attention block |
| `TRGTInternalRiskEncoder` | `modules/backbone.py` | internal risk encoder |

## 6. Runtime Resolution

Runtime id mapping lives in [../experiment/models/graph.py](../experiment/models/graph.py):

`dyrift_gnn -> DyRIFTTrainer`

这保证了 CLI、checkpoint 和论文方法名的一致性。

## 7. Mainline Call Chain

当前主线调用链可以概括为：

1. `experiment/mainline.py` or `experiment/suite.py`
2. `experiment/config_loader.py`
3. `experiment/models/runtime.py`
4. `experiment/models/graph.py`
5. `DyRIFTTrainer`
6. `DyRIFTModel`
7. `TRGTTemporalRelationAttentionBlock`
8. bridge, drift, prototype, pseudo-contrastive, internal risk, and cold-start logic from `engine.py` and `modules/`

## 8. Result And Audit Files

| File | Role |
| --- | --- |
| [results/accepted_mainline_summary.json](results/accepted_mainline_summary.json) | accepted mainline summary |
| [results/leakage_audit.json](results/leakage_audit.json) | accepted mainline hard-leakage audit |
| [results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv) | accepted mainline AUC table |
| [results/comparison_auc.csv](results/comparison_auc.csv) | comparison-study AUC table |
| [results/ablation_auc.csv](results/ablation_auc.csv) | ablation-study AUC table |
| [results/progressive_auc.csv](results/progressive_auc.csv) | progressive-study AUC table |
| [results/supplementary_auc.csv](results/supplementary_auc.csv) | supplementary-study AUC table |
| [results/epoch_log_manifest.csv](results/epoch_log_manifest.csv) | epoch/log/curve manifest for plotting and review |
