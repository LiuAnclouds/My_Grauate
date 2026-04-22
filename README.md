# DyRIFT-GNN

Dynamic Risk-Informed Fraud Graph Neural Network for dynamic-graph financial fraud detection.

`DyRIFT-GNN` 是本项目的最终毕业设计方法。模型以 `TRGT` (`Temporal-Relational Graph Transformer`) 为动态图主干，在统一 UTPM 特征契约下完成 XinYe DGraph、Elliptic Transactions、Elliptic++ Transactions 三个数据集的训练、验证和复现实验。

最终路线是单路纯 GNN：

`dataset-local preprocessing -> UTPM feature contract -> TRGT backbone -> DyRIFT-GNN risk modules -> fraud probability`

训练和推理不依赖外部分类头或二阶段融合器。

## Project Cards

| Card | Description |
| --- | --- |
| [Method Overview](docs/dyrift_gnn_method.md) | DyRIFT-GNN 的整体方法、输入输出、部署路径 |
| [TRGT Backbone](docs/trgt_backbone.md) | Temporal-Relational Graph Transformer 主干说明 |
| [Model Modules](docs/dyrift_modules.md) | bridge、drift expert、prototype、pseudo-contrastive、risk fusion、cold-start residual |
| [Code Reference](docs/code_reference.md) | 关键文件、类、函数和调用关系 |
| [Training And Configs](docs/training_and_configs.md) | 复现实验命令、三数据集参数文件、输出路径 |
| [Experiment Results](docs/thesis_experiments.md) | 最终指标、GNN 对比、消融表 |
| [Leakage Audit](experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md) | 最终 hard-leakage 审计报告 |
| [Final Summary JSON](experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json) | 最终三数据集 suite summary |
| [Metrics CSV](docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv) | 三数据集最终指标汇总 |
| [Epoch Metrics CSV](docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv) | 三数据集逐 epoch 训练日志 |

## Final Results

| Dataset | Val AUC | Val PR-AUC | Val AP |
| --- | ---: | ---: | ---: |
| XinYe DGraph | 0.790455 | 0.045843 | 0.045998 |
| Elliptic Transactions | 0.821329 | 0.432221 | 0.396340 |
| Elliptic++ Transactions | 0.821953 | 0.471452 | 0.438596 |
| Macro Average | 0.811246 | 0.316505 | 0.293645 |

All numbers are from the final pure-GNN suite:

`thesis_dyrift_gnn_trgt_deploy_pure_v1`

The official runtime model id is `dyrift_gnn`. The paper-facing method name is `DyRIFT-GNN`, and the backbone name is `TRGT`.

## Architecture

| Layer | Name | Role |
| --- | --- | --- |
| Input contract | UTPM | Unified feature schema for all datasets |
| Backbone | TRGT | Temporal-relation multi-head attention over sampled dynamic subgraphs |
| Full model | DyRIFT-GNN | TRGT plus risk-aware fraud modules |
| Output | single GNN probability | Binary fraud probability for target nodes |

Core modules:

- `TRGTTemporalRelationAttentionBlock`: relation-aware temporal attention message passing.
- `TRGTInternalRiskEncoder`: internal multi-scale risk representation from sampled subgraphs.
- `TargetContextFusionHead`: target-level temporal-normality bridge.
- `TargetTimeDriftExpertAdapter`: time-drift expert adaptation.
- `PrototypeMemoryBank`: prototype memory regularization.
- `pseudo-contrastive temporal mining`: time-balanced hard sample mining during training.
- `context-conditioned cold-start residual`: cold-start correction inside the pure-GNN path.

## Code Layout

| Path | Role |
| --- | --- |
| [experiment/training/trgt_backbone.py](experiment/training/trgt_backbone.py) | TRGT backbone blocks and risk encoder |
| [experiment/training/dyrift_model.py](experiment/training/dyrift_model.py) | DyRIFT-GNN model facade and factory |
| [experiment/training/dyrift_training.py](experiment/training/dyrift_training.py) | Final DyRIFT-GNN training wrapper |
| [experiment/training/gnn_models.py](experiment/training/gnn_models.py) | Shared graph training runtime, losses, sampling, metrics |
| [experiment/training/run_thesis_mainline.py](experiment/training/run_thesis_mainline.py) | Single-dataset build/train entry |
| [experiment/training/run_thesis_suite.py](experiment/training/run_thesis_suite.py) | Three-dataset suite runner |
| [experiment/training/audit_thesis_leakage.py](experiment/training/audit_thesis_leakage.py) | Hard-leakage audit |

## Dataset Hyperparameters

The architecture is shared. Dataset-level tuning is isolated into three small JSON files:

| Dataset | Config |
| --- | --- |
| XinYe DGraph | [xinye_dgraph.json](experiment/training/configs/dyrift_gnn/xinye_dgraph.json) |
| Elliptic Transactions | [elliptic_transactions.json](experiment/training/configs/dyrift_gnn/elliptic_transactions.json) |
| Elliptic++ Transactions | [ellipticpp_transactions.json](experiment/training/configs/dyrift_gnn/ellipticpp_transactions.json) |

The suite-level manifest is:

[thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json](experiment/training/configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json)

## Reproduce

Build unified features:

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py \
  build_features \
  --phase both
```

Run the final suite:

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py \
  --suite-name thesis_dyrift_gnn_trgt_deploy_pure_v1 \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/training/configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json \
  --seeds 42 \
  --skip-existing
```

Run hard-leakage audit:

```bash
conda run -n Graph --no-capture-output python3 experiment/training/audit_thesis_leakage.py \
  --suite-summary experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json
```

## Integrity

The final suite declares:

- `deployment_path=single_gnn_end_to_end`
- `dataset_isolation=true`
- `cross_dataset_training=false`
- `same_architecture_across_datasets=true`

The hard-leakage audit reports `hard_leakage_detected=false`.

## Citation Name

Use this naming in the thesis and slides:

- Full model: `Dynamic Risk-Informed Fraud Graph Neural Network (DyRIFT-GNN)`
- Backbone: `Temporal-Relational Graph Transformer (TRGT)`
