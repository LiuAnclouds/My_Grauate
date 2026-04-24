# DyRIFT-GNN

[English](README.md) | 中文说明

`DyRIFT-GNN` 是当前仓库用于毕业设计的论文主线方法，面向基于动态图异常检测的金融反欺诈任务。模型主干为 `TRGT`，并且在三个数据集上坚持同一套纯 GNN 训练与推理架构：

- XinYe DGraph
- Elliptic Transactions
- Elliptic++ Transactions

最终部署路线：

`数据集本地预处理 -> UTPM 统一特征契约 -> TRGT 主干 -> DyRIFT 模块 -> 欺诈概率`

最终推理路径是单模型纯 GNN：

- 不依赖外部树模型
- 不依赖 teacher 分支
- 不依赖第二阶段融合器

## 主结果

| 数据集 | Val AUC | 结果目录 |
| --- | ---: | --- |
| XinYe DGraph | 0.792851 | `experiment/outputs/training/models/dyrift_gnn/full_xinye_repro_v1` |
| Elliptic Transactions | 0.821329 | `experiment/outputs/elliptic_transactions/training/models/dyrift_gnn/probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1` |
| Elliptic++ Transactions | 0.821953 | `experiment/outputs/ellipticpp_transactions/training/models/dyrift_gnn/probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1` |
| 宏平均 | 0.812044 | `docs/results/accepted_mainline_summary.json` |

运行时模型名：`dyrift_gnn`  
论文方法名：`Dynamic Risk-Informed Fraud Graph Neural Network (DyRIFT-GNN)`  
主干名：`Temporal-Relational Graph Transformer (TRGT)`

## 方法概览

- 主干：`TRGT`，负责时序关系感知的动态图消息传递。
- 推理期模块：target-context bridge、drift expert、internal risk fusion，以及数据集条件启用的 cold-start residual。
- 训练期方法：prototype memory 和 pseudo-contrastive temporal mining。
- 输入契约：三个数据集原始字段不同，但最终都映射到统一 `UTPM` 语义族；保留数据集级超参数，不拆模型架构。

## 文档导航

| 文档 | 说明 |
| --- | --- |
| [复现指南](docs/reproducibility.md) | 环境安装、特征构建、实验命令和结果文件 |
| [模型执行流程](docs/model_execution_flow.md) | 从原始动态图到欺诈概率的工程执行链路 |
| [论文方法说明](docs/thesis_method.md) | 论文主线、约束、统一架构与部署路径 |
| [方法卡片](docs/dyrift_gnn_method.md) | DyRIFT-GNN 简版方法卡 |
| [TRGT 主干](docs/trgt_backbone.md) | 主干结构与时序关系注意力 |
| [模块说明](docs/dyrift_modules.md) | 模块与训练方法拆分，以及消融证据 |
| [实验结果](docs/thesis_experiments.md) | 主结果、对比实验、消融实验、补充实验 |
| [训练与配置](docs/training_and_configs.md) | 命令、配置文件与输出布局 |
| [代码索引](docs/code_reference.md) | 代码目录与调用链 |
| [Studies 工作区](experiment/studies/README.md) | 对比、消融、递进式实验、补充实验 |
| [泄露审计](docs/leakage_audit.md) | 当前 accepted 主结果的硬泄露审计 |
| [主结果 AUC 表](docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv) | 三数据集 accepted AUC 汇总 |
| [对比实验 AUC 表](docs/results/comparison_auc.csv) | 对比实验 AUC 汇总 |
| [消融实验 AUC 表](docs/results/ablation_auc.csv) | 减法消融 AUC 汇总 |
| [递进实验 AUC 表](docs/results/progressive_auc.csv) | 方法递进式 AUC 汇总 |
| [补充实验 AUC 表](docs/results/supplementary_auc.csv) | XinYe `phase1+phase2` 联合训练补充实验 |
| [训练日志清单](docs/results/epoch_log_manifest.csv) | 每个实验的 epoch、日志和曲线路径 |

## 目录结构

| 路径 | 作用 |
| --- | --- |
| [experiment/mainline.py](experiment/mainline.py) | 单数据集特征构建与训练入口 |
| [experiment/suite.py](experiment/suite.py) | 三数据集统一 rerun 入口 |
| [experiment/audit.py](experiment/audit.py) | 硬泄露审计入口 |
| [experiment/configs/](experiment/configs) | 当前维护中的三数据集配置 |
| [experiment/datasets/](experiment/datasets) | 数据集注册、原始数据约定、预处理脚本 |
| [experiment/features/](experiment/features) | 统一特征构建与缓存工具 |
| [experiment/models/](experiment/models) | 运行时、引擎、主干、模块和预设 |
| [experiment/studies/](experiment/studies) | 独立的对比、消融、递进和补充实验 |
| [experiment/utils/](experiment/utils) | 路径、切分、IO、采样工具 |
| [docs/](docs) | 论文方法、实验和代码说明文档 |

## 复现命令

完整环境配置和复现清单见 [复现指南](docs/reproducibility.md)。下面是最小运行命令。

统一构建特征：

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  build_features \
  --phase both
```

运行当前维护的主线配置：

```bash
conda run -n Graph --no-capture-output python3 experiment/suite.py \
  --suite-name dyrift_mainline_rerun \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/configs/dyrift_suite.json \
  --seeds 42
```

运行单个独立 study：

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/comparisons/tgat_style_reference/run.py
```

重新生成 accepted 主结果泄露审计：

```bash
conda run -n Graph --no-capture-output python3 experiment/audit.py \
  --suite-summary experiment/outputs/reports/dyrift_gnn_accepted_mainline/summary.json
```

## 完整性约束

- `deployment_path=single_gnn_end_to_end`
- `dataset_isolation=true`
- `cross_dataset_training=false`
- `same_architecture_across_datasets=true`
- `hard_leakage_detected=false`
- XinYe `phase1+phase2` 联合训练补充实验单独存放，不作为正式无泄露论文主结果。
