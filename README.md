# Graduation Project

基于动态图异常检测的金融反欺诈毕业设计项目。

当前仓库已经收口为一条统一、可审计、三数据集可复现的主线：三个数据集先经过各自的数据清洗与统一特征映射，再进入同一套纯 GNN 主架构 `m8_utgt`。最终结果不依赖外部 residual、XGBoost 或 teacher 推理分支，部署路径就是单路 UTGT。

## Quick Links

| Card | What It Opens |
| --- | --- |
| [Method Overview](docs/thesis_method.md) | 统一纯 GNN 方法定义、模块说明、部署口径 |
| [Experiment Results](docs/thesis_experiments.md) | 主结果、GNN 对比、消融与说明 |
| [Mainline Guide](experiment/training/README_thesis_mainline.md) | 复现实验命令、套件入口、指标读取方式 |
| [Final Pure-GNN Summary](experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/summary.json) | 当前统一纯 GNN 三数据集结果 |
| [Final Leakage Audit](experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/leakage_audit.md) | 最终 pure-GNN 套件硬泄露审计 |
| [Final Metrics CSV](docs/results/thesis_m8_utgt_deploy_pure_eppcold_v1_metrics.csv) | 三数据集最终指标汇总 |
| [Epoch Metrics CSV](docs/results/thesis_m8_utgt_deploy_pure_eppcold_v1_epoch_metrics.csv) | 三数据集逐 epoch 训练日志合并表 |
| [Dataset Hparams](experiment/training/configs/thesis_dataset_hparams.pure_gnn_eppcold_v1.json) | 数据集级超参数模板 |
| [Shared-module Ablation](experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md) | 共享模块消融报告 |

## Final Thesis Route

最终采用的主线不是三套策略，而是一套统一架构在三个数据集上分别训练：

| Item | Choice |
| --- | --- |
| Unified backbone family | `m8_utgt` |
| Unified preset family | `utgt_temporal_shift_deploy_v1` |
| Unified deployment path | single pure-GNN path |
| Unified feature schema family | UTPM contract with dataset-local subsets |
| Core innovation modules | temporal-relation attention, temporal-normality bridge, drift-expert adaptation, prototype memory, pseudo-contrastive temporal mining, internal causal risk fusion, context-conditioned cold-start residual |
| Final suite | [thesis_m8_utgt_deploy_pure_eppcold_v1](experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/summary.json) |
| Leakage audit | [leakage_audit.md](experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/leakage_audit.md) |

这里不再做外部融合。训练与推理都是同一条 `m8_utgt` 路径，数据集之间只允许合理的超参数差异，不允许改成三套不同模型。

注意：最终 EPP run name 中的 `hybrid120` 只表示内部邻居采样器使用 recent/random 混合采样窗口，不表示外部 hybrid 模型、XGBoost 分支或二阶段分类器。

## Result Snapshot

| Dataset | Final Pure GNN Val AUC |
| --- | ---: |
| XinYe DGraph | 0.790455 |
| Elliptic | 0.821329 |
| Elliptic++ | 0.821953 |
| Macro Val AUC | 0.811246 |

读取方式：

- 这四个数都来自同一套纯 GNN 架构。
- 三个数据集没有混训，也没有第二个模型参与最终推理。
- 数据集之间只做输入容量、上下文特征组和图超参数的合理调优。

## GNN Comparison Snapshot

| Model | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical strong GNN `m5_temporal_graphsage` | 0.794628 | 0.793990 | 0.782830 | 0.790483 | 历史强 GNN 参考线 |
| Legacy thesis GNN `m7_utpm` | 0.776439 | 0.812635 | 0.777611 | 0.788895 | 旧主干 |
| Early pure UTGT `m8_utgt` | 0.772707 | 0.751369 | 0.777344 | 0.767140 | 初始纯 UTGT 基线 |
| Final pure UTGT `m8_utgt` | 0.790455 | 0.821329 | 0.821953 | 0.811246 | 当前论文主结果 |

## Why ET And EPP Are Lower Than The Historical 0.9+ Runs

如果你记得之前 ET / EPP 有过 `0.9+`，那批结果不是当前这条纯 GNN 路线的同口径结果。

- 之前的高分主要来自 `graphprop + tree/residual` 或 teacher-guided 的双路径方案。
- 当前最终路线锁成了单模型 pure GNN，可部署但更严格。
- ET / EPP 这两个数据集上，图传播式结构信号本来就特别强；最终版本通过纯 GNN 内部的时间上下文桥接和冷启动残差专家把 EPP 从旧版纯 GNN 的 `0.783441` 提升到 `0.821953`。
- 现在这版结果更适合答辩时讲成“统一单模型动态图 GNN 方法”，而不是“两模型融合系统”。

## Innovation Groups

| Group | Evidence |
| --- | --- |
| Temporal relation attention backbone | `m7_utpm -> m8_utgt` 完成统一动态图主干现代化 |
| Temporal-normality bridge | 目标节点级上下文桥接，保留纯 GNN 单路推理 |
| Drift-expert adaptation | 用时间漂移专家调节不同时间段的上下文融合 |
| Prototype memory | 共享模块消融保留，作为结构正则与类别稳定器 |
| Pseudo-contrastive temporal mining | 去掉后宏平均下降 `0.006182`，是最稳定有效的共享模块 |
| Internal causal risk fusion | 在纯 GNN 内部显式建模多尺度风险差分，不依赖外部 teacher 推理 |
| Context-conditioned cold-start residual | 修复 EPP 晚期冷启动节点消息不足问题，EPP `43-49` 天段 AUC 从 `0.5000` 提升到 `0.6088` |

## Leakage Guardrails

最终套件已经通过硬泄露审计：

- [Final Pure-GNN Audit](experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/leakage_audit.md)
- [Final Pure-GNN Audit JSON](experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/leakage_audit.json)

审计结论：

- 未发现 `train / val / test_pool / external` 交叉。
- 验证集节点与官方 `phase1_val` 对齐。
- 未发现跨数据集混训、跨数据集缓存复用或验证/测试标签回流训练。

## Main Entry

- 主训练入口：[run_thesis_mainline.py](experiment/training/run_thesis_mainline.py)
- 纯 GNN 套件：[run_thesis_suite.py](experiment/training/run_thesis_suite.py)
- 泄露审计脚本：[audit_thesis_leakage.py](experiment/training/audit_thesis_leakage.py)
