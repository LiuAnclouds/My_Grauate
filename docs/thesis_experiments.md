# Thesis Experiments

## Quick Links

- [Back to README](../README.md)
- [Reproducibility Guide](reproducibility.md)
- [Experiment Reproduction](experiment_reproduction.md)
- [Model Execution Flow](model_execution_flow.md)
- [Method Overview](thesis_method.md)
- [Accepted Leakage Audit](leakage_audit.md)
- [Mainline AUC CSV](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv)
- [Comparison AUC CSV](results/comparison_auc.csv)
- [Ablation AUC CSV](results/ablation_auc.csv)
- [Progressive AUC CSV](results/progressive_auc.csv)
- [Supplementary AUC CSV](results/supplementary_auc.csv)
- [Presentation AUC CSV](results/presentation_auc_percent.csv)
- [Experiment Epoch Policy CSV](results/experiment_epoch_policy.csv)
- [Historical External Records](results/historical_external_records.csv)
- [Epoch Log Manifest](results/epoch_log_manifest.csv)

## 1. Accepted Mainline Result

当前论文正式主结果采用三条已经保存的 accepted run artifact：

- XinYe: `experiment/outputs/training/models/dyrift_gnn/full_xinye_repro_v1`
- ET: `experiment/outputs/elliptic_transactions/training/models/dyrift_gnn/probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1`
- EPP: `experiment/outputs/ellipticpp_transactions/training/models/dyrift_gnn/probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1`

| Setting | XinYe | ET | EPP | Macro Val AUC |
| --- | ---: | ---: | ---: | ---: |
| Full DyRIFT-GNN | 79.2851% | 82.1329% | 82.1953% | 81.2044% |

这张表的含义：

- 三个数据集共享同一套 `DyRIFT-GNN / TRGT` 架构。
- 最终推理只走单模型纯 GNN 路径。
- 允许数据集级超参数不同，但不允许拆成三套模型策略。

## 2. Comparison Experiments

这里的输入统一指同一套 `build_features --phase both` 构建出的统一语义特征缓存。不同模型可以做自己的入口映射，但不是各自重做一套特征工程。

| Setting | XinYe | ET | EPP | Macro Val AUC | Delta vs Full | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full DyRIFT-GNN | 79.2851% | 82.1329% | 82.1953% | 81.2044% | +0.0000 pp | 正式论文主结果 |
| Plain TRGT Backbone | 79.0742% | 80.0629% | 78.4006% | 79.1792% | -2.0252 pp | 不带 DyRIFT 增强模块的纯主干 |
| TGAT-style Reference | 78.9445% | 80.0629% | 78.3644% | 79.1239% | -2.0805 pp | 同组时序注意力 GNN 对比 |
| Temporal GraphSAGE Reference | 78.8309% | 77.3516% | 78.0595% | 78.0807% | -3.1238 pp | 同组动态图 GNN 对比 |
| XGBoost Same Input | 74.5771% | 90.4028% | 94.1352% | 86.3717% | +5.1673 pp | 非 GNN 同输入参考，不是最终部署路线 |

解释方式：

- 正式主结果相对 `Plain TRGT Backbone` 的宏平均提升是 `+2.0252` 个百分点。
- 正式主结果相对 `TGAT-style Reference` 的宏平均提升是 `+2.0805` 个百分点。
- 正式主结果相对 `Temporal GraphSAGE Reference` 的宏平均提升是 `+3.1238` 个百分点。
- `XGBoost Same Input` 只作为同输入的非 GNN 补充参考线，不参与最终方法选型，也不用于说明 GNN 方法优越性，因为它不是统一纯 GNN 部署路线。

## 3. Subtractive Ablation

主消融表使用减法设计，直接验证最终共享模块是否有效。

| Setting | XinYe | ET | EPP | Macro Val AUC | Delta vs Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full DyRIFT-GNN | 79.2851% | 82.1329% | 82.1953% | 81.2044% | +0.0000 pp |
| w/o Target-Context Bridge | 79.0284% | 80.0629% | 78.3473% | 79.1462% | -2.0582 pp |
| w/o Drift Expert | 79.0578% | 78.5032% | 80.3772% | 79.3127% | -1.8917 pp |
| w/o Prototype Memory | 79.1456% | 82.1686% | 81.9782% | 81.0975% | -0.1070 pp |
| w/o Pseudo-Contrastive Temporal Mining | 78.9999% | 82.1820% | 78.9550% | 80.0456% | -1.1588 pp |

从主消融表能读出的结论：

- `Target-Context Bridge` 和 `Drift Expert` 是宏平均掉点最大的两个共享组件。
- `Pseudo-Contrastive` 对 EPP 的提升最明显，是训练期最有效的共享方法之一。
- `Prototype Memory` 的宏平均增益较小，但在 EPP 上仍然是正贡献。
- `Internal Risk Fusion` 和 `Cold-Start Residual` 没有进入主消融表，因为它们不是三个最终 profile 都共同启用的组件。

## 4. Progressive Method Table

补充方法表使用递进设计，用来说明模型是如何一步步从 `TRGT` 长成 `DyRIFT-GNN` 的。

| Setting | XinYe | ET | EPP | Macro Val AUC | Delta vs Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain TRGT Backbone | 79.0742% | 80.0629% | 78.4006% | 79.1792% | -2.0252 pp |
| TRGT + Bridge | 78.8639% | 78.4225% | 78.3053% | 78.5306% | -2.6738 pp |
| TRGT + Bridge + Drift Expert | 78.8363% | 81.1953% | 78.3624% | 79.4647% | -1.7397 pp |
| TRGT + Bridge + Drift Expert + Prototype Memory | 78.9463% | 80.9195% | 78.3048% | 79.3902% | -1.8142 pp |
| TRGT + Bridge + Drift Expert + Prototype Memory + Pseudo-Contrastive | 79.0680% | 81.6581% | 78.3399% | 79.6886% | -1.5158 pp |
| Full DyRIFT-GNN | 79.2851% | 82.1329% | 82.1953% | 81.2044% | +0.0000 pp |

说明：

- 前五行来自 `experiment/outputs/studies/progressive/` 的统一 study 输出。
- 最后一行使用 accepted 主结果 artifact，因为它才是论文最终采用的 full model。

## 5. Supplementary XinYe Phase1+Phase2 Diagnostics

补充实验不是正式论文主线。它们用于回答一个诊断问题：如果把 XinYe phase2 的标注训练节点引入联合训练，模型是否能同时维持 phase1 验证表现并学习 phase2 排序。

| Setting | Train Scope | Checkpoint Scope | Phase1 Val AUC | Phase2 Train AUC | Phase2 Holdout AUC | Note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Official XinYe Mainline | `phase1.train` | `phase1.val` | 79.2851% | n/a | n/a | 正式无泄露论文主线 |
| Joint Phase1+Phase2 Train | `phase1.train + phase2.train` | `phase1.val` | 79.1441% | 71.6531% | n/a | 补充实验，不切 phase2 holdout |
| Phase-Aware Balanced | `phase1.train + 50% phase2.train` | `phase1.val` | 78.9344% | 63.5207% | 63.6328% | 加 phase indicator，phase2 holdout 仅诊断 |
| Phase-Aware DualVal | `phase1.train + 50% phase2.train` | `phase1.val + phase2.holdout` | 78.4233% | 70.9306% | 70.6197% | phase2 能提升，但牺牲 phase1 val |

补充实验的关键事实：

- 这些实验是 from-scratch joint training，不是 warmup 或 fine-tune。
- 它们没有跨 phase 人工造边，只做 disjoint union。
- `phase2.holdout` 来自 `phase2.train_mask` 的分层切分，不是官方 test pool。
- 它们使用了 `phase2` 的标注训练节点，因此不作为正式无泄露主结果。
- DualVal 将 phase2 AUC 从约 63.63% 提升到约 70.62%，但 phase1 val 从 78.9344% 降到 78.4233%，说明 phase1/phase2 存在明显阶段漂移。

输出位置：

- `experiment/outputs/studies/supplementary/xinye_phase12_joint_train_phase1_val/summary.json`
- `experiment/outputs/studies/supplementary/xinye_phase12_joint_train_phase1_val/xinye_dgraph/summary.json`
- `experiment/outputs/studies/supplementary/xinye_phase12_phase_aware_balanced/xinye_dgraph/summary.json`
- `experiment/outputs/studies/supplementary/xinye_phase12_phase_aware_dualval/xinye_dgraph/summary.json`

## 6. Historical External Record

该记录来自历史竞赛/外部评测，不覆盖当前仓库可复现的 `DyRIFT-GNN / TRGT` 主线 artifact。它可以用于论文或答辩中说明历史调参和参赛阶段曾达到的外部评测水平，但不能和当前 accepted mainline 的 summary.json 混写。

| Run ID | Timestamp | Dataset | Metric | Value | Source |
| --- | --- | --- | --- | ---: | --- |
| `XXiRer_47f488` | 2025-11-13 23:01:21 | XinYe DGraph | Val AUC | 81.5585% | user-provided competition evaluation record |

CSV: [Historical External Records](results/historical_external_records.csv)

## 7. Saved Result Files

后续画图和论文制表直接用下面这些文件：

- [Mainline AUC CSV](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv)
- [Comparison AUC CSV](results/comparison_auc.csv)
- [Ablation AUC CSV](results/ablation_auc.csv)
- [Progressive AUC CSV](results/progressive_auc.csv)
- [Supplementary AUC CSV](results/supplementary_auc.csv)
- [Presentation AUC CSV](results/presentation_auc_percent.csv)
- [Experiment Epoch Policy CSV](results/experiment_epoch_policy.csv)
- [Training Policy Summary JSON](results/training_policy_summary.json)
- [Historical External Records](results/historical_external_records.csv)
- [Epoch Log Manifest](results/epoch_log_manifest.csv)
- [Studies Workspace](../experiment/studies/README.md)
- [Accepted Leakage Audit JSON](results/leakage_audit.json)

## 8. Epoch Policy

当前维护中的正式 rerun 和所有 study rerun 统一采用：

| Item | Value |
| --- | ---: |
| Max epochs | 70 |
| Minimum early-stop epoch | 30 |
| Default graph patience | 10 |
| XGBoost boosting rounds | 70 |
| XGBoost early-stopping rounds | 30 |

`docs/results/experiment_epoch_policy.csv` 记录计划训练策略；`docs/results/epoch_log_manifest.csv` 记录已经保存 artifact 的真实 epoch、日志和曲线路径。两者不混写，避免为了展示而改动历史训练记录。
