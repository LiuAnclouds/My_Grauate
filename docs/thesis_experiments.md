# Thesis Experiments

## Quick Links

- [Back to README](../README.md)
- [Reproducibility Guide](reproducibility.md)
- [Model Execution Flow](model_execution_flow.md)
- [Method Overview](thesis_method.md)
- [Accepted Leakage Audit](leakage_audit.md)
- [Mainline AUC CSV](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv)
- [Comparison AUC CSV](results/comparison_auc.csv)
- [Ablation AUC CSV](results/ablation_auc.csv)
- [Progressive AUC CSV](results/progressive_auc.csv)
- [Supplementary AUC CSV](results/supplementary_auc.csv)
- [Epoch Log Manifest](results/epoch_log_manifest.csv)

## 1. Accepted Mainline Result

当前论文正式主结果采用三条已经保存的 accepted run artifact：

- XinYe: `experiment/outputs/training/models/dyrift_gnn/full_xinye_repro_v1`
- ET: `experiment/outputs/elliptic_transactions/training/models/dyrift_gnn/probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1`
- EPP: `experiment/outputs/ellipticpp_transactions/training/models/dyrift_gnn/probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1`

| Setting | XinYe | ET | EPP | Macro Val AUC |
| --- | ---: | ---: | ---: | ---: |
| Full DyRIFT-GNN | 0.792851 | 0.821329 | 0.821953 | 0.812044 |

这张表的含义：

- 三个数据集共享同一套 `DyRIFT-GNN / TRGT` 架构。
- 最终推理只走单模型纯 GNN 路径。
- 允许数据集级超参数不同，但不允许拆成三套模型策略。

## 2. Comparison Experiments

这里的输入统一指同一套 `build_features --phase both` 构建出的统一语义特征缓存。不同模型可以做自己的入口映射，但不是各自重做一套特征工程。

| Setting | XinYe | ET | EPP | Macro Val AUC | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Full DyRIFT-GNN | 0.792851 | 0.821329 | 0.821953 | 0.812044 | 正式论文主结果 |
| Plain TRGT Backbone | 0.790742 | 0.800629 | 0.784006 | 0.791792 | 不带 DyRIFT 增强模块的纯主干 |
| TGAT-style Reference | 0.789445 | 0.800629 | 0.783644 | 0.791239 | 同组时序注意力 GNN 对比 |
| Temporal GraphSAGE Reference | 0.788309 | 0.773516 | 0.780595 | 0.780807 | 同组动态图 GNN 对比 |
| XGBoost Same Input | 0.745771 | 0.904028 | 0.941352 | 0.863717 | 非 GNN 同输入参考，不是最终部署路线 |

解释方式：

- 正式主结果相对 `Plain TRGT Backbone` 的宏平均提升是 `+0.020252`。
- 正式主结果相对 `TGAT-style Reference` 的宏平均提升是 `+0.020805`。
- 正式主结果相对 `Temporal GraphSAGE Reference` 的宏平均提升是 `+0.031238`。
- `XGBoost Same Input` 只作为同输入的非 GNN 参考线，不参与最终方法选型，因为它不是统一纯 GNN 部署路线。

## 3. Subtractive Ablation

主消融表使用减法设计，直接验证最终共享模块是否有效。

| Setting | XinYe | ET | EPP | Macro Val AUC | Delta vs Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full DyRIFT-GNN | 0.792851 | 0.821329 | 0.821953 | 0.812044 | +0.000000 |
| w/o Target-Context Bridge | 0.790284 | 0.800629 | 0.783473 | 0.791462 | -0.020582 |
| w/o Drift Expert | 0.790578 | 0.785032 | 0.803772 | 0.793127 | -0.018917 |
| w/o Prototype Memory | 0.791456 | 0.821686 | 0.819782 | 0.810975 | -0.001070 |
| w/o Pseudo-Contrastive Temporal Mining | 0.789999 | 0.821820 | 0.789550 | 0.800456 | -0.011588 |

从主消融表能读出的结论：

- `Target-Context Bridge` 和 `Drift Expert` 是宏平均掉点最大的两个共享组件。
- `Pseudo-Contrastive` 对 EPP 的提升最明显，是训练期最有效的共享方法之一。
- `Prototype Memory` 的宏平均增益较小，但在 EPP 上仍然是正贡献。
- `Internal Risk Fusion` 和 `Cold-Start Residual` 没有进入主消融表，因为它们不是三个最终 profile 都共同启用的组件。

## 4. Progressive Method Table

补充方法表使用递进设计，用来说明模型是如何一步步从 `TRGT` 长成 `DyRIFT-GNN` 的。

| Setting | XinYe | ET | EPP | Macro Val AUC |
| --- | ---: | ---: | ---: | ---: |
| Plain TRGT Backbone | 0.790742 | 0.800629 | 0.784006 | 0.791792 |
| TRGT + Bridge | 0.788639 | 0.784225 | 0.783053 | 0.785306 |
| TRGT + Bridge + Drift Expert | 0.788363 | 0.811953 | 0.783624 | 0.794647 |
| TRGT + Bridge + Drift Expert + Prototype Memory | 0.789463 | 0.809195 | 0.783048 | 0.793902 |
| TRGT + Bridge + Drift Expert + Prototype Memory + Pseudo-Contrastive | 0.790680 | 0.816581 | 0.783399 | 0.796886 |
| Full DyRIFT-GNN | 0.792851 | 0.821329 | 0.821953 | 0.812044 |

说明：

- 前五行来自 `experiment/outputs/studies/progressive/` 的统一 study 输出。
- 最后一行使用 accepted 主结果 artifact，因为它才是论文最终采用的 full model。

## 5. Supplementary XinYe Phase1+Phase2 Diagnostics

补充实验不是正式论文主线。它们用于回答一个诊断问题：如果把 XinYe phase2 的标注训练节点引入联合训练，模型是否能同时维持 phase1 验证表现并学习 phase2 排序。

| Setting | Train Scope | Checkpoint Scope | Phase1 Val AUC | Phase2 Train AUC | Phase2 Holdout AUC | Note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Official XinYe Mainline | `phase1.train` | `phase1.val` | 0.792851 | n/a | n/a | 正式无泄露论文主线 |
| Joint Phase1+Phase2 Train | `phase1.train + phase2.train` | `phase1.val` | 0.791441 | 0.716531 | n/a | 补充实验，不切 phase2 holdout |
| Phase-Aware Balanced | `phase1.train + 50% phase2.train` | `phase1.val` | 0.789344 | 0.635207 | 0.636328 | 加 phase indicator，phase2 holdout 仅诊断 |
| Phase-Aware DualVal | `phase1.train + 50% phase2.train` | `phase1.val + phase2.holdout` | 0.784233 | 0.709306 | 0.706197 | phase2 能提升，但牺牲 phase1 val |

补充实验的关键事实：

- 这些实验是 from-scratch joint training，不是 warmup 或 fine-tune。
- 它们没有跨 phase 人工造边，只做 disjoint union。
- `phase2.holdout` 来自 `phase2.train_mask` 的分层切分，不是官方 test pool。
- 它们使用了 `phase2` 的标注训练节点，因此不作为正式无泄露主结果。
- DualVal 将 phase2 AUC 从约 0.636 提升到约 0.706，但 phase1 val 从 0.789344 降到 0.784233，说明 phase1/phase2 存在明显阶段漂移。

输出位置：

- `experiment/outputs/studies/supplementary/xinye_phase12_joint_train_phase1_val/summary.json`
- `experiment/outputs/studies/supplementary/xinye_phase12_joint_train_phase1_val/xinye_dgraph/summary.json`
- `experiment/outputs/studies/supplementary/xinye_phase12_phase_aware_balanced/xinye_dgraph/summary.json`
- `experiment/outputs/studies/supplementary/xinye_phase12_phase_aware_dualval/xinye_dgraph/summary.json`

## 6. Saved Result Files

后续画图和论文制表直接用下面这些文件：

- [Mainline AUC CSV](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv)
- [Comparison AUC CSV](results/comparison_auc.csv)
- [Ablation AUC CSV](results/ablation_auc.csv)
- [Progressive AUC CSV](results/progressive_auc.csv)
- [Supplementary AUC CSV](results/supplementary_auc.csv)
- [Epoch Log Manifest](results/epoch_log_manifest.csv)
- [Studies Workspace](../experiment/studies/README.md)
- [Accepted Leakage Audit JSON](results/leakage_audit.json)
