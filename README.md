# Graduation Project

基于动态图异常检测的金融反欺诈毕业设计项目。

当前仓库已经收敛为一条统一主线：

- 三个数据集分别预处理，但全部映射到同一输入契约 `utpm_unified`
- 训练协议统一为 `phase1_train -> phase1_val -> test_pool`
- 主模型必须是动态图 GNN
- 允许混合决策，但最终结果必须保持 `GNN-primary`
- 不做跨数据集联合训练，不做 split 泄露

## Quick Links

| Entry | Purpose |
| --- | --- |
| [Method Overview](docs/thesis_method.md) | 统一架构、teacher 含义、为什么这不是“两套模型” |
| [Experiment Results](docs/thesis_experiments.md) | 主结果、对比模型、消融实验、取舍说明 |
| [Mainline Guide](experiment/training/README_thesis_mainline.md) | 复现实验命令、推荐套件、legacy 支撑实验 |
| [Dataset Hparam Profile](experiment/training/configs/thesis_dataset_hparams.search_v1.json) | 统一架构下的数据集级调参模板 |
| [Recommended Result JSON](experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/summary.json) | 当前推荐论文主结果 |
| [Recommended Leakage Audit](experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/leakage_audit.md) | 新主结果的硬泄露审计 |
| [Pure Teacher Backbone JSON](experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_e8_s42_v1/summary.json) | 不加决策层时的纯 GNN 结果 |
| [AUC-first Appendix JSON](experiment/outputs/thesis_suite/thesis_m8_utgt_graphpropblend091/summary.json) | 只追 AUC 的附录结果，不作为论文主线 |
| [Shared-module Ablation Report](experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md) | 共享主干模块的逐项消融证据 |

## Recommended Mainline

当前推荐主线不是旧的 `m7 + alpha=0.82`，而是：

| Item | Choice |
| --- | --- |
| Unified input contract | `utpm_unified` |
| Main GNN backbone | `m8_utgt` |
| GNN preset | `utgt_temporal_shift_teacher_v1` |
| Training-time guidance | 数据集内、只读的 graphprop logits，用于 target-context、rank distill、hard negative guidance |
| Inference decision | fixed logit fusion |
| Blend weight | `alpha=0.4999`，即 `50.01% GNN + 49.99% secondary` |
| Recommended suite | [thesis_m8_utgt_teacher_gnnprimary04999](experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/summary.json) |

最终决策公式为：

`sigmoid(0.5001 * logit(p_gnn) + 0.4999 * logit(p_secondary))`

这里必须说明白：

- `teacher` 指的是训练期的只读教师信号，不是另一套主模型
- `secondary-only` 指的是单独使用 graphprop 分支做预测，它不是 GNN
- 当前论文主结果是 `GNN-primary blend 0.4999`，不是 `secondary-only`

## Architecture vs Hyperparameters

现在仓库明确区分两件事：

- 必须统一：`utpm_unified` 输入契约、`m8_utgt` 主干家族、teacher 读法、secondary 家族、GNN-primary 决策公式
- 可以按数据集调：`attr_proj_dim`、`hidden_dim`、`rel_dim`、`fanouts`、`batch_size`、`epochs`、`learning_rate`、`weight_decay`、`dropout`、低层 `graph_config_overrides`、`blend_alpha`

统一批跑时优先使用：

- [thesis_dataset_hparams.search_v1.json](experiment/training/configs/thesis_dataset_hparams.search_v1.json)

这样做的含义是：

- 不是三个数据集三套模型
- 而是一套统一架构下的 dataset-local hyperparameter tuning

## Result Snapshot

推荐主线在 `phase1_val` 上的结果如下：

| Dataset | Pure Teacher GNN | Secondary-only Graphprop | Recommended GNN-primary Blend |
| --- | ---: | ---: | ---: |
| XinYe DGraph | 0.783101 | 0.795130 | 0.794914 |
| Elliptic | 0.785398 | 0.968031 | 0.891093 |
| Elliptic++ | 0.783195 | 0.963736 | 0.893422 |
| Macro Val AUC | 0.783898 | 0.908965 | 0.859810 |

解释口径：

- 三个数据集最终主结果都已经稳定在 `0.79+` 区间，其中 XinYe 为 `0.794914`
- `secondary-only` 在 ET/EPP 上更强，但它不是 GNN，所以只能作为上界分支或 appendix
- `alpha=0.4999` 仍然保持 GNN 主导，不会退化成“树模型为主、GNN陪跑”

## Comparison Snapshot

| Model | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical strong GNN `m5_temporal_graphsage` | 0.794628 | 0.793990 | 0.782830 | 0.790483 | 历史 GNN 基线 |
| Legacy thesis GNN `m7_utpm` | 0.776439 | 0.812635 | 0.777611 | 0.788895 | 旧主干 |
| Pure UTGT `m8_utgt` | 0.772707 | 0.751369 | 0.777344 | 0.767140 | 只换 attention 主干但不加 teacher |
| Teacher-guided pure UTGT | 0.783101 | 0.785398 | 0.783195 | 0.783898 | 纯 GNN 推荐主干 |
| Recommended `m8_utgt` GNN-primary blend `0.4999` | 0.794914 | 0.891093 | 0.893422 | 0.859810 | 当前论文主结果 |

## Innovation Groups

当前需要交代清楚的创新组一共 6 个：

| Group | Evidence |
| --- | --- |
| Temporal relation attention backbone | `m7_utpm -> m8_utgt` 在同一输入契约下完成主干现代化 |
| Prototype memory | 保留 legacy shared-module ablation 作为直接证据 |
| Pseudo-contrastive temporal mining | 去掉后宏平均下降 `0.006182`，是最稳定有效的共享主干模块 |
| Drift residual target context | 更偏稳健性与上下文校准，而不是直接 AUC 驱动 |
| Teacher-guided temporal normality bridge | pure `m8_utgt` 到 teacher-guided pure `m8_utgt`，宏平均提升 `0.016758` |
| Graphprop residual correction + fixed fusion | teacher pure GNN 到 recommended blend，宏平均再提升 `0.073213` |

这套表述比“我们发明了一个全新底层 GNN 家族”更准确，也比“只是把树模型堆在外面”更扎实。

## Leakage Guardrails

推荐主线已经重新做过硬泄露审计：

- [leakage_audit.md](experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/leakage_audit.md)
- [leakage_audit.json](experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/leakage_audit.json)

审计结论：

- 未发现 `train / val / test_pool / external` 交叉
- 未发现跨数据集混训
- teacher 与 secondary 都严格限制为当前数据集、`phase1_train` 拟合、只读加载
- 验证集只用于推理与评估，不回流训练

## Main Entry

- 主训练入口：[run_thesis_mainline.py](experiment/training/run_thesis_mainline.py)
- 纯 GNN 套件：[run_thesis_suite.py](experiment/training/run_thesis_suite.py)
- GNN-primary hybrid 套件：[run_thesis_hybrid_suite.py](experiment/training/run_thesis_hybrid_suite.py)
- 固定融合脚本：[run_thesis_hybrid_blend.py](experiment/training/run_thesis_hybrid_blend.py)
- 泄露审计脚本：[audit_thesis_leakage.py](experiment/training/audit_thesis_leakage.py)
