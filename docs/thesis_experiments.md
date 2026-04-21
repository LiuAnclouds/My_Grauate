# Thesis Experiments

## Quick Links

- [Back to README](../README.md)
- [Method Overview](thesis_method.md)
- [Mainline Guide](../experiment/training/README_thesis_mainline.md)
- [Recommended Result JSON](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/summary.json)
- [Pure Teacher Backbone JSON](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_e8_s42_v1/summary.json)
- [AUC-first Appendix JSON](../experiment/outputs/thesis_suite/thesis_m8_utgt_graphpropblend091/summary.json)
- [Recommended Leakage Audit](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/leakage_audit.md)
- [Shared-module Ablation Report](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md)

## 1. Main Result

当前推荐论文主结果：

- [thesis_m8_utgt_teacher_gnnprimary04999](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/summary.json)

| Dataset | Pure Teacher GNN | Secondary-only Graphprop | Recommended GNN-primary Blend | Gain vs Pure Teacher GNN |
| --- | ---: | ---: | ---: | ---: |
| XinYe DGraph | 0.783101 | 0.795130 | 0.794914 | +0.011813 |
| Elliptic | 0.785398 | 0.968031 | 0.891093 | +0.105696 |
| Elliptic++ | 0.783195 | 0.963736 | 0.893422 | +0.110227 |
| Macro Val AUC | 0.783898 | 0.908965 | 0.859810 | +0.075912 |

这张表的正确解释是：

- 第二列是纯 GNN，已经是统一主干结果
- 第三列不是 GNN，而是单独的 graphprop 分支
- 第四列才是最终论文主结果，因为它满足 `GNN-primary`

## 2. Comparison Models

这里固定给出 5 条主要对比线，避免只和 1 个弱 baseline 比。

| Model | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Type |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical strong GNN `m5_temporal_graphsage` | 0.794628 | 0.793990 | 0.782830 | 0.790483 | 历史强 GNN 基线 |
| Legacy thesis GNN `m7_utpm` | 0.776439 | 0.812635 | 0.777611 | 0.788895 | 旧主干 |
| Pure UTGT `m8_utgt` | 0.772707 | 0.751369 | 0.777344 | 0.767140 | 只做主干替换 |
| Teacher-guided pure UTGT | 0.783101 | 0.785398 | 0.783195 | 0.783898 | 纯 GNN 推荐主干 |
| Recommended `m8_utgt` GNN-primary blend `0.4999` | 0.794914 | 0.891093 | 0.893422 | 0.859810 | 当前论文主结果 |

对比结论：

- 单纯把 `m7` 换成 `m8` 并不会自动提高指标
- `teacher-guided` 后，pure GNN 宏平均从 `0.767140` 提升到 `0.783898`
- 再叠加 strict-frontier GNN-primary residual correction 后，宏平均提升到 `0.859810`

## 3. Backbone-level Ablation

这部分用来回答“新主线里的几个创新组是否真的有效”。

| Setting | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Pure `m8_utgt` | 0.772707 | 0.751369 | 0.777344 | 0.767140 | 只换 attention 主干，不加 teacher |
| Teacher-guided pure `m8_utgt` | 0.783101 | 0.785398 | 0.783195 | 0.783898 | 训练期 teacher guidance 有效 |
| Pure `m8_utgt` + blend `0.48` | 0.791590 | 0.849702 | 0.883761 | 0.841684 | 说明 residual correction 本身有效 |
| Teacher `m8_utgt` + blend `0.48` | 0.794762 | 0.885479 | 0.891094 | 0.857111 | 更保守的 GNN-primary 版本 |
| Teacher `m8_utgt` + blend `0.4999` | 0.794914 | 0.891093 | 0.893422 | 0.859810 | 推荐主结果 |
| AUC-first blend `0.91` | 0.794897 | 0.965251 | 0.958475 | 0.906208 | appendix only，不作为论文主线 |

从这张表能直接读出的结论：

1. `teacher guidance` 贡献为 `0.016758` 宏平均提升
2. 从保守版 `alpha=0.48` 推到 strict-frontier `alpha=0.4999` 后，宏平均还能再提升 `0.002698`
3. `alpha=0.91` 的确更高，但已经偏向 secondary-dominant，不适合作为“GNN 主模型”主线

## 4. Shared-module Ablation

共享主干模块的逐项消融保留在 legacy `m7` 主干上，因为 `prototype memory / pseudo-contrastive / drift residual` 的实现仍然是共用模块。

- [report.md](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md)
- [results_long.csv](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/results_long.csv)
- [results_macro.csv](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/results_macro.csv)

| Setting | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Delta vs Legacy `m7` |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy `m7` backbone | 0.776439 | 0.812635 | 0.777611 | 0.788895 | +0.000000 |
| no prototype memory | 0.777391 | 0.812275 | 0.778365 | 0.789344 | +0.000449 |
| no pseudo-contrastive mining | 0.775860 | 0.794927 | 0.777350 | 0.782712 | -0.006182 |
| no drift residual context | 0.777244 | 0.816354 | 0.779095 | 0.790898 | +0.002003 |

阅读方式：

- `pseudo-contrastive temporal mining` 是当前最稳定有效的共享主干模块
- `prototype memory` 更像弱正则，不能夸成主要 AUC 来源
- `drift residual target context` 更适合表述为稳健性模块，而不是单看验证 AUC 的增益模块

## 5. Why Secondary-only Is Not The Main Model

这个问题在答辩里一定会被问到：

- 为什么 ET/EPP 上 `secondary-only` 比 `blend` 还高？

答案是：

1. `secondary-only` 不是第二个 GNN，而是 graphprop tree 分支
2. ET/EPP 的传播结构让 graphprop 单列非常强
3. 但论文主模型硬约束是“必须是动态图 GNN”
4. 所以 `secondary-only` 只能作为上界、ablation 或 appendix
5. 推荐主结果必须使用 `alpha=0.4999` 的 strict-frontier `GNN-primary blend`

## 6. Hard-Leakage Audit

推荐主线已经按新 summary 重新审计：

- [leakage_audit.md](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/leakage_audit.md)

审计结论：

- 三个数据集都没有发现 `train / val / test_pool / external` 交叉
- secondary 训练节点都严格属于各自数据集的 `phase1_train`
- hybrid 验证 bundle 与官方 split 完全对齐
- teacher 与 secondary 都没有跨数据集缓存复用
