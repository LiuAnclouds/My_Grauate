# Thesis Experiments

## Quick Links

- [Back to README](../README.md)
- [Method Overview](thesis_method.md)
- [Mainline Guide](../experiment/training/README_thesis_mainline.md)
- [Final Pure-GNN Summary](../experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/summary.json)
- [Final Pure-GNN Audit](../experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/leakage_audit.md)
- [Final Metrics CSV](results/thesis_m8_utgt_deploy_pure_eppcold_v1_metrics.csv)
- [Epoch Metrics CSV](results/thesis_m8_utgt_deploy_pure_eppcold_v1_epoch_metrics.csv)
- [Shared-module Ablation Report](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md)

## 1. Final Main Result

当前论文主结果：

- [thesis_m8_utgt_deploy_pure_eppcold_v1](../experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/summary.json)

| Dataset | Final Pure GNN |
| --- | ---: |
| XinYe DGraph | 0.790455 |
| Elliptic | 0.821329 |
| Elliptic++ | 0.821953 |
| Macro Val AUC | 0.811246 |

这张表的正确解释：

- 这四个数来自同一套纯 GNN `DyRIFT-GNN / TRGT` 主线。
- 不存在第二个外部模型参与最终推理。
- 三个数据集共享同一主架构，只在合理超参数上分开调优。

## 2. GNN-family Comparison

这里放 GNN 同组对比，避免只拿非 GNN 模型上界当主比较。

| Model | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Type |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical strong GNN `m5_temporal_graphsage` | 0.794628 | 0.793990 | 0.782830 | 0.790483 | 历史强 GNN 参考 |
| Legacy thesis GNN `m7_utpm` | 0.776439 | 0.812635 | 0.777611 | 0.788895 | 旧主干 |
| Early pure TRGT `m8_utgt` | 0.772707 | 0.751369 | 0.777344 | 0.767140 | 初始纯 TRGT 基线 |
| Final `DyRIFT-GNN` / `TRGT` | 0.790455 | 0.821329 | 0.821953 | 0.811246 | 当前论文主结果 |

历史强 GNN 参考线来源：

- XinYe: `experiment/outputs/training/models/m5_temporal_graphsage/plan69a_m5_tcc_driftexpert015_entreg090w005_xinye_v1/summary.json`
- Elliptic: `experiment/outputs/elliptic_transactions/training/models/m5_temporal_graphsage/probe_proto_bucketadv_dpdisc_ctxadaptive_elliptic_v2/summary.json`
- Elliptic++: `experiment/outputs/ellipticpp_transactions/training/models/m5_temporal_graphsage/plan63_epplus_clean_teacher_semrank_v1/summary.json`

结论：

- 单纯换成早期 `TRGT` 不会自动赢，early pure `m8_utgt` 宏平均只有 `0.767140`。
- 继续沿着纯 GNN 路线做统一输入重构、桥接、冷启动增强和容量调优后，宏平均提升到 `0.811246`。
- 当前最终结果已经超过历史强 GNN 宏平均 `0.790483`。

## 3. Mainline Ablation

这部分回答“统一主线里的关键变化是否有效”。

| Setting | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Legacy `m7_utpm` | 0.776439 | 0.812635 | 0.777611 | 0.788895 | 旧 thesis 主干 |
| Early pure `TRGT` | 0.772707 | 0.751369 | 0.777344 | 0.767140 | 只换 backbone 的初始版本 |
| Final `DyRIFT-GNN` | 0.790455 | 0.821329 | 0.821953 | 0.811246 | 当前统一纯 GNN 主线 |

从这张表能直接读出的结论：

- `TRGT` 的价值不是“换个 Transformer 就天然变强”。
- GNN 侧真正有效的是统一输入重构、temporal-normality bridge、drift-expert 适配、prototype/pseudo-contrastive 正则和内部风险分支的组合。
- 这条单模型纯 GNN 路线已经把三数据集宏平均拉到 `0.811246`。

## 4. Shared-module Ablation

共享模块逐项消融保留在 legacy `m7` 主干上，因为 `prototype memory / pseudo-contrastive / drift bridge` 的实现仍然是共用模块。

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

- `pseudo-contrastive temporal mining` 是当前最稳定有效的共享主干模块。
- `prototype memory` 更适合作为结构正则，不应夸成主要 AUC 来源。
- `drift bridge` 更偏稳健性与上下文校准，而不是单看验证 AUC 的增益模块。

## 5. Why ET And EPP No Longer Show 0.9+

如果你之前看到 ET / EPP 有过 `0.9+`，那批结果不是当前 pure-GNN 主线的同口径结果。

| Route | ET | EPP | Reading |
| --- | ---: | ---: | --- |
| Historical hybrid / residual upper-bound | 0.903161 | 0.901002 | 含外部分支，不是当前最终路线 |
| Final pure GNN | 0.821329 | 0.821953 | 单模型、可部署、口径统一 |

所以这里不是“纯 GNN 突然退化了”，而是：

- 比较对象变了。
- 之前高分使用了额外图传播或树模型校正。
- 现在最终路线锁成单模型纯 GNN，指标更真实，也更适合答辩时讲统一方法。

## 6. Hard-Leakage Audit

最终主结果已经重新审计：

- [Final Pure-GNN Audit](../experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/leakage_audit.md)
- [Final Pure-GNN Audit JSON](../experiment/outputs/thesis_suite/thesis_m8_utgt_deploy_pure_eppcold_v1/leakage_audit.json)

审计结论：

- `hard_leakage_detected = false`
- 三个数据集都没有发现 `train / val / test_pool / external` 交叉。
- GNN 验证 bundle 与官方 `phase1_val` 完全对齐。
- 没有跨数据集缓存复用，也没有把验证/测试标签回流训练。
