# Thesis Experiments

## Quick Links

- [Back to README](../README.md)
- [Method Overview](thesis_method.md)
- [Mainline Guide](../experiment/training/README_thesis_mainline.md)
- [Official Result JSON](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json)
- [Backbone Ablation Report](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md)
- [Backbone Ablation CSV](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/results_long.csv)
- [Leakage Audit](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md)

## 1. Main Result

官方套件文件：

- [thesis_m7_v4_graphpropblend082](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json)

| Dataset | Official GNN Backbone | Secondary-only (non-GNN graphprop) | Official GNN-primary Blend | Gain vs Pure GNN |
| --- | ---: | ---: | ---: | ---: |
| XinYe DGraph | 0.776439 | 0.794888 | 0.795293 | +0.018854 |
| Elliptic | 0.812635 | 0.968319 | 0.949436 | +0.136801 |
| Elliptic++ | 0.777611 | 0.963736 | 0.946584 | +0.168973 |

宏平均：

- `Official GNN Backbone`: `0.788895`
- `Secondary-only (non-GNN graphprop)`: `0.908981`
- `Official GNN-primary Blend`: `0.897104`

## 2. Design Decision

这个问题不能回避：

1. 从纯验证集数值看，第二列 `Secondary-only` 确实整体更强。
2. 但第二列不是第二个 GNN，而是非 GNN 的 graphprop tree 分支。
3. 论文硬约束是“主模型必须是动态图 GNN”，所以它不能直接被写成 official main result。
4. 因此，正式结论要分成两层：
   - `Secondary-only`：作为强上界分支和 ablation，对外公开
   - `Official GNN-primary Blend`：作为满足论文约束的正式主线结果

如果只想冲当前 `val_auc`，第二列更优。
如果要同时满足“统一架构 + GNN 为主”的论文定义，第三列才是 official result。

## 3. Comparison Models

这里不再只放 1 条弱对比线，而是固定展示 3 个 baseline 加 1 个 official main result。

| Model | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical strong GNN `m5_temporal_graphsage` | 0.794628 | 0.793990 | 0.782830 | 0.790483 | 历史强 GNN 基线 |
| Official pure `m7_utpm` | 0.776439 | 0.812635 | 0.777611 | 0.788895 | 去掉二级校正后的纯主干 |
| Weak hybrid `alpha=0.35` | 0.784396 | 0.841749 | 0.838915 | 0.821687 | 弱融合对照 |
| Official GNN-primary blend `alpha=0.82` | 0.795293 | 0.949436 | 0.946584 | 0.897104 | 当前 official thesis result |

对应文件：

- Historical strong GNN:
  - `experiment/outputs/training/models/m5_temporal_graphsage/plan69a_m5_tcc_driftexpert015_entreg090w005_xinye_v1/summary.json`
  - `experiment/outputs/elliptic_transactions/training/models/m5_temporal_graphsage/probe_proto_bucketadv_dpdisc_ctxadaptive_elliptic_v2/summary.json`
  - `experiment/outputs/ellipticpp_transactions/training/models/m5_temporal_graphsage/plan63_epplus_clean_teacher_semrank_v1/summary.json`
- Official pure backbone:
  - `thesis_xy_m7_v4_unified_s42_e8`
  - `thesis_et_m7_v4_unified_s42_e8`
  - `thesis_epp_m7_v4_unified_s42_e8`
- Weak hybrid:
  - [thesis_m7_v4_xgbblend035](../experiment/outputs/thesis_suite/thesis_m7_v4_xgbblend035/summary.json)

## 4. Ablation

### 4.1 Completed Decision-Layer Ablation

| Setting | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| `m7_utpm` only | 0.776439 | 0.812635 | 0.777611 | 0.788895 | 去掉整个二级校正层 |
| `secondary-only` | 0.794888 | 0.968319 | 0.963736 | 0.908981 | 保留 graphprop 分支，移除 GNN 输出 |
| weak hybrid `alpha=0.35` | 0.784396 | 0.841749 | 0.838915 | 0.821687 | 融合太弱，无法稳定吸收强 secondary |
| official blend `alpha=0.82` | 0.795293 | 0.949436 | 0.946584 | 0.897104 | 当前 official thesis result |

结论：

- XinYe 上，official blend 比 `secondary-only` 略强，说明 GNN 主干仍有正向贡献。
- Elliptic / Elliptic++ 上，`secondary-only` 更强，说明 graphprop 分支在这两个数据集上接近上界。
- `alpha=0.35` 明显不足，说明如果保留 GNN-primary 叙事，残差分支必须占更高权重。

### 4.2 Completed Backbone-Module Ablation

官方主干三模块消融已经完成，聚合产物位于：

- [report.md](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md)
- [results_long.csv](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/results_long.csv)
- [results_macro.csv](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/results_macro.csv)

| Setting | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Delta vs Official Backbone |
| --- | ---: | ---: | ---: | ---: | ---: |
| official backbone | 0.776439 | 0.812635 | 0.777611 | 0.788895 | +0.000000 |
| no prototype memory | 0.777391 | 0.812275 | 0.778365 | 0.789344 | +0.000449 |
| no pseudo-contrastive mining | 0.775860 | 0.794927 | 0.777350 | 0.782712 | -0.006182 |
| no drift residual context | 0.777244 | 0.816354 | 0.779095 | 0.790898 | +0.002003 |

解读要实话实说：

- `pseudo-contrastive temporal mining` 是当前单种子 `phase1_val` 下最明确有效的主干创新，去掉后宏平均下降 `0.006182`，其中 Elliptic 单数据集下降 `0.017709`。
- `prototype memory` 在当前单种子验证上几乎中性，去掉后宏平均只波动 `+0.000449`，不能夸成主要增益来源。
- `drift residual target context` 在当前单种子验证上也没有带来直接 AUC 提升，去掉后宏平均反而上升 `+0.002003`；它更适合被表述为上下文校准/稳健性设计，而不是主增益模块。
- 论文级主结果的主要跃升仍然来自决策层两项创新：`graphprop residual head` 和 `fixed logit fusion`。

### 4.3 Innovation Module Count

当前 thesis mainline 需要单独交代的创新模块一共 5 个：

| Module | Layer | Current Ablation Coverage | Official Status |
| --- | --- | --- | --- |
| `prototype memory` | GNN 主干 | 已完成 official tri-dataset ablation | 已完成 |
| `pseudo-contrastive temporal mining` | GNN 主干 | 已完成 official tri-dataset ablation | 已完成 |
| `drift residual target context` | GNN 主干 | 已完成 official tri-dataset ablation | 已完成 |
| `graphprop residual head` | 决策层 | 已完成并在上表公开 | 已完成 |
| `fixed logit fusion` | 决策层 | 已完成并在上表公开 | 已完成 |

也就是说：

- `utpm_unified` 是统一输入契约，不算 ablation 模块。
- 现在仓库里已经同时有完整的“决策层消融”和“主干内部模块消融”。
- 三个主干模块都不能省，因为它们决定了你对创新性陈述能否做到逐项举证。

## 5. Hard-Leakage Audit Summary

审计文件：

- [leakage_audit.md](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md)

审计结论：

- 三个数据集都没有发现 `train/val/test_pool/external` 交叉
- secondary 训练节点都严格属于各自数据集的 `phase1_train`
- validation 预测 bundle 与官方 split 完全对齐
- 没有跨数据集输出目录和缓存复用
