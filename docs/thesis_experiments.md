# Thesis Experiments

## Quick Links

- [Back to README](../README.md)
- [Method Overview](thesis_method.md)
- [Mainline Guide](../experiment/training/README_thesis_mainline.md)
- [Official Result JSON](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json)
- [Leakage Audit](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md)

## 1. Main Result

官方套件文件：

- [thesis_m7_v4_graphpropblend082](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json)

| Dataset | Official GNN Backbone | Secondary-only (non-GNN graphprop) | Official GNN-primary Blend | Gain vs Pure GNN |
| --- | ---: | ---: | ---: | ---: |
| XinYe DGraph | 0.777741 | 0.794888 | 0.795293 | +0.017552 |
| Elliptic | 0.801914 | 0.968319 | 0.949436 | +0.147522 |
| Elliptic++ | 0.778276 | 0.963736 | 0.946584 | +0.168308 |

宏平均：

- `Official GNN Backbone`: `0.785977`
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
| Official pure `m7_utpm` | 0.777741 | 0.801914 | 0.778276 | 0.785977 | 去掉二级校正后的纯主干 |
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
| `m7_utpm` only | 0.777741 | 0.801914 | 0.778276 | 0.785977 | 去掉整个二级校正层 |
| `secondary-only` | 0.794888 | 0.968319 | 0.963736 | 0.908981 | 保留 graphprop 分支，移除 GNN 输出 |
| weak hybrid `alpha=0.35` | 0.784396 | 0.841749 | 0.838915 | 0.821687 | 融合太弱，无法稳定吸收强 secondary |
| official blend `alpha=0.82` | 0.795293 | 0.949436 | 0.946584 | 0.897104 | 当前 official thesis result |

结论：

- XinYe 上，official blend 比 `secondary-only` 略强，说明 GNN 主干仍有正向贡献。
- Elliptic / Elliptic++ 上，`secondary-only` 更强，说明 graphprop 分支在这两个数据集上接近上界。
- `alpha=0.35` 明显不足，说明如果保留 GNN-primary 叙事，残差分支必须占更高权重。

### 4.2 Innovation Module Count

当前 thesis mainline 需要单独交代的创新模块一共 5 个：

| Module | Layer | Current Ablation Coverage | Official Status |
| --- | --- | --- | --- |
| `prototype memory` | GNN 主干 | 代码已支持，尚缺 official tri-dataset ablation 结果 | 待补完整存档 |
| `pseudo-contrastive temporal mining` | GNN 主干 | 代码已支持，尚缺 official tri-dataset ablation 结果 | 待补完整存档 |
| `drift residual target context` | GNN 主干 | 代码已支持，尚缺 official tri-dataset ablation 结果 | 待补完整存档 |
| `graphprop residual head` | 决策层 | 已完成并在上表公开 | 已完成 |
| `fixed logit fusion` | 决策层 | 已完成并在上表公开 | 已完成 |

也就是说：

- `utpm_unified` 是统一输入契约，不算 ablation 模块。
- 现在仓库里已经有完整的“决策层消融”。
- 但真正答辩级别的“主干内部模块消融”还应该把 `prototype / pseudo-contrastive / drift residual` 三项单独跑齐。
- 这三项不能省，因为它们才是你主干创新性最核心的证据。

## 5. Hard-Leakage Audit Summary

审计文件：

- [leakage_audit.md](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md)

审计结论：

- 三个数据集都没有发现 `train/val/test_pool/external` 交叉
- secondary 训练节点都严格属于各自数据集的 `phase1_train`
- validation 预测 bundle 与官方 split 完全对齐
- 没有跨数据集输出目录和缓存复用
