# Thesis Experiments

## 1. Main Result

官方套件文件：

- `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json`

| Dataset | Pure M7 GNN | Graphprop Secondary | Final Hybrid | Gain vs Pure GNN |
| --- | ---: | ---: | ---: | ---: |
| XinYe DGraph | 0.777741 | 0.794888 | 0.795293 | +0.017552 |
| Elliptic | 0.801914 | 0.968319 | 0.949436 | +0.147522 |
| Elliptic++ | 0.778276 | 0.963736 | 0.946584 | +0.168308 |

## 2. Comparison Models

这里放 3 条效果更弱或更旧的对比线。

### 2.1 Historical M5 Family Best Saved Run

| Dataset | Run | Val AUC |
| --- | --- | ---: |
| XinYe DGraph | `plan69a_m5_tcc_driftexpert015_entreg090w005_xinye_v1` | 0.794628 |
| Elliptic | `probe_proto_bucketadv_dpdisc_ctxadaptive_elliptic_v2` | 0.793990 |
| Elliptic++ | `plan63_epplus_clean_teacher_semrank_v1` | 0.782830 |

### 2.2 Pure Official M7 Backbone

| Dataset | Run | Val AUC |
| --- | --- | ---: |
| XinYe DGraph | `thesis_xy_m7_v4_unified_s42_e8` | 0.777741 |
| Elliptic | `thesis_et_m7_v4_unified_s42_e8` | 0.801914 |
| Elliptic++ | `thesis_epp_m7_v4_unified_s42_e8` | 0.778276 |

### 2.3 Old Weak Hybrid

历史弱融合文件：

- `experiment/outputs/thesis_suite/thesis_m7_v4_xgbblend035/summary.json`

| Dataset | Val AUC |
| --- | ---: |
| XinYe DGraph | 0.784396 |
| Elliptic | 0.841749 |
| Elliptic++ | 0.838915 |

## 3. Ablation

| Setting | XinYe | Elliptic | Elliptic++ |
| --- | ---: | ---: | ---: |
| `m7_utpm` only | 0.777741 | 0.801914 | 0.778276 |
| graphprop secondary only | 0.794888 | 0.968319 | 0.963736 |
| weak hybrid `alpha=0.35` | 0.784396 | 0.841749 | 0.838915 |
| final hybrid `alpha=0.82` | 0.795293 | 0.949436 | 0.946584 |

结论：

- XinYe 上，最终收益主要来自 graphprop residual 对时间漂移排序误差的校正
- Elliptic / Elliptic++ 上，secondary 非常强，但固定融合仍然优于单纯保留纯 GNN
- `alpha=0.35` 明显不足，说明 residual head 在最终架构里必须占据更高权重

## 4. Hard-Leakage Audit Summary

审计文件：

- `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md`

审计结论：

- 三个数据集都没有发现 `train/val/test_pool/external` 交叉
- secondary 训练节点都严格属于各自数据集的 `phase1_train`
- validation 预测 bundle 与官方 split 完全对齐
- 没有跨数据集输出目录和缓存复用
