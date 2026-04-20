# Graduation Project

基于动态图神经网络异常检测的金融反欺诈毕业设计项目。

当前仓库只保留一条统一主线：

- 三个数据集分别预处理，但全部映射到同一输入契约 `utpm_unified`
- 主模型统一为动态图 GNN `m7_utpm`
- 统一训练协议为 `phase1_train -> phase1_val -> test_pool`
- 二级分支只允许作为 GNN 主干的残差校正，不允许变成“每个数据集一套策略”

## Navigation

| Entry | Purpose |
| --- | --- |
| [Method Overview](docs/thesis_method.md) | 统一架构、创新模块、为什么 official 结果不是直接选第二列 |
| [Experiment Results](docs/thesis_experiments.md) | 主结果、3 个 baseline、消融矩阵、设计取舍 |
| [Mainline Guide](experiment/training/README_thesis_mainline.md) | 训练命令、复现实验、模块级 ablation 命令 |
| [Official Result JSON](experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json) | 官方三数据集结果文件 |
| [Leakage Audit](experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md) | 硬泄露审计与 split 隔离证据 |

## Official Thesis Contract

- 同一输入特征契约：`utpm_unified`
- 同一主模型族：`m7_utpm`
- 同一训练验证协议：`phase1_train -> phase1_val -> test_pool`
- 同一二级决策原则：`GNN-primary + graphprop residual correction`
- 数据集彼此隔离，不做跨数据集联合训练

## Result Snapshot

官方三数据集套件：

- [thesis_m7_v4_graphpropblend082](experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json)

当前 `val_auc`：

| Dataset | Official GNN Backbone | Secondary-only (non-GNN graphprop) | Official GNN-primary Blend |
| --- | ---: | ---: | ---: |
| XinYe DGraph | 0.777741 | 0.794888 | 0.795293 |
| Elliptic | 0.801914 | 0.968319 | 0.949436 |
| Elliptic++ | 0.778276 | 0.963736 | 0.946584 |

结果解释必须说明白：

- 如果只追求当前验证集 AUC，`Secondary-only` 的数值更强，三数据集宏平均为 `0.908981`。
- 但 `Secondary-only` 不是第二个 GNN，而是非 GNN 的 graphprop tree 分支，所以它不能直接替代论文主模型。
- 正式论文主线仍然使用 `Official GNN-primary Blend`，因为你的硬约束是“主模型必须是动态图 GNN”，graphprop 只能作为 leakage-safe 残差校正。

## Comparison Snapshot

| Model | XinYe | Elliptic | Elliptic++ | Macro Val AUC | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical strong GNN `m5_temporal_graphsage` | 0.794628 | 0.793990 | 0.782830 | 0.790483 | 历史强基线 |
| Official pure `m7_utpm` | 0.777741 | 0.801914 | 0.778276 | 0.785977 | 去掉残差分支后的纯主干 |
| Weak hybrid `alpha=0.35` | 0.784396 | 0.841749 | 0.838915 | 0.821687 | 说明融合策略太弱时收益不稳定 |
| Official GNN-primary blend `alpha=0.82` | 0.795293 | 0.949436 | 0.946584 | 0.897104 | 当前 official thesis result |

## Innovation Modules

真正需要逐项做消融的创新模块一共 5 个：

1. 原型记忆与时间桶约束 `prototype memory`
2. 伪对比时间挖掘 `pseudo-contrastive temporal mining`
3. 漂移残差上下文适配 `drift residual target context`
4. `graphprop residual head`
5. 固定 logit 融合 `fixed logit fusion`

说明：

- `utpm_unified` 是统一输入契约，不计入 ablation 模块数。
- 前 3 项属于 GNN 主干内部创新。
- 后 2 项属于 GNN 主导的决策层创新。

## Hard-Leakage Audit

硬泄露审计已经落盘：

- [leakage_audit.md](experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md)
- [leakage_audit.json](experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.json)

审计结论：

- 未发现 `train/val/test_pool/external` 交叉
- 未发现跨数据集混训
- 二级模型训练严格限制在当前数据集的 `phase1_train`
- `val` 只用于推理与评估，不回流训练

## Main Entry

- 主训练入口：[run_thesis_mainline.py](experiment/training/run_thesis_mainline.py)
- 纯 GNN 套件：[run_thesis_suite.py](experiment/training/run_thesis_suite.py)
- 官方 hybrid 套件：[run_thesis_hybrid_suite.py](experiment/training/run_thesis_hybrid_suite.py)
- 泄露审计脚本：[audit_thesis_leakage.py](experiment/training/audit_thesis_leakage.py)
