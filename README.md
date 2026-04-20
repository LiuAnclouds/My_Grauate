# Graduation Project

基于动态图神经网络异常检测的金融反欺诈毕业设计项目。

当前仓库已经收敛到一条统一主线：

- 三个数据集分别做数据准备，但全部映射到同一输入契约 `utpm_unified`
- 主模型统一为 `m7_utpm`
- 统一训练/验证协议为 `phase1_train -> phase1_val -> test_pool`
- 最终决策统一为 `GNN + graphprop residual` 的固定 logit 融合

## Official Result

官方三数据集套件：

- `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json`

当前 `val_auc`：

| Dataset | Pure M7 GNN | Graphprop Secondary | Final Hybrid |
| --- | ---: | ---: | ---: |
| XinYe DGraph | 0.777741 | 0.794888 | 0.795293 |
| Elliptic | 0.801914 | 0.968319 | 0.949436 |
| Elliptic++ | 0.778276 | 0.963736 | 0.946584 |

## Hard-Leakage Audit

硬泄露审计已经落盘：

- `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md`
- `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.json`

审计结论：

- 未发现 `train/val/test_pool/external` 交叉
- 未发现跨数据集混训
- 二级模型训练行严格限制在当前数据集的 `phase1_train`
- `val` 仅用于推理和评估，不回流到训练

## Main Entry

- 主训练入口：`experiment/training/run_thesis_mainline.py`
- 主线套件入口：`experiment/training/run_thesis_suite.py`
- 官方 hybrid 套件：`experiment/training/run_thesis_hybrid_suite.py`
- 泄露审计脚本：`experiment/training/audit_thesis_leakage.py`

## Docs

- 方法说明：`docs/thesis_method.md`
- 实验表与消融：`docs/thesis_experiments.md`
- 训练主线说明：`experiment/training/README_thesis_mainline.md`
