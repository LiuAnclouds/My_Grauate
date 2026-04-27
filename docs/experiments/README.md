# 实验指标与计划
正式训练以 30 epoch 作为上限；验证集 AUC 达到当前最佳值后，连续 5 个 epoch 没有提升即提前停止。

最终展示指标统一为 AUC。每次训练都会打印 epoch 结果，并将 `train_loss`、`val_loss`、`train_auc`、`val_auc` 保存到 CSV。早停规则为：验证集 AUC 达到当前最佳值后，连续 5 个 epoch 未出现有效提升即停止，不再设置固定最小训练轮数。

## 当前 Full Model

| 数据集 | DyRIFT-TGAT AUC |
| --- | ---: |
| XinYe DGraph | 0.7928507857 |
| Elliptic Transactions | 0.8213290877 |
| Elliptic++ Transactions | 0.8370458443 |

## 正式对比实验

| 方法 | 作用 |
| --- | --- |
| `Linear Same-Feature Baseline` | 使用相同节点特征但不使用图边和时间邻域，验证图结构建模价值 |
| `Temporal GraphSAGE Reference` | 使用时间切分和邻居采样的基础时序 GNN，验证注意力与漂移模块贡献 |
| `TGAT Backbone Reference` | 使用 TGAT 时间图注意力主干但不加入 DyRIFT 模块，作为直接 backbone 对照 |
| `DyRIFT-TGAT` | 最终模型 |

## 正式消融实验

| 方法 | 作用 |
| --- | --- |
| `DyRIFT-TGAT w/o Target-Context Bridge` | 去掉目标上下文桥接 |
| `DyRIFT-TGAT w/o Drift Expert` | 去掉漂移专家 |
| `DyRIFT-TGAT w/o Prototype Memory` | 去掉原型记忆 |

旧版 plain backbone、XGBoost、逐步加模块实验和单独 pseudo-contrastive 消融不再作为默认正式表格展示。

## 快机器复现实验顺序

1. 先运行三个数据集的 `DyRIFT-TGAT` full model，确认 AUC 不低于当前记录。
2. 再运行三个对比实验：Linear Same-Feature、Temporal GraphSAGE、TGAT Backbone。
3. 最后运行三个消融实验：w/o Target-Context Bridge、w/o Drift Expert、w/o Prototype Memory。
4. 每个实验完成后运行 `python experiments/reporting/recover_results.py` 刷新 `docs/generated/` 下的 CSV 表格。

整套正式实验可直接运行：

```powershell
python experiments/run_official.py --stage all --device cuda
```

如果只先跑对比或消融：

```powershell
python experiments/run_official.py --stage comparisons --device cuda
python experiments/run_official.py --stage ablations --device cuda
```

## 输出文件

单次训练输出：

```text
outputs/train/<experiment>/<model>/<dataset>/epoch_metrics.csv
outputs/train/<experiment>/<model>/<dataset>/best_model.pt
outputs/train/<experiment>/<model>/<dataset>/last_model.pt
outputs/train/<experiment>/<model>/<dataset>/model.pt
```

汇总结果：

```text
docs/generated/results_summary.csv
docs/generated/comparison_table.csv
docs/generated/ablation_table.csv
```

## 重建汇总

```powershell
python experiments/reporting/recover_results.py
```

该命令只从已有训练输出恢复 CSV 表格，不生成额外 Markdown 文档。
