# DyRIFT-TGAT

本项目用于动态图反欺诈异常检测实验，核心方法为单一纯 GNN 模型 `DyRIFT-TGAT`。模型以 TGAT 时间图注意力主干为基础，加入目标上下文桥接、漂移专家和原型记忆等动态图风险漂移建模模块。最终展示指标统一使用 AUC。
正式训练以 30 epoch 作为上限，验证集 AUC 连续 5 个 epoch 没有提升即提前停止。

## AUC 结果

| 数据集 | DyRIFT-TGAT AUC |
| --- | ---: |
| XinYe DGraph | 0.7928507857 |
| Elliptic Transactions | 0.8213290877 |
| Elliptic++ Transactions | 0.8370458443 |

三组结果来自同一模型框架；不同数据集的输入维度不同，属于数据集特征空间差异。节点 `ID` 仅用于索引、对齐和切分，不作为数值特征输入模型。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `train.py` | 单数据集特征构建与 DyRIFT-TGAT 训练入口 |
| `src/dyrift/` | 数据处理、特征、模型、工具和结果汇总代码 |
| `experiments/` | 对比实验和消融实验入口 |
| `configs/templates/` | 可公开的参数模板 |
| `configs/private/` | 本地参数文件目录，默认不提交 |
| `data/raw/` | 本地原始或预处理数据，默认不提交 |
| `outputs/features/` | 特征缓存 |
| `outputs/train/` | 训练输出 |
| `docs/features/README.md` | 特征构建说明 |
| `docs/modules/README.md` | 模块说明 |
| `docs/experiments/README.md` | 实验指标与复现实验计划 |

## 环境

```bash
pip install -r requirements.txt
```

建议使用独立 Conda 或 venv 环境。CUDA、PyTorch 和本机显卡驱动版本按实际机器匹配即可。

## 运行流程

准备数据后，先运行分析和特征构建：

```powershell
$env:GRADPROJ_ACTIVE_DATASET="xinye_dgraph"
python src/dyrift/analysis/run_analysis.py
python train.py build_features --outdir outputs/features/<dataset>/<feature_cache>
```

训练时使用本地参数文件：

```powershell
$env:GRADPROJ_ACTIVE_DATASET="xinye_dgraph"
python train.py train --parameter-file configs/private/xinye_dgraph.json
```

三个数据集分别切换 `GRADPROJ_ACTIVE_DATASET`、特征目录和本地参数文件即可。公开仓库只保留流程和模板，具体训练参数保留在本地。

## 输出规范

训练输出统一写入：

```text
outputs/train/<experiment>/<model>/<dataset>/
```

默认保留：

| 文件 | 内容 |
| --- | --- |
| `epoch_metrics.csv` | 每个 epoch 的 `train_loss`、`val_loss`、`train_auc`、`val_auc` |
| `best_model.pt` | 最佳 validation AUC epoch 的模型权重 |
| `last_model.pt` | 最后一轮模型权重 |
| `model.pt` | 兼容旧加载逻辑的最佳权重别名 |

汇总 CSV 可通过下面命令重建：

```powershell
python experiments/reporting/recover_results.py
```

生成文件位于 `docs/generated/`，只用于结果表格同步，不生成额外 Markdown 报告。
