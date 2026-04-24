# DyRIFT-GNN Graduation Project

本仓库是毕设项目代码，面向动态图反欺诈实验。当前版本只保留项目自身实现、单数据集训练入口、对比实验、消融实验和必要复现说明。

## 目录

| 路径 | 内容 |
| --- | --- |
| `train.py` | 单数据集训练入口 |
| `src/dyrift/` | 项目源码，包含分析、数据处理、特征、模型和通用工具 |
| `configs/train/` | 三个数据集各自的训练参数 |
| `data/raw/` | 本地原始/预处理数据，默认不提交 |
| `experiments/` | 对比、消融、渐进实验配置和入口 |
| `outputs/analysis/` | 数据分析输出 |
| `outputs/features/` | 特征缓存 |
| `outputs/train/` | 训练结果 |
| `docs/reproducibility.md` | 复现流程 |

## 训练输出

训练结果统一放在：

```text
outputs/train/<实验名称>/<模型名称>/<数据集名称>/epoch_metrics.csv
```

`epoch_metrics.csv` 的列固定为：

```text
epoch,train_loss,train_auc,val_loss,val_auc
```

## 单数据集训练

每次只跑一个数据集，通过环境变量指定数据集，再使用对应参数文件：

```bash
GRADPROJ_ACTIVE_DATASET=xinye_dgraph \
python3 train.py train --parameter-file configs/train/xinye_dgraph.json

GRADPROJ_ACTIVE_DATASET=elliptic_transactions \
python3 train.py train --parameter-file configs/train/elliptic_transactions.json

GRADPROJ_ACTIVE_DATASET=ellipticpp_transactions \
python3 train.py train --parameter-file configs/train/ellipticpp_transactions.json
```

## 实验入口

实验也一次只跑一个数据集：

```bash
python3 experiments/comparisons/xgboost_same_input/run.py --dataset xinye_dgraph
python3 experiments/ablations/without_prototype_memory/run.py --dataset elliptic_transactions
python3 experiments/progressive/trgt_bridge_drift/run.py --dataset ellipticpp_transactions
```

更多复现细节见 `docs/reproducibility.md`。
