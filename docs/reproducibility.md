# Reproducibility

所有命令默认在仓库根目录执行。项目不提供隐藏默认训练参数，参数保存在 `configs/train/*.json` 和 `experiments/common/dataset_parameters.json`。

## 环境

```bash
conda activate Graph
pip install -r requirements.txt
```

## 数据

本地数据放在 `data/raw/`，该目录不提交到 Git。Elliptic 数据可重新准备：

```bash
python3 src/dyrift/data_processing/scripts/prepare_elliptic.py
python3 src/dyrift/data_processing/scripts/prepare_ellipticpp.py
```

## 分析与特征

```bash
GRADPROJ_ACTIVE_DATASET=xinye_dgraph python3 src/dyrift/analysis/run_analysis.py
GRADPROJ_ACTIVE_DATASET=xinye_dgraph python3 train.py build_features --outdir outputs/features/xinye_dgraph/features_ap32
```

把 `GRADPROJ_ACTIVE_DATASET` 和 `--outdir` 换成对应数据集即可。

## 主模型训练

```bash
GRADPROJ_ACTIVE_DATASET=xinye_dgraph \
python3 train.py train --parameter-file configs/train/xinye_dgraph.json
```

结果目录：

```text
outputs/train/full_dyrift_gnn/dyrift_gnn/xinye_dgraph/
```

核心曲线表：

```text
outputs/train/full_dyrift_gnn/dyrift_gnn/xinye_dgraph/epoch_metrics.csv
```

## 对比和消融

每次指定一个数据集：

```bash
python3 experiments/comparisons/plain_trgt_backbone/run.py --dataset xinye_dgraph
python3 experiments/comparisons/xgboost_same_input/run.py --dataset xinye_dgraph
python3 experiments/ablations/without_target_context_bridge/run.py --dataset xinye_dgraph
```

输出路径同样是：

```text
outputs/train/<实验名称>/<模型名称>/<数据集名称>/epoch_metrics.csv
```
