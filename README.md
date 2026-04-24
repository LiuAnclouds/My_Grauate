# DyRIFT-GNN

毕业设计项目：面向动态图金融反欺诈的 `DyRIFT-GNN`。当前最终主线使用 `TRGT`
(`Temporal-Relational Graph Transformer`) 作为图神经网络主干，并在三个数据集上保持同一套纯 GNN 训练与推理路线。

## Main Result

| Dataset | Val AUC |
| --- | ---: |
| XinYe DGraph | 79.2851% |
| Elliptic Transactions | 82.1329% |
| Elliptic++ Transactions | 82.1953% |
| Macro Average | 81.2044% |

核心约束：

- 最终部署路径是 `dataset-local preprocessing -> UTPM features -> TRGT -> DyRIFT-GNN -> fraud probability`。
- 不使用外部树模型、teacher 分支或二阶段融合器作为最终推理路线。
- 三个数据集隔离训练，不做跨数据集联合训练。
- 维护中的 rerun 策略为 `max_epochs=70`、`min_early_stop_epoch=30`。

## Repository Layout

| Path | Role |
| --- | --- |
| `mainline.py` | 单数据集特征构建与训练入口 |
| `suite.py` | 三数据集主线 rerun 入口 |
| `audit.py` | accepted 主结果硬泄露审计 |
| `sync_results.py` | 从真实 `summary.json` 同步 `docs/results/` 表格 |
| `configs/` | 主线配置、数据集参数、训练策略和显式 train 参数 |
| `datasets/` | 数据集注册、原始数据约定、下载与预处理脚本 |
| `features/` | UTPM 特征构建与图缓存 |
| `models/` | TRGT、DyRIFT-GNN、训练引擎和预设 |
| `studies/` | 对比、消融、递进和补充实验入口 |
| `docs/` | 精简论文说明、复现说明和自动结果表 |
| `outputs/` | 本地生成 artifact；除 accepted report 外默认不纳入 Git |

## Quick Start

```bash
conda create -n Graph python=3.10 -y
conda run -n Graph --no-capture-output pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n Graph --no-capture-output pip install -r requirements.txt
```

放置数据：

| Dataset | Expected location |
| --- | --- |
| XinYe DGraph | `datasets/raw/xinye_dgraph/phase1_gdata.npz` and `phase2_gdata.npz` |
| Elliptic Transactions | `datasets/raw/elliptic_transactions/prepared/` |
| Elliptic++ Transactions | `datasets/raw/ellipticpp_transactions/prepared/` |

构建特征：

```bash
conda run -n Graph --no-capture-output python3 mainline.py build_features --phase both
```

运行当前主线：

```bash
conda run -n Graph --no-capture-output python3 suite.py \
  --suite-name dyrift_mainline_rerun \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams configs/dyrift_suite.json \
  --seeds 42
```

单数据集显式参数训练：

```bash
conda run -n Graph --no-capture-output python3 mainline.py \
  train \
  --parameter-file configs/parameters/xinye_dgraph_train.json
```

切换当前数据集：

```bash
export GRADPROJ_ACTIVE_DATASET=xinye_dgraph
export GRADPROJ_ACTIVE_DATASET=elliptic_transactions
export GRADPROJ_ACTIVE_DATASET=ellipticpp_transactions
```

## Experiments

对比实验：

```bash
conda run -n Graph --no-capture-output python3 studies/comparisons/plain_trgt_backbone/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/comparisons/tgat_style_reference/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/comparisons/temporal_graphsage_reference/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/comparisons/xgboost_same_input/run.py
```

消融实验：

```bash
conda run -n Graph --no-capture-output python3 studies/ablations/without_target_context_bridge/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/ablations/without_drift_expert/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/ablations/without_prototype_memory/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/ablations/without_pseudo_contrastive/run.py --device cuda
```

递进实验：

```bash
conda run -n Graph --no-capture-output python3 studies/progressive/trgt_bridge/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/progressive/trgt_bridge_drift/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/progressive/trgt_bridge_drift_prototype/run.py --device cuda
conda run -n Graph --no-capture-output python3 studies/progressive/trgt_bridge_drift_prototype_pseudocontrastive/run.py --device cuda
```

XinYe 补充诊断：

```bash
conda run -n Graph --no-capture-output python3 studies/supplementary/xinye_phase12_joint_train_phase1_val/run.py --device cuda
```

同步结果表：

```bash
python3 sync_results.py
python3 sync_results.py --check
```

泄露审计：

```bash
conda run -n Graph --no-capture-output python3 audit.py \
  --suite-summary outputs/reports/accepted_mainline/summary.json
```

## Kept Documents

- `docs/reproducibility.md`: 环境、数据、主线和 study 复现命令。
- `docs/thesis_method.md`: 方法说明和论文口径。
- `docs/thesis_experiments.md`: 主结果、对比、消融、递进和补充实验表。
- `docs/results/`: 自动同步的 CSV/JSON 结果表。
- `outputs/reports/accepted_mainline/`: 当前 accepted 主结果和泄露审计 artifact。
