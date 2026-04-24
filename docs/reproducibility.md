# Reproducibility

本页是唯一复现说明，覆盖环境、数据、主线 rerun、study rerun、结果同步和泄露审计。所有命令默认在仓库根目录执行。

## Environment

```bash
conda create -n Graph python=3.10 -y
conda run -n Graph --no-capture-output pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n Graph --no-capture-output pip install -r requirements.txt
```

GPU 检查：

```bash
nvidia-smi
conda run -n Graph --no-capture-output python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Data

| Dataset | Expected location |
| --- | --- |
| XinYe DGraph | `datasets/raw/xinye_dgraph/phase1_gdata.npz` and `phase2_gdata.npz` |
| Elliptic Transactions | `datasets/raw/elliptic_transactions/prepared/` |
| Elliptic++ Transactions | `datasets/raw/ellipticpp_transactions/prepared/` |

切换单数据集调试时使用：

```bash
export GRADPROJ_ACTIVE_DATASET=xinye_dgraph
export GRADPROJ_ACTIVE_DATASET=elliptic_transactions
export GRADPROJ_ACTIVE_DATASET=ellipticpp_transactions
```

## Feature Build

```bash
conda run -n Graph --no-capture-output python3 mainline.py build_features --phase both
```

三个数据集使用统一 `UTPM` 特征语义，但原始解析和缓存路径保持数据集隔离。

## Mainline Rerun

```bash
conda run -n Graph --no-capture-output python3 suite.py \
  --suite-name dyrift_mainline_rerun \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams configs/dyrift_suite.json \
  --seeds 42
```

维护策略：

| Item | Value |
| --- | ---: |
| Max epochs | 70 |
| Minimum early-stop epoch | 30 |
| Default graph patience | 10 |
| XGBoost boosting rounds | 70 |
| XGBoost early stopping rounds | 30 |

单数据集显式参数训练：

```bash
conda run -n Graph --no-capture-output python3 mainline.py \
  train \
  --parameter-file configs/parameters/xinye_dgraph_train.json
```

## Study Reruns

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

XinYe phase1+phase2 补充诊断：

```bash
conda run -n Graph --no-capture-output python3 studies/supplementary/xinye_phase12_joint_train_phase1_val/run.py --device cuda
```

这些补充诊断使用额外 phase2 标注训练节点，不作为正式无泄露主结果。

## Results

正式表格从真实 `summary.json` 同步生成：

```bash
python3 sync_results.py
python3 sync_results.py --check
```

常用结果文件：

| File | Purpose |
| --- | --- |
| `docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv` | 主结果 |
| `docs/results/comparison_auc.csv` | 对比实验 |
| `docs/results/ablation_auc.csv` | 消融实验 |
| `docs/results/progressive_auc.csv` | 递进实验 |
| `docs/results/supplementary_auc.csv` | XinYe 补充诊断 |
| `docs/results/presentation_auc_percent.csv` | 论文表格百分数和百分点差值 |
| `docs/results/experiment_epoch_policy.csv` | 计划 epoch 策略 |
| `docs/results/epoch_log_manifest.csv` | 已保存 artifact 的日志、曲线和 summary 路径 |

## Leakage Audit

```bash
conda run -n Graph --no-capture-output python3 audit.py \
  --suite-summary outputs/reports/accepted_mainline/summary.json
```

accepted 审计文件：

| File | Purpose |
| --- | --- |
| `outputs/reports/accepted_mainline/leakage_audit.md` | 可读审计报告 |
| `outputs/reports/accepted_mainline/leakage_audit.json` | 机器可读审计结果 |
| `docs/results/leakage_audit.json` | 同步到论文结果区的审计快照 |

验收约束：

```text
deployment_path=single_gnn_end_to_end
dataset_isolation=true
cross_dataset_training=false
same_architecture_across_datasets=true
hard_leakage_detected=false
```
