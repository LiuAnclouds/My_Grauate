# Training And Configs

For installation and mainline reproduction, see [Reproducibility Guide](reproducibility.md). For the full experiment matrix, see [Experiment Reproduction](experiment_reproduction.md). For the model data flow, see [Model Execution Flow](model_execution_flow.md).

This document describes how the current `DyRIFT-GNN / TRGT` pipeline is organized, how to rerun it, and where the saved artifacts live.

## 1. Environment

Use the project conda environment from repository root:

```bash
conda run -n Graph --no-capture-output python3 ...
```

## 2. Runner Surface

| File | Purpose |
| --- | --- |
| [../experiment/mainline.py](../experiment/mainline.py) | build features and train one dataset |
| [../experiment/suite.py](../experiment/suite.py) | run the maintained three-dataset rerun config |
| [../experiment/audit.py](../experiment/audit.py) | verify hard-leakage constraints |
| [../experiment/studies/README.md](../experiment/studies/README.md) | isolated studies workspace |

## 3. Build Features

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  build_features \
  --phase both
```

这一步只会构建各数据集自己的 feature cache 和 graph cache，不会把原始数据混在一起。

## 4. Current Maintained Mainline Config

当前维护中的配置文件：

| File | Role |
| --- | --- |
| [../experiment/configs/dyrift_suite.json](../experiment/configs/dyrift_suite.json) | shared defaults and dataset file references |
| [../experiment/configs/xinye_dgraph.json](../experiment/configs/xinye_dgraph.json) | XinYe profile |
| [../experiment/configs/elliptic_transactions.json](../experiment/configs/elliptic_transactions.json) | ET profile |
| [../experiment/configs/ellipticpp_transactions.json](../experiment/configs/ellipticpp_transactions.json) | EPP profile |

当前 repo 维护的 rerun config 统一把 `epochs` 设为 `30`，同时允许早停。accepted 论文主结果 artifact 已经单独保存，直接通过结果表引用。

## 5. Run Mainline Rerun

```bash
conda run -n Graph --no-capture-output python3 experiment/suite.py \
  --suite-name dyrift_mainline_rerun \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/configs/dyrift_suite.json \
  --seeds 42
```

`dyrift_gnn` 是正式 runtime id，对应 `DyRIFTTrainer`。

## 6. What Can Vary By Dataset

架构固定，但这些超参数允许按数据集单独调：

- `attr_proj_dim`
- `feature_profile`
- `feature_subdir`
- `hidden_dim`
- `rel_dim`
- `fanouts`
- `dropout`
- `attention_num_heads`
- `recent_window`
- `recent_ratio`
- `time_decay_strength`
- `prototype_loss_weight`
- `pseudo_contrastive_weight`
- `cold_start_residual_strength`

这属于同一模型族下的 dataset-local tuning，不是换架构。

## 7. Final Profiles In The Maintained Configs

| Dataset | Key Choices |
| --- | --- |
| XinYe DGraph | `attr_proj_dim=32`, `hidden_dim=128`, `rel_dim=32`, `fanouts=[15,10]`, `epochs=30` |
| Elliptic Transactions | `attr_proj_dim=64`, `hidden_dim=160`, `rel_dim=48`, `attention_num_heads=4`, `epochs=30` |
| Elliptic++ Transactions | `attr_proj_dim=96`, `hidden_dim=192`, `rel_dim=64`, `attention_num_heads=16`, `cold_start_residual_strength=0.35`, `epochs=30` |

## 8. Run Comparison, Ablation, And Progressive Studies

对比实验：

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/comparisons/tgat_style_reference/run.py
```

减法消融：

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/ablations/without_drift_expert/run.py
```

递进式方法实验：

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/progressive/trgt_bridge_drift_prototype_pseudocontrastive/run.py
```

## 9. Run Supplementary XinYe Joint Train

这些补充实验是 XinYe phase1/phase2 诊断实验，不是正式论文主线。它们使用 phase2 标注节点，因此只用于解释跨阶段分布漂移和 checkpoint 选择 trade-off。

基础 from-scratch `phase1.train + phase2.train` 联合训练：

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/supplementary/xinye_phase12_joint_train_phase1_val/run.py \
  --device cuda
```

基础联合训练固定：

- `train = phase1.train + phase2.train`
- `val = phase1.val`

Phase-aware balanced 和 dual-validation 诊断结果已经作为 archived output 记录在结果表中。对应 exploratory runner 不再作为维护中的实验代码保留。它们从 `phase2.train_mask` 中切出一部分 labeled holdout，不使用官方 test pool 标签，但仍不作为无泄露主线。

输出到：

- `experiment/outputs/studies/supplementary/xinye_phase12_joint_train_phase1_val/`
- `experiment/outputs/studies/supplementary/xinye_phase12_phase_aware_balanced/` archived diagnostic output
- `experiment/outputs/studies/supplementary/xinye_phase12_phase_aware_dualval/` archived diagnostic output

## 10. Run Leakage Audit

```bash
conda run -n Graph --no-capture-output python3 experiment/audit.py \
  --suite-summary experiment/outputs/reports/dyrift_gnn_accepted_mainline/summary.json
```

预期结论：

- `hard_leakage_detected=false`
- no `train/val/test_pool/external` overlap
- no cross-dataset training
- single pure-GNN deployment path only

## 11. Output Files

| File | Role |
| --- | --- |
| [results/accepted_mainline_summary.json](results/accepted_mainline_summary.json) | accepted mainline summary |
| [leakage_audit.md](leakage_audit.md) | accepted mainline audit report |
| [results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv) | accepted mainline AUC table |
| [results/comparison_auc.csv](results/comparison_auc.csv) | comparison-study AUC table |
| [results/ablation_auc.csv](results/ablation_auc.csv) | ablation-study AUC table |
| [results/progressive_auc.csv](results/progressive_auc.csv) | progressive-study AUC table |
| [results/supplementary_auc.csv](results/supplementary_auc.csv) | supplementary-study AUC table |
| [results/epoch_log_manifest.csv](results/epoch_log_manifest.csv) | epoch/log/curve manifest for all kept experiments |
