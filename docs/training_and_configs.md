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
| [../experiment/configs/training_policy.json](../experiment/configs/training_policy.json) | maintained epoch and early-stopping policy |

当前 repo 维护的 rerun config 统一把 `epochs` 设为 `70`，并通过 `GraphModelConfig.min_early_stop_epoch=30` 保证图模型不会在第 30 轮之前早停。accepted 论文主结果 artifact 已经单独保存，直接通过结果表引用。

历史已保存 artifact 的 epoch 曲线保持真实落盘状态，不手动补 synthetic epoch。新的正式 rerun 和 study rerun 使用 70/30 策略重新生成曲线。

新的 mainline rerun 使用 `{suite_name}_{dataset_short}_70e_min30` 作为 run name 模板，避免覆盖旧 accepted artifact。

## 5. Direct Train Parameters

`experiment/mainline.py train` 使用 `TrainParameters` / `Parameter` 容器接管训练参数。这个入口不再把论文超参数写成代码默认值；除了 `epochs` 和输出目录这类运行控制项有安全 fallback，模型、preset、feature profile、feature dir、batch size、hidden dim、relation dim、fanouts、seeds 等核心训练参数必须来自 JSON 或 CLI。`device` 也可以由 JSON 或 CLI 指定，不指定时交给训练器自动选择。

单数据集 JSON 参数文件存放在：

| File | Dataset |
| --- | --- |
| [../experiment/configs/parameters/xinye_dgraph_train.json](../experiment/configs/parameters/xinye_dgraph_train.json) | XinYe DGraph |
| [../experiment/configs/parameters/elliptic_transactions_train.json](../experiment/configs/parameters/elliptic_transactions_train.json) | Elliptic Transactions |
| [../experiment/configs/parameters/ellipticpp_transactions_train.json](../experiment/configs/parameters/ellipticpp_transactions_train.json) | Elliptic++ Transactions |

示例：

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  train \
  --parameter-file experiment/configs/parameters/xinye_dgraph_train.json
```

CLI 参数会覆盖 JSON：

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  train \
  --parameter-file experiment/configs/parameters/xinye_dgraph_train.json \
  --epochs 90 \
  --graph-config-override min_early_stop_epoch=40
```

只检查参数解析、不训练：

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  train \
  --parameter-file experiment/configs/parameters/xinye_dgraph_train.json \
  --dry-run
```

完全不用 JSON 时，也可以显式传入所有核心参数：

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  train \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --run-name manual_cli_run \
  --feature-profile utpm_shift_enhanced \
  --feature-dir experiment/outputs/training/features_ap32 \
  --seeds 42 \
  --batch-size 512 \
  --hidden-dim 128 \
  --rel-dim 32 \
  --fanouts 15 10
```

## 6. Run Mainline Rerun

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

## 7. What Can Vary By Dataset

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
- `min_early_stop_epoch`

这属于同一模型族下的 dataset-local tuning，不是换架构。

## 8. Final Profiles In The Maintained Configs

| Dataset | Key Choices |
| --- | --- |
| XinYe DGraph | `attr_proj_dim=32`, `hidden_dim=128`, `rel_dim=32`, `fanouts=[15,10]`, `epochs=70`, `min_early_stop_epoch=30` |
| Elliptic Transactions | `attr_proj_dim=64`, `hidden_dim=160`, `rel_dim=48`, `attention_num_heads=4`, `epochs=70`, `min_early_stop_epoch=30` |
| Elliptic++ Transactions | `attr_proj_dim=96`, `hidden_dim=192`, `rel_dim=64`, `attention_num_heads=16`, `cold_start_residual_strength=0.35`, `epochs=70`, `min_early_stop_epoch=30` |

## 9. Run Comparison, Ablation, And Progressive Studies

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

## 10. Run Supplementary XinYe Joint Train

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

## 11. Run Leakage Audit

```bash
conda run -n Graph --no-capture-output python3 experiment/audit.py \
  --suite-summary experiment/outputs/reports/dyrift_gnn_accepted_mainline/summary.json
```

预期结论：

- `hard_leakage_detected=false`
- no `train/val/test_pool/external` overlap
- no cross-dataset training
- single pure-GNN deployment path only

## 12. Output Files

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
| [results/experiment_epoch_policy.csv](results/experiment_epoch_policy.csv) | planned 70-epoch / min-30 early-stop policy for maintained reruns |
| [results/training_policy_summary.json](results/training_policy_summary.json) | machine-readable policy summary for thesis reproduction |

## 13. Result Table Sync

Tracked result tables are generated from saved local `summary.json` artifacts:

```bash
python3 experiment/sync_results.py
python3 experiment/sync_results.py --check
```

The sync command updates the decimal CSVs, the percentage presentation table, the study snapshot, the epoch/log manifest, and the machine-readable training-policy summary. It does not synthesize missing epochs or rewrite saved training curves; `docs/results/epoch_log_manifest.csv` remains the observed artifact record, while `docs/results/experiment_epoch_policy.csv` remains the maintained 70/30 rerun policy.
