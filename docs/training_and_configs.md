# Training And Configs

This document describes how to reproduce the final `DyRIFT-GNN / TRGT` suite and how dataset-local hyperparameters are organized.

## 1. Environment

Use the project environment:

```bash
conda run -n Graph --no-capture-output python3 ...
```

The main training scripts assume repository root as the working directory.

## 2. Build Features

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py \
  build_features \
  --phase both
```

This builds dataset-local graph and feature caches. The raw datasets are not mixed.

## 3. Run Final Suite

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py \
  --suite-name thesis_dyrift_gnn_trgt_deploy_pure_v1 \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/training/configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json \
  --seeds 42 \
  --skip-existing
```

The `dyrift_gnn` argument is the official runtime model id. Internally it resolves to `DyRIFTGNNExperiment`.

## 4. Run Leakage Audit

```bash
conda run -n Graph --no-capture-output python3 experiment/training/audit_thesis_leakage.py \
  --suite-summary experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json
```

Expected output:

- `hard_leakage_detected=false`
- no train/val/test_pool overlap
- no cross-dataset training
- single GNN deployment path only

## 5. Config Layout

The final suite uses one manifest plus three dataset files.

Manifest:

- [thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json)

Dataset files:

- [xinye_dgraph.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/configs/dyrift_gnn/xinye_dgraph.json)
- [elliptic_transactions.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/configs/dyrift_gnn/elliptic_transactions.json)
- [ellipticpp_transactions.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/configs/dyrift_gnn/ellipticpp_transactions.json)

The manifest uses `mainline.dataset_files` to load dataset-specific JSON files.

## 6. Shared Defaults

The manifest stores shared defaults such as:

- `run_name_template`
- `feature_profile`
- `target_context_groups`
- `epochs`
- `batch_size`
- `hidden_dim`
- `rel_dim`
- `fanouts`

Each dataset file can override these values without changing the model architecture.

## 7. Dataset-Specific Tuning

The following hyperparameters are allowed to differ by dataset:

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
- module weights such as `prototype_loss_weight` and `pseudo_contrastive_weight`

This is dataset-local hyperparameter tuning, not a change of model family.

## 8. Final Profiles

| Dataset | Key Choices |
| --- | --- |
| XinYe DGraph | `attr_proj_dim=32`, `hidden_dim=128`, `rel_dim=32`, `fanouts=[15,10]` |
| Elliptic Transactions | `attr_proj_dim=64`, `hidden_dim=160`, `rel_dim=48`, `attention_num_heads=4` |
| Elliptic++ Transactions | `attr_proj_dim=96`, `hidden_dim=192`, `rel_dim=64`, `attention_num_heads=16`, `cold_start_residual_strength=0.35` |

## 9. Output Files

Suite summary:

- [summary.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json)

Audit:

- [leakage_audit.md](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md)
- [leakage_audit.json](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json)

Metrics:

- [metrics.csv](/home/moonxkj/Desktop/MyWork/Graduation_Project/docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv)
- [epoch_metrics.csv](/home/moonxkj/Desktop/MyWork/Graduation_Project/docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv)
