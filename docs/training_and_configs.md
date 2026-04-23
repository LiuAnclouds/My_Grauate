# Training And Configs

This document describes how to reproduce the final `DyRIFT-GNN / TRGT` suite and how the experiment pipeline is organized.

## 1. Environment

Use the project environment from repository root:

```bash
conda run -n Graph --no-capture-output python3 ...
```

## 2. Runner Surface

| File | Purpose |
| --- | --- |
| [../experiment/mainline.py](../experiment/mainline.py) | build unified features and train one dataset |
| [../experiment/suite.py](../experiment/suite.py) | run the tri-dataset thesis suite |
| [../experiment/audit.py](../experiment/audit.py) | verify hard-leakage constraints |

## 3. Build Features

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  build_features \
  --phase both
```

This builds dataset-scoped feature and graph caches. Raw datasets are never mixed.

## 4. Run Final Suite

```bash
conda run -n Graph --no-capture-output python3 experiment/suite.py \
  --suite-name thesis_dyrift_gnn_trgt_deploy_pure_v1 \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/dyrift_suite.json \
  --seeds 42 \
  --skip-existing
```

`dyrift_gnn` is the official runtime id and resolves to `DyRIFTTrainer`.

## 5. Run Leakage Audit

```bash
conda run -n Graph --no-capture-output python3 experiment/audit.py \
  --suite-summary experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json
```

Expected audit conclusions:

- `hard_leakage_detected=false`
- no `train/val/test_pool/external` overlap
- no cross-dataset training
- single pure-GNN deployment path only

## 6. Config Layout

The final suite uses one shared manifest plus three dataset files:

| File | Role |
| --- | --- |
| [../experiment/dyrift_suite.json](../experiment/dyrift_suite.json) | shared defaults and dataset file references |
| [../experiment/xinye_dgraph.json](../experiment/xinye_dgraph.json) | XinYe tuning |
| [../experiment/elliptic_transactions.json](../experiment/elliptic_transactions.json) | ET tuning |
| [../experiment/ellipticpp_transactions.json](../experiment/ellipticpp_transactions.json) | EPP tuning |

The manifest uses `mainline.dataset_files` to load the per-dataset JSON files.

## 7. What Can Vary By Dataset

The architecture stays fixed, but dataset-local hyperparameters can differ:

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

This is per-dataset tuning under one model family, not a change of architecture.

## 8. Final Profiles

| Dataset | Key Choices |
| --- | --- |
| XinYe DGraph | `attr_proj_dim=32`, `hidden_dim=128`, `rel_dim=32`, `fanouts=[15,10]` |
| Elliptic Transactions | `attr_proj_dim=64`, `hidden_dim=160`, `rel_dim=48`, `attention_num_heads=4` |
| Elliptic++ Transactions | `attr_proj_dim=96`, `hidden_dim=192`, `rel_dim=64`, `attention_num_heads=16`, `cold_start_residual_strength=0.35` |

## 9. Output Files

| File | Role |
| --- | --- |
| [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json) | suite summary |
| [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md) | audit report |
| [../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json) | audit JSON |
| [results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv) | dataset metrics |
| [results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv) | epoch logs |
