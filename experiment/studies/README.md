# Experiment Studies Workspace

This workspace isolates comparison, ablation, progressive, and supplementary experiments from the production `DyRIFT-GNN / TRGT` training path.

## Rules

- Do not modify the production `experiment/mainline.py` route just to run a study.
- Every study owns its own `config.json` and `run.py`.
- All outputs go to `experiment/outputs/studies/`.
- Each study writes:
  - dataset-level `summary.json`
  - per-seed `epoch_metrics.csv`
  - per-seed `fit_summary.json`
  - study-level `auc_summary.csv`
  - study-level `seed_overview.csv`
  - study-level `epoch_metrics_all.csv`

## Layout

- `common/`: shared launcher, graph runner, XGBoost same-input runner, and XinYe joint-train supplementary runner
- `comparisons/`: comparison experiments against the final DyRIFT route
- `ablations/`: subtractive ablations on the final DyRIFT route
- `progressive/`: step-by-step method-building studies
- `supplementary/`: non-mainline supplementary studies

## Current Kept Studies

- Comparisons:
  - `plain_trgt_backbone`
  - `tgat_style_reference`
  - `temporal_graphsage_reference`
  - `xgboost_same_input`
- Ablations:
  - `without_target_context_bridge`
  - `without_drift_expert`
  - `without_prototype_memory`
  - `without_pseudo_contrastive`
- Progressive:
  - `trgt_bridge`
  - `trgt_bridge_drift`
  - `trgt_bridge_drift_prototype`
  - `trgt_bridge_drift_prototype_pseudocontrastive`
- Supplementary:
  - `xinye_phase12_joint_train_phase1_val`

## Run

Use the project conda environment:

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/comparisons/tgat_style_reference/run.py
```

Single dataset:

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/ablations/without_drift_expert/run.py \
  --dataset elliptic_transactions
```

Supplementary XinYe joint train:

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/supplementary/xinye_phase12_joint_train_phase1_val/run.py \
  --device cuda
```
