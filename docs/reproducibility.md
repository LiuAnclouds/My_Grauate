# Reproducibility Guide

This guide is the engineering entry point for reproducing the DyRIFT-GNN thesis experiments. It separates the accepted leakage-free mainline from diagnostic studies so that results are easy to audit.

## 1. Hardware And Runtime

Recommended hardware:

| Item | Recommendation |
| --- | --- |
| GPU | NVIDIA GPU with at least 12 GB memory; RTX 4070 Super class is enough for the saved profiles |
| CPU/RAM | 8+ CPU cores and 32 GB RAM recommended for feature construction |
| OS | Linux workstation or server |
| Python | 3.10 |
| CUDA | Use a CUDA-enabled PyTorch build that matches the installed NVIDIA driver |

The project was run in a conda environment named `Graph`. The exact local workstation used `torch 2.10.0+cu128` with an NVIDIA 575-series driver. A compatible PyTorch CUDA build is sufficient; the project does not require compiling custom CUDA kernels.

## 2. Environment Setup

Create the environment from a clean shell:

```bash
conda create -n Graph python=3.10 -y
conda activate Graph
```

Install PyTorch according to the local CUDA driver. One common CUDA 12.x setup is:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the Python packages used by the feature, training, and evaluation scripts:

```bash
pip install -r requirements.txt
```

Verify GPU visibility before running experiments:

```bash
nvidia-smi
conda run -n Graph --no-capture-output python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected result: `torch.cuda.is_available()` should print `True`.

## 3. Dataset Layout

Raw/prepared dataset contracts live under:

| Dataset | Expected location |
| --- | --- |
| XinYe DGraph | `experiment/datasets/raw/xinye_dgraph/phase1_gdata.npz` and `phase2_gdata.npz` |
| Elliptic Transactions | `experiment/datasets/raw/elliptic_transactions/prepared/` |
| Elliptic++ Transactions | `experiment/datasets/raw/ellipticpp_transactions/prepared/` |

The repository uses dataset-local raw preprocessing, then maps all datasets into the same UTPM feature family. Dataset-local preprocessing is allowed; cross-dataset training is not used.

## 4. Build Unified Feature Caches

Run from repository root:

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  build_features \
  --phase both
```

This builds graph caches and unified feature caches for the current dataset selection. It does not mix datasets together.

For dataset-specific runs, set the dataset environment variable used by the registry before building or training:

```bash
export DGRAPH_DATASET=xinye_dgraph
export DGRAPH_DATASET=elliptic_transactions
export DGRAPH_DATASET=ellipticpp_transactions
```

## 5. Reproduce The Accepted Mainline

The accepted thesis route is single-model pure GNN deployment:

```text
dataset-local preprocessing -> UTPM features -> TRGT backbone -> DyRIFT-GNN modules -> fraud probability
```

Run the maintained three-dataset suite:

```bash
conda run -n Graph --no-capture-output python3 experiment/suite.py \
  --suite-name dyrift_mainline_rerun \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/configs/dyrift_suite.json \
  --seeds 42
```

The accepted saved artifacts used for the thesis tables are:

| Dataset | Val AUC | Saved artifact |
| --- | ---: | --- |
| XinYe DGraph | 0.792851 | `experiment/outputs/training/models/dyrift_gnn/full_xinye_repro_v1` |
| Elliptic Transactions | 0.821329 | `experiment/outputs/elliptic_transactions/training/models/dyrift_gnn/probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1` |
| Elliptic++ Transactions | 0.821953 | `experiment/outputs/ellipticpp_transactions/training/models/dyrift_gnn/probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1` |

The macro average is `0.812044`.

## 6. Reproduce Studies

Comparison studies:

```bash
conda run -n Graph --no-capture-output python3 experiment/studies/comparisons/plain_trgt_backbone/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/comparisons/tgat_style_reference/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/comparisons/temporal_graphsage_reference/run.py --device cuda
```

Subtractive ablations:

```bash
conda run -n Graph --no-capture-output python3 experiment/studies/ablations/without_target_context_bridge/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/ablations/without_drift_expert/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/ablations/without_prototype_memory/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/ablations/without_pseudo_contrastive/run.py --device cuda
```

Progressive method-building studies:

```bash
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/plain_trgt_backbone/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/trgt_bridge/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/trgt_bridge_drift/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/trgt_bridge_drift_prototype/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/trgt_bridge_drift_prototype_pseudocontrastive/run.py --device cuda
```

Supplementary XinYe phase studies:

```bash
conda run -n Graph --no-capture-output python3 experiment/studies/supplementary/xinye_phase12_joint_train_phase1_val/run.py --device cuda
```

The maintained supplementary runner is diagnostic. Archived phase-aware diagnostic outputs are kept in the result CSV files, but their exploratory runners are not part of the maintained code path. These studies use additional phase2 labels and are not the official leakage-free mainline.

## 7. Result Files

Use these files for tables and figures:

| File | Purpose |
| --- | --- |
| `docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv` | accepted mainline AUC table |
| `docs/results/comparison_auc.csv` | baseline comparison table |
| `docs/results/ablation_auc.csv` | subtractive ablation table |
| `docs/results/progressive_auc.csv` | progressive method-building table |
| `docs/results/supplementary_auc.csv` | supplementary XinYe phase diagnostics |
| `docs/results/epoch_log_manifest.csv` | per-run epoch log, train log, curve, and summary paths |

Each training run stores at least:

| Artifact | Meaning |
| --- | --- |
| `summary.json` | final scalar metrics and run metadata |
| `seed_42/epoch_metrics.csv` | per-epoch metrics for plotting |
| `seed_42/train.log` | training log with module diagnostics |
| `seed_42/training_curves.png` | generated curve figure |
| prediction `.npz` files | node ids, labels, and predicted probabilities for saved splits |

## 8. Leakage Audit

Regenerate the accepted-mainline leakage audit:

```bash
conda run -n Graph --no-capture-output python3 experiment/audit.py \
  --suite-summary experiment/outputs/reports/dyrift_gnn_accepted_mainline/summary.json
```

Expected conclusion:

```text
hard_leakage_detected=false
cross_dataset_training=false
deployment_path=single_gnn_end_to_end
```

The accepted audit report is stored at `docs/leakage_audit.md` and `docs/results/leakage_audit.json`.
