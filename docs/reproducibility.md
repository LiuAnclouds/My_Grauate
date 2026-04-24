# Reproducibility Guide

This guide is the engineering entry point for reproducing the maintained `DyRIFT-GNN / TRGT` thesis mainline from a clean workstation. It covers repository checkout, conda setup, dependency installation, dataset placement, unified feature construction, and the accepted mainline rerun.

For the full experiment matrix, see [Experiment Reproduction](experiment_reproduction.md). That separate guide contains comparison, ablation, progressive, supplementary, and leakage-audit commands.

## 1. Clone The Repository

Use SSH if your GitHub key is configured:

```bash
git clone git@github.com:LiuAnclouds/My_Grauate.git
cd My_Grauate
```

HTTPS is also valid:

```bash
git clone https://github.com/LiuAnclouds/My_Grauate.git
cd My_Grauate
```

All commands below assume the current directory is the repository root.

## 2. Hardware And Runtime

Recommended runtime:

| Item | Recommendation |
| --- | --- |
| GPU | NVIDIA GPU with at least 12 GB memory; RTX 4070 Super class is sufficient for the saved profiles |
| CPU/RAM | 8+ CPU cores and 32 GB RAM recommended for feature construction |
| OS | Linux workstation or server |
| Python | 3.10 |
| CUDA | CUDA-enabled PyTorch build matching the installed NVIDIA driver |

The local thesis runs used a conda environment named `Graph`. The project does not compile custom CUDA kernels, so a compatible CUDA PyTorch wheel is enough.

## 3. Create Conda Environment

Create a clean environment:

```bash
conda create -n Graph python=3.10 -y
```

Install PyTorch. For CUDA 12.x drivers, this wheel set is the maintained default:

```bash
conda run -n Graph --no-capture-output pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

If your workstation uses a different CUDA driver or CPU-only runtime, install PyTorch from the official selector and keep the same environment name.

Install project dependencies:

```bash
conda run -n Graph --no-capture-output pip install -r requirements.txt
```

Verify GPU visibility:

```bash
nvidia-smi
conda run -n Graph --no-capture-output python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected result on a GPU workstation: `torch.cuda.is_available()` prints `True` and the GPU name is shown.

## 4. Dataset Layout

Place raw or prepared data under the dataset contract directories:

| Dataset | Expected location |
| --- | --- |
| XinYe DGraph | `experiment/datasets/raw/xinye_dgraph/phase1_gdata.npz` and `phase2_gdata.npz` |
| Elliptic Transactions | `experiment/datasets/raw/elliptic_transactions/prepared/` |
| Elliptic++ Transactions | `experiment/datasets/raw/ellipticpp_transactions/prepared/` |

The project uses dataset-local raw parsing, then maps all datasets into the same `UTPM` feature contract. This preserves a unified model input while avoiding cross-dataset data mixing.

## 5. Build Unified Feature Caches

Run feature construction after datasets are in place:

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  build_features \
  --phase both
```

This builds graph caches and unified feature caches. It does not merge the three datasets into one training set.

For one-dataset debugging, set the dataset environment variable before running a command:

```bash
export DGRAPH_DATASET=xinye_dgraph
export DGRAPH_DATASET=elliptic_transactions
export DGRAPH_DATASET=ellipticpp_transactions
```

## 6. Run The Accepted Mainline Rerun

The maintained thesis route is:

```text
dataset-local preprocessing -> UTPM features -> TRGT backbone -> DyRIFT-GNN modules -> fraud probability
```

Run the three-dataset suite:

```bash
conda run -n Graph --no-capture-output python3 experiment/suite.py \
  --suite-name dyrift_mainline_rerun \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/configs/dyrift_suite.json \
  --seeds 42
```

Maintained reruns use:

| Policy Item | Value |
| --- | ---: |
| Max epochs | 70 |
| Minimum early-stop epoch | 30 |
| Default graph patience | 10 |

The policy is stored in `experiment/configs/training_policy.json` and summarized in `docs/results/experiment_epoch_policy.csv`. Historical saved artifacts are not rewritten; rerunning the commands above regenerates epoch curves under this policy.

Dataset-level feature and training parameters are explicit in `experiment/configs/dyrift_suite.json`, `experiment/configs/xinye_dgraph.json`, `experiment/configs/elliptic_transactions.json`, and `experiment/configs/ellipticpp_transactions.json`. The runner does not require hidden thesis-only defaults.

Rerun outputs are written to suite-scoped names such as `dyrift_mainline_rerun_xy_70e_min30`, so previous saved results are not overwritten.

For single-dataset training, use the explicit `Parameter` JSON interface. The train entrypoint does not provide hidden model/data defaults; core fields must come from this JSON file or from CLI flags, while operational controls such as `epochs` can fall back to the maintained policy.

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  train \
  --parameter-file experiment/configs/parameters/xinye_dgraph_train.json
```

The official thesis result table is:

| Dataset | Val AUC |
| --- | ---: |
| XinYe DGraph | 79.2851% |
| Elliptic Transactions | 82.1329% |
| Elliptic++ Transactions | 82.1953% |
| Macro Average | 81.2044% |

## 7. Result Files

Use these files for tables and figures:

| File | Purpose |
| --- | --- |
| `docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv` | accepted mainline AUC table |
| `docs/results/comparison_auc.csv` | baseline comparison table |
| `docs/results/ablation_auc.csv` | subtractive ablation table |
| `docs/results/progressive_auc.csv` | progressive method-building table |
| `docs/results/supplementary_auc.csv` | supplementary XinYe phase diagnostics |
| `docs/results/presentation_auc_percent.csv` | percentage-format AUC and percentage-point deltas for thesis tables |
| `docs/results/experiment_epoch_policy.csv` | maintained 70-epoch / min-30 early-stop policy |
| `docs/results/training_policy_summary.json` | machine-readable policy summary |
| `docs/results/epoch_log_manifest.csv` | per-run epoch log, train log, curve, and summary paths |

Each training artifact normally contains `summary.json`, `seed_42/epoch_metrics.csv`, `seed_42/train.log`, `seed_42/training_curves.png`, and saved prediction files.

## 8. Integrity Constraints

The accepted thesis route must satisfy:

```text
deployment_path=single_gnn_end_to_end
dataset_isolation=true
cross_dataset_training=false
same_architecture_across_datasets=true
hard_leakage_detected=false
```

XinYe `phase1+phase2` joint-training studies are diagnostic supplements only. They use additional phase2 labels and are not the official leakage-free mainline.
