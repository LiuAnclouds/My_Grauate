# Experiment Reproduction

This document lists the commands for reproducing the experiment matrix after the environment and unified feature caches are prepared. Start with [Reproducibility Guide](reproducibility.md) if the repository, conda environment, dependencies, or datasets are not ready.

All commands run from the repository root and use the same unified feature contract built by:

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  build_features \
  --phase both
```

`same input` means the models consume features produced by the same build pipeline. A comparison model may apply its own lightweight tensor or matrix adapter at the model entry, but it must not introduce a separate dataset-specific feature engineering route.

## 1. Accepted Mainline

Run the maintained three-dataset `DyRIFT-GNN / TRGT` suite:

```bash
conda run -n Graph --no-capture-output python3 experiment/suite.py \
  --suite-name dyrift_mainline_rerun \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/configs/dyrift_suite.json \
  --seeds 42
```

Accepted result table:

| Setting | XinYe | ET | EPP | Macro Val AUC |
| --- | ---: | ---: | ---: | ---: |
| Full DyRIFT-GNN | 79.2851% | 82.1329% | 82.1953% | 81.2044% |

Accepted artifacts:

| Dataset | Saved artifact |
| --- | --- |
| XinYe DGraph | `experiment/outputs/training/models/dyrift_gnn/full_xinye_repro_v1` |
| Elliptic Transactions | `experiment/outputs/elliptic_transactions/training/models/dyrift_gnn/probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1` |
| Elliptic++ Transactions | `experiment/outputs/ellipticpp_transactions/training/models/dyrift_gnn/probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1` |

## 2. Comparison Studies

Purpose: compare the full model against a plain backbone, two GNN references, and a non-GNN same-input reference.

```bash
conda run -n Graph --no-capture-output python3 experiment/studies/comparisons/plain_trgt_backbone/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/comparisons/tgat_style_reference/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/comparisons/temporal_graphsage_reference/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/comparisons/xgboost_same_input/run.py
```

Saved summary table: `docs/results/comparison_auc.csv`.

Thesis presentation table: `docs/results/presentation_auc_percent.csv`.

Current comparison results:

| Setting | XinYe | ET | EPP | Macro Val AUC | Delta vs Full | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full DyRIFT-GNN | 79.2851% | 82.1329% | 82.1953% | 81.2044% | +0.0000 pp | official mainline |
| Plain TRGT Backbone | 79.0742% | 80.0629% | 78.4006% | 79.1792% | -2.0252 pp | backbone only |
| TGAT-style Reference | 78.9445% | 80.0629% | 78.3644% | 79.1239% | -2.0805 pp | temporal-attention GNN reference |
| Temporal GraphSAGE Reference | 78.8309% | 77.3516% | 78.0595% | 78.0807% | -3.1238 pp | temporal neighbor-aggregation GNN reference |
| XGBoost Same Input | 74.5771% | 90.4028% | 94.1352% | 86.3717% | +5.1673 pp | non-GNN reference, not deployment route |

## 3. Subtractive Ablations

Purpose: start from the full model and remove one shared component or method at a time.

```bash
conda run -n Graph --no-capture-output python3 experiment/studies/ablations/without_target_context_bridge/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/ablations/without_drift_expert/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/ablations/without_prototype_memory/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/ablations/without_pseudo_contrastive/run.py --device cuda
```

Saved summary table: `docs/results/ablation_auc.csv`.

Current subtractive results:

| Setting | XinYe | ET | EPP | Macro Val AUC | Delta vs Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full DyRIFT-GNN | 79.2851% | 82.1329% | 82.1953% | 81.2044% | +0.0000 pp |
| w/o Target-Context Bridge | 79.0284% | 80.0629% | 78.3473% | 79.1462% | -2.0582 pp |
| w/o Drift Expert | 79.0578% | 78.5032% | 80.3772% | 79.3127% | -1.8917 pp |
| w/o Prototype Memory | 79.1456% | 82.1686% | 81.9782% | 81.0975% | -0.1070 pp |
| w/o Pseudo-Contrastive Temporal Mining | 78.9999% | 82.1820% | 78.9550% | 80.0456% | -1.1588 pp |

## 4. Progressive Method-Building Studies

Purpose: show how the method grows from `TRGT` to `DyRIFT-GNN`.

```bash
conda run -n Graph --no-capture-output python3 experiment/studies/comparisons/plain_trgt_backbone/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/trgt_bridge/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/trgt_bridge_drift/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/trgt_bridge_drift_prototype/run.py --device cuda
conda run -n Graph --no-capture-output python3 experiment/studies/progressive/trgt_bridge_drift_prototype_pseudocontrastive/run.py --device cuda
```

The first row is the same plain backbone reference used in the comparison table, so it intentionally reuses `experiment/studies/comparisons/plain_trgt_backbone/run.py`.

Saved summary table: `docs/results/progressive_auc.csv`.

Current progressive results:

| Setting | XinYe | ET | EPP | Macro Val AUC |
| --- | ---: | ---: | ---: | ---: |
| Plain TRGT Backbone | 79.0742% | 80.0629% | 78.4006% | 79.1792% |
| TRGT + Bridge | 78.8639% | 78.4225% | 78.3053% | 78.5306% |
| TRGT + Bridge + Drift Expert | 78.8363% | 81.1953% | 78.3624% | 79.4647% |
| TRGT + Bridge + Drift Expert + Prototype Memory | 78.9463% | 80.9195% | 78.3048% | 79.3902% |
| TRGT + Bridge + Drift Expert + Prototype Memory + Pseudo-Contrastive | 79.0680% | 81.6581% | 78.3399% | 79.6886% |
| Full DyRIFT-GNN | 79.2851% | 82.1329% | 82.1953% | 81.2044% |

## 5. Supplementary XinYe Phase Study

Purpose: diagnose whether adding XinYe phase2 labeled training nodes improves phase2 adaptation without damaging the official phase1 validation target.

```bash
conda run -n Graph --no-capture-output python3 experiment/studies/supplementary/xinye_phase12_joint_train_phase1_val/run.py --device cuda
```

Saved summary table: `docs/results/supplementary_auc.csv`.

Important interpretation:

- These are diagnostic supplementary runs, not the official leakage-free thesis mainline.
- The maintained runner trains from scratch on `phase1.train + phase2.train` and still checkpoints on phase1 validation.
- Archived phase-aware diagnostic outputs are kept in the CSV tables, but their exploratory runners are not part of the maintained code path.

## 6. Leakage Audit

Regenerate the accepted-mainline audit:

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

Accepted audit files:

| File | Purpose |
| --- | --- |
| `docs/leakage_audit.md` | human-readable audit report |
| `docs/results/leakage_audit.json` | machine-readable audit output |

## 7. Epoch Logs And Curves

Every official table row should have an entry in:

```text
docs/results/epoch_log_manifest.csv
```

Use that manifest to locate:

- `summary.json` for scalar metrics.
- `seed_42/epoch_metrics.csv` for per-epoch AUC curves.
- `seed_42/train.log` for training diagnostics.
- `seed_42/training_curves.png` for saved figures.
- prediction `.npz` files for node ids, labels, and probabilities.
