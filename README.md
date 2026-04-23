# DyRIFT-GNN

English | [中文说明](README.zh-CN.md)

`DyRIFT-GNN` is the thesis mainline in this repository for dynamic-graph financial fraud detection. It uses `TRGT` (`Temporal-Relational Graph Transformer`) as the backbone and keeps one unified pure-GNN architecture across:

- XinYe DGraph
- Elliptic Transactions
- Elliptic++ Transactions

Main deployment path:

`dataset-local preprocessing -> UTPM unified feature contract -> TRGT backbone -> DyRIFT modules -> fraud probability`

The final inference route is single-model pure GNN. No external tree model, teacher branch, or second-stage fusion model is used at deployment.

## Mainline Result

| Dataset | Val AUC | Artifact |
| --- | ---: | --- |
| XinYe DGraph | 0.792851 | `experiment/outputs/training/models/dyrift_gnn/full_xinye_repro_v1` |
| Elliptic Transactions | 0.821329 | `experiment/outputs/elliptic_transactions/training/models/dyrift_gnn/probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1` |
| Elliptic++ Transactions | 0.821953 | `experiment/outputs/ellipticpp_transactions/training/models/dyrift_gnn/probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1` |
| Macro Average | 0.812044 | `docs/results/accepted_mainline_summary.json` |

Runtime id: `dyrift_gnn`  
Paper-facing method: `Dynamic Risk-Informed Fraud Graph Neural Network (DyRIFT-GNN)`  
Backbone name: `Temporal-Relational Graph Transformer (TRGT)`

## Model Summary

- Backbone: `TRGT`, a temporal-relation graph transformer for dynamic neighborhood message passing.
- Inference-time modules: target-context bridge, drift expert, internal risk fusion, and dataset-conditional cold-start residual.
- Training-time methods: prototype memory and pseudo-contrastive temporal mining.
- Input contract: all datasets are mapped into the same `UTPM` semantic family; only raw preprocessing and hyperparameters remain dataset-local.

## Documentation

| Card | Description |
| --- | --- |
| [Chinese README](README.zh-CN.md) | Chinese project overview |
| [Thesis Method](docs/thesis_method.md) | thesis-facing method, constraints, and deployment path |
| [DyRIFT Method Card](docs/dyrift_gnn_method.md) | compact model identity and method card |
| [TRGT Backbone](docs/trgt_backbone.md) | backbone structure and temporal-relation attention |
| [DyRIFT Modules](docs/dyrift_modules.md) | module vs method split and ablation evidence |
| [Thesis Experiments](docs/thesis_experiments.md) | mainline, comparison, ablation, progressive, and supplementary tables |
| [Training And Configs](docs/training_and_configs.md) | commands, config files, and output layout |
| [Code Reference](docs/code_reference.md) | package layout and call chain |
| [Studies Workspace](experiment/studies/README.md) | isolated comparison, ablation, progressive, and supplementary experiments |
| [Leakage Audit](docs/leakage_audit.md) | accepted mainline hard-leakage audit |
| [Mainline AUC CSV](docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv) | accepted three-dataset AUC table |
| [Comparison AUC CSV](docs/results/comparison_auc.csv) | comparison-study AUC table |
| [Ablation AUC CSV](docs/results/ablation_auc.csv) | subtractive ablation AUC table |
| [Progressive AUC CSV](docs/results/progressive_auc.csv) | progressive method-building table |
| [Supplementary AUC CSV](docs/results/supplementary_auc.csv) | XinYe `phase1+phase2` joint-train supplement |
| [Epoch Log Manifest](docs/results/epoch_log_manifest.csv) | per-experiment epoch, log, and curve paths |

## Repository Layout

| Path | Role |
| --- | --- |
| [experiment/mainline.py](experiment/mainline.py) | single-dataset feature build and train entry |
| [experiment/suite.py](experiment/suite.py) | three-dataset mainline rerun entry |
| [experiment/audit.py](experiment/audit.py) | hard-leakage audit |
| [experiment/configs/](experiment/configs) | maintained per-dataset rerun configs |
| [experiment/datasets/](experiment/datasets) | dataset registry, raw-data contract, and preparation scripts |
| [experiment/features/](experiment/features) | unified feature cache and normalizer utilities |
| [experiment/models/](experiment/models) | runtime, engine, backbone, model modules, and presets |
| [experiment/studies/](experiment/studies) | isolated comparison, ablation, progressive, and supplementary studies |
| [experiment/utils/](experiment/utils) | path, split, IO, and sampling helpers |
| [docs/](docs) | thesis-facing method, experiment, and code documents |

## Reproduce

Build unified features:

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  build_features \
  --phase both
```

Run the current maintained mainline config:

```bash
conda run -n Graph --no-capture-output python3 experiment/suite.py \
  --suite-name dyrift_mainline_rerun \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/configs/dyrift_suite.json \
  --seeds 42
```

Run one isolated study:

```bash
conda run -n Graph --no-capture-output python3 \
  experiment/studies/comparisons/tgat_style_reference/run.py
```

Regenerate the accepted-mainline leakage audit:

```bash
conda run -n Graph --no-capture-output python3 experiment/audit.py \
  --suite-summary experiment/outputs/reports/dyrift_gnn_accepted_mainline/summary.json
```

## Integrity

- `deployment_path=single_gnn_end_to_end`
- `dataset_isolation=true`
- `cross_dataset_training=false`
- `same_architecture_across_datasets=true`
- `hard_leakage_detected=false`
- Supplementary XinYe `phase1+phase2` joint training is stored separately and is explicitly not the official leakage-free thesis mainline.
