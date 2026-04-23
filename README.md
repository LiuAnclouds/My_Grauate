# DyRIFT-GNN

`DyRIFT-GNN` is the final thesis method in this repository. It uses `TRGT` (`Temporal-Relational Graph Transformer`) as the dynamic-graph backbone and keeps one unified training and inference route for XinYe DGraph, Elliptic Transactions, and Elliptic++ Transactions.

Final pipeline:

`dataset-local preprocessing -> UTPM feature contract -> TRGT backbone -> DyRIFT-GNN modules -> fraud probability`

The deployed path is single-model pure GNN. No external tree model, teacher branch, or second-stage classifier is required at inference time.

## Project Cards

| Card | Description |
| --- | --- |
| [Method Overview](docs/dyrift_gnn_method.md) | DyRIFT-GNN method, input contract, deployment path |
| [TRGT Backbone](docs/trgt_backbone.md) | backbone structure and message passing |
| [Model Modules](docs/dyrift_modules.md) | bridge, drift expert, prototype, pseudo-contrastive, risk fusion, cold-start residual |
| [Code Reference](docs/code_reference.md) | package layout, main classes, call chain |
| [Training And Configs](docs/training_and_configs.md) | commands, config files, output layout |
| [Experiment Results](docs/thesis_experiments.md) | final AUC table, GNN comparison, ablations |
| [Experiment Workspace](experiment/README.md) | experiment-level folder layout and responsibility split |
| [Dataset Workspace](experiment/datasets/README.md) | dataset registry, preparation scripts, and raw-data layout |
| [Pipeline Guide](experiment/README_pipeline.md) | engineering-facing guide for the final experiment pipeline |
| [Leakage Audit](experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md) | final hard-leakage audit |
| [Suite Summary](experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json) | final tri-dataset suite summary |
| [Metrics CSV](docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv) | final dataset-level metrics |
| [Epoch Metrics CSV](docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv) | aggregated epoch logs |

## Final Results

| Dataset | Val AUC | Val PR-AUC | Val AP |
| --- | ---: | ---: | ---: |
| XinYe DGraph | 0.790455 | 0.045843 | 0.045998 |
| Elliptic Transactions | 0.821329 | 0.432221 | 0.396340 |
| Elliptic++ Transactions | 0.821953 | 0.471452 | 0.438596 |
| Macro Average | 0.811246 | 0.316505 | 0.293645 |

Official runtime id: `dyrift_gnn`  
Paper-facing method: `Dynamic Risk-Informed Fraud Graph Neural Network (DyRIFT-GNN)`  
Backbone name: `Temporal-Relational Graph Transformer (TRGT)`

## Repository Layout

| Path | Role |
| --- | --- |
| [experiment/mainline.py](experiment/mainline.py) | single-dataset feature build and train entry |
| [experiment/suite.py](experiment/suite.py) | tri-dataset suite runner |
| [experiment/audit.py](experiment/audit.py) | hard-leakage audit |
| [experiment/datasets/core/registry.py](experiment/datasets/core/registry.py) | active dataset registry and raw/prepared path contract |
| [experiment/datasets/scripts/prepare_elliptic.py](experiment/datasets/scripts/prepare_elliptic.py) | Elliptic dataset preparation entry |
| [experiment/datasets/scripts/prepare_ellipticpp.py](experiment/datasets/scripts/prepare_ellipticpp.py) | Elliptic++ dataset preparation entry |
| [experiment/models/engine.py](experiment/models/engine.py) | shared graph engine, loss, sampling, evaluation |
| [experiment/models/runtime.py](experiment/models/runtime.py) | runtime bundle builder |
| [experiment/models/presets.py](experiment/models/presets.py) | official preset definitions |
| [experiment/config_loader.py](experiment/config_loader.py) | suite and dataset hyperparameter loading |
| [experiment/features/features.py](experiment/features/features.py) | UTPM feature cache and normalizer utilities |
| [experiment/models/graph.py](experiment/models/graph.py) | experiment-class resolution and graph contexts |
| [experiment/models/modules/backbone.py](experiment/models/modules/backbone.py) | TRGT backbone blocks and internal risk encoder |
| [experiment/models/modules/model.py](experiment/models/modules/model.py) | DyRIFT-GNN model facade |
| [experiment/models/modules/trainer.py](experiment/models/modules/trainer.py) | DyRIFT-GNN trainer wrapper |
| [experiment/models/modules/bridge.py](experiment/models/modules/bridge.py) | target-context bridge |
| [experiment/models/modules/memory.py](experiment/models/modules/memory.py) | prototype and normal-alignment memory |
| [experiment/utils/common.py](experiment/utils/common.py) | paths, split loading, metrics, IO helpers |
| [experiment/utils/sampling.py](experiment/utils/sampling.py) | sampling-profile helpers |

## Config Files

The model family is shared across all datasets. Dataset-local tuning lives in separate JSON files.

| File | Role |
| --- | --- |
| [experiment/configs/dyrift_suite.json](experiment/configs/dyrift_suite.json) | suite manifest with shared defaults and dataset file references |
| [experiment/configs/xinye_dgraph.json](experiment/configs/xinye_dgraph.json) | XinYe profile |
| [experiment/configs/elliptic_transactions.json](experiment/configs/elliptic_transactions.json) | ET profile |
| [experiment/configs/ellipticpp_transactions.json](experiment/configs/ellipticpp_transactions.json) | EPP profile |

## Reproduce

Build unified features:

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  build_features \
  --phase both
```

Run the final suite:

```bash
conda run -n Graph --no-capture-output python3 experiment/suite.py \
  --suite-name thesis_dyrift_gnn_trgt_deploy_pure_v1 \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/configs/dyrift_suite.json \
  --seeds 42 \
  --skip-existing
```

Run hard-leakage audit:

```bash
conda run -n Graph --no-capture-output python3 experiment/audit.py \
  --suite-summary experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json
```

## Integrity

- `deployment_path=single_gnn_end_to_end`
- `dataset_isolation=true`
- `cross_dataset_training=false`
- `same_architecture_across_datasets=true`
- `hard_leakage_detected=false`
