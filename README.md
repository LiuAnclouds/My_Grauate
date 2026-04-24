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

| Dataset | Val AUC |
| --- | ---: |
| XinYe DGraph | 79.2851% |
| Elliptic Transactions | 82.1329% |
| Elliptic++ Transactions | 82.1953% |
| Macro Average | 81.2044% |

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
| [Reproducibility Guide](docs/reproducibility.md) | clone, conda environment, dependency installation, data layout, feature build, and mainline rerun |
| [Experiment Reproduction](docs/experiment_reproduction.md) | accepted mainline, comparison, ablation, progressive, supplementary, and audit commands |
| [Model Execution Flow](docs/model_execution_flow.md) | end-to-end engineering flow from raw graph to fraud probability |
| [Thesis Method](docs/thesis_method.md) | thesis-facing method, constraints, and deployment path |
| [DyRIFT Method Card](docs/dyrift_gnn_method.md) | compact model identity and method card |
| [TRGT Backbone](docs/trgt_backbone.md) | backbone structure and temporal-relation attention |
| [DyRIFT Modules](docs/dyrift_modules.md) | module vs method split and ablation evidence |
| [Thesis Experiments](docs/thesis_experiments.md) | mainline, comparison, ablation, progressive, and supplementary tables |
| [Training And Configs](docs/training_and_configs.md) | commands, config files, and output layout |
| [Code Reference](docs/code_reference.md) | package layout and call chain |
| [Studies Workspace](experiment/studies/README.md) | isolated comparison, ablation, progressive, and supplementary experiments |
| [Leakage Audit](docs/leakage_audit.md) | accepted mainline hard-leakage audit |
| [Training Policy JSON](experiment/configs/training_policy.json) | maintained `70` epoch / min-`30` early-stop policy for future reruns |
| [Mainline AUC CSV](docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv) | accepted three-dataset AUC table |
| [Comparison AUC CSV](docs/results/comparison_auc.csv) | comparison-study AUC table |
| [Ablation AUC CSV](docs/results/ablation_auc.csv) | subtractive ablation AUC table |
| [Progressive AUC CSV](docs/results/progressive_auc.csv) | progressive method-building table |
| [Supplementary AUC CSV](docs/results/supplementary_auc.csv) | XinYe `phase1+phase2` joint-train supplement |
| [Presentation AUC CSV](docs/results/presentation_auc_percent.csv) | percentage-format AUC and percentage-point deltas for thesis tables |
| [Experiment Epoch Policy CSV](docs/results/experiment_epoch_policy.csv) | per-study planned max epochs and minimum early-stop epoch |
| [Historical External Records](docs/results/historical_external_records.csv) | user-provided competition/external evaluation records kept separate from reproducible mainline artifacts |
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

The full checklist is split into two documents:

- [Reproducibility Guide](docs/reproducibility.md): clone the repository, create the conda environment, install dependencies, place datasets, build features, and run the accepted mainline.
- [Experiment Reproduction](docs/experiment_reproduction.md): rerun comparison, ablation, progressive, supplementary, and leakage-audit studies.

Minimal end-to-end setup from a clean machine:

```bash
git clone git@github.com:LiuAnclouds/My_Grauate.git
cd My_Grauate

conda create -n Graph python=3.10 -y
conda run -n Graph --no-capture-output pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n Graph --no-capture-output pip install -r requirements.txt
conda run -n Graph --no-capture-output python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Build unified features after raw datasets are placed under `experiment/datasets/raw/`:

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

The maintained rerun policy is `max_epochs=70` with `min_early_stop_epoch=30`. Existing accepted artifacts keep their actual saved epoch logs; rerunning the commands above regenerates curves under suite-scoped names such as `dyrift_mainline_rerun_xy_70e_min30`.

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
