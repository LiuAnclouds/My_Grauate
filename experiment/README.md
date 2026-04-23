# Experiment Workspace

This directory is the runnable thesis workspace for the final `DyRIFT-GNN / TRGT` route. The layout is split by responsibility so feature building, model code, evaluation, and dataset contracts are no longer coupled inside a single `training/` package.

## Layout

| Path | Role |
| --- | --- |
| [datasets/README.md](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/README.md) | dataset registry, download helpers, preparation scripts, and ignored raw data roots |
| [eda](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/eda) | leakage-safe exploratory analysis and split construction |
| [README_pipeline.md](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/README_pipeline.md) | CLI entrypoints for feature build, suite execution, and experiment workflow |
| [features](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/features) | unified UTPM feature construction and runtime feature stores |
| [models](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/models) | TRGT backbone, DyRIFT-GNN modules, graph runtime, and presets |
| [mainline.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/mainline.py) | single-dataset build/train entry |
| [suite.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/suite.py) | tri-dataset suite runner |
| [audit.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/audit.py) | leakage audit entry |
| [config_loader.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/config_loader.py) | suite manifest loader and dataset-local hparam merge logic |
| [configs](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/configs) | parameter manifests and dataset-local JSON hyperparameter files |
| [utils](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/utils) | shared metrics, split IO, path helpers, and sampling helpers |
| [outputs](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs) | ignored generated artifacts, logs, checkpoints, feature caches, and reports |

## Rules

- `datasets/` owns dataset contracts, registry, download flow, and raw/prepared benchmark files.
- `eda/` consumes the dataset contract and produces split/profile artifacts with no cross-dataset mixing.
- `features/` builds the unified input contract consumed by all three datasets.
- `models/` owns the pure-GNN runtime, backbone, modules, and graph execution logic.
- Root-level `experiment/*.py` files are the only public CLI surface for build/train/suite/audit execution.
- `experiment/configs/*.json` holds the suite manifest plus dataset-local hyperparameters.
- `outputs/` is generated state only and is ignored by Git.
