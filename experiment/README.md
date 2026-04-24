# Experiment Workspace

This directory is the runnable thesis workspace for the final `DyRIFT-GNN / TRGT` route. The layout is split by responsibility so feature building, model code, evaluation, and dataset contracts are no longer coupled inside a single `training/` package.

## Layout

| Path | Role |
| --- | --- |
| [datasets/README.md](datasets/README.md) | dataset registry, download helpers, preparation scripts, and ignored raw data roots |
| [eda](eda) | leakage-safe exploratory analysis and split construction |
| [README_pipeline.md](README_pipeline.md) | CLI entrypoints for feature build, suite execution, and experiment workflow |
| [features](features) | unified UTPM feature construction and runtime feature stores |
| [models](models) | TRGT backbone, DyRIFT-GNN modules, graph runtime, and presets |
| [mainline.py](mainline.py) | single-dataset build/train entry |
| [suite.py](suite.py) | tri-dataset suite runner |
| [audit.py](audit.py) | leakage audit entry |
| [config_loader.py](config_loader.py) | suite manifest loader and dataset-local hparam merge logic |
| [configs](configs) | parameter manifests and dataset-local JSON hyperparameter files |
| [utils](utils) | shared metrics, split IO, path helpers, and sampling helpers |
| [outputs/README.md](outputs/README.md) | organized local artifact layout for reports, caches, studies, and accepted runs |

## Rules

- `datasets/` owns dataset contracts, registry, download flow, and raw/prepared benchmark files.
- `eda/` consumes the dataset contract and produces split/profile artifacts with no cross-dataset mixing.
- `features/` builds the unified input contract consumed by all three datasets.
- `models/` owns the pure-GNN runtime, backbone, modules, and graph execution logic.
- Root-level `experiment/*.py` files are the only public CLI surface for build/train/suite/audit execution.
- `experiment/configs/*.json` holds the suite manifest plus dataset-local hyperparameters.
- `outputs/` is generated state; keep only the accepted mainline, maintained feature caches, EDA outputs, and organized study outputs there.
