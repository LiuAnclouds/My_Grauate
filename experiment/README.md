# Experiment Workspace

This directory contains the runnable thesis workspace. The layout is intentionally split by responsibility.

## Layout

| Path | Role |
| --- | --- |
| [datasets/README.md](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/README.md) | dataset registry, download helpers, preparation scripts, and ignored raw data roots |
| [eda](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/eda) | leakage-safe exploratory analysis and split construction |
| [training/README.md](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/README.md) | DyRIFT-GNN/TRGT training pipeline |
| [outputs](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/outputs) | ignored generated artifacts, logs, checkpoints, feature caches, and reports |

## Rules

- `datasets/` owns dataset contracts, registry, download flow, and raw/prepared benchmark files.
- `eda/` consumes the active dataset contract and produces split and profile artifacts.
- `training/` consumes the dataset contract plus EDA split outputs and runs the final pure-GNN pipeline.
- `outputs/` is generated state only and is ignored by Git.
