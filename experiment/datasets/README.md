# Dataset Workspace

This directory is the single dataset domain for the thesis. It replaces the old split between `experiment/dataset` and `experiment/datasets`.

## Layout

| Path | Role |
| --- | --- |
| [core](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/core) | dataset registry, prepared-graph contracts, download helpers, and shared Elliptic preparation logic |
| [scripts](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/scripts) | CLI scripts for download and dataset preparation |
| [docs/dataset_selection.md](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/docs/dataset_selection.md) | benchmark selection notes |
| [docs/remote_downloads.md](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/docs/remote_downloads.md) | mirror-aware download rules |
| `raw/` | ignored raw and prepared dataset files for XinYe, Elliptic, and Elliptic++ |

## Raw Data Layout

The ignored `raw/` subtree now uses one dataset per directory:

- `raw/xinye_dgraph/`
- `raw/elliptic_transactions/`
- `raw/ellipticpp_transactions/`
- `raw/archive/`

## Public Entry Points

- Registry: [core/registry.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/core/registry.py)
- Contracts: [core/contracts.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/core/contracts.py)
- Downloader: [core/downloads.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/core/downloads.py)
- Elliptic prepare: [scripts/prepare_elliptic.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/scripts/prepare_elliptic.py)
- Elliptic++ prepare: [scripts/prepare_ellipticpp.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/scripts/prepare_ellipticpp.py)
