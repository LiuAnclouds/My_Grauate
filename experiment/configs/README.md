# Config Files

This directory holds parameter manifests only.

- `dyrift_suite.json`: shared suite defaults and dataset-file references
- `xinye_dgraph.json`: XinYe dataset-local hyperparameters
- `elliptic_transactions.json`: ET dataset-local hyperparameters
- `ellipticpp_transactions.json`: EPP dataset-local hyperparameters

Public training and evaluation entrypoints stay at `experiment/mainline.py`, `experiment/suite.py`, and `experiment/audit.py`.
