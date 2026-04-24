# Config Files

This directory holds parameter manifests only.

- `dyrift_suite.json`: shared suite defaults and dataset-file references
- `xinye_dgraph.json`: XinYe dataset-local hyperparameters
- `elliptic_transactions.json`: ET dataset-local hyperparameters
- `ellipticpp_transactions.json`: EPP dataset-local hyperparameters
- `training_policy.json`: maintained epoch and early-stopping policy for mainline and study reruns
- `parameters/`: explicit single-dataset `mainline.py train --parameter-file` manifests

Public training and evaluation entrypoints stay at `experiment/mainline.py`, `experiment/suite.py`, and `experiment/audit.py`.

The maintained policy uses `max_epochs=70` and `min_early_stop_epoch=30`. Saved historical artifacts keep their observed epoch logs; this directory defines future rerun behavior rather than rewriting old curves.

The direct `experiment/mainline.py train` entrypoint does not inject thesis hyperparameters as code defaults. Provide a JSON file from `parameters/` or pass the same fields through CLI flags.
