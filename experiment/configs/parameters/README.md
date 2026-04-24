# Train Parameter Files

These files are explicit single-dataset `experiment/mainline.py train` parameter manifests.

The train entrypoint intentionally does not inject thesis hyperparameters as code defaults. Use one of these JSON files, or pass the same fields through CLI flags. Only operational controls such as `epochs` and `outdir` have maintained fallbacks.

## Files

| File | Dataset |
| --- | --- |
| `xinye_dgraph_train.json` | XinYe DGraph |
| `elliptic_transactions_train.json` | Elliptic Transactions |
| `ellipticpp_transactions_train.json` | Elliptic++ Transactions |

## Example

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  train \
  --parameter-file experiment/configs/parameters/xinye_dgraph_train.json
```

CLI flags override JSON values, so a quick override can be written as:

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  train \
  --parameter-file experiment/configs/parameters/xinye_dgraph_train.json \
  --epochs 90 \
  --graph-config-override min_early_stop_epoch=40
```

The same entrypoint can also run without a JSON file when every required core field is explicit:

```bash
conda run -n Graph --no-capture-output python3 experiment/mainline.py \
  train \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --run-name manual_cli_run \
  --feature-profile utpm_shift_enhanced \
  --feature-dir experiment/outputs/training/features_ap32 \
  --seeds 42 \
  --batch-size 512 \
  --hidden-dim 128 \
  --rel-dim 32 \
  --fanouts 15 10
```

Required core fields are `model`, `preset`, `run_name`, `feature_profile`, `feature_dir`, `seeds`, `batch_size`, `hidden_dim`, `rel_dim`, and `fanouts`.
