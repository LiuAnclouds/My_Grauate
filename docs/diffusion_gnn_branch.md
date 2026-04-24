# Diffusion-GNN Branch Notes

This branch keeps the original DyRIFT-GNN/TRGT training path intact and adds a
lightweight diffusion regularizer over target embeddings.

## Reference Mapping

- `reference/DiffGAD/diffusion_model.py` inspired the EDM-style denoising loss:
  random log-normal noise level, EDM preconditioning, and MLP denoiser.
- `reference/WWW25-Grad` was reviewed as a relation-generation option, but it
  requires DGL, improved-diffusion, and graph generation plumbing. That is too
  invasive for a two-week thesis fallback, so it is not copied into this branch.

## Implemented Change

- New module: `models/components/diffusion.py`
- New config fields under `GraphModelConfig`:
  - `embedding_diffusion_weight`
  - `embedding_diffusion_start_epoch`
  - `embedding_diffusion_dim`
  - `embedding_diffusion_p_mean`
  - `embedding_diffusion_p_std`
  - `embedding_diffusion_sigma_data`
  - `embedding_diffusion_min_batch_size`
- New preset: `dyrift_trgt_diffusion_v1`
- New train parameter file:
  `configs/train/xinye_dgraph_diffusion.json`

The loss is only used during training. Inference still uses the DyRIFT-GNN
network checkpoint, so serving and prediction behavior do not depend on the
auxiliary denoiser.

## Quick Validation

Environment:

```bash
conda run -n Graph python3 train.py train --parameter-file configs/train/xinye_dgraph_diffusion.json --dry-run
```

Smoke comparison on `xinye_dgraph`, seed `42`, `128` train nodes, `128` val
nodes, `2` epochs, hidden dim `64`, fanouts `5 3`:

| Run | Validation AUC | Note |
| --- | ---: | --- |
| `deploy_smoke` | 0.552846 | DyRIFT-GNN deploy preset |
| `diffusion_smoke` | 0.551220 | Diffusion enabled, very small sample |

Medium comparison on `xinye_dgraph`, seed `42`, `2048` train nodes, `2048` val
nodes, `5` epochs, hidden dim `64`, fanouts `5 3`:

| Run | Diffusion weight | Validation AUC |
| --- | ---: | ---: |
| `deploy_medium` | 0.000 | 0.716258 |
| `diffusion_medium_w005` | 0.005 | 0.716948 |
| `diffusion_medium_w010` | 0.010 | 0.716990 |

Interpretation: the medium run shows a small positive validation-AUC change
when the diffusion denoising regularizer is enabled. It is not yet a full
paper-level result; the next step is to run the full `70` epoch setting in
`configs/train/xinye_dgraph_diffusion.json`.
