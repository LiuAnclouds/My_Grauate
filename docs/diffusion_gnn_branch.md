# Diffusion-GNN Branch Notes

This branch is a copied working branch for diffusion experiments. The original
baseline project under `/home/moonxkj/Desktop/MyWork/Graduation_Project` is not
modified.

## Reference Mapping

| Source | Used part | Branch implementation |
| --- | --- | --- |
| `reference/DiffGAD/diffusion_model.py` | EDM loss, log-normal noise, EDM preconditioning, MLP denoiser, reconstruction score | `models/components/diffusion.py` learns to denoise DyRIFT target embeddings and exposes a reconstruction score |
| `reference/DiffGAD/DiffGAD.py` | prototype-conditioned diffusion over latent embeddings | optional `embedding_diffusion_proto_*` config uses normal-target embedding EMA as the prototype condition |
| `reference/WWW25-Grad/models/SupGCL.py` | high-pass graph signal, contrastive view idea | `features/features.py::neighbor_highpass` stores center-minus-neighbor residual statistics |
| `reference/WWW25-Grad/models/GuiDDPM.py` | DDPM over fixed-size adjacency subgraphs | reviewed but not directly copied; generating transaction edges would violate dynamic split semantics |
| `reference/WWW25-Grad/models/WeightedFusion.py` | relation-wise polynomial/weighted fusion | kept as a later ablation candidate; DyRIFT already has relation-gated attention |

## Implemented Modules

- `models/components/diffusion.py`: DiffGAD-style EDM embedding denoiser,
  reconstruction score, prototype-conditioned denoising hook.
- `models/components/diffusion_runtime.py`: target selection, score loss,
  EMA score calibration, DE-GAD-style view contrast, prediction score blend,
  prototype EMA state, and best-epoch runtime-state rollback.
- `features/features.py`: `utpm_diffusion` now uses fused temporal behavior
  features plus `neighbor_highpass`.
- `configs/train/xinye_dgraph_diffusion.json`: XinYE full-run profile using
  batch size 512, diffusion dim 256, score blend, and the new fused feature
  foundation.

## Method Boundary

The branch separates paper-faithful pieces from engineering adaptations:

- Paper-faithful: latent denoising, reconstruction score, prototype-conditioned
  denoiser, diffusion-enhanced view contrast.
- Engineering adaptation: detach mode, EMA score calibration, delayed inference
  blend, and best-epoch runtime-state rollback.
- Not used: generated transaction edges or DDPM adjacency synthesis, because
  those can leak future topology or alter the chronological fraud split.

## Current XinYE Ledger

Historical reference:

| Run | Best val AUC | Note |
| --- | ---: | --- |
| `fullsplit_xy_base_5e` | 0.789734 @ epoch 3 | old baseline-style run, b512, pseudo contrast on |
| `stable_xy_diff_detach_ema_d256_score001_blend01_b512_70e` | 0.791934 @ epoch 6 | best completed diffusion branch run before fused features |

Active full run:

```bash
GRADPROJ_ACTIVE_DATASET=xinye_dgraph conda run -n Graph python3 train.py train \
  --parameter-file configs/train/xinye_dgraph_diffusion.json \
  --experiment-name stable_xy_utpmdiff_fused_detach_ema_d256_score001_blend01_b512_70e \
  --run-name stable_xy_utpmdiff_fused_detach_ema_d256_score001_blend01_b512_70e \
  --epochs 70 --batch-size 512 --skip-final-predictions --device cuda
```

Early curve:

| Epoch | Val AUC | Diffusion active |
| ---: | ---: | --- |
| 1 | 0.771073 | no |
| 2 | 0.780869 | no |

Diffusion starts at epoch 5 and inference blending starts at epoch 8, so the
decisive comparison is the best AUC after epoch 8.
