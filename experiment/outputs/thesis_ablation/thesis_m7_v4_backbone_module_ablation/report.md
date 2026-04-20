# Thesis Backbone Ablation Report

- Created at: `2026-04-20T06:16:28+00:00`
- Output root: `experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation`

## Backbone Ablation Table

| Setting | xinye_dgraph | elliptic_transactions | ellipticpp_transactions | Macro Val AUC |
| --- | ---: | ---: | ---: | ---: |
| Official Backbone | 0.776439 | 0.812635 | 0.777611 | 0.788895 |
| No Prototype Memory | 0.777391 | 0.812275 | 0.778365 | 0.789344 |
| No Pseudo-Contrastive Mining | 0.775860 | 0.794927 | 0.777350 | 0.782712 |
| No Drift Residual Context | 0.777244 | 0.816354 | 0.779095 | 0.790898 |

## Decision Reference

| Setting | xinye_dgraph | elliptic_transactions | ellipticpp_transactions | Macro Val AUC |
| --- | ---: | ---: | ---: | ---: |
| Official GNN-primary Blend | 0.795293 | 0.949436 | 0.946584 | 0.897104 |

## Commands

- `official_backbone`: `/home/moonxkj/miniconda3/envs/Graph/bin/python3 /home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/run_thesis_suite.py --suite-name thesis_m7_v4_unified_e8 --model m7_utpm --preset utpm_temporal_shift_v4 --feature-profile utpm_unified --device cuda --epochs 8 --seeds 42 --skip-existing`
- `ablate_no_prototype`: `/home/moonxkj/miniconda3/envs/Graph/bin/python3 /home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/run_thesis_suite.py --suite-name thesis_m7_v4_ablate_noprototype --model m7_utpm --preset utpm_temporal_shift_v4 --feature-profile utpm_unified --device cuda --epochs 8 --seeds 42 --skip-existing --graph-config-override prototype_multiclass_num_classes=0 --graph-config-override prototype_loss_weight=0.0 --graph-config-override prototype_loss_weight_schedule=none --graph-config-override prototype_loss_min_weight=0.0 --graph-config-override prototype_neighbor_blend=0.0 --graph-config-override prototype_global_blend=0.0 --graph-config-override prototype_consistency_weight=0.0 --graph-config-override prototype_separation_weight=0.0`
- `ablate_no_pseudocontrast`: `/home/moonxkj/miniconda3/envs/Graph/bin/python3 /home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/run_thesis_suite.py --suite-name thesis_m7_v4_ablate_nopseudocontrast --model m7_utpm --preset utpm_temporal_shift_v4 --feature-profile utpm_unified --device cuda --epochs 8 --seeds 42 --skip-existing --graph-config-override pseudo_contrastive_weight=0.0`
- `ablate_no_drift_residual`: `/home/moonxkj/miniconda3/envs/Graph/bin/python3 /home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/training/run_thesis_suite.py --suite-name thesis_m7_v4_ablate_nodriftresidual --model m7_utpm --preset utpm_temporal_shift_v4 --feature-profile utpm_unified --device cuda --epochs 8 --seeds 42 --skip-existing --graph-config-override target_context_fusion=none --graph-config-override target_time_adapter_strength=0.0 --graph-config-override target_time_expert_entropy_weight=0.0 --graph-config-override normal_bucket_align_weight=0.0 --graph-config-override normal_bucket_adv_weight=0.0 --graph-config-override context_residual_scale=0.0 --graph-config-override context_residual_clip=0.0 --graph-config-override context_residual_budget=0.0 --graph-config-override context_residual_budget_weight=0.0 --graph-config-override context_residual_budget_schedule=none --graph-config-override context_residual_budget_min_weight=0.0 --graph-config-override context_residual_budget_release_epochs=0 --graph-config-override context_residual_budget_release_delay_epochs=0`
