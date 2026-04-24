# Experiment Outputs

`experiment/outputs/` is local generated state. Most files are ignored by Git, but the directory is kept organized so the current thesis artifacts are easy to find.

## Current Layout

| Path | Role |
| --- | --- |
| `reports/accepted_mainline/` | canonical accepted DyRIFT-GNN/TRGT report for the current thesis version |
| `reports/suites/<suite_name>/` | future suite reruns produced by `experiment/suite.py` |
| `training/models/dyrift_gnn/full_xinye_repro_v1/` | accepted XinYe DGraph run |
| `elliptic_transactions/training/models/dyrift_gnn/probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1/` | accepted Elliptic Transactions run |
| `ellipticpp_transactions/training/models/dyrift_gnn/probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1/` | accepted Elliptic++ Transactions run |
| `training/features_ap32/` | maintained XinYe feature cache |
| `elliptic_transactions/training/features_ap64/` | maintained ET feature cache |
| `ellipticpp_transactions/training/features_ap96/` | maintained EPP feature cache |
| `eda/`, `elliptic_transactions/eda/`, `ellipticpp_transactions/eda/` | dataset profiles and split artifacts |
| `studies/comparisons/` | comparison-study outputs used by `experiment/sync_results.py` |
| `studies/ablations/` | subtractive ablation outputs used by `experiment/sync_results.py` |
| `studies/progressive/` | progressive module-building outputs used by `experiment/sync_results.py` |
| `studies/supplementary/` | XinYe supplementary diagnostics |

## Removed Local Clutter

The cleanup removed old local search artifacts from this directory, including `thesis_suite/`, `thesis_ablation/`, old `m7_*` and `m8_*` exploratory runs, XGBoost search model directories, blend caches, stale feature caches, smoke runs, and loose logs.
