# Thesis Leakage Audit

- Suite: `thesis_m8_utgt_deploy_pure_eppcold_v1`
- Mode: `mainline`
- Scope: direct train/val/test/external overlap and cross-dataset isolation.
- Conclusion: no hard leakage was detected in the audited thesis suite.

| Dataset | Train | Val | Test Pool | External | Teacher Dirs | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| xinye_dgraph | 661334 | 166013 | 354578 | 0 | 0 | pass |
| elliptic_transactions | 36591 | 9973 | 157205 | 0 | 0 | pass |
| ellipticpp_transactions | 36591 | 9973 | 157205 | 0 | 0 | pass |

## Hard-Leakage Checklist

- `train`, `val`, `test_pool`, and `external` id sets are pairwise disjoint for every dataset.
- Backbone `phase1_train`, `phase1_val`, and `test_pool` prediction bundles exactly match the official split ids.
- No teacher prediction directory is configured for the audited pure-GNN deployment suite.
- No cross-dataset prediction path is reused by the audited pure GNN mainline.
