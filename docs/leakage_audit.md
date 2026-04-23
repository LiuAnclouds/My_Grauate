# Thesis Leakage Audit

- Suite: `dyrift_gnn_accepted_mainline`
- Mode: `pure_gnn_mainline`
- Scope: direct train/val/test/external overlap and cross-dataset isolation.
- Conclusion: no hard leakage was detected in the audited thesis suite.

| Dataset | Train | Val | Test Pool | External | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| xinye_dgraph | 661334 | 166013 | 354578 | 0 | pass |
| elliptic_transactions | 36591 | 9973 | 157205 | 0 | pass |
| ellipticpp_transactions | 36591 | 9973 | 157205 | 0 | pass |

## Hard-Leakage Checklist

- `train`, `val`, `test_pool`, and `external` id sets are pairwise disjoint for every dataset.
- DyRIFT-GNN `phase1_train`, `phase1_val`, and `test_pool` prediction bundles exactly match the official split ids.
- Every run directory stays inside its dataset-scoped output namespace.
- No cross-dataset prediction path, external classifier, or second-stage model is used by the audited pure-GNN mainline.
