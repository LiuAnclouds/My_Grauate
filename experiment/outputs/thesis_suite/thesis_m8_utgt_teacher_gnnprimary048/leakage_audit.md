# Thesis Leakage Audit

- Suite: `thesis_m8_utgt_teacher_gnnprimary048`
- Scope: direct train/val/test/external overlap and cross-dataset isolation.
- Conclusion: no hard leakage was detected in the audited official suite.

| Dataset | Train | Val | Test Pool | External | Secondary Train | Selection Mode | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| xinye_dgraph | 661334 | 166013 | 354578 | 0 | 661334 | split_train_ids | pass |
| elliptic_transactions | 36591 | 9973 | 157205 | 0 | 36591 | split_train_ids | pass |
| ellipticpp_transactions | 36591 | 9973 | 157205 | 0 | 36591 | split_train_ids | pass |

## Hard-Leakage Checklist

- `train`, `val`, `test_pool`, and `external` id sets are pairwise disjoint for every dataset.
- Every secondary training row is a subset of the dataset's `phase1_train` split.
- Secondary and hybrid validation bundles exactly match the official validation ids.
- Run directories stay inside dataset-scoped output namespaces; no cross-dataset prediction path is reused.
