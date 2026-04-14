# Fixed Thesis Solution

## Current XinYe Benchmark Choice

The fixed thesis-facing solution remains:

- run name: `plan53a_rank_35e_53_v1`
- summary: `experiment/outputs/training/blends/plan53a_rank_35e_53_v1/summary.json`

This is not a pure XGBoost baseline.

## Method Composition

The final score comes from a legal leakage-safe ensemble of heterogeneous experts:

- dynamic GNN expert:
  - `m5_temporal_graphsage`
- tabular / graph-feature experts:
  - XGBoost-based historical graph-feature branches
- final blend:
  - rank-mean ensemble on validation-safe predictions

## Why This Is Reasonable For The Thesis

- It keeps the dynamic-graph core in the final method, so the thesis is still centered on dynamic graph anomaly detection.
- It uses a stronger engineering strategy than a single model, which is realistic for anti-fraud systems.
- The blend is leakage-safe because it combines already produced validation/external predictions instead of training on external labels.

## Generic Continuation Path

For new datasets, the recommended first generic path is:

1. write a dataset adapter that outputs the shared `phase1/phase2 npz` contract
2. run EDA and build features under dataset-scoped outputs
3. start with `m5_temporal_graphsage`
4. only re-enable dataset-specific auxiliary branches when the new dataset actually contains compatible auxiliary labels
