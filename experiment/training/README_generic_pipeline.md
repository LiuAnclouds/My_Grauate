# Generic Dynamic-Graph Pipeline

For the graduation-thesis official path, prefer:

- `experiment/training/README_thesis_mainline.md`
- `experiment/training/thesis_contract.py`
- `experiment/training/run_thesis_mainline.py`
- `experiment/training/run_thesis_recipe.py`
- `experiment/training/run_thesis_suite.py`
- `experiment/training/thesis_presets.py`

`run_training.py` remains a compatibility hub for older experiments and should not be treated as the thesis mainline.
The official multi-dataset suite intentionally excludes teacher-bound presets so the thesis conclusion stays on one pure shared model path.

`run_training.py` no longer accepts `m7_utpm`.
The thesis path must be entered directly through:

- `run_thesis_mainline.py`
- `run_thesis_suite.py`
- `run_thesis_hybrid_suite.py`

## 1. Shared Input Contract

Every dataset adapter should emit the same two-phase contract:

- `phase1_gdata.npz`
- `phase2_gdata.npz`

Each file contains:

- `x`: node feature matrix, shape `[num_nodes, raw_feature_dim]`
- `y`: node labels using the shared convention
  - `1`: fraud / illicit
  - `0`: normal / licit
  - `-100`: unknown or intentionally hidden label
  - `2/3/...`: optional auxiliary historical labels, only when a dataset really has them
- `edge_index`: directed edges, shape `[num_edges, 2]`
- `edge_type`: positive integer relation ids starting from `1`
- `edge_timestamp`: positive integer event time for each edge
- `train_mask`: node ids with visible binary labels `{0, 1}`
- `test_mask`: node ids whose labels are hidden as `-100`

This keeps training code independent from the raw CSV format of any specific dataset.

## 2. Schema Derivation

The feature builder no longer assumes a fixed XinYe schema. It derives runtime schema from the active dataset:

- `raw_feature_dim` comes from `x.shape[1]`
- `edge_type_count` comes from `max(edge_type)`
- `background_labels` and optional `strong_pairs` come from `experiment/datasets/registry.py`

That means the feature width changes automatically with the dataset:

- raw blocks scale with `raw_feature_dim`
- relation blocks scale with `edge_type_count`
- auxiliary-label statistics scale with registered `background_labels`
- strong pair interactions are only enabled when the dataset registry explicitly declares them

## 3. Feature Transformation

`features.py` converts the shared contract into two stable cache layers:

- `core_features.npy`
  - raw node features
  - missingness indicators
  - degree / direction statistics
  - edge-type statistics
  - temporal counts, snapshot blocks, recent-window blocks
  - optional auxiliary-label context statistics
- `neighbor_features.npy`
  - offline 1-hop neighbor aggregations

It also writes `feature_manifest.json`, which is now the authoritative schema/output contract for training.

## 4. Graph Transformation

The same build step writes graph caches under `graph/`:

- CSR-style inbound / outbound adjacency arrays
- edge relation ids
- edge timestamps
- `first_active.npy`
- `node_time_bucket.npy`

These are consumed directly by `m4_graphsage`, `m5_temporal_graphsage`, and `m6_temporal_gat`.

## 5. Output Contract

Outputs are dataset-scoped through `experiment/datasets/registry.py`:

- XinYe keeps the legacy layout under `experiment/outputs/{eda,training}`
- new datasets write under `experiment/outputs/<dataset_name>/{eda,training}`

Stable output artifacts are:

- EDA:
  - `dataset_summary.json`
  - `recommended_split.json`
- feature cache:
  - `core_features.npy`
  - `neighbor_features.npy`
  - `feature_manifest.json`
- training:
  - `summary.json`
  - `phase1_val_predictions.npz`
  - `phase2_external_predictions.npz`

## 6. What Stays Dataset-Specific

Only dataset adapters and registry metadata should contain dataset-specific rules:

- raw file download / parsing
- label mapping into the shared contract
- optional auxiliary-label semantics
- optional strong feature pairs

The generic GNN path should stay in:

- `run_training.py`
- `graph_runtime.py`
- `features.py`
- `gnn_models.py`

The thesis mainline-specific preset surface should stay in:

- `thesis_contract.py`
- `run_thesis_mainline.py`
- `thesis_presets.py`

XinYe-specific multiclass background probe scripts can remain as separate experimental branches, but they should not define the repository-wide data contract.

## 7. Reusable Recipe Entry

For the shared dynamic-GNN path, prefer the recipe wrapper instead of manually typing long commands:

- inspect a recipe:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_recipe.py show --dataset xinye_dgraph --recipe mainline_temporal_m5`
- build shared feature caches:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_recipe.py build_features --dataset elliptic_transactions --recipe mainline_temporal_m5`
- launch a recipe run:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_recipe.py train --dataset ellipticpp_transactions --recipe context_motifadapt_m5 --run-name ellipticpp_ctx_motifadapt_v1`

Current shared recipes live in:

- `experiment/training/recipes.py`
- `experiment/training/run_recipe.py`

Official thesis recipes have been split out of the legacy wrapper surface:

- `experiment/training/run_thesis_recipe.py`

Current recipe families:

- `baseline_m5_unified`
- `thesis_m7_utpm`
- `mainline_temporal_m5`
- `context_motifadapt_m5`
- `prototype_temporal_bucket_m5`
