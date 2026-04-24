# Model Execution Flow

This document explains the engineering execution chain of the accepted `DyRIFT-GNN / TRGT` route. It is written to make the project inspectable as both a thesis method and a runnable research codebase.

## 1. End-To-End Route

The accepted deployment path is:

```text
raw dynamic graph
  -> dataset-local contract validation
  -> UTPM unified feature cache
  -> GraphPhaseContext runtime bundle
  -> TRGT temporal-relational message passing
  -> DyRIFT target-context and risk modules
  -> fraud probability
```

There is no external teacher model, tree model, or second-stage classifier in final inference.

## 2. Main Entrypoints

| Entrypoint | Role |
| --- | --- |
| `experiment/mainline.py` | build features and train/evaluate one dataset |
| `experiment/suite.py` | run the maintained three-dataset suite |
| `experiment/audit.py` | verify hard-leakage constraints |
| `experiment/studies/*/run.py` | isolated comparison, ablation, progressive, and supplementary studies |

## 3. Data And Feature Flow

### Step 1: Dataset Registry

`experiment/datasets/core/registry.py` defines dataset names, display names, and phase artifacts. The important dataset keys are:

| Key | Dataset |
| --- | --- |
| `xinye_dgraph` | XinYe DGraph |
| `elliptic_transactions` | Elliptic Transactions |
| `ellipticpp_transactions` | Elliptic++ Transactions |

### Step 2: Dataset Contract

Each phase must provide a dynamic graph contract with node features, edges, labels, train ids, and test-pool ids. Contract validation prevents invalid ids, duplicated masks, and train/test overlap.

### Step 3: UTPM Feature Cache

`experiment/features/features.py` builds the unified feature cache. Dataset-local raw fields are allowed, but the output feature groups are mapped into a shared semantic family.

Common feature group roles:

| Group type | Role |
| --- | --- |
| attribute projection | compact raw attribute representation |
| graph statistics | degree, transaction, and local structural descriptors |
| temporal descriptors | activity and recency information |
| relation-aware descriptors | edge-type and direction summaries |
| target context groups | target-node context used by bridge modules |

### Step 4: Runtime Bundle

`experiment/models/runtime.py` builds a `RuntimeBundle` containing:

| Object | Meaning |
| --- | --- |
| `GraphModelConfig` | model and training hyperparameters |
| `feature_groups` | selected base feature groups |
| `feature_normalizer_state` | feature normalization fitted on training ids |
| `phase1_context` / `phase2_context` | graph, labels, feature stores, and target context stores |
| `target_context_input_dim` | target-context bridge input width |
| `num_relations` | relation count used by TRGT |

The accepted mainline fits normalizers only from the training split of the corresponding dataset phase.

## 4. TRGT Backbone

`TRGT` stands for `Temporal-Relational Graph Transformer`.

Its role is to transform a target node and sampled temporal neighbors into a hidden representation:

```text
node features + sampled temporal neighbors + relation ids + relative time
  -> relation-aware message projection
  -> temporal encoding
  -> attention aggregation
  -> target embedding
```

The backbone is not a generic static GNN. It explicitly uses:

| Signal | Purpose |
| --- | --- |
| edge direction | distinguish incoming and outgoing risk paths |
| edge type / relation id | encode transaction relation semantics |
| edge timestamp / relative time | support dynamic graph ordering |
| recent-neighbor sampling | emphasize recent temporal evidence |
| consistency-aware sampling | stabilize temporal neighborhoods |

## 5. DyRIFT-GNN Modules

`DyRIFT-GNN` is the full model built around TRGT.

| Component | Inference-time? | Role |
| --- | --- | --- |
| TRGT Backbone | yes | temporal-relational GNN representation |
| Target-Context Bridge | yes | fuses target-node context with GNN embedding |
| Drift Expert | yes | adapts target representation under temporal drift |
| Internal Risk Fusion | yes, dataset profile dependent | injects model-internal risk residuals |
| Cold-Start Residual | yes, EPP profile | compensates late/cold nodes in EPP |
| Prototype Memory | no | training-time representation regularizer |
| Pseudo-Contrastive Temporal Mining | no | training-time hard temporal contrastive mining |

The training-only methods shape the representation space during optimization. They are not used as a second inference branch.

## 6. Training Loop

`experiment/models/engine.py` owns the training loop.

Per epoch:

1. Build balanced target batches from training ids.
2. Sample temporal-relational subgraphs around batch targets.
3. Tensorize node features, relation ids, relative times, and target-context features.
4. Run TRGT and DyRIFT modules.
5. Compute supervised BCE loss plus enabled training-time regularizers.
6. Evaluate the validation split and save the best checkpoint by validation AUC.
7. Write `epoch_metrics.csv`, `epoch_metrics.jsonl`, `train.log`, and curve artifacts.

Important implementation detail: class imbalance is handled by balanced target sampling and positive weighting. The whole training set is not used as a dense full-batch loss every epoch.

## 7. Inference Loop

Inference uses the same graph context and the same single model checkpoint:

```text
target ids
  -> temporal neighbor sampling
  -> feature lookup and normalization
  -> TRGT forward pass
  -> DyRIFT module fusion
  -> sigmoid fraud probability
```

The saved prediction files contain:

| Field | Meaning |
| --- | --- |
| `node_ids` | evaluated target node ids |
| `y_true` | labels for the evaluated split, if available |
| `probability` | predicted fraud probability |

## 8. Mainline Vs Studies

| Route | Purpose | Can be thesis main result? |
| --- | --- | --- |
| accepted mainline | leakage-free three-dataset result | yes |
| comparisons | baseline reference under same feature cache | yes, as comparison |
| ablations | verify module contribution | yes, as ablation |
| progressive studies | show method construction trend | yes, as supplementary method table |
| XinYe phase1+phase2 studies | diagnose cross-phase distribution shift | no, diagnostic only |

The phase1+phase2 studies use additional phase2 labels and therefore are not used as the official leakage-free thesis mainline. They are useful for explaining phase drift and checkpoint-selection trade-offs.

## 9. Accepted Results And Artifacts

| Dataset | Val AUC | Epoch log |
| --- | ---: | --- |
| XinYe DGraph | 79.2851% | `experiment/outputs/training/models/dyrift_gnn/full_xinye_repro_v1/seed_42/epoch_metrics.csv` |
| Elliptic Transactions | 82.1329% | `experiment/outputs/elliptic_transactions/training/models/dyrift_gnn/probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1/seed_42/epoch_metrics.csv` |
| Elliptic++ Transactions | 82.1953% | `experiment/outputs/ellipticpp_transactions/training/models/dyrift_gnn/probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1/seed_42/epoch_metrics.csv` |

The full manifest is `docs/results/epoch_log_manifest.csv`.

Maintained reruns use `max_epochs=70` and `min_early_stop_epoch=30`; the policy table is `docs/results/experiment_epoch_policy.csv`. Existing saved epoch logs remain the observed artifact records.

## 10. Engineering Guarantees

The maintained codebase is organized around these guarantees:

| Guarantee | Implementation |
| --- | --- |
| single deployment path | final inference uses `dyrift_gnn` only |
| dataset isolation | no cross-dataset training or mixed feature cache |
| unified input contract | all datasets go through UTPM feature groups |
| split auditability | masks and prediction bundles are saved and audited |
| result traceability | every kept experiment has summary, epoch metrics, logs, and curves |
