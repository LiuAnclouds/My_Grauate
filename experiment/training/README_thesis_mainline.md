# Thesis Mainline

## Quick Links

- [Repository README](../../README.md)
- [Method Overview](../../docs/thesis_method.md)
- [Experiment Table](../../docs/thesis_experiments.md)
- [Final Pure-GNN Summary](../outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json)
- [Final Pure-GNN Audit](../outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md)
- [Final Metrics CSV](../../docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv)
- [Epoch Metrics CSV](../../docs/results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv)
- [Dataset Hparams](configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json)

## Recommended Surface

当前推荐主线已经固定为一条统一的 **deployable pure-GNN DyRIFT-GNN**：

- 统一方法名：`DyRIFT-GNN`
- 统一 backbone：`TRGT`
- 统一运行入口：`dyrift_gnn`
- 统一纯 GNN preset：`dyrift_trgt_deploy_v1`
- 统一推理路径：single pure-GNN path
- 统一论文主结果：[thesis_dyrift_gnn_trgt_deploy_pure_v1](../outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json)
- 统一硬泄露审计：[leakage_audit.md](../outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md)

## Unified Architecture, Separate Tuning

thesis runner 支持一份 manifest 加三个独立 JSON 管理数据集级超参数：

- 架构固定：`DyRIFT-GNN` / `TRGT` / dataset-local UTPM feature schema / single-model inference
- 允许分别调：`attr_proj_dim`、`hidden_dim`、`rel_dim`、`fanouts`、`attention_num_heads`、`batch_size`、`epochs`、`learning_rate`、`weight_decay`、`dropout`、`graph_config_overrides`
- 主线 manifest：`configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json`
- XinYe profile：`configs/dyrift_gnn/xinye_dgraph.json`
- ET profile：`configs/dyrift_gnn/elliptic_transactions.json`
- EPP profile：`configs/dyrift_gnn/ellipticpp_transactions.json`

如果不传 `--dataset-hparams`，suite 会退回全局统一超参。

## What The Current Model Actually Is

| Part | Current Choice | Meaning |
| --- | --- | --- |
| Input | UTPM unified schema family | 三个数据集共享统一输入语义 |
| Full model | `DyRIFT-GNN` | 动态风险感知反欺诈图神经网络 |
| Backbone | `TRGT` | Temporal-Relational Graph Transformer，多头时序关系注意力主干 |
| GNN modules | prototype memory / pseudo-contrastive temporal mining / temporal-normality bridge / drift-expert adaptation / internal causal risk fusion / context-conditioned cold-start residual | 论文核心创新 |
| Final decision | single pure-GNN output | 无外部 residual、无外部 tree head |

这意味着：

- 训练和推理是一条路。
- 没有第二个模型参与最终部署。
- 数据集差异只体现在合理超参数和特征容量上。

## Recommended Commands

### 1. Build Unified Features

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py \
  build_features \
  --phase both
```

### 2. Run Final Pure-GNN Suite

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py \
  --suite-name thesis_dyrift_gnn_trgt_deploy_pure_v1 \
  --model dyrift_gnn \
  --preset dyrift_trgt_deploy_v1 \
  --feature-profile utpm_shift_enhanced \
  --dataset-hparams experiment/training/configs/thesis_dataset_hparams.dyrift_gnn_trgt_deploy_pure_v1.json \
  --seeds 42 \
  --skip-existing
```

输出：

- `experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json`

### 3. Audit Hard Leakage For The Final Suite

```bash
conda run -n Graph --no-capture-output python3 experiment/training/audit_thesis_leakage.py \
  --suite-summary experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json
```

输出：

- `experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md`
- `experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json`

## Current Recommended Metrics

当前最终主结果验证集 AUC：

- XinYe: `0.7904545115949035`
- Elliptic: `0.821329087693758`
- Elliptic++: `0.821953270227715`
- Macro: `0.8112456231721256`

从这组数可以直接说明：

- 统一纯 GNN 主线已经稳定超过历史强 GNN 的宏平均。
- 当前主线是可部署的单模型路径。
- 论文叙事可以完全围绕纯 GNN 展开，不需要再解释旧的双路推理方案。
- 三数据集训练日志和逐 epoch 指标已经汇总到 `../../docs/results/` 下的两个 CSV，可直接用于画图和制表。

## File Hotspots

后续如果继续只改 thesis 主线，优先改这些文件：

- `experiment/training/run_thesis_mainline.py`
- `experiment/training/run_thesis_suite.py`
- `experiment/training/dyrift_training.py`
- `experiment/training/dyrift_model.py`
- `experiment/training/trgt_backbone.py`
- `experiment/training/thesis_presets.py`
- `experiment/training/thesis_contract.py`
- `experiment/training/audit_thesis_leakage.py`

## Code Names

- `TRGTExperiment`: TRGT backbone training base for temporal-relation attention GNN runs.
- `DyRIFTGNNModel`: final DyRIFT-GNN model facade.
- `build_dyrift_gnn_model`: final model factory used by the trainer.
- `DyRIFTGNNExperiment`: final training wrapper for `dyrift_gnn`.
- `TRGTTemporalRelationAttentionBlock`: backbone attention block.
- `TRGTInternalRiskEncoder`: internal multi-scale risk encoder.

Older aliases such as `TemporalRelationGraphTransformerExperiment`, `DyRIFTGraphModel`, and `build_dyrift_model` are kept only for backward compatibility.
