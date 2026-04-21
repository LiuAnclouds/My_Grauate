# Thesis Mainline

## Quick Links

- [Repository README](../../README.md)
- [Method Overview](../../docs/thesis_method.md)
- [Experiment Table](../../docs/thesis_experiments.md)
- [Recommended Result JSON](../outputs/thesis_suite/thesis_m8_utgt_teacher_hpsearch1_gnnprimary04999/summary.json)
- [Pure Teacher Backbone JSON](../outputs/thesis_suite/thesis_m8_utgt_teacher_hpsearch1_e8_s42_v1/summary.json)
- [Recommended Leakage Audit](../outputs/thesis_suite/thesis_m8_utgt_teacher_hpsearch1_gnnprimary04999/leakage_audit.md)
- [Dataset Hparam Profile](configs/thesis_dataset_hparams.search_v1.json)

## Recommended Surface

当前推荐主线只保留一条统一路径：

- 统一输入契约：`utpm_unified`
- 统一主干家族：`m8_utgt`
- 统一 teacher preset：`utgt_temporal_shift_teacher_v1`
- 统一 residual family：`graphprop + XGBoost`
- 统一融合规则：`alpha=0.4999`

这里的 `alpha` 是 secondary 权重：

- `alpha=0.4999` = `50.01% GNN + 49.99% secondary`
- `alpha=0.91` = `9% GNN + 91% secondary`

因此：

- `thesis_m8_utgt_teacher_hpsearch1_gnnprimary04999` 是论文主结果
- `thesis_m8_utgt_graphpropblend091` 只是 appendix

## Unified Architecture, Separate Tuning

现在 thesis runner 支持一份统一 JSON profile 管理数据集级超参：

- 架构固定：`m8_utgt` / `utpm_unified` / dataset-local teacher / GNN-primary hybrid
- 允许分别调：`attr_proj_dim`、`hidden_dim`、`rel_dim`、`fanouts`、`batch_size`、`epochs`、`learning_rate`、`weight_decay`、`dropout`、`graph_config_overrides`、`blend_alpha`
- 推荐 profile：`configs/thesis_dataset_hparams.search_v1.json`

如果不传 `--dataset-hparams`，suite 会退回全局统一超参。

## What The Current Model Actually Is

把推荐主线拆开看：

| Part | Current Choice | Meaning |
| --- | --- | --- |
| Input | `utpm_unified` | 三个数据集统一输入契约 |
| Main GNN | `m8_utgt` | 多头时序关系注意力主干 |
| Shared backbone modules | `prototype memory` / `pseudo-contrastive temporal mining` / `drift residual target context` | GNN 主干内部创新 |
| Teacher guidance | dataset-local graphprop logits | 训练期辅助上下文与蒸馏信号 |
| Secondary branch | `graphprop + XGBoost` | 推理期 residual correction |
| Final decision | fixed logit fusion | 构成推荐主结果 |

这意味着：

- `secondary-only` 不是第二个 GNN
- `teacher` 不是另一套独立主模型
- 推荐结果仍然是一套统一的 GNN 主线，只是在训练期和决策层用到了数据集内、泄露安全的辅助分支

## Recommended Commands

### 1. Build Unified Features

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py \
  build_features \
  --phase both
```

### 2. Run Pure Teacher-guided GNN Suite

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py \
  --suite-name thesis_m8_utgt_teacher_hpsearch1_e8_s42_v1 \
  --model m8_utgt \
  --preset utgt_temporal_shift_teacher_v1 \
  --feature-profile utpm_unified \
  --dataset-hparams experiment/training/configs/thesis_dataset_hparams.search_v1.json \
  --epochs 8 \
  --seeds 42 \
  --skip-existing
```

输出：

- `experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_hpsearch1_e8_s42_v1/summary.json`

### 3. Run Recommended GNN-primary Blend

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_hybrid_suite.py \
  --suite-name thesis_m8_utgt_teacher_hpsearch1_gnnprimary04999 \
  --base-model m8_utgt \
  --base-run-name-template thesis_m8_utgt_teacher_hpsearch1_e8_s42_v1_{dataset_short} \
  --dataset-hparams experiment/training/configs/thesis_dataset_hparams.search_v1.json \
  --blend-alpha 0.4999 \
  --skip-existing
```

输出：

- `experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_hpsearch1_gnnprimary04999/summary.json`

### 4. Run AUC-first Appendix

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_hybrid_suite.py \
  --suite-name thesis_m8_utgt_graphpropblend091 \
  --base-model m8_utgt \
  --base-run-name-template thesis_m8_utgt_e8_s42_v1_{dataset_short} \
  --dataset-hparams experiment/training/configs/thesis_dataset_hparams.search_v1.json \
  --blend-alpha 0.91 \
  --skip-existing
```

输出：

- `experiment/outputs/thesis_suite/thesis_m8_utgt_graphpropblend091/summary.json`

### 5. Audit Hard Leakage

```bash
conda run -n Graph --no-capture-output python3 experiment/training/audit_thesis_leakage.py \
  --suite-summary experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_hpsearch1_gnnprimary04999/summary.json
```

输出：

- `experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_hpsearch1_gnnprimary04999/leakage_audit.md`
- `experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_hpsearch1_gnnprimary04999/leakage_audit.json`

## Current Recommended Metrics

推荐主线当前验证集 AUC：

- XinYe: `0.7960912312634558`
- Elliptic: `0.8976530494361453`
- Elliptic++: `0.8991860885961518`

纯 teacher-guided GNN：

- XinYe: `0.7923906331620709`
- Elliptic: `0.784233801549338`
- Elliptic++: `0.7837331225324542`

从这两组数可以直接说明：

- teacher guidance 本身有效
- fixed logit residual correction 也有效
- 最终主线确实保持了 GNN 为主，同时把 XinYe 推到 `0.7960+`

## Legacy Supporting Experiments

以下内容保留为支撑性材料，不再作为论文最终主结论：

- legacy `m7_utpm` pure backbone
- legacy `m7` module ablations
- old `alpha=0.82` hybrid

对应产物：

- `experiment/outputs/thesis_suite/thesis_m7_v4_unified_e8/summary.json`
- `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json`
- `experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md`

如果还需要补共享模块消融图，可以直接使用：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_backbone_ablation.py \
  --skip-existing
```

## File Hotspots

后续如果继续只改 thesis 主线，优先改这些文件：

- `experiment/training/run_thesis_mainline.py`
- `experiment/training/run_thesis_suite.py`
- `experiment/training/run_thesis_hybrid_suite.py`
- `experiment/training/thesis_hparam_profiles.py`
- `experiment/training/run_thesis_hybrid_blend.py`
- `experiment/training/thesis_contract.py`
- `experiment/training/thesis_presets.py`
- `experiment/training/thesis_runtime.py`
- `experiment/training/prediction_signal_utils.py`
