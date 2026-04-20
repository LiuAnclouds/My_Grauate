# Thesis Mainline

## Quick Links

- [Repository README](../../README.md)
- [Method Overview](../../docs/thesis_method.md)
- [Experiment Table](../../docs/thesis_experiments.md)
- [Official Result JSON](../outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json)
- [Backbone Ablation Report](../outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md)

## Official Surface

毕业设计的正式实验主线只保留一条统一路径，不再接受 teacher 依赖或“一个数据集一套策略”的分叉：

- 纯 GNN 主干:
  - `experiment/training/run_thesis_mainline.py`
  - `experiment/training/run_thesis_suite.py`
- 官方 recipe 入口:
  - `experiment/training/run_thesis_recipe.py`
- GNN 主导的统一混合决策层:
  - `experiment/training/run_thesis_graphprop_secondary.py`
  - `experiment/training/run_thesis_hybrid_blend.py`
  - `experiment/training/run_thesis_hybrid_suite.py`
- 单一真相源:
  - `experiment/training/thesis_contract.py`

官方配置固定为：

- baseline:
  - `m5_temporal_graphsage` + `unified_baseline`
- thesis backbone:
  - `m7_utpm` + `utpm_temporal_shift_v4`
- official final decision layer:
  - fixed logit blend with `alpha=0.82`
  - secondary model: leakage-safe `phase1_train` graphprop XGBoost

额外说明：

- `secondary-only` 那一列不是第二个 GNN，而是非 GNN 的 graphprop 分支。
- 如果只追 `val_auc`，它在 Elliptic / Elliptic++ 上更强。
- 如果要维持“GNN 是论文主模型”的主线，official 结果就必须使用 `GNN-primary blend`。

## Unified Design

三个数据集都必须遵守同一实验合同：

- 同一输入契约:
  - `utpm_unified`
- 同一主模型族:
  - `m7_utpm`
- 同一训练协议:
  - `train -> val -> test_pool`
- 同一混合决策规则:
  - `GNN logit * 0.18 + graphprop logit * 0.82`

这里的“统一模型”含义是：

- 结构统一
- 训练协议统一
- 特征契约统一
- 数据集彼此隔离

不是把三个数据集混在一起做联合训练。

## Leakage Guardrails

必须保持以下约束：

- 特征归一化只用当前数据集的 `phase1_train` 节点拟合。
- 图标签上下文只允许使用时间阈值之前的可见监督，不能把未来标签回流到训练侧。
- `val` 和 `test_pool` 只做推理与评估，不参与模型或二级决策层拟合。
- hybrid 的二级模型只在当前数据集的 `phase1_train` 上训练。
- 每个数据集单独使用自己的缓存目录、模型目录、预测目录。
- 硬泄露审计文件位于 `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md`。

## Commands

- build feature cache:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py build_features --phase both`
- run unified GNN backbone:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py train --model m7_utpm --preset utpm_temporal_shift_v4 --run-name thesis_xy_m7_v4_unified_s42_e8 --device cuda --epochs 8 --seeds 42`
- run pure-GNN tri-dataset suite:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py --suite-name thesis_m7_v4_unified_e8 --model m7_utpm --preset utpm_temporal_shift_v4 --feature-profile utpm_unified --epochs 8 --seeds 42`
- run official tri-dataset hybrid suite:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_hybrid_suite.py --suite-name thesis_m7_v4_graphpropblend082 --blend-alpha 0.82`
- inspect official thesis recipe:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_recipe.py show --dataset xinye_dgraph --recipe thesis_m7_utpm`

## Ablation Recipes

主干内部 3 个核心创新模块都支持通过 `--graph-config-override` 做 official tri-dataset ablation。

一键编排并导出可画图汇总：

- `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_backbone_ablation.py --skip-existing`

聚合输出目录：

- `experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/`
  - `report.md`
  - `results_long.csv`
  - `results_macro.csv`
  - `results.json`

- remove prototype memory:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py --suite-name thesis_m7_v4_ablate_noprototype --model m7_utpm --preset utpm_temporal_shift_v4 --feature-profile utpm_unified --epochs 8 --seeds 42 --graph-config-override prototype_loss_weight=0.0 --graph-config-override prototype_neighbor_blend=0.0 --graph-config-override prototype_global_blend=0.0 --graph-config-override prototype_consistency_weight=0.0 --graph-config-override prototype_separation_weight=0.0`
- remove pseudo-contrastive temporal mining:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py --suite-name thesis_m7_v4_ablate_nopseudocontrast --model m7_utpm --preset utpm_temporal_shift_v4 --feature-profile utpm_unified --epochs 8 --seeds 42 --graph-config-override pseudo_contrastive_weight=0.0`
- remove drift residual target context:
  - `conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py --suite-name thesis_m7_v4_ablate_nodriftresidual --model m7_utpm --preset utpm_temporal_shift_v4 --feature-profile utpm_unified --epochs 8 --seeds 42 --graph-config-override target_context_fusion=none --graph-config-override target_time_adapter_strength=0.0 --graph-config-override normal_bucket_align_weight=0.0 --graph-config-override context_residual_clip=0.0 --graph-config-override context_residual_budget=0.0 --graph-config-override context_residual_budget_weight=0.0 --graph-config-override context_residual_budget_min_weight=0.0`

决策层 ablation 已经有现成结果：

- pure backbone:
  - `thesis_m7_v4_unified_e8`
- weak hybrid:
  - `thesis_m7_v4_xgbblend035`
- official hybrid:
  - `thesis_m7_v4_graphpropblend082`

## Current Result

官方最终套跑结果位于：

- `experiment/outputs/thesis_suite/thesis_m7_v4_xgbblend035/summary.json`
  - 历史版本
- `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json`
  - 当前 official

当前验证集 AUC：

- XinYe:
  - `0.7952929882597335`
- Elliptic:
  - `0.949435862593804`
- Elliptic++:
  - `0.9465836685331215`

当前主干三模块 official tri-dataset ablation：

- official backbone:
  - macro `0.788895`
- no prototype memory:
  - macro `0.789344`
- no pseudo-contrastive mining:
  - macro `0.782712`
- no drift residual context:
  - macro `0.790898`

结论必须写清楚：

- `secondary-only` 数值更强，不应该被隐藏。
- 但它不是论文主模型，因为它不是 GNN。
- official thesis mainline 仍然以 `m7_utpm` 为主，只把 graphprop 当作 residual correction。
- 当前单种子 `phase1_val` 下，主干内部最明确有效的是 `pseudo-contrastive temporal mining`。

## Legacy Boundary

以下内容只保留为历史比较材料，不再定义毕业设计主结论：

- `experiment/training/run_training.py`
- `experiment/training/run_xgb_*`
- 旧的 OOF / stack / teacher / context bridge 分支
- 旧 recipe 组合与探索性 preset

后续如果继续改 thesis 主线，优先只改：

- `experiment/training/run_thesis_mainline.py`
- `experiment/training/run_thesis_suite.py`
- `experiment/training/run_thesis_recipe.py`
- `experiment/training/run_thesis_hybrid_blend.py`
- `experiment/training/run_thesis_hybrid_suite.py`
- `experiment/training/thesis_contract.py`
- `experiment/training/thesis_presets.py`
- `experiment/training/thesis_runtime.py`
