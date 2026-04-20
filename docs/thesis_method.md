# Thesis Method

## Quick Links

- [Back to README](../README.md)
- [Experiment Table](thesis_experiments.md)
- [Official Result JSON](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/summary.json)
- [Backbone Ablation Report](../experiment/outputs/thesis_ablation/thesis_m7_v4_backbone_module_ablation/report.md)
- [Leakage Audit](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md)

## 1. Problem Setting

目标不是给三个数据集各做一套“特供策略”，而是在严格数据隔离前提下，用一套统一动态图异常检测架构完成训练、验证和推理。

统一约束：

- 数据集彼此隔离，不做跨数据集联合训练
- 各数据集各自预处理后，全部映射到同一特征契约 `utpm_unified`
- 主模型必须是动态图神经网络
- 二级决策层允许存在，但只能作为 GNN 主干的残差校正

## 2. Official Architecture

统一流程如下：

1. 各数据集构建自己的 `graph_gdata/phase_gdata` 与统一特征缓存
2. 根据各自 `recommended_split.json` 读取 `phase1_train / phase1_val / test_pool`
3. 用统一动态图 GNN 主干训练，当前 official 为 `m7_utpm`，现代化候选为 `m8_utgt`
4. 仅在当前数据集的 `phase1_train` 上训练 `graphprop residual` tree 分支
5. 用固定 logit 融合形成最终预测

这里“统一模型”的正确含义是：

- 统一输入契约
- 统一主模型族
- 统一训练协议
- 统一决策规则
- 但不混数据集

## 3. Main GNN Backbone

当前 official 主模型为 `m7_utpm`，统一现代化候选为 `m8_utgt`。

主干固定项：

- 共享输入契约：`utpm_unified`
- 共享目标上下文组：
  - `graph_time_detrend`
  - `neighbor_similarity`
  - `activation_early`
- 统一超参数：
  - hidden dim = `128`
  - relation dim = `32`
  - fanouts = `(15, 10)`
  - epochs = `8`

这个主干负责学习动态图中的时间漂移、邻域关系与行为上下文，是论文的主模型，而不是附属模型。

### 3.1 Backbone Modernization

为回应“不能只是旧 GraphSAGE 外挂一些模块”的问题，当前代码里新增了真正统一的候选主干 `m8_utgt`。

它和 `m7_utpm` 的关系不是“另一套实验分支”，而是：

- 同一个输入契约：`utpm_unified`
- 同一个训练协议：`phase1_train -> phase1_val -> test_pool`
- 同一组 thesis 模块：`prototype memory`、`pseudo-contrastive temporal mining`、`drift residual target context`
- 只替换主干内部的局部聚合算子

具体来说：

| Backbone | Local Aggregation | Role |
| --- | --- | --- |
| `m7_utpm` | GraphSAGE-style relation aggregation | 当前 official 主干 |
| `m8_utgt` | multi-head temporal relation attention | 现代化候选主干 |

因此这次重构的创新点不是“给不同数据集换不同 trick”，而是在统一合同下，把主干升级成更接近时序图 Transformer 的关系注意力结构。

## 4. Innovation Modules

当前 thesis mainline 里真正需要做消融的创新模块一共 5 个：

| Layer | Module | Role |
| --- | --- | --- |
| GNN backbone | `prototype memory` | 用时间桶原型约束稳定类别结构 |
| GNN backbone | `pseudo-contrastive temporal mining` | 增强时间漂移下的正负分离 |
| GNN backbone | `drift residual target context` | 对时间漂移和上下文偏移做残差适配 |
| Decision layer | `graphprop residual head` | 用泄露安全的 graphprop 分支纠正 GNN 局部排序误差 |
| Decision layer | `fixed logit fusion` | 保证 graphprop 只是残差校正，而不是重新训练一个 stacker |

这意味着：

- `utpm_unified` 是统一输入契约，不计入 ablation 模块数。
- 你的方法不是老式的“单纯 GraphSAGE/GAT + 手工特征”。
- 但也不应该夸成“提出了一个全新的基础图神经网络家族”。
- 更准确的表述是：这是一个 thesis-level 的统一动态图反欺诈架构，把现代时序 GNN 主干、原型记忆、伪对比约束和 graphprop 残差校正整合到了同一 leakage-safe 合同里。

当前官方 tri-dataset 主干消融的直接证据是：

- `pseudo-contrastive temporal mining` 去掉后宏平均从 `0.788895` 降到 `0.782712`，是最明确有效的主干模块。
- `prototype memory` 与 `drift residual target context` 在当前单种子 `phase1_val` 上更接近弱正则/上下文校准，数值变化都在 `0.002` 量级内。
- 决策层两项创新 `graphprop residual head + fixed logit fusion` 仍然是 official result 大幅跃升的主要来源。

## 5. Why The Second Column Is Stronger But Not The Official Main Result

这个问题必须说清楚，因为现在 GitHub 上最容易被误读的地方就在这里。

结果上：

- `Secondary-only (non-GNN graphprop)` 在 Elliptic 和 Elliptic++ 上确实比 `Official GNN-primary Blend` 更高。
- 三数据集宏平均里，`Secondary-only` 也是当前数值最高的一列，达到 `0.908981`。

但它不是 official main result，原因也很明确：

1. 第二列不是第二个 GNN，而是非 GNN 的 graphprop tree 分支。
2. 你的论文硬约束是“主模型必须是动态图 GNN”，所以不能把树模型直接写成主模型。
3. 因此，第二列应该被定义为“强上界 secondary branch / ablation”，而不是被包装成论文主干。
4. 正式论文主线使用 `GNN-primary Blend`，因为它满足“GNN 为主、graphprop 为残差校正”的设计契约。

换句话说：

- 如果只追数值，第二列更强。
- 如果要满足论文主模型定义，official 结果就必须是第三列。

## 6. Effective Gain

`UTPM Dynamic GNN + Graphprop Residual Correction`

在官方主线下，纯 GNN 到 official blend 的提升如下：

| Dataset | Pure `m7_utpm` | Official GNN-primary Blend | Gain |
| --- | ---: | ---: | ---: |
| XinYe DGraph | 0.776439 | 0.795293 | +0.018854 |
| Elliptic | 0.812635 | 0.949436 | +0.136801 |
| Elliptic++ | 0.777611 | 0.946584 | +0.168973 |

因此，graphprop 分支不是替代 GNN，而是官方主架构中的残差校正层。

## 7. Why This Is Not Hard Leakage

本项目采用的是“无硬泄露”标准：

- `train/val/test_pool/external` 节点集合互不重叠
- 特征归一化只在 `phase1_train` 上拟合
- secondary 只在 `phase1_train` 上训练
- `val` 只参与评估与固定权重选择，不参与二级模型拟合
- 不存在跨数据集特征缓存、预测缓存或训练样本复用

正式审计文件：

- [leakage_audit.md](../experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md)

## 8. Repro Commands

构建缓存：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py build_features --phase both
```

训练纯 GNN 官方主干：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py \
  --suite-name thesis_m7_v4_unified_e8 \
  --model m7_utpm \
  --preset utpm_temporal_shift_v4 \
  --feature-profile utpm_unified \
  --epochs 8 \
  --seeds 42
```

训练 transformer-style 候选主干：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py \
  --suite-name thesis_m8_utgt_e8 \
  --model m8_utgt \
  --preset utgt_temporal_shift_v1 \
  --feature-profile utpm_unified \
  --epochs 8 \
  --seeds 42
```

训练官方 hybrid 套件：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_hybrid_suite.py \
  --suite-name thesis_m7_v4_graphpropblend082 \
  --blend-alpha 0.82
```

训练官方主干三模块消融并导出可画图汇总：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_backbone_ablation.py \
  --skip-existing
```

生成硬泄露审计：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/audit_thesis_leakage.py
```
