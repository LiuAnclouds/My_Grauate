# Thesis Method

## Quick Links

- [Back to README](../README.md)
- [Experiment Table](thesis_experiments.md)
- [Recommended Result JSON](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/summary.json)
- [Pure Teacher Backbone JSON](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_e8_s42_v1/summary.json)
- [Recommended Leakage Audit](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/leakage_audit.md)

## 1. Problem Setting

目标不是给三个数据集各做一套“特供策略”，而是在严格数据隔离前提下，用一套统一动态图异常检测架构完成训练、验证和推理。

统一约束：

- 数据集彼此隔离，不做跨数据集联合训练
- 各数据集各自预处理后，全部映射到同一特征契约 `utpm_unified`
- 主模型必须是动态图神经网络
- 二级决策层允许存在，但只能作为 GNN 主干的残差校正
- 不允许把 `secondary-only` 包装成论文主模型

## 2. Final Recommended Architecture

当前推荐主线如下：

1. 各数据集分别构建自己的图缓存与特征缓存
2. 所有数据都映射到统一输入契约 `utpm_unified`
3. 使用统一动态图主干 `m8_utgt`，preset 为 `utgt_temporal_shift_teacher_v1`
4. 训练期间读取当前数据集、`phase1_train` 拟合得到的 graphprop logits，作为只读 teacher 信号
5. 推理阶段再使用同一 graphprop 家族产生的 secondary logit，做固定残差校正
6. 通过 `alpha=0.4999` 的 logit-space 固定融合得到最终预测

也就是说，推荐主线不是“先树模型、后 GNN”，也不是“两套模型并列投票”，而是：

`统一输入契约 + teacher-guided dynamic GNN + leakage-safe residual correction`

## 3. What The Run Names Actually Mean

这几个名词一定要分清：

| Name | Meaning | Is It The Main Model? |
| --- | --- | --- |
| `thesis_m8_utgt_teacher_e8_s42_v1` | 纯 GNN 主干结果，teacher 只在训练期提供辅助信号 | 是 |
| `secondary-only` | 单独使用 graphprop 分支做预测 | 否 |
| `thesis_m8_utgt_teacher_gnnprimary04999` | 最终论文主结果，`50.01% GNN + 49.99% secondary` | 是 |
| `thesis_m8_utgt_graphpropblend091` | 只追 AUC 的 appendix 结果，`9% GNN + 91% secondary` | 否 |

其中 `teacher` 的准确含义是：

- teacher 预测来自当前数据集、`phase1_train` 拟合出的 graphprop 模型
- 这些预测在 GNN 训练时只读加载，不重新在验证集标签上拟合
- teacher 参与的是 target-context、rank distill、hard negative guidance
- teacher 不是第二套主模型，也不是额外的数据集分支

## 4. Backbone And Decision Formula

推荐主线的主干与决策层如下：

| Layer | Current Choice | Role |
| --- | --- | --- |
| Input contract | `utpm_unified` | 三个数据集统一输入空间 |
| Main GNN backbone | `m8_utgt` | 多头时序关系注意力主干 |
| Shared backbone modules | `prototype memory` + `pseudo-contrastive temporal mining` + `drift residual target context` | 处理时间漂移、类别结构和上下文偏移 |
| Teacher guidance | dataset-local graphprop logits | 训练期辅助目标与难负样本引导 |
| Residual decision branch | leakage-safe `graphprop + XGBoost` | 推理期残差校正 |
| Final output | fixed logit fusion | 保持 GNN-primary |

最终公式是：

`p_final = sigmoid(0.5001 * logit(p_gnn) + 0.4999 * logit(p_secondary))`

这里 `alpha=0.4999` 表示 secondary 权重，而不是 GNN 权重。

## 5. Innovation Groups And Evidence

当前论文最稳妥的创新叙述应该按“创新组”来讲，而不是按零散超参数来讲。

| Innovation Group | Evidence | Current Reading |
| --- | --- | --- |
| Temporal relation attention backbone | `m8_utgt` 在统一合同下替换 `m7_utpm` 的局部聚合器 | 主干现代化本身成立 |
| Prototype memory | legacy shared-module ablation | 更像结构正则与类别稳定器 |
| Pseudo-contrastive temporal mining | 去掉后宏平均下降 `0.006182` | 是最明确有效的共享主干模块 |
| Drift residual target context | legacy shared-module ablation | 更偏稳健性与上下文校准 |
| Teacher-guided temporal normality bridge | pure `m8_utgt` -> teacher pure `m8_utgt`，宏平均 `+0.016758` | 说明 teacher guidance 有效 |
| Graphprop residual correction + fixed fusion | teacher pure `m8` -> recommended blend，宏平均 `+0.075912` | 是最终主结果跃升的关键模块 |

这里需要实话实说：

- `m8_utgt` 单独替换主干并不会自动变强，pure `m8_utgt` 宏平均只有 `0.767140`
- 真正有效的是 `UTGT + teacher guidance + residual correction` 这个统一组合
- 所以论文创新点不能写成“attention 一上去就全赢了”，而应该写成“在统一动态图合同下，把 attention 主干、teacher-guided temporal bridge 与 leakage-safe residual correction 有机整合起来”

## 6. Why Secondary-only Is Higher On ET/EPP

这个问题必须直接回答：

1. `secondary-only` 不是 GNN，而是 graphprop tree 分支
2. ET 和 EPP 的传播结构对 graphprop 更友好，所以单列分数会更高
3. 但它不满足“主模型必须是动态图 GNN”的论文硬约束
4. 因此它只能作为上界分支、ablation 或 appendix，不能直接当论文主结果

换句话说：

- 如果只追验证集 AUC，可以展示 `alpha=0.91` 或 `secondary-only`
- 如果要同时满足“统一架构 + GNN 为主 + 可答辩”，就应该选择 `alpha=0.4999` 的推荐主线

## 7. Why This Is Still One Unified Model

有人会把现在这套方法误解成“两套模型”。这个说法不成立，原因有 4 个：

1. 三个数据集都映射到同一输入契约 `utpm_unified`
2. 三个数据集都使用同一 GNN 主干家族 `m8_utgt`
3. teacher、secondary、blend 的规则在三个数据集上完全一致
4. 所有训练与推理都严格在各自数据集内部完成

因此这里的“统一模型”含义是：

- 统一输入
- 统一主干
- 统一训练协议
- 统一决策规则
- 但不混数据集

## 8. Hard-Leakage Guardrails

推荐主线已经重新审计过：

- [leakage_audit.md](../experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/leakage_audit.md)

审计结论：

- `train / val / test_pool / external` 四个 split 两两不重叠
- secondary 训练样本严格属于当前数据集的 `phase1_train`
- hybrid 与 secondary 的验证 bundle 都与官方 `phase1_val` 完全对齐
- 所有路径都在数据集作用域内，未复用跨数据集缓存或预测
- teacher 信号来自数据集内、只读的训练期输出，不回流验证标签

## 9. Repro Commands

构建特征缓存：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py build_features --phase both
```

运行纯 teacher-guided GNN 套件：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_suite.py \
  --suite-name thesis_m8_utgt_teacher_e8_s42_v1 \
  --model m8_utgt \
  --preset utgt_temporal_shift_teacher_v1 \
  --feature-profile utpm_unified \
  --epochs 8 \
  --seeds 42 \
  --skip-existing
```

运行推荐论文主结果：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_hybrid_suite.py \
  --suite-name thesis_m8_utgt_teacher_gnnprimary04999 \
  --base-model m8_utgt \
  --base-run-name-template thesis_m8_utgt_teacher_e8_s42_v1_{dataset_short} \
  --blend-alpha 0.4999 \
  --skip-existing
```

运行只追 AUC 的 appendix 结果：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_hybrid_suite.py \
  --suite-name thesis_m8_utgt_graphpropblend091 \
  --base-model m8_utgt \
  --base-run-name-template thesis_m8_utgt_e8_s42_v1_{dataset_short} \
  --blend-alpha 0.91 \
  --skip-existing
```

生成硬泄露审计：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/audit_thesis_leakage.py \
  --suite-summary experiment/outputs/thesis_suite/thesis_m8_utgt_teacher_gnnprimary04999/summary.json
```
