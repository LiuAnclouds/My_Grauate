# Thesis Method

## 1. Problem Setting

目标是用一套统一的动态图异常检测架构，在三个金融/反洗钱图数据集上完成训练、验证和推理，而不是为每个数据集单独发明一套模型。

统一约束：

- 数据集彼此隔离，不做跨数据集联合训练
- 各数据集经过各自预处理后，全部映射到同一特征契约 `utpm_unified`
- 主模型必须是动态图神经网络
- 二级决策层允许存在，但只能作为 GNN 主干的残差校正

## 2. Unified Pipeline

统一流程如下：

1. 各数据集各自构建 `graph_gdata/phase_gdata` 与统一特征缓存
2. 根据各自 `recommended_split.json` 读取 `phase1_train / phase1_val / test_pool`
3. 用 `m7_utpm` 训练动态图 GNN 主干
4. 在相同数据集的 `phase1_train` 上训练 graphprop residual XGBoost
5. 用固定 logit 融合得到最终预测

统一而不混训，才是这里“统一模型”的正确定义。

## 3. Main Model

主模型为 `m7_utpm`。

核心点：

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

这个主干负责学习动态图中的时间漂移、邻域关系与行为上下文，是论文主模型。

## 4. Effective Innovation

最终主线的有效创新不在于“再加一套完全不同的 teacher”，而在于：

- 用统一 GNN 主干负责动态图表征
- 用 leakage-safe graph propagation residual head 纠正主干在时间漂移下的局部排序误差
- 用固定 logit 融合而不是再训练一层会吃掉验证集的 stacker

可以把它概括为：

`UTPM Dynamic GNN + Graphprop Residual Correction`

为什么这个创新是有效的：

- XinYe 上，纯 `m7_utpm` 为 `0.777741`，最终 hybrid 提升到 `0.795293`
- Elliptic 上，从 `0.801914` 提升到 `0.949436`
- Elliptic++ 上，从 `0.778276` 提升到 `0.946584`

因此，graphprop 分支不是替代 GNN，而是作为统一主架构中的残差校正模块。

## 5. Why This Is Not Hard Leakage

本项目最终采用的是“无硬泄露”标准：

- `train/val/test_pool/external` 节点集合互不重叠
- 特征归一化只在 `phase1_train` 上拟合
- secondary 只在 `phase1_train` 上训练
- `val` 只参与评估和固定权重选择，不参与二级模型拟合
- 不存在跨数据集特征缓存、预测缓存或训练样本复用

正式审计文件：

- `experiment/outputs/thesis_suite/thesis_m7_v4_graphpropblend082/leakage_audit.md`

## 6. Repro Commands

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

训练官方 hybrid 套件：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_hybrid_suite.py \
  --suite-name thesis_m7_v4_graphpropblend082 \
  --blend-alpha 0.82
```

生成硬泄露审计：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/audit_thesis_leakage.py
```
