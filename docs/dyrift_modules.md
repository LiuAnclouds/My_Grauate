# DyRIFT-GNN Modules

This document explains the components used by the final `DyRIFT-GNN` route and, more importantly, separates real model modules from training-time methods.

## 1. Module vs Method Map

| Item | Type | Code | Final Scope | Evidence |
| --- | --- | --- | --- | --- |
| TRGT backbone | model backbone | [backbone.py](../experiment/models/modules/backbone.py) | all datasets | main architecture |
| Target-Context Bridge | model module | [bridge.py](../experiment/models/modules/bridge.py) | all datasets | main ablation: `-0.020582` macro |
| Drift Expert | model module | [engine.py](../experiment/models/engine.py) | all datasets | main ablation: `-0.018917` macro |
| Prototype Memory | training-time method | [memory.py](../experiment/models/modules/memory.py) | all datasets | main ablation: `-0.001070` macro |
| Pseudo-Contrastive Temporal Mining | training-time method | [engine.py](../experiment/models/engine.py) | all datasets | main ablation: `-0.011588` macro |
| Internal Risk Fusion | model module | [backbone.py](../experiment/models/modules/backbone.py) | XinYe and EPP final profiles | dataset-conditional |
| Cold-Start Residual | model module | [engine.py](../experiment/models/engine.py) | EPP final profile | dataset-conditional |

主消融只覆盖三个最终 profile 都共同启用的部分：

- Target-Context Bridge
- Drift Expert
- Prototype Memory
- Pseudo-Contrastive Temporal Mining

`Internal Risk Fusion` 和 `Cold-Start Residual` 不放进主消融表，是因为它们不是三数据集共同启用的统一组件。

## 2. TRGT Backbone

`TRGT` 是 `DyRIFT-GNN` 的动态图主干：

- relation-aware，因为边有关系嵌入
- time-aware，因为边时间和目标时间位置参与编码
- target-specific，因为注意力按目标节点聚合

关键类：

- `TRGTTemporalRelationAttentionBlock`
- `TRGTInternalRiskEncoder`

## 3. Target-Context Bridge

Bridge 把目标节点级上下文特征和主干 GNN 表示在模型内部融合。

典型上下文组：

- `graph_stats`
- `graph_time_detrend`
- `neighbor_similarity`
- `activation_early`

它属于推理期模型结构，而不是额外的第二模型。

## 4. Drift Expert

金融图的行为会随时间漂移。`Drift Expert` 根据目标时间位置和上下文特征做时间漂移适配。

从消融结果看，它是宏平均掉点第二大的共享组件，尤其影响 ET 的时间泛化能力。

## 5. Prototype Memory

`Prototype Memory` 是训练期正则器：

- 保存类别原型
- 拉近同类 embedding
- 拉开异类 embedding

它的作用是表示稳定化，不是推理期第二分类头。

## 6. Pseudo-Contrastive Temporal Mining

`Pseudo-Contrastive` 是训练期难样本挖掘方法：

- 从无标签池上用当前模型分数做高置信伪划分
- 构造时间均衡的对比损失
- 强化时间漂移下的异常和正常分离

它不使用验证和测试标签。

## 7. Internal Risk Fusion

`Internal Risk Fusion` 在模型内部学习风险差分信号，例如：

- 入向和出向差异
- 短窗和长窗差异
- 一跳和二跳风险差异
- 方向不对称

这些信号以残差形式并回 GNN 表示，不需要外接第二个分类器。

## 8. Cold-Start Residual

`Cold-Start Residual` 只在最终 EPP profile 启用，用来补偿晚期低支持度节点的消息不足。

对应超参数：

```json
"cold_start_residual_strength": 0.35
```

它仍然属于同一架构内部的模块开关，不是换模型。

## 9. Deployment Path

推理时始终只有：

`features + graph -> DyRIFT-GNN -> fraud probability`

不会额外调用外部树模型或第二阶段融合器。 
