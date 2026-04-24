# DyRIFT-GNN Method

## 1. Identity

Full model name:

- `Dynamic Risk-Informed Fraud Graph Neural Network`
- short name: `DyRIFT-GNN`

Backbone name:

- `Temporal-Relational Graph Transformer`
- short name: `TRGT`

Runtime id:

- `dyrift_gnn`

论文里的方法名是 `DyRIFT-GNN`，工程里的运行入口名是 `dyrift_gnn`。

## 2. End-to-End Route

正式部署路径：

`raw dataset -> dataset-local preprocessing -> UTPM contract -> TRGT -> DyRIFT modules -> fraud probability`

这条路线是纯 GNN：

- 推理期没有外部分类器
- 推理期没有 teacher 分支
- 推理期没有第二阶段融合模型

## 3. Unified Input Contract

三个数据集原始 schema 不同，但最终都被映射到同一语义输入族：

- node attribute statistics
- graph structural statistics
- temporal activity statistics
- relation-aware interaction statistics
- target-context groups for bridge fusion

当前主线使用的 profile 主要是：

- `utpm_shift_enhanced`
- `utpm_shift_compact`

统一的是语义，不是原始列名。

## 4. Module And Method Split

| Item | Category | Inference Used | Role |
| --- | --- | --- | --- |
| `TRGT` backbone | backbone | yes | 时序关系消息传递 |
| `Target-Context Bridge` | model module | yes | 目标级上下文融合 |
| `Drift Expert` | model module | yes | 时间漂移适配 |
| `Internal Risk Fusion` | model module | yes | 内部多尺度风险残差 |
| `Cold-Start Residual` | model module | yes | 晚期冷启动补偿 |
| `Prototype Memory` | training-time method | no | 表示空间稳定化 |
| `Pseudo-Contrastive Temporal Mining` | training-time method | no | 时间均衡难样本挖掘 |

这里最关键的区分是：

- `Bridge / Drift / Internal Risk / Cold-Start` 属于模型结构。
- `Prototype / Pseudo-Contrastive` 属于训练期方法，不会在推理时再额外走一条第二模型分支。

## 5. Core Implementation

主要代码：

- [../experiment/models/modules/backbone.py](../experiment/models/modules/backbone.py)
- [../experiment/models/modules/model.py](../experiment/models/modules/model.py)
- [../experiment/models/modules/trainer.py](../experiment/models/modules/trainer.py)
- [../experiment/models/modules/bridge.py](../experiment/models/modules/bridge.py)
- [../experiment/models/modules/memory.py](../experiment/models/modules/memory.py)
- [../experiment/models/engine.py](../experiment/models/engine.py)

关键类：

- `TRGTTemporalRelationAttentionBlock`
- `TRGTInternalRiskEncoder`
- `DyRIFTModel`
- `DyRIFTTrainer`

## 6. Accepted Result

Validation AUC:

- XinYe DGraph: `79.2851%`
- Elliptic Transactions: `82.1329%`
- Elliptic++ Transactions: `82.1953%`
- Macro: `81.2044%`

对应结果表：

- [Mainline AUC CSV](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv)
- [Epoch Policy CSV](results/experiment_epoch_policy.csv)
- [Experiment Table](thesis_experiments.md)

维护中的 rerun 策略为 `70` 个最大 epoch，且图模型第 `30` 个 epoch 前不允许早停。

## 7. Leakage Guardrails

正式主线保证：

- 各数据集独立训练
- 不做跨数据集标签和预测复用
- 验证和测试标签不回流训练
- 推理期不依赖外部模型

Accepted audit：

- [leakage_audit.md](leakage_audit.md)
- [results/leakage_audit.json](results/leakage_audit.json)
