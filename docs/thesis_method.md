# Thesis Method

## Quick Links

- [Back to README](../README.md)
- [Reproducibility Guide](reproducibility.md)
- [Model Execution Flow](model_execution_flow.md)
- [Experiment Table](thesis_experiments.md)
- [DyRIFT Method Card](dyrift_gnn_method.md)
- [TRGT Backbone](trgt_backbone.md)
- [Accepted Leakage Audit](leakage_audit.md)
- [Mainline AUC CSV](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv)

## 1. Problem Setting

目标不是给三个数据集各做一套特供模型，而是在严格数据隔离前提下，用一套统一的动态图异常检测架构完成训练、验证和推理。

主线约束：

- 三个数据集彼此隔离，不做跨数据集联合训练。
- 原始清洗可以数据集本地化，但最终必须映射到同一套 `UTPM` 语义输入契约。
- 主体模型固定为动态图 GNN，当前正式主线为 `DyRIFT-GNN`，主干为 `TRGT`。
- 数据集级超参数允许分别调优，例如 `attr_proj_dim`、`hidden_dim`、`rel_dim`、`fanouts`、`attention_num_heads`、`dropout`。
- 禁止训练集、验证集、测试池和外部集节点交叉，也禁止验证和测试标签回流训练。

正式论文主线是：

`统一特征映射 + DyRIFT-GNN + TRGT + 单路纯 GNN 部署`

## 2. Architecture Stack

| Layer | Current Choice | Role |
| --- | --- | --- |
| Dataset preprocessing | dataset-local processors | 处理不同原始字段，但输出同一语义家族 |
| Unified input contract | `UTPM` | 三个数据集共享同一输入语义 |
| Backbone | `TRGT` | 时序关系图 Transformer 主干 |
| Full model | `DyRIFT-GNN` | 动态风险感知反欺诈图神经网络 |
| Final decision | single pure-GNN probability | 训练和推理都只走一条 GNN 路径 |

这不是三个数据集三套模型，因为主干、模块组织、训练协议和推理路径都固定一致。不同数据集只在合理超参数上分开调优。

## 3. Base Features And Target Context

当前输入分成两个角色，但都来自同一份统一特征缓存：

- `base features`：所有节点共享的主干输入，包括属性统计、图结构统计、时间统计和关系统计。
- `target context groups`：目标节点级辅助上下文，包括 `graph_stats`、`graph_time_detrend`、`neighbor_similarity`、`activation_early` 这类目标上下文组。

它们的区别不是“一个人工一个自动”，而是：

- `base features` 直接进入 GNN 主干，服务所有节点。
- `target context groups` 在目标节点级别进入 bridge 分支，和 GNN 表示再融合。

这仍然是单模型纯 GNN，因为 bridge 输入不来自外部模型预测，也不依赖标签泄露特征。

## 4. What Is A Module And What Is A Method

这是当前论文写法里最容易混淆的一点。

| Item | Type | Used At Inference | Final Scope | Role |
| --- | --- | --- | --- | --- |
| TRGT Backbone | 模型主干 | 是 | 全部数据集 | 时序关系消息传递 |
| Target-Context Bridge | 模型模块 | 是 | 全部数据集 | 目标级上下文融合 |
| Drift Expert | 模型模块 | 是 | 全部数据集 | 时间漂移适配 |
| Internal Risk Fusion | 模型模块 | 是 | XinYe 与 EPP 最终 profile | 多尺度风险残差建模 |
| Cold-Start Residual | 模型模块 | 是 | EPP 最终 profile | 晚期冷启动补偿 |
| Prototype Memory | 训练期方法 | 否 | 全部数据集 | 表示空间正则与类别稳定 |
| Pseudo-Contrastive Temporal Mining | 训练期方法 | 否 | 全部数据集 | 时间均衡难样本挖掘 |

更准确的表述是：

- `Bridge`、`Drift Expert`、`Internal Risk Fusion`、`Cold-Start Residual` 属于模型内部结构。
- `Prototype Memory` 和 `Pseudo-Contrastive` 属于训练协议里的增强方法，不是推理期第二条模型分支。

## 5. Why It Is Still Pure GNN

最终部署和推理只有一条 `DyRIFT-GNN / TRGT` 路径：

- 没有外部树模型参与最终预测。
- 没有 teacher 分支参与最终推理。
- 没有第二阶段分类器或加权融合器。

因此正式主线是可部署的单模型纯 GNN。

## 6. Innovation Summary

当前可以稳定写进论文的创新点是：

- 统一 `UTPM` 输入契约，让三个动态图数据集共享同一语义输入家族。
- `TRGT` 时序关系主干，负责动态图邻域的关系感知和时间感知消息传递。
- `Target-Context Bridge`，把目标节点级时序上下文与 GNN 主干表示在模型内融合。
- `Drift Expert`，用时间位置和上下文去适配时间分布漂移。
- `Internal Risk Fusion`，在模型内部学习多尺度风险差分，而不是外接第二个分类器。
- `Prototype Memory` 与 `Pseudo-Contrastive Temporal Mining`，作为训练协议增强表示稳定性和时间漂移鲁棒性。

不要把方法写成“换了个 Transformer 就变强”。更准确的说法是：在统一动态图输入契约下，`TRGT` 主干和 `DyRIFT` 风险模块共同构成了一套可审计、可部署的动态图反欺诈框架。

## 7. Final Accepted Result

| Dataset | Val AUC |
| --- | ---: |
| XinYe DGraph | 0.792851 |
| Elliptic Transactions | 0.821329 |
| Elliptic++ Transactions | 0.821953 |
| Macro | 0.812044 |

补充说明：

- XinYe accepted run 使用 `full_xinye_repro_v1`。
- ET final profile 关闭了 `internal_risk_fusion`，但仍然保持同一 `DyRIFT-GNN / TRGT` 架构。
- EPP final profile 在同一架构内启用了 `cold_start_residual_strength=0.35`。

## 8. Leakage Guardrails

当前 accepted 主结果已经重新做过一致性审计：

- [leakage_audit.md](leakage_audit.md)
- [leakage_audit.json](results/leakage_audit.json)

结论：

- `hard_leakage_detected = false`
- `train / val / test_pool / external` 两两不重叠
- 预测 bundle 与官方 split 对齐
- 没有跨数据集缓存复用，也没有验证和测试标签回流训练

## 9. Supplementary XinYe Joint Training

补充实验中，我额外做了三个 `phase1/phase2` 诊断版本：

| Setting | Phase1 Val AUC | Phase2 Train AUC | Phase2 Holdout AUC | Meaning |
| --- | ---: | ---: | ---: | --- |
| Joint Phase1+Phase2 Train | 0.791441 | 0.716531 | n/a | from-scratch joint train, phase1 validation |
| Phase-Aware Balanced | 0.789344 | 0.635207 | 0.636328 | phase indicator plus balanced phase2 subset, phase1 checkpoint |
| Phase-Aware DualVal | 0.784233 | 0.709306 | 0.706197 | checkpoint selected by phase1 val plus phase2 holdout |

这些实验说明：

- `phase2` 标注节点直接加入训练并不会自然提高 `phase1.val`。
- 只按 `phase1.val` 选择 checkpoint 会牺牲 phase2 排序能力。
- 加入 phase2 holdout 参与 checkpoint 选择能提高 phase2 AUC，但会降低 phase1 val AUC。
- 因此 XinYe 的 phase1/phase2 存在明显阶段漂移，正式论文主线继续采用无泄露 `phase1.train -> phase1.val` 口径。

因此这些结果只作为补充诊断保留，不替换正式主结果。
