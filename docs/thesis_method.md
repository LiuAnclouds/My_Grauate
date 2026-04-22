# Thesis Method

## Quick Links

- [Back to README](../README.md)
- [Experiment Table](thesis_experiments.md)
- [Mainline Guide](../experiment/training/README_thesis_mainline.md)
- [Final Pure-GNN Summary](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json)
- [Final Pure-GNN Audit](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md)
- [Final Metrics CSV](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_metrics.csv)
- [Epoch Metrics CSV](results/thesis_dyrift_gnn_trgt_deploy_pure_v1_epoch_metrics.csv)

## 1. Problem Setting

目标不是给三个数据集各做一套“特供策略”，而是在严格数据隔离前提下，用一套统一动态图异常检测架构完成训练、验证和推理。

统一约束：

- 三个数据集彼此隔离，不做跨数据集联合训练。
- 各数据集可以有自己的原始清洗流程，但最终必须映射到同一套 UTPM 特征语义。
- 主体模型必须是动态图 GNN，当前固定为 `DyRIFT-GNN`；backbone 为 `TRGT`，运行入口名为 `dyrift_gnn`。
- 数据集级超参数允许分别调，例如 `attr_proj_dim`、`hidden_dim`、`rel_dim`、`fanouts`、`attention_num_heads`、`dropout`。
- 禁止训练集、验证集、测试池和外部集节点交叉，也禁止用验证/测试标签反哺训练。

最终论文主线采用：

`统一特征映射 + DyRIFT-GNN 模型 + TRGT 主干 + 单路部署推理`

## 2. Final Architecture

| Layer | Current Choice | Role |
| --- | --- | --- |
| Dataset preprocessing | dataset-local processors | 处理不同原始字段，但输出同一语义契约 |
| Input contract family | UTPM schema with dataset-local subsets | 三个数据集共享同一输入语义 |
| Primary GNN | `DyRIFT-GNN` | 动态风险感知反欺诈图神经网络 |
| Backbone | `TRGT` | Temporal-Relational Graph Transformer，多头时序关系注意力动态图主干 |
| GNN modules | temporal-normality bridge, drift-expert adaptation, prototype memory, pseudo-contrastive temporal mining, internal causal risk fusion, context-conditioned cold-start residual | 论文主要创新模块 |
| Final decision | single pure-GNN probability | 训练和推理都只走 `DyRIFT-GNN` |

这不是三个数据集三套模型，因为 `DyRIFT-GNN` 模型、`TRGT` 主干、输入语义、训练协议和推理路径都固定一致。不同数据集只在合理超参数上分别调优。

## 3. Base Features And Prior Bridge

当前方法里要区分两类输入角色：

- `base`：所有节点共享的主干输入，是统一后的节点属性、图结构统计、时间统计和关系统计。
- `bridge context`：目标节点级别的辅助上下文，是从同一份统一特征里抽取出来的目标上下文组，再经过模型内 bridge 编码器融合到目标表示。

这里的 bridge context 不是外部模型预测，也不是标签泄露特征。它本质上仍然来自当前节点及其局部时间图结构的无标签统计，例如：

- `graph_stats`
- `graph_time_detrend`
- `neighbor_similarity`
- `activation_early`

这样做的目的，是把“统一主干表征”和“目标节点特异性时序上下文”解耦，避免把所有东西简单拼成一个大平面输入。

## 4. Why This Is Still Pure GNN

最终训练和推理都只有一条 `DyRIFT-GNN / TRGT` 路径：

- 没有独立外部分类头参与最终输出。
- 没有 residual 分支参与推理。
- 没有外部模型预测在验证或测试时作为输入特征注入。

因此这条路线是可部署的单模型纯 GNN。

## 5. Innovation Groups

| Innovation Group | Evidence | Current Reading |
| --- | --- | --- |
| Temporal relation attention backbone | `m7_utpm -> TRGT` 完成主干现代化 | 解决统一动态图关系建模问题 |
| Temporal-normality bridge | pure-GNN 主线显式使用 | 在目标节点级别融合时序上下文先验 |
| Drift-expert adaptation | `drift_expert` adapter | 适配不同时间段的分布漂移 |
| Prototype memory | 共享模块消融 | 更偏结构正则与类别稳定器 |
| Pseudo-contrastive temporal mining | 去掉后宏平均下降 `0.006182` | 当前最稳定有效的共享模块 |
| Internal causal risk fusion | pure-GNN 最优配置保留该分支 | 在模型内部做多尺度风险差分建模 |
| Context-conditioned cold-start residual | EPP 最终版启用 | 只在纯 GNN 内部补偿晚期冷启动节点的消息缺失 |

论文里不要写成“attention 一换就全赢”。更准确的表述是：在统一动态图输入契约下，将时序关系注意力主干、temporal-normality bridge、drift-expert 适配、prototype memory、pseudo-contrastive mining 和 internal causal risk fusion 组合成一套可审计的动态图反欺诈框架。

## 6. Final Pure-GNN Results

| Dataset | Val AUC |
| --- | ---: |
| XinYe DGraph | 0.790455 |
| Elliptic | 0.821329 |
| Elliptic++ | 0.821953 |
| Macro | 0.811246 |

补充说明：

- XinYe 和 ET 直接采用当前各自最优 pure-GNN run。
- EPP 在保持同一 `DyRIFT-GNN / TRGT` 架构前提下，启用了 `cold_start_residual_strength=0.35` 的上下文冷启动残差专家。
- EPP run name 中的 `mixed120` 仅表示内部邻居采样策略是 mixed recent/random sampler。
- 这不是换第二个模型，而是同一 DyRIFT-GNN / TRGT 主干中的数据集级超参数开关。

## 7. Hard-Leakage Guardrails

最终主结果已审计：

- [leakage_audit.md](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.md)
- [leakage_audit.json](../experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/leakage_audit.json)

审计结论：

- `hard_leakage_detected = false`
- `train / val / test_pool / external` 四个 split 两两不重叠。
- GNN 的验证节点与官方 `phase1_val` 对齐。
- 所有路径都在各自数据集作用域内，没有跨数据集缓存复用。

## 8. Repro Commands

构建统一特征缓存：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/run_thesis_mainline.py \
  build_features --phase both
```

运行 final pure-GNN suite：

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

生成硬泄露审计：

```bash
conda run -n Graph --no-capture-output python3 experiment/training/audit_thesis_leakage.py \
  --suite-summary experiment/outputs/thesis_suite/thesis_dyrift_gnn_trgt_deploy_pure_v1/summary.json
```
