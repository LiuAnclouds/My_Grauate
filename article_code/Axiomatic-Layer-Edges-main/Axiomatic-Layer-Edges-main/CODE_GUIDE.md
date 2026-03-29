# Axiomatic Layer Edges - 代码结构文档

> 论文: "Explanations of GNN on Evolving Graphs via Axiomatic Layer Edges" (ICLR 2025)
>
> 核心思想: 提出了一种基于公理化层边(Axiomatic Layer Edges)的方法，用于解释演化图上的GNN预测变化。通过将GNN输出差异分解为各层边的贡献，并使用凸优化选择最重要的边作为解释。

---

## 一、项目总览

```
Axiomatic-Layer-Edges-main/
├── train.py                  # 训练入口（统一调度各任务的训练）
├── main_explain.py           # 解释入口（统一调度各任务的解释）
├── select_args.py            # 超参数选择（按数据集选择layer edge/edge数量）
├── train/                    # 训练模块
│   ├── train_graph.py        # 图分类训练
│   ├── train_link.py         # 链接预测训练
│   ├── train_link_utils.py   # 链接预测工具类（数据集、模型定义）
│   ├── train_pheme.py        # Pheme谣言检测训练
│   ├── train_weibo.py        # Weibo谣言检测训练
│   ├── train_yelp.py         # Yelp欺诈检测训练
│   └── train_yelp_utils.py   # Yelp工具类（数据读取、特征、模型）
├── explain/                  # 解释模块
│   ├── explain_utils.py      # 核心解释工具函数
│   ├── graph_explain.py      # 图分类解释
│   ├── link_explain.py       # 链接预测解释
│   ├── rumor_explain.py      # 谣言检测解释
│   └── yelp_explain.py       # Yelp欺诈解释
├── case_study/               # 案例研究（BA-Shapes实验）
│   ├── gen_data_circle.py    # 生成Circle motif数据
│   ├── gen_data_house.py     # 生成House motif数据
│   ├── train_GCN.py          # 训练案例研究的GCN
│   ├── explain.py            # 案例研究解释
│   └── utils.py              # 案例研究工具函数
├── running_time/             # 运行时间测试
│   ├── train_GCN.py          # 训练运行时间测试模型
│   ├── explain.py            # 运行时间测试解释
│   └── utils.py              # 运行时间测试工具
└── data/                     # 数据集目录
```

---

## 二、入口文件

### 功能类型: 程序入口 / 参数解析

| 文件名 | 函数名 | 作用 |
|--------|--------|------|
| `train.py` | `main(args)` | 训练入口：根据 `--dataset` 参数分发到对应任务的训练函数 |
| `main_explain.py` | `main(args)` | 解释入口：根据 `--dataset` 参数分发到对应任务的解释函数 |
| `select_args.py` | `select_path_number(dataset, targetpath)` | 根据数据集名称和候选layer edge数量，返回要选择的layer edge数量列表 |
| `select_args.py` | `select_edge_number(dataset, targetpath)` | 根据数据集名称和候选edge数量，返回要选择的input edge数量列表 |

**数据集分派逻辑:**
- **节点分类**: `pheme` → `train_pheme` / `rumor_explain`, `weibo` → `train_weibo` / `rumor_explain`, `Chi`/`NYC` → `train_yelp` / `yelp_explain`
- **链接预测**: `bitcoinalpha`/`bitcoinotc`/`UCI` → `train_link` / `link_explain`
- **图分类**: `mutag`/`clintox`/`IMDB-BINARY`/`REDDIT-BINARY` → `train_graph` / `graph_explain`

---

## 三、训练模块 (`train/`)

### 功能类型: 模型定义与训练

#### `train/train_graph.py` — 图分类

| 函数/类名 | 作用 |
|-----------|------|
| `train_GCN` (class) | 训练用GCN模型（带自环和归一化），2层GCNConv + global_mean_pool，用于图分类 |
| `train_GCN.forward(x, edge_index, batch)` | 前向传播：GCNConv → ReLU → GCNConv → 全局平均池化 |
| `train_GCN.back(x, edge_index_1, edge_index_2)` | 返回中间层的激活前/后值，用于计算ReLU delta |
| `GCN` (class) | 解释用GCN模型（无自环、无归一化），用于可控的layer edge解释 |
| `GCN.forward(x, edge_index, edge_weight)` | 前向传播，接受自定义edge_weight |
| `GCN.back(x, edge_index, edge_weight)` | 返回激活前后值 |
| `GCN.pre_forward(x, edge_index, edge_weight)` | 返回未经readout的节点级logits |
| `GCN.verify_layeredge(x, edge_index1, edge_index2, edge_weight1, edge_weight2)` | 验证layer edge解释：两层分别使用不同的edge_index和edge_weight |
| `train(model, optimizer, train_number, criterion, dataset)` | 训练一个epoch |
| `acc_val(model, train_number, dataset)` | 计算验证集准确率 |
| `acc_train(model, train_number, dataset)` | 计算训练集准确率 |
| `NormalizedDegree` (class) | 数据变换：将节点度数归一化为特征 |
| `initializeNodes(dataset)` | 为没有节点特征的图数据集初始化节点特征（OneHot度数或归一化度数） |
| `train_all(args)` | 完整训练流程：加载数据集 → 训练 → 保存模型 |

#### `train/train_link.py` — 链接预测

| 函数/类名 | 作用 |
|-----------|------|
| `Net_link_train` (class) | 训练用链接预测模型：2层GCNConv编码器 + 线性层解码器 |
| `Net_link_train.encode(edgeindex)` | 编码：GCNConv → ReLU → GCNConv |
| `Net_link_train.decode(z, pos_edge_index, neg_edge_index)` | 解码：拼接两端节点嵌入 → 线性层 → 二分类 |
| `get_link_labels(pos, neg)` | 生成链接标签（正边=1，负边=0） |
| `train(edgeindex, model, data, optimizer)` | 训练一个epoch：负采样 → 编解码 → 交叉熵损失 |
| `test(model, data, edgeindex)` | 测试：计算AUC和F1 |
| `train_all(args)` | 完整训练流程：按时间切片训练（bitcoin按年，UCI按周） |

#### `train/train_link_utils.py` — 链接预测工具

| 函数/类名 | 作用 |
|-----------|------|
| `SynGraphDataset` (class) | 继承InMemoryDataset，加载CSV格式的动态图数据集 |
| `link_load_data(path)` | 从CSV读取图数据，构建NetworkX图，提取8维节点特征（投票统计、度数特征等），提取时间戳 |
| `link_read_data(folder, prefix)` | 读取数据并转为PyG Data对象 |
| `Net_link` (class) | 解释用链接预测模型（无归一化、无自环），支持逐层edge_weight |
| `Net_link.encode(x, edge_index1, edge_index2, edge_weight1, edge_weight2)` | 编码（两层可用不同edge_index/weight） |
| `Net_link.decode(z, pos_edge_index)` | 解码：拼接 → 线性层 |
| `Net_link.back(x, edge_index_1, edge_index_2, edgeweight1, edgeweight2)` | 返回中间层激活值 |
| `split_edge(start, end, flag, clear_time, num_nodes)` | 按时间窗口（年/月/周）切分边 |
| `clear_time(time_dict)` | 将时间戳转换为 (年, 月编号, 周编号) 三元组 |
| `clear_time_UCI(time_dict)` | UCI数据集专用的时间清洗函数 |

#### `train/train_pheme.py` — Pheme谣言检测

| 函数/类名 | 作用 |
|-----------|------|
| `Net_rumor` (class) | 谣言检测GCN模型（无归一化版本），用于解释 |
| `Net_rumor.forward(sentence, edge_index_1, edge_index_2, edgeweight1, edgeweight2)` | 前向传播：Embedding → BiLSTM → GCNConv → ReLU → GCNConv |
| `Net_rumor.back(x, edge_index_1, edge_index_2, edgeweight1, edgeweight2)` | 返回中间层激活值 |
| `Net_rumor.feature(sentence)` | 从句子提取特征：Embedding → BiLSTM → concat hidden |
| `Net_rumor.forward_v2(x, ...)` | 从已提取的特征开始前向传播（跳过Embedding和BiLSTM） |
| `Net` (class) | 训练用谣言检测模型（带归一化版本） |
| `accuracy_list(pred, true)` | 计算准确率 |
| `train_all(args)` | Pheme完整训练：加载GloVe → 加载JSON数据 → 按类别分层划分 → 逐样本训练 |
| `train(model, train_idx, optimizer, ...)` | 训练单个样本 |
| `test(model, test_list, ...)` | 测试集评估 |

#### `train/train_weibo.py` — Weibo谣言检测

| 函数/类名 | 作用 |
|-----------|------|
| `Net_rumor` (class) | 与Pheme的`Net_rumor`结构相同，用于解释 |
| `Net` (class) | 训练用模型（与Pheme的`Net`结构相同） |
| `train_all(args)` | Weibo完整训练：加载中文词嵌入 → batch训练 |
| `train(epoch, model, batch_list, ...)` | 按batch训练 |
| `val(model, val_list, ...)` | 验证集评估 |
| `test(model, test_list, ...)` | 测试集评估 |

#### `train/train_yelp.py` — Yelp欺诈检测

| 函数/类名 | 作用 |
|-----------|------|
| `train_all(args)` | Yelp完整训练：加载特征/标签 → 按月构建邻接矩阵 → 训练GCN |
| `train(epoch, model, optimizer, ...)` | 按月份时间窗口训练 |
| `test(adj, model, ...)` | 测试集评估（准确率 + review/user AUC） |

#### `train/train_yelp_utils.py` — Yelp工具

| 函数/类名 | 作用 |
|-----------|------|
| `read_data(tvt, urp, city_name, ...)` | 读取训练/验证/测试的评论、用户、商品列表 |
| `read_user_prod(review_list)` | 从评论列表提取用户和商品ID |
| `feature_matrix(features, p_train, p_val, p_test)` | 构建特征矩阵，按评论/用户分离 |
| `onehot_label(ground_truth, list_idx)` | 构建one-hot标签，推断用户标签 |
| `construct_adj_matrix(...)` | 按时间窗口构建邻接矩阵（评论-用户-商品异构图） |
| `construct_edge(...)` | 按时间窗口构建边列表 |
| `GCN` (class) | Yelp训练用GCN：特征变换层 + 2层GCNConv |
| `GCN.feature(features, nums)` | 将原始特征变换为统一维度 |
| `GCN_test` (class) | Yelp解释用GCN（无归一化） |
| `sparse_mx_to_torch_sparse_tensor(sparse_mx)` | 稀疏矩阵转PyTorch稀疏张量 |
| `accuracy(output, labels)` | 计算分类准确率 |
| `auc_score(output, ground_truth, ...)` | 计算AUC分数 |

---

## 四、解释模块 (`explain/`)

### 功能类型: 核心算法 — Axiomatic Layer Edge 解释

#### `explain/explain_utils.py` — 核心工具函数

| 函数/类名 | 作用 |
|-----------|------|
| **图构建** | |
| `rumor_construct_adj_matrix(edges_index, x)` | 从边列表构建对称归一化邻接矩阵 $\hat{A} = D^{-1/2}AD^{-1/2}$ |
| `yelp_construct_adj_matrix(edges_index, x)` | 构建CSR格式的归一化邻接矩阵 |
| `normalize(mx)` | 对称归一化：$D^{-1/2} M D^{-1/2}$ |
| `matrixtodict(nonzero)` | 将邻接矩阵的非零元素转为字典形式（邻接表） |
| **变化检测** | |
| `difference_weight(edgeindex1, edgeindex2, adj_new, adj_old)` | 比较新旧邻接矩阵，找出权重发生变化的边 |
| `clear(edges)` | 去重：去除重复的无向边 |
| **路径搜索** | |
| `dfs(start, index, end, graph, length, path, paths)` | DFS搜索：找到从start到end的所有固定长度路径 |
| `dfs2(start, index, graph, length, path, paths)` | DFS搜索：找到从start出发的所有固定长度路径 |
| `from_edge_findpaths(edge_list, graph)` | 从变化边列表出发，找到所有经过变化边的2跳路径 |
| `findnewpath(addedgelist, graph, layernumbers, goal)` | 找到目标节点相关的、经过变化边的所有路径 |
| `reverse_paths(pathlist)` | 将路径列表中所有路径反转 |
| `find_target_changed_paths(changed_path, node)` | 筛选出终点为目标节点的路径 |
| `find_target_changed_edges(target_path, changededgelist)` | 从目标路径中提取涉及的变化边 |
| `find_target_changed_layer_edegs(target_changed_edgelist, layernumbers)` | 将变化边扩展为layer edge（边 + 层号） |
| **核心贡献计算** | |
| `contribution_layeredge(paths, adj_start, adj_end, addedgelist, relu_delta, relu_start, relu_end, x_tensor, W1, W2)` | **核心函数**：计算每条layer edge对输出变化的贡献。使用Shapley值分配公式，将路径贡献在路径上的两条layer edge之间公平分配 |
| `contribution_edge(paths, adj_start, adj_end, ...)` | 计算每条input edge（不区分层）对输出变化的贡献 |
| **凸优化选择** | |
| `solve_layeredge(select_number_path, goal, edge_result_dict, edgelist, old_tensor, output_new)` | 节点分类的layer edge选择：使用CVXPY求解整数规划，最小化KL散度选择最重要的layer edges |
| `solve_edge(select_number_path, goal, ...)` | 节点分类的input edge选择 |
| `solve_layeredge_graph(select_number_path, edge_result_dict, edgelist, gcn_old_tensor, output_new)` | 图分类的layer edge选择（需要对所有节点做平均） |
| `solve_edge_graph(select_number_path, ...)` | 图分类的input edge选择 |
| `solve_layeredge_link(select_number_path, ...)` | 链接预测的layer edge选择 |
| `solve_edge_link(select_number_path, ...)` | 链接预测的input edge选择 |
| **评估辅助** | |
| `from_layeredges_to_evaulate(select_layeredges_list, ...)` | 将选出的layer edges应用到旧图，构建评估用的edge_index和edge_weight（两层分别不同） |
| `from_edges_to_evaulate(select_edges_list, ...)` | 将选出的input edges应用到旧图，构建评估用的edge_index和edge_weight |
| `map_target(result_dict, target_node)` | 从全节点贡献字典中提取目标节点的贡献向量 |
| `mlp_contribution(result_dict, W)` | 将GCN输出层的贡献通过MLP权重矩阵映射（链接预测用） |
| `merge_result(dict1, dict2)` | 合并两个贡献字典（链接预测中合并两端节点的贡献） |
| **数学工具** | |
| `softmax(x)` | 数值稳定的softmax |
| `smooth(arr, eps)` | 平滑处理（避免log(0)） |
| `KL_divergence(P, Q)` | 计算KL散度 $D_{KL}(P \| Q)$ |
| **子图处理** | |
| `k_hop_subgraph(node_idx, num_hops, edge_index, ...)` | 提取k跳子图（来自PyG的改写版本） |
| `maybe_num_nodes(edge_index, num_nodes)` | 推断节点数量 |
| `split(start, end, edge_result, all_nodes)` | 按时间索引切分边，构建edge_index |
| `split_link_edge(start, end, edge_result, all_nodes)` | 链接预测专用的边切分 |
| `subfeaturs(features, mapping)` | 按映射关系提取子图特征矩阵 |
| `subadj_map(union, edge_index_old, edge_index_new)` | 将全局节点ID映射为子图局部ID |
| `subadj_map_link(union, ...)` | 链接预测专用的子图映射 |
| **模型** | |
| `GCN_test` (class) | 评估用GCN（两层可用不同edge_index/weight），用于验证layer edge解释效果 |

#### `explain/graph_explain.py` — 图分类解释

| 函数/类名 | 作用 |
|-----------|------|
| `gen_graph_data` (class) | 图分类数据生成器 |
| `gen_graph_data.gen_original_edge()` | 提取图的原始特征和边 |
| `gen_graph_data.pertub_edges(data, edge_index_all)` | 按时间滑动窗口将边划分为旧图和新图，计算变化边 |
| `gen_graph_data.gen_model(model_path, dataset)` | 加载训练好的GCN模型 |
| `gen_graph_data.gen_parameters(model, x, edges_new, edges_old, ew1, ew2)` | 计算ReLU激活比率：relu_delta, relu_end, relu_start |
| `explain_graph(args)` | **图分类解释主流程**：遍历所有图 → 计算变化边/路径 → 贡献分解 → 验证求和性质 → 凸优化选择 → KL散度评估 → 保存结果 |

#### `explain/link_explain.py` — 链接预测解释

| 函数/类名 | 作用 |
|-----------|------|
| `gen_link_data` (class) | 链接预测数据生成器 |
| `gen_link_data.load_data()` | 加载数据集，按时间切分边 |
| `gen_link_data.gen_new_edge(target_edge, ...)` | 对目标边提取k跳子图，按时间窗口划分新旧图 |
| `gen_link_data.gen_model(data, hidden)` | 加载训练好的链接预测模型 |
| `gen_link_data.gen_parameters(model, features, ...)` | 计算ReLU激活比率 |
| `explain_link(args)` | **链接预测解释主流程**：加载数据 → 提取子图 → 分别计算两端节点的贡献 → 通过MLP权重合并 → 凸优化选择 → 评估 |

#### `explain/rumor_explain.py` — 谣言检测解释

| 函数/类名 | 作用 |
|-----------|------|
| `gen_rumor_data` (class) | 谣言数据生成器 |
| `gen_rumor_data.gen_edge_index_old(file_index)` | 从JSON加载传播图的早期边 (edges_2) |
| `gen_rumor_data.gen_edge_index_new(file_index)` | 从JSON加载传播图的完整边 (edges_4) |
| `gen_rumor_data.find_changed_edges(...)` | 找出新旧图之间的变化边 |
| `gen_rumor_data.gen_idxlist()` | 生成文件索引映射 |
| `gen_rumor_data.gen_parameters(file_index, model, ...)` | 加载句子 → 提取BiLSTM特征 → 计算ReLU激活比率和GCN权重 |
| `gen_rumor_data.gen_embedding()` | 加载预训练词嵌入 |
| `gen_rumor_data.gen_model(embedding_tensor)` | 创建并加载谣言检测模型 |
| `gen_rumor_data.gen_evaluate_model(model)` | 创建评估用GCN_test模型（从原模型复制权重） |
| `explain_rumor(args)` | **谣言检测解释主流程**：随机抽样100个传播图 → 计算路径和贡献 → 验证 → 凸优化选择 → 评估 |

#### `explain/yelp_explain.py` — Yelp欺诈解释

| 函数/类名 | 作用 |
|-----------|------|
| `gen_Yelp_data` (class) | Yelp数据生成器 |
| `gen_Yelp_data.gen_data()` | 加载特征、标签、评论时间，构建边 |
| `gen_Yelp_data.gen_adj(goal, edges_old, ...)` | 对目标节点提取k跳子图 → 按时间窗口划分 → 构建新旧邻接矩阵 |
| `gen_Yelp_data.gen_parameters(features, ...)` | 计算ReLU激活比率 |
| `gen_Yelp_data.gen_model(features, nums)` | 加载训练好的GCN模型并提取特征和权重 |
| `gen_Yelp_data.gen_evaluate_model(model)` | 创建评估用GCN_test模型 |
| `explain_yelp(args)` | **Yelp解释主流程**：随机抽样评论和用户 → 提取子图 → 计算贡献 → 验证 → 凸优化选择 → 评估 |

---

## 五、案例研究 (`case_study/`)

### 功能类型: BA-Shapes Motif 检测实验

| 文件名 | 函数/类名 | 作用 |
|--------|-----------|------|
| `gen_data_circle.py` | `gen_negative_graph(node_idx)` | 在BA图+Circle motif上：随机删除1条motif内边，扰动motif外边权重，生成正负样本对 |
| `gen_data_circle.py` | `initializeNodes(dataset)` | 初始化节点特征（OneHot度数编码） |
| `gen_data_house.py` | `gen_negative_graph(node_idx)` | 在BA图+House motif上：随机删除1条motif内边，扰动motif外边权重，生成正负样本对 |
| `gen_data_house.py` | `initializeNodes(dataset)` | 初始化节点特征 |
| `train_GCN.py` | `GCN` (class) | 案例研究GCN模型（与主项目GCN结构一致） |
| `train_GCN.py` | `train()` | 训练：正样本标签1，负样本标签0 |
| `train_GCN.py` | `acc_train()` / `acc_val()` | 计算训练/验证准确率 |
| `explain.py` | `GCN` (class) | 案例研究解释用GCN |
| `explain.py` | `gen_parameters(model, ...)` | 计算ReLU激活比率和GCN权重 |
| `explain.py` | `__main__` | 对所有正→负预测变化的图，选择最重要的1条layer edge，检查是否落在motif内，计算解释准确率 |
| `utils.py` | 与`explain_utils.py`类似 | 案例研究专用的工具函数（接受edge_weight参数的版本） |

---

## 六、核心算法流程

### 解释流程（以图分类为例）

```
1. 加载训练好的GCN模型
2. 构建新旧图（按时间窗口划分边）
3. 计算变化边列表 changededgelist
4. 计算ReLU激活比率：relu_delta, relu_start, relu_end
5. 搜索经过变化边的所有路径（DFS）
6. 核心分解：contribution_layeredge()
   - 将 f(G_new) - f(G_old) 分解为各 layer edge 的贡献
   - 对路径上两条 layer edge 使用 Shapley 值公平分配
7. 验证：所有 layer edge 贡献之和 = 真实输出差异
8. 凸优化选择：solve_layeredge_graph()
   - 选择 k 条 layer edges 使 KL(P_new || P_selected) 最小
   - 使用 MOSEK 求解器
9. 评估：用选出的 layer edges 更新旧图，计算新输出与真实新输出的 KL 散度
10. 保存结果到 JSON
```

### 关键数学概念

- **Layer Edge**: 边 (u,v) 在第 l 层的实例，记为 (u, v, l)。同一条输入边在不同GCN层中作为不同的layer edge
- **路径贡献**: 对于路径 [v0, v1, v2]，其贡献 = adj变化 × ReLU激活比率 × 特征 × 权重
- **Shapley分配**: 路径贡献在其两条layer edge之间按Shapley值公平分配
- **求和性质**: 所有layer edge贡献之和 = GNN输出差异（公理保证）
- **凸优化**: 最小化 KL散度 来选择最重要的 k 条 edges

---

## 七、数据集说明

| 数据集 | 任务类型 | 说明 |
|--------|----------|------|
| MUTAG | 图分类 | 化学分子诱变性分类 |
| ClinTox | 图分类 | 药物毒性分类 |
| IMDB-BINARY | 图分类 | 电影评论二分类 |
| REDDIT-BINARY | 图分类 | Reddit帖子二分类 |
| Bitcoinalpha | 链接预测 | 比特币信任网络 |
| Bitcoinotc | 链接预测 | 比特币OTC信任网络 |
| UCI | 链接预测 | UCI消息网络 |
| Pheme | 节点分类 | 谣言传播图（英文） |
| Weibo | 节点分类 | 谣言传播图（中文） |
| Chi / NYC | 节点分类 | Yelp欺诈评论检测 |

---

## 八、依赖库

- PyTorch + PyTorch Geometric (PyG)
- CVXPY + MOSEK（凸优化求解器）
- DIG (dig.xgraph.dataset) — 用于ClinTox数据集
- NLTK — 用于文本预处理（谣言检测）
- NetworkX, Pandas, NumPy, SciPy
