# tunedGNN 论文深度笔记

> 这份笔记以 `article/tunedGNN.pdf` 为依据重写，目标是形成一份适合飞书阅读、适合后续反复回看的**研究型深度笔记**。  
> 写法上继续保持三点：**结构完整、解释扎实、对比突出**。  
> 文中默认区分三类内容：`论文原文要点`、`理解性解释`、`证据边界与合理推断`；后两者服务于理解，不等同于论文逐字复述。

## 一、前言

### 1. 论文定位

| 项目 | 内容 |
| --- | --- |
| 论文标题 | Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification |
| 文件名 | `article/tunedGNN.pdf` |
| 作者 | Yuankai Luo, Lei Shi, Xiao-Ming Wu |
| 单位 | Beihang University; The Hong Kong Polytechnic University |
| 会议信息 | NeurIPS 2024, Datasets and Benchmarks Track |
| 本地文件 | `article/tunedGNN.pdf` |
| 项目代码 | `https://github.com/LUOyk1999/tunedGNN` |
| 实现框架 | PyG + DGL |
| 主要任务 | node classification |
| 数据范围 | 18 个数据集，覆盖 homophilous / heterophilous / large-scale graphs |
| 一句话核心判断 | 这篇论文真正要纠偏的不是某个新模型的输赢，而是**classic GNN 是否长期因为配置不足的评测设置而被低估**。 |

这篇论文更适合放在 **benchmark 纠偏、经验评测升级、baseline 重新估值** 这类工作里理解，而不是把它当成提出新传播算子的新模型论文来读。它没有发明新的 message passing 机制，也没有设计新的图 Transformer 结构；它做的是另一件在实验科学里同样重要的事：**把比较协议重新置于更对称、更完整、更有说服力的轨道上**。

如果把它压缩成一句更适合后续回看的定位，可以写成：

> 本文不是在证明 classic GNN 理论上一定优于 GT，而是在证明：**在 node classification 上，GT 对 GNN 的优势很可能被过去不充分的 GNN 调参与不对称比较放大了。**

### 2. 阅读主线

本文最值得抓住的是三条主线：

* **问题主线**：GT 在 node classification 上看起来长期强于 classic GNN，到底是范式本身更强，还是 classic GNN 长期没有被按足够强的配置来评测。
* **方法主线**：作者并不改 GCN、GraphSAGE、GAT 的传播核心，而是系统重调 `normalization`、`dropout`、`residual connection` 与 `network depth`。
* **证据主线**：18 个数据集上的重评估结果，加上 Table 5-7 的消融、Figure 1 的深度分析，以及 Appendix B 的补证，是否足以支撑“GT 优势被高估”这一判断。

本文最核心的收束句可以先记成：

> **在 node classification 上，classic GNN 并没有像很多文献叙事里那样天然落后；只要给它们足够对称、足够完整的调参与训练配置，它们可以在 18 个数据集中的 17 个上达到第一。**

## 二、前备知识

### 1. 节点分类任务与 message passing 骨架

先把本文默认依赖的任务形式立住。对这篇论文来说，最关键的不是某个模型名，而是 node classification 的统一骨架。

记图为：

$$
G = (V, E, X, Y)
$$

其中：

* $V$ 是节点集合。
* $E \subseteq V \times V$ 是边集合。
* $X \in \mathbb{R}^{|V| \times d}$ 是节点特征矩阵。
* $Y \in \mathbb{R}^{|V| \times C}$ 是标签矩阵。

经典 message passing GNN 的统一写法是：

$$
h_v^l = \mathrm{UPDATE}^l \left( h_v^{l-1}, \mathrm{AGG}^l \left( \left\{ h_u^{l-1} \mid u \in \mathcal{N}(v) \right\} \right) \right)
$$

这里最需要讲清楚的不是公式形式本身，而是它在本文里的作用：

* $h_v^{l-1}$ 是节点 $v$ 在上一层的表示。
* $\mathcal{N}(v)$ 是邻域。
* $\mathrm{AGG}$ 负责把邻居信息聚起来。
* $\mathrm{UPDATE}$ 负责把自信息和邻域信息合成新的表示。

最后一层输出再接一个分类头：

$$
\hat y_v = g \left( h_v^L \right)
$$

训练目标就是在训练节点集合上最小化分类损失。  
这层前备知识之所以必须写出来，是因为本文后面所有争论都建立在同一个前提上：**作者比较的不是“有没有消息传播”，而是“在同样都是 message passing 的前提下，classic GNN 到底被没被认真调过”。**

### 2. 三类 classic GNN 代表了什么

本文选择 `GCN`、`GraphSAGE`、`GAT`，不是随手挑三种老模型，而是刻意覆盖了三种常见的 node-level GNN 路线。

| 模型 | 核心机制 | 在本文中的代表意义 | 典型长处 | 典型短板 |
| --- | --- | --- | --- | --- |
| GCN | 归一化邻域聚合 | 最经典、最朴素的 spectral-style message passing | 结构简单、训练稳定、成本低 | 深层时容易过平滑 |
| GraphSAGE | 自表示与邻域均值分开映射 | 归纳式、大图友好的邻域采样路线 | 可扩展、适合大图 | 表达能力依赖配置与深度 |
| GAT | 邻域注意力加权 | 局部 attention 式 message passing | 邻居权重可学习 | 训练更敏感，参数和稳定性更依赖 recipe |

三者的典型传播形式可以简写为：

$$
\text{GCN: } h_v^l = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\hat d_u \hat d_v}} h_u^{l-1} W^l \right)
$$

$$
\text{GraphSAGE: } h_v^l = \sigma \left( h_v^{l-1} W_1^l + \left( \mathrm{mean}_{u \in \mathcal{N}(v)} h_u^{l-1} \right) W_2^l \right)
$$

$$
\text{GAT: } h_v^l = \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu}^l h_u^{l-1} W^l \right)
$$

这三种形式在本文里分别承担了不同角色：

* `GCN` 用来检验最基础的 classic GNN 是否被低估。
* `GraphSAGE` 用来检验在大图和归纳场景下，classic GNN 是否仍然很强。
* `GAT` 用来检验“局部注意力式 GNN”在认真调参后能否在部分数据集上追平或超过若干 GT baseline。

更重要的是，本文的比较逻辑并不是“只有某一种 old baseline 偶然被调强了”，而是：**三条常见 classic GNN 路线都在系统调参后出现了显著改观。**

### 3. 四类关键超参为什么会左右结论

本文真正强调的不是新模块，而是四类大家都认识、但过去并没有被系统拉齐的训练配置。

| 超参或配置 | 在模型中的位置 | 论文给出的核心作用 | 如果不讲清楚会卡在哪个 `Work` 模块 |
| --- | --- | --- | --- |
| normalization | 每层传播后、激活前 | 稳定训练分布，尤其对大图更重要 | `方法论创新` 与 `补充分析` |
| dropout | 激活后 | 抑制共适应，持续提升泛化 | `补充分析` |
| residual connection | 层间直连 | 稳定深层训练，尤其异质图更关键 | `主结果` 与 `补充分析` |
| network depth | 层数搜索空间 | 影响 receptive field 与异质图建模能力 | `实验设计` 与 `补充分析` |

以 GCN 为例，论文显式给出了三个增强写法：

$$
h_v^l = \sigma \left( \mathrm{Norm} \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\hat d_u \hat d_v}} h_u^{l-1} W^l \right) \right)
$$

$$
h_v^l = \mathrm{Dropout} \left( \sigma \left( \mathrm{Norm} \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\hat d_u \hat d_v}} h_u^{l-1} W^l \right) \right) \right)
$$

$$
h_v^l = \mathrm{Dropout} \left( \sigma \left( \mathrm{Norm} \left( h_v^{l-1} W_r^l + \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\hat d_u \hat d_v}} h_u^{l-1} W^l \right) \right) \right)
$$

这里必须抓住一个非常重要的点：  
**本文不是在宣称这些组件“新”，而是在强调过去很多 GNN baseline 并没有以一套完整、对称、被认真搜索过的训练 recipe 参与比较。**

### 4. 同质图、异质图与大图为什么要分开看

本文把 18 个数据集拆成三类：

| 类型 | 代表数据集 | 论文要回答的问题 |
| --- | --- | --- |
| homophilous graphs | Cora, CiteSeer, PubMed, Computer, Photo, CS, Physics, WikiCS | classic GNN 在传统 node benchmark 上是否真的落后于 GT |
| heterophilous graphs | Squirrel, Chameleon, Amazon-Ratings, Roman-Empire, Minesweeper, Questions | classic GNN 是否真的只能处理同质图 |
| large-scale graphs | ogbn-proteins, ogbn-arxiv, ogbn-products, pokec | 大图和长程依赖场景下，message passing 是否已经失效 |

这三类必须分开看的原因是：

* 同质图最容易出现“GT 更强”的经典叙事。
* 异质图是 classic GNN 被批评最重的区域之一。
* 大图则直接关联训练稳定性、显存、采样和长程依赖。

因此，本文的证据链并不是单点突破，而是三段推进：

1. 先证明在传统同质图上，classic GNN 的“弱”可能被夸大了。
2. 再证明在异质图上，classic GNN 也不一定系统性处于劣势。
3. 最后证明在大图上，message passing 依然可以非常强。

## 三、Question&Study

### 1. 比较缺口

本文要修正的比较缺口可以概括成两层。

第一层缺口是**配置不对称**。GT 往往是在较完整的现代训练 recipe 下被报告成绩的，而 classic GNN 在很多对比里只是朴素版本。

第二层缺口是**搜索不完整**。过去不少工作没有系统搜索 classic GNN 的 `normalization`、`dropout`、`residual` 和 `depth`，但这些配置恰恰会强烈影响 node classification 结果。

换句话说，社区里长期流行的判断其实是：

> “GT 在 node classification 上显著优于 classic GNN。”

而本文要追问的是：

> 这个结论到底是模型范式差异，还是经典 GNN 长期没有按足够公平的方式进入比较。

### 2. 核心问题

本文的核心问题可以压缩为一句话：

> **在 node classification 上，如果把 classic GNN 认真调到位，它们是否仍然明显弱于 GT？**

更具体地说，作者要回答四个子问题：

1. 在 homophilous graphs 上，classic GNN 能否回到一线。
2. 在 heterophilous graphs 上，classic GNN 是否真的不适用。
3. 在 large-scale graphs 上，message passing 是否已经被 GT 淘汰。
4. 哪些超参是真正决定 classic GNN 强弱的关键因素。

### 3. 方法主张

本文的方法主张并不复杂，但非常关键：

* 不改 `GCN`、`GraphSAGE`、`GAT` 的基本传播规则。
* 对 classic GNN 做系统超参搜索。
* 对 GT baseline 也用相近搜索空间重训，避免单边重调。
* 在 18 个数据集上统一比较。

因此，本文的创新性质更接近：

| 层面 | 内容 | 性质判断 |
| --- | --- | --- |
| 模型层 | 仍是 classic GNN | 不是新算子创新 |
| 训练层 | 系统搜索关键超参 | 经验性 recipe 纠偏 |
| 评测层 | 统一搜索空间和运行环境 | 这是最核心的贡献 |

### 4. 创新性质

如果必须用一句话说“这篇论文创新在哪里”，更稳的说法是：

> **它的创新不在结构发明，而在经验研究的严谨性升级。**

这种创新类型通常容易被低估，因为它不像新模型那样显眼，但它对后续研究的影响反而很深：  
如果 baseline 本身就配置不足，那么很多“新模型更强”的结论，根本不够干净。

### 5. 证据边界

本文能支持的判断很强，但边界也必须写清楚。

它能支持的是：

* 在本文覆盖的 18 个 node classification benchmark 上，认真调过的 classic GNN 非常强。
* GT 在该任务上的优势至少没有过去一些叙事里那么绝对。
* `normalization`、`dropout`、`residual` 和 `depth` 对结果影响非常大。

它不能支持的是：

* classic GNN 在所有图任务上都优于 GT。
* node classification 的结论可以直接外推到 graph classification、link prediction 或动态图。
* 本文已经从理论上解释了为什么 GT 在这里没有形成稳定优势。

论文自己在限制部分也承认，它只研究了 node classification，没有覆盖 graph-level 与 edge-level 任务。  
这点对你后续做动态图异常检测尤其重要，因为它意味着：**这篇论文更适合提供“如何重新看待 baseline”的方法论启发，而不是直接提供动态图任务的现成结论。**

## 四、Work

### 1. 规则

正文真正的主轴，是把“classic GNN 为什么会被重新估值”这件事讲清楚，而不是简单复述某几张榜单。

对本文来说，最稳定的规则可以写成下面三条：

* **规则一：比较的是 node classification，不是 graph-level。** 这点必须明确，因为很多 GT 的代表性结果最早来自 graph-level 任务，不能直接迁移到 node-level 叙事里。
* **规则二：比较的是完整训练协议，不只是传播核心。** 过去很多比较把传播规则差异和 recipe 差异混在了一起。
* **规则三：最终结论必须由跨三类数据集的一整条证据链支撑。** 单看同质图、异质图或大图中的任何一块，都不足以支撑“GT 优势被高估”这个强判断。

如果把这三条规则再压缩成一句话，就是：

> 本文真正改写的不是某一个榜单，而是 **node classification 上 baseline 应该怎样被公平构造和公平比较**。

### 2. 方法论创新

这篇论文的方法论创新，本质上是把 classic GNN 的实验协议补完整。

#### 2.1 不改传播核，只改评测严谨度

作者没有引入新的 GNN block，也没有提出新的 global attention 模块。  
他们保留了 `GCN`、`GraphSAGE` 和 `GAT` 的原始传播逻辑，只把下面这些量系统纳入搜索：

| 项目 | 搜索空间 |
| --- | --- |
| learning rate | `0.001`, `0.005`, `0.01` |
| hidden dim | `64`, `256`, `512` |
| normalization | `None`, `BN`, `LN` |
| residual | `False`, `True` |
| dropout | `0.2`, `0.3`, `0.5`, `0.7` |
| depth | `1` 到 `10` 层 |
| 训练上限 | 2500 epochs |
| 结果统计 | 5 个随机种子，报告均值与标准差 |

这张表的意义并不只是罗列超参，而是说明：

* 作者把 classic GNN 从“随便给一个常见配置”升级到了“系统搜索后的强 baseline”。
* 他们还重训了 baseline GT，而不是只重调 GNN。
* 这使得后面的对比更像同配方比较，而不是单边精调与不对称配置。

#### 2.2 为什么这一步足以构成论文的主体贡献

这一步看上去不新，但其实很重。因为在经验研究里，下面这三件事经常被混为一谈：

1. 模型范式真的更强。
2. 训练 recipe 更成熟。
3. 报告方式更激进，调参更充分。

本文的贡献就在于把它们重新拆开。  
尤其在 node classification 上，`normalization`、`dropout`、`residual`、`depth` 这几个因素既足够基础，又足够强烈地改变结果，因此如果它们不对齐，很多高阶结论都并不干净。

> **所以，这篇论文最“硬”的地方不是模型，而是证据生产方式。**

### 3. 实验设计

#### 3.1 数据覆盖面

作者把实验设计成一个覆盖面很广的 node classification benchmark。

| 数据族 | 数据集 | 评价重点 |
| --- | --- | --- |
| Homophilous | Cora, CiteSeer, PubMed, Computer, Photo, CS, Physics, WikiCS | classic GNN 在传统任务上是否真的不如 GT |
| Heterophilous | Squirrel, Chameleon, Amazon-Ratings, Roman-Empire, Minesweeper, Questions | classic GNN 是否天然不适合异质图 |
| Large-scale | ogbn-proteins, ogbn-arxiv, ogbn-products, pokec | message passing 在大图上是否仍有竞争力 |

这组设计的价值在于：  
它不是只挑一种容易赢的场景，而是横跨传统图、异质图和百万级大图。只要结论在这三类场景里都能成立，说服力就会明显增强。

#### 3.2 对比对象

对比对象并不只是一批 GT，而是一组层次化 baseline：

| 类别 | 代表模型 |
| --- | --- |
| classic GNN | `GCN`, `GraphSAGE`, `GAT` |
| scalable GT | `SGFormer`, `Polynormer`, `GOAT`, `NodeFormer`, `NAGphormer` |
| 其他强 GT | `GraphGPS`, `Exphormer` |
| heterophily 专门模型 | `H2GCN`, `CPGNN`, `GPRGNN`, `FSGNN`, `GloGNN` |

这意味着本文不是在拿 classic GNN 和弱 GT 比，而是在拿它们和当时 node classification 上的一线 GT 与异质图专门模型比。

#### 3.3 公平性来自哪里

实验设计里最关键的不是“用了多少模型”，而是公平性来源。

公平性主要来自四点：

* **统一搜索空间**：classic GNN 与 GT 使用相近的搜索框架。
* **统一训练环境**：相同硬件环境与相近训练协议。
* **统一统计方式**：5 个随机种子，报告均值与标准差。
* **统一大图训练策略说明**：ogbn-proteins 用 neighbor sampling，pokec 与 ogbn-products 用随机分区 mini-batch。

如果少了这些条件，后面的“classic GNN 竞争力被恢复”就很容易沦为口号式判断；正是因为这些条件被补齐了，后面的结果才开始具有纠偏意味。

### 4. 主结果

主结果部分不能只写“谁第一”，而必须回答：**这些结果分别修正了哪种旧认识。**

在进入 Table 2-4 之前，先用一张总览表把三类场景里的“对比焦点”和“证据边界”固定下来：

| 场景 | 旧认识 | 本文修正后的更稳判断 | 最关键证据 | 仍不能直接推出 |
| --- | --- | --- | --- | --- |
| 同质图 | GT 在传统 node benchmark 上天然压过 classic GNN | 差距没有想象中稳，classic GNN 在同配方下足以回到第一梯队 | Table 2 中 `GCN*`、`GraphSAGE*`、`GAT*` 的系统性提升 | classic GNN 因而在所有任务上都优于 GT |
| 异质图 | classic GNN 只适合同质图 | 在补齐 residual 与深度搜索后，classic GNN 不再系统性出局 | Table 3 中 `Roman-Empire`、`Minesweeper` 等数据集上的显著改观 | 异质图已经不再需要专门模型 |
| 大图 | 大图天然属于 GT，message passing 会失效 | tuned classic GNN 仍具强竞争力，且可训练性本身就是比较的一部分 | Table 4 中 `ogbn-proteins`、`ogbn-products`、`pokec` 的结果与 `OOM` 现象 | classic GNN 已经解决所有长程依赖问题 |

#### 4.1 Table 2：Homophilous Graphs

> 对应主文 Table 2。下表保留最能体现对比关系的关键行。

| 模型 | Cora | CiteSeer | PubMed | Computer | Photo | CS | Physics | WikiCS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GraphGPS* | 83.87 | 72.73 | 79.94 | 91.79 | 94.89 | 94.04 | 96.71 | 78.66 |
| SGFormer* | 84.82 | 72.72 | 80.60 | 92.42 | 95.58 | 95.71 | 96.75 | 80.05 |
| Polynormer* | 83.43 | 72.19 | 79.35 | 93.78 | 96.57 | 95.42 | 97.18 | 80.26 |
| GCN | 81.60 | 71.60 | 78.80 | 89.65 | 92.70 | 92.92 | 96.18 | 77.47 |
| GCN* | 85.10 | 73.14 | 81.12 | 93.99 | 96.10 | 96.17 | 97.46 | 80.30 |
| GraphSAGE | 82.68 | 71.93 | 79.41 | 91.20 | 94.59 | 93.91 | 96.49 | 74.77 |
| GraphSAGE* | 83.88 | 72.26 | 79.72 | 93.25 | 96.78 | 96.38 | 97.19 | 80.69 |
| GAT | 83.00 | 72.10 | 79.00 | 90.78 | 93.87 | 93.61 | 96.17 | 76.91 |
| GAT* | 84.46 | 72.22 | 80.28 | 94.09 | 96.60 | 96.21 | 97.25 | 81.07 |

从这张表里真正要读出来的，不是“classic GNN 全面碾压 GT”，而是下面三点：

* **GCN* 的提升非常结构性**。它在 8 个同质图数据集上全部提升，幅度从 `+1.28` 到 `+4.34` 不等。
* **GraphSAGE* 和 GAT* 也不是局部偶然变好**。尤其 `GraphSAGE*` 在 `WikiCS` 上从 `74.77` 提升到 `80.69`，`GAT*` 在 `WikiCS` 上从 `76.91` 提升到 `81.07`。
* **同质图上很多原先看起来属于 GT 的优势，其实掺进了大量 GNN 配置不足因素。**

这张表最支持的判断是：

> **在传统 node classification benchmark 上，classic GNN 的“落后”并没有过去很多结论写得那么稳。**

但边界也要写清楚：

* 这张表说明的是同质图 benchmark 上的对比被重新改写了。
* 它还不足以单独推出 classic GNN 在异质图和大图上也同样强。

#### 4.2 Table 3：Heterophilous Graphs

> 对应主文 Table 3。这里最值得看的是 classic GNN 是否能修正“只能处理同质图”的默认批评。

| 模型 | Squirrel | Chameleon | Amazon-Ratings | Roman-Empire | Minesweeper | Questions |
| --- | --- | --- | --- | --- | --- | --- |
| GraphGPS* | 39.81 | 41.55 | 53.27 | 82.72 | 90.75 | 72.56 |
| SGFormer* | 42.65 | 45.21 | 54.14 | 80.01 | 91.42 | 73.81 |
| Polynormer* | 41.97 | 41.97 | 54.96 | 92.66 | 97.49 | 78.94 |
| GCN | 38.67 | 41.31 | 48.70 | 73.69 | 89.75 | 76.09 |
| GCN* | 45.01 | 46.29 | 53.80 | 91.27 | 97.86 | 79.02 |
| GraphSAGE | 36.09 | 37.77 | 53.63 | 85.74 | 93.51 | 76.44 |
| GraphSAGE* | 40.78 | 44.81 | 55.40 | 91.06 | 97.77 | 77.21 |
| GAT | 35.62 | 39.21 | 52.70 | 88.75 | 93.91 | 76.79 |
| GAT* | 41.73 | 44.13 | 55.54 | 90.63 | 97.73 | 77.95 |

这张表的冲击力比同质图更大，因为异质图本来就是 classic GNN 最容易被视为不适用的区域。

关键读法有四点：

* `GCN*` 在 `Squirrel`、`Chameleon`、`Minesweeper`、`Questions` 上都拿到很强成绩，在 `Roman-Empire` 上更是从 `73.69` 提到 `91.27`，提升 `17.58`。
* `GraphSAGE*` 和 `GAT*` 也都在多个异质图上显著上涨，说明这不是单模型现象。
* `Polynormer*` 依然在 `Roman-Empire` 上最好，但 classic GNN 已经不再只是边缘 baseline。
* 论文借这张表要反驳的不是“GT 完全无效”，而是“classic GNN 只适合同质图”这一过强断言。

更稳的收束句是：

> **异质图并没有自动把 classic GNN 判出局；过去很多关于异质图的悲观判断，至少在本文的比较协议下被明显削弱了。**

这里还有一个后文会回钩的要点：  
`Roman-Empire` 的提升非常大，而论文后面明确指出，这类提升与 `residual connection` 有强关系。这说明异质图上的结论，不该被简单理解成“GNN 天然变强了”，而更像是“正确的深层训练配置终于起作用了”。

#### 4.3 Table 4：Large-scale Graphs

> 对应主文 Table 4。这一块最重要，因为它直接触到“大图场景下 message passing 是否还有效”。

| 模型 | ogbn-proteins | ogbn-arxiv | ogbn-products | pokec |
| --- | --- | --- | --- | --- |
| GraphGPS* | 77.15 | 71.23 | OOM | OOM |
| SGFormer* | 79.92 | 72.76 | 81.54 | 82.44 |
| Polynormer* | 79.53 | 73.40 | 83.82 | 86.06 |
| GCN | 72.51 | 71.74 | 75.64 | 75.45 |
| GCN* | 77.29 | 73.53 | 82.33 | 86.33 |
| GraphSAGE | 77.68 | 71.49 | 78.29 | 75.63 |
| GraphSAGE* | 82.21 | 73.00 | 83.89 | 85.97 |
| GAT | 72.02 | 71.95 | 79.45 | 72.23 |
| GAT* | 85.01 | 73.30 | 80.99 | 86.19 |

这张表支持的判断比前两张更接近“现实工程判断”。

主要结论有五点：

* `GCN*` 在 `ogbn-arxiv` 与 `pokec` 上非常强，`pokec` 上从 `75.45` 直接提升到 `86.33`。
* `GraphSAGE*` 在 `ogbn-products` 上达到 `83.89`，略高于 `Polynormer*` 的 `83.82`。
* `GAT*` 在 `ogbn-proteins` 上达到 `85.01`，明显高于多种 GT baseline。
* 一部分 GT 在大图上直接 `OOM`，这意味着“大图上更先进”不能只看理论表达力，还必须看训练可行性。
* **message passing 在大图上并没有像很多叙事里那样自然失效。**

这张表最应该被读成：

> **大图场景并没有自动站在 GT 一边；在 node classification 上，classic GNN 依然保有非常强的性能与可训练性。**

但这一块也要收住：

* 这里证明的是 node classification 大图 benchmark 上的竞争力。
* 它不能直接推出 classic GNN 已经解决了所有 long-range dependency 问题。

### 5. 补充分析

补充分析的作用不是重复主结果，而是解释：**为什么这些 tuned classic GNN 会集体变强。**

#### 5.1 Table 5-7：四条超参观察

论文把消融结果压成了四条 observation。它们其实构成了一条非常清楚的机制链。

| Observation | 论文原文结论 | 最关键的证据 |
| --- | --- | --- |
| 1 | `normalization` 对大图更重要，对小图相对不敏感 | `GraphSAGE*` 在 `ogbn-proteins` 去掉 normalization 后从 `82.21` 降到 `77.42`；`GAT*` 从 `85.01` 降到 `80.32` |
| 2 | `dropout` 在三类图上都持续有益 | `GraphSAGE*` 在 `PubMed` 从 `79.72` 降到 `77.02`；在 `Roman-Empire` 从 `91.06` 降到 `84.49` |
| 3 | `residual connection` 在异质图和部分大图上特别关键 | `GCN*` 在 `Roman-Empire` 从 `91.27` 掉到 `74.84`，下降 `16.43` |
| 4 | 更深的网络对异质图更有帮助 | Figure 1 与 Table 12 都显示异质图最优层数常高于同质图 |

这四条 observation 其实把本文最关键的经验判断写得很清楚：

* **大图结果更依赖 normalization。**
* **泛化稳定性更依赖 dropout。**
* **异质图与深层训练更依赖 residual。**
* **异质图往往需要更深的 receptive field。**

换句话说，本文不是简单告诉你“要调参”，而是给出了更具体的经验分工。

#### 5.2 为什么异质图更依赖深层和残差

Figure 1 与 Table 12 一起看，读法更稳。

Table 12 给出的补证是：

| 模型与层数 | Roman-Empire | Minesweeper |
| --- | --- | --- |
| GCN* (12 层) | 90.68 | 97.76 |
| GCN* (15 层) | 90.74 | 97.65 |
| GCN* (20 层) | 90.43 | 97.52 |
| GraphSAGE* (12 层) | 90.96 | 97.02 |
| GraphSAGE* (15 层) | 90.78 | 97.77 |
| GraphSAGE* (20 层) | 90.22 | 97.73 |

这组结果说明：

* 异质图上的最优层数经常明显高于同质图常见的 `2-5` 层。
* 但“越深越好”也不成立，超过一定范围后收益很小，甚至开始回落。
* 因此，更稳的经验结论不是“堆到几十层”，而是：**异质图常常需要中等偏深的 classic GNN，并配合 residual 才训得稳。**

这也解释了为什么作者在 A.2 里特地把 heterophilous graphs 的层数搜索扩展到 `12/15/20`。  
不是为了追求极深网络的叙事，而是因为实验上确实看到异质图对深度更敏感。

#### 5.3 Appendix B：正文之外最值得保留的三个补证

Appendix B 里不是所有内容都值得搬进主结论，但有三块确实很重要。

| 附录部分 | 直接信息 | 在全文中的作用 |
| --- | --- | --- |
| B.1 | 加入 edge features 的 `GAT*` 在 `ogbn-proteins` 上可到 `87.82`，高于 `DeeperGCN` 的 `85.80` | 说明“更深”不是唯一出路，公平比较还要看输入设定 |
| B.2 | 异质图在 `12/15/20` 层时仍可维持高表现，但收益趋缓 | 支撑“异质图更适合更深网络，但不必无限加深” |
| B.3 | 早期把 `JK` 作为搜索项时结果也不错，但更细致调参后发现 `JK` 并非必要 | 说明真正关键的未必是加花样模块，而是认真搜索基础配置 |

这里尤其值得强调 B.1 的公平性含义。  
正文里没有拿 `DeeperGCN` 做主比较，是因为 `DeeperGCN` 在 `ogbn-proteins` 上用了 edge features，而正文主对比并没有给所有 baseline 统一引入这项输入。作者因此把它放到附录里，避免混入口径不一致的主结果。这是本文比较严谨的一面。

同时，这里还有一个细节值得客观记录：  
Appendix B.1 的表格给出 `GAT* (with edge features)` 在 `ogbn-proteins` 上是 `87.82`，但表后文字提到的是 `87.47`。这说明附录里存在一个轻微数值不一致。它不影响“显著高于 DeeperGCN”的方向性判断，但从笔记角度应该明确记下来。

#### 5.4 B.3 对 JK 的处理，其实很能说明这篇论文的气质

我认为 B.3 很值得单独保留，因为它透露出作者的实验态度。

作者早期把 `Jumping Knowledge` 也当成超参的一部分，早期结果见 Table 13-15。后面更细致搜索后，他们发现：

* 不用 `JK` 的结果通常已经与用 `JK` 接近。
* 某些数据集甚至不用 `JK` 更好。
* 所以正式论文里把 `JK` 从主搜索空间里去掉了。

这一点重要，不是因为 `JK` 本身，而是因为它说明：

> **作者最后保留的不是“更复杂”的方案，而是“更必要”的方案。**

这使得本文不像一篇为了刷新榜单而不断堆叠技巧的论文，更像一篇在不断剥离干扰因素、试图把真正关键变量留下来的 benchmark 论文。

#### 5.5 机制层收束：GT 优势为什么会被高估

把主文和附录合起来看，本文最后支持的不是“GT 全面失效”，而是更克制也更有说服力的判断：

* 过去不少对比里，classic GNN 的默认配置太弱。
* node classification 对基础训练配置非常敏感。
* 异质图和大图的结论尤其容易被 `residual`、`normalization` 和 `depth` 改写。
* 因此，GT 的领先里混入了相当一部分“比较方式优势”。

更稳的最终读法是：

> **这篇论文不是在否定 GT，而是在把 classic GNN 重新置于经过充分调参与公平比较后的竞争框架中。**

## 五、Conclusion

### 1. 核心结论

本文最稳定、最能被证据直接支撑的结论是：

> **在 node classification 上，classic GNN 是强 baseline；GT 对 classic GNN 的优势至少被过去一些不对称比较高估了。**

这不是空泛结论，而是被三类数据集、18 个 benchmark、以及 Table 5-7 的消融共同支撑的。

### 2. 主要判断

结合主文与附录，我认为本文真正建立起来的判断有四条：

* **第一，classic GNN 在 node classification 上被长期低估。** 这不是情绪性表述，而是由 `17/18` 的顶尖结果和大幅性能提升支撑的。
* **第二，GT 的优势并不消失，但没有原先一些叙事里那么绝对。** 尤其在 heterophilous 与 large-scale graphs 上，这一点更明显。
* **第三，训练配置本身就是模型比较的一部分。** 对基础 GNN 而言，`normalization`、`dropout`、`residual`、`depth` 不是细枝末节，而是决定结论方向的主因子。
* **第四，这篇论文最重要的输出其实是经验研究标准。** 它要求后续工作不能再拿配置不足的 GNN 去充当新模型的衬托对象。

### 3. 可借鉴点

如果把这篇论文当成后续做题或做研究的方法论素材，我认为最值得借鉴的是三点：

* **先把 baseline 做厚，再讨论新模型是否真的更强。**
* **对图模型的比较，不能只比传播核，还要比 recipe 是否对称。**
* **node classification 上的很多旧结论都值得在同配方下重检一次。**

对你自己的研究语境来说，这三点尤其有价值，因为动态图异常检测同样很容易出现“新模型赢了，但 baseline 其实没调到位”的问题。

### 4. 局限与改进

本文也有明确局限，不能省略。

1. 它只研究了 node classification，没有覆盖 graph classification 与 link prediction。
2. 它没有给出 GT 优势被高估的严格理论解释，核心仍是经验结论。
3. 它没有覆盖动态图、时序图、异构动态图或异常检测场景。
4. 它证明了 tuned classic GNN 很强，但没有证明 GT 在所有更复杂任务里都会失去优势。

因此，更准确的外推方式应该是：

* 把这篇论文视为 **baseline 重新审查的方法论模板**；
* 而不是把它直接当作“classic GNN 全面优于 Transformer”的普适结论。

### 5. 全文概括

这篇论文最有价值的地方，不在于提出了新的 GNN 模块，而在于重新规定了一个更高的经验研究标准：  
**在 node classification 上，只有当 classic GNN 也被认真调参、被完整比较时，GT 的优势才算是真正成立的优势。**

如果把全文再压成一句以后最方便回看的话，我会记成：

> **tunedGNN 真正改写的不是某个模型榜单，而是“classic GNN 到底有没有被公平比较过”这个更基础的问题。**
