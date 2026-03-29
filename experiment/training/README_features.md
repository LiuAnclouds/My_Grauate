# features.py 代码讲解

本文档对应 [features.py](./features.py)。目标不是做一页摘要，而是按 `teach-code-mentor` 的方式，把这份“特征缓存与图缓存构建代码”拆成可以真正读懂、真正复现的教程。

这份文档按两层来写：

1. 按实际执行路径讲：`build_features` 时这份文件到底做了什么
2. 按函数逐段讲：每个函数和数据结构的职责、输入输出、控制流、数据流、设计原因

如果你准备继续读训练框架，这份文件应该放在 [README_gnn_models.md](./README_gnn_models.md) 之前读。因为 `gbdt_models.py` 和 `gnn_models.py` 都建立在这里产出的缓存和接口之上。

## 1. 入口与目标

这份文件解决的是一个很具体的问题：

```text
原始 DGraph npz 数据太大，直接每次训练时临时现算特征和邻接结构成本太高
-> 所以先离线构建：
   1. 节点级核心特征缓存
   2. 邻域聚合特征缓存
   3. CSR 风格图结构缓存
   4. 一份 manifest 元数据
-> 后续训练阶段只需要按节点 id 直接取行
```

你可以把它理解成：

```text
raw npz dataset
-> feature engineering
-> feature cache / graph cache / manifest
-> FeatureStore / GraphCache
-> LightGBM / GNN 训练
```

## 2. 运行主线

这份文件真正的主路径不是“从上到下按文件顺序执行”，而是：

```text
run_training.py build_features
-> build_feature_artifacts(...)
-> build_phase_feature_artifacts(...)
-> 计算核心特征
-> 可选计算邻域特征
-> 写入图缓存
-> 写 feature_manifest.json
-> 训练时通过 FeatureStore / GraphCache 读取
```

如果你只想先抓住骨架，记住下面这 5 个符号就够了：

- `FeatureStore`
- `GraphCache`
- `build_phase_feature_artifacts(...)`
- `load_graph_cache(...)`
- `default_feature_groups(...)`

## 3. 文件里的核心对象与常量

### 3.1 常量

代码：

```python
RAW_FEATURE_COUNT = 17
NUM_EDGE_TYPES = 11
NUM_TIME_WINDOWS = 4
STRONG_PAIRS = ((2, 3), (6, 8), (15, 16))
```

这 4 个常量控制了整份文件的大方向：

- `RAW_FEATURE_COUNT = 17`
  说明原始节点特征 `x` 有 17 列。

- `NUM_EDGE_TYPES = 11`
  说明边类型共有 11 类。

- `NUM_TIME_WINDOWS = 4`
  说明时间相关特征默认按 4 个窗口构建，`M5` 也会沿用这套时间窗口。

- `STRONG_PAIRS`
  这是前期 EDA 后手工定下来的强特征组合对：
  - `x2/x3`
  - `x6/x8`
  - `x15/x16`

这些常量的作用不是“把逻辑写死”，而是把当前 benchmark 框架的设计选择集中放在文件最前面，方便统一修改。

### 3.2 `GraphCache`

代码：

```python
@dataclass(frozen=True)
class GraphCache:
    phase: str
    num_nodes: int
    max_day: int
    num_edge_types: int
    num_relations: int
    time_windows: list[dict[str, int]]
    out_ptr: np.ndarray
    out_neighbors: np.ndarray
    out_edge_type: np.ndarray
    out_edge_timestamp: np.ndarray
    in_ptr: np.ndarray
    in_neighbors: np.ndarray
    in_edge_type: np.ndarray
    in_edge_timestamp: np.ndarray
    first_active: np.ndarray
    node_time_bucket: np.ndarray
```

它不是训练逻辑，而是一个“图缓存读取结果包”。

它把后续 GNN 需要的所有结构化数组放在一起：

- 出边 CSR 缓存：`out_ptr/out_neighbors/out_edge_type/out_edge_timestamp`
- 入边 CSR 缓存：`in_ptr/in_neighbors/in_edge_type/in_edge_timestamp`
- 时间相关元数据：`max_day/time_windows/first_active/node_time_bucket`
- 图规模信息：`num_nodes/num_edge_types/num_relations`

如果你看过 [README_gnn_models.md](./README_gnn_models.md)，会发现 `sample_relation_subgraph(...)` 读的就是这里的这些字段。

设计原因：

- 不在训练时直接保留原始 `edge_index`
- 改成 CSR 风格指针数组
- 让“取某个节点的所有入边/出边”变成局部切片，而不是全图扫描

### 3.3 `FeatureStore`

代码：

```python
class FeatureStore:
    def __init__(
        self,
        phase: str,
        selected_groups: list[str],
        outdir: Path = FEATURE_OUTPUT_ROOT,
    ) -> None:
        self.phase = phase
        self.phase_dir = outdir / phase
        self.manifest = load_feature_manifest(phase, outdir=outdir)
        self.selected_groups = selected_groups
        self.core = np.load(self.phase_dir / self.manifest["core_file"], mmap_mode="r")
        self.neighbor = None
        neighbor_file = self.manifest.get("neighbor_file")
        if neighbor_file and (self.phase_dir / neighbor_file).exists():
            self.neighbor = np.load(self.phase_dir / neighbor_file, mmap_mode="r")
        self._group_specs = self._resolve_group_specs(selected_groups)
        self.feature_names = [
            name
            for spec in self._group_specs
            for name in spec["names"]
        ]
```

`FeatureStore` 是训练阶段最重要的读取器。它不负责构建特征，而负责：

- 读 manifest
- 打开 memmap 特征文件
- 根据指定特征组解析列区间
- 在训练时按 `node_ids` 抽取对应行

它的工作目标是：

```text
给定 phase + feature_groups + node_ids
-> 返回模型可直接使用的特征矩阵
```

继续看它的两个关键方法：

```python
def _resolve_group_specs(self, selected_groups: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for group_name in selected_groups:
        if group_name in self.manifest["core_groups"]:
            spec = dict(self.manifest["core_groups"][group_name])
            spec["source"] = "core"
            specs.append(spec)
            continue
        if group_name in self.manifest["neighbor_groups"]:
            if self.neighbor is None:
                raise FileNotFoundError(
                    f"{self.phase}: neighbor feature file is missing, run build_features first."
                )
            spec = dict(self.manifest["neighbor_groups"][group_name])
            spec["source"] = "neighbor"
            specs.append(spec)
            continue
        raise KeyError(f"{self.phase}: unknown feature group '{group_name}'")
    return specs
```

这个函数做的是“特征组解析”：

- 如果组名在 `core_groups` 里，就从核心特征矩阵读取
- 如果组名在 `neighbor_groups` 里，就从邻域特征矩阵读取
- 如果请求了邻域组但文件没构建，就直接报错

然后是训练时真正被频繁调用的 `take_rows(...)`：

```python
def take_rows(self, node_ids: np.ndarray) -> np.ndarray:
    rows = np.asarray(node_ids, dtype=np.int32)
    blocks: list[np.ndarray] = []
    for spec in self._group_specs:
        matrix = self.core if spec["source"] == "core" else self.neighbor
        blocks.append(
            np.asarray(matrix[rows, spec["start"] : spec["end"]], dtype=np.float32)
        )
    if not blocks:
        raise ValueError("No feature groups selected.")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
```

数据流是：

```text
node_ids
-> 对每个 group 找到对应矩阵和列范围
-> 按行切片
-> 按列拼接
-> 输出 [len(node_ids), input_dim] 的特征矩阵
```

这就是为什么后面的 `LightGBMExperiment.fit(...)` 和 `BaseGraphSAGEExperiment._tensorize_subgraph(...)` 都能用同一个读取接口。

`input_dim` 属性则是一个很直接的辅助接口：

```python
@property
def input_dim(self) -> int:
    return int(sum(spec["end"] - spec["start"] for spec in self._group_specs))
```

作用是给 GNN 构造网络时直接拿输入维度。

## 4. 时间窗、分组定义与列布局

### 4.1 `_build_edge_time_windows(...)`

代码：

```python
def _build_edge_time_windows(
    timestamps: np.ndarray,
    n_windows: int = NUM_TIME_WINDOWS,
) -> list[dict[str, int]]:
    quantiles = np.quantile(timestamps, np.linspace(0.0, 1.0, num=n_windows + 1))
    windows: list[dict[str, int]] = []
    for idx in range(n_windows):
        start_day = int(np.floor(quantiles[idx]))
        end_day = int(np.ceil(quantiles[idx + 1]))
        if idx > 0 and start_day <= windows[-1]["end_day"]:
            start_day = windows[-1]["end_day"] + 1
        if end_day < start_day:
            end_day = start_day
        windows.append(
            {
                "window_idx": idx + 1,
                "start_day": start_day,
                "end_day": end_day,
            }
        )
    windows[-1]["end_day"] = int(np.max(timestamps))
    return windows
```

这个函数的任务是：

```text
输入所有边时间戳
-> 按分位数切成 4 个时间窗
-> 输出每个时间窗的 [start_day, end_day]
```

为什么用分位数而不是均匀切天数：

- 数据时间分布可能不均匀
- 如果直接按日期均匀切，某些窗可能几乎没边
- 分位数切法能让每个时间窗边量更接近

代码里的两个保护分支要注意：

- `if idx > 0 and start_day <= windows[-1]["end_day"]`
  作用是防止相邻时间窗发生重叠。

- `if end_day < start_day`
  作用是防止边界被量化后出现非法区间。

最后一句：

```python
windows[-1]["end_day"] = int(np.max(timestamps))
```

是为了保证最后一个窗口一定真正覆盖到最大时间。

### 4.2 `_assign_node_time_bucket(...)`

代码：

```python
def _assign_node_time_bucket(
    first_active: np.ndarray,
    time_windows: list[dict[str, int]],
) -> np.ndarray:
    buckets = np.zeros(first_active.shape[0], dtype=np.int8)
    for window in time_windows:
        mask = (first_active >= window["start_day"]) & (first_active <= window["end_day"])
        buckets[mask] = int(window["window_idx"] - 1)
    return buckets
```

这个函数回答的问题是：

```text
每个节点属于哪个时间桶
```

当前实现采用的规则是：

- 节点按 `first_active` 归桶
- 第 1 个窗口映射成桶 `0`
- 第 2 个窗口映射成桶 `1`
- 以此类推

这一步主要服务于 `M5` 的时间分 batch 训练，而不是说模型只看 `first_active`。  
`last_active` 和 `active_span` 仍然会作为时间特征进入模型。

### 4.3 `_group_definition(...)`

代码：

```python
def _group_definition() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    core_groups: dict[str, list[str]] = {
        "raw_x": [f"x{i}" for i in range(RAW_FEATURE_COUNT)],
        "missing_mask": [f"x{i}_is_neg1" for i in range(RAW_FEATURE_COUNT)],
        "missing_summary": ["missing_count"],
        "strong_combo": [],
        "graph_direction": [
            "indegree",
            "outdegree",
            "total_degree",
            "out_over_in_plus1",
            "in_over_out_plus1",
        ],
        "edge_type": [],
        "background": [
            "bg_in_count",
            "bg_out_count",
            "bg_total_count",
            "bg_in_ratio",
            "bg_out_ratio",
            "bg_total_ratio",
            "touch_background",
        ],
        "time": [
            "first_active",
            "last_active",
            "active_span",
        ],
    }
    for left, right in STRONG_PAIRS:
        core_groups["strong_combo"].extend(
            [
                f"x{left}_x{right}_mean",
                f"x{left}_x{right}_absdiff",
            ]
        )
    for edge_type in range(1, NUM_EDGE_TYPES + 1):
        core_groups["edge_type"].extend(
            [
                f"in_type_{edge_type}_count",
                f"out_type_{edge_type}_count",
            ]
        )
        core_groups["background"].extend(
            [
                f"bg_in_type_{edge_type}_count",
                f"bg_out_type_{edge_type}_count",
            ]
        )
    for window_idx in range(1, NUM_TIME_WINDOWS + 1):
        core_groups["time"].extend(
            [
                f"window_{window_idx}_in_count",
                f"window_{window_idx}_out_count",
                f"window_{window_idx}_total_count",
            ]
        )
    core_groups["time"].append("early_late_total_ratio")

    neighbor_groups = {"neighbor": []}
    for reducer in ("mean", "max", "missing_ratio"):
        for prefix in ("in", "out"):
            neighbor_groups["neighbor"].extend(
                [f"{prefix}_neighbor_{reducer}_x{i}" for i in range(RAW_FEATURE_COUNT)]
            )
    return core_groups, neighbor_groups
```

这是整份文件最重要的“列名设计函数”。它并不计算值，而是定义：

- 每个特征组有哪些列
- 每列叫什么名字
- 哪些列属于核心缓存
- 哪些列属于邻域缓存

你可以把它理解成：

```text
feature schema 定义器
```

这里最值得注意的组有：

- `raw_x`
  原始 17 维节点特征

- `missing_mask`
  17 维缺失标记

- `missing_summary`
  总缺失数

- `strong_combo`
  `x2/x3`、`x6/x8`、`x15/x16` 的均值与差值

- `graph_direction`
  入度、出度、总度、入出度比值

- `edge_type`
  11 类边型的入边/出边计数

- `background`
  与背景节点接触的计数和比例

- `time`
  首次活跃、最后活跃、活跃跨度、时间窗边计数、早晚活跃比

- `neighbor`
  1-hop 入邻居/出邻居的 mean、max、missing_ratio 聚合

设计原因：

- 把“特征算值逻辑”和“特征组组织逻辑”分开
- 后续模型只需要说“我要哪些组”，不需要知道具体列偏移

### 4.4 `_allocate_group_spans(...)`

代码：

```python
def _allocate_group_spans(groups: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    spans: dict[str, dict[str, Any]] = {}
    col = 0
    for group_name, feature_names in groups.items():
        spans[group_name] = {
            "start": col,
            "end": col + len(feature_names),
            "names": feature_names,
        }
        col += len(feature_names)
    return spans
```

这个函数的作用是把“列名列表”转换成“列区间定义”。

输入：

```text
group_name -> [feature_name1, feature_name2, ...]
```

输出：

```text
group_name -> {
  start: 起始列号,
  end: 结束列号,
  names: 列名列表
}
```

为什么这一步必要：

- 特征矩阵最终是一个大二维数组
- 训练时需要知道每个组在这个大矩阵里的列范围
- manifest 里保存的就是这里产出的 `start/end/names`

如果没有这个函数，后面 `FeatureStore.take_rows(...)` 根本没法做到按组抽列。

## 5. 图缓存写入：把边改造成 CSR 风格

### 5.1 `_write_graph_arrays(...)`

代码：

```python
def _write_graph_arrays(
    phase_dir: Path,
    prefix: str,
    centers: np.ndarray,
    neighbors: np.ndarray,
    edge_type: np.ndarray,
    edge_timestamp: np.ndarray,
    num_nodes: int,
) -> dict[str, str]:
    counts = np.bincount(centers, minlength=num_nodes).astype(np.int64, copy=False)
    ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    order = np.argsort(centers, kind="stable")
    neighbor_ordered = neighbors[order].astype(np.int32, copy=False)
    edge_type_ordered = edge_type[order].astype(np.int16, copy=False)
    edge_timestamp_ordered = edge_timestamp[order].astype(np.int32, copy=False)

    graph_dir = ensure_dir(phase_dir / "graph")
    ptr_path = graph_dir / f"{prefix}_ptr.npy"
    neighbor_path = graph_dir / f"{prefix}_neighbors.npy"
    edge_type_path = graph_dir / f"{prefix}_edge_type.npy"
    edge_time_path = graph_dir / f"{prefix}_edge_timestamp.npy"
    np.save(ptr_path, ptr)
    np.save(neighbor_path, neighbor_ordered)
    np.save(edge_type_path, edge_type_ordered)
    np.save(edge_time_path, edge_timestamp_ordered)
    return {
        f"{prefix}_ptr": str(ptr_path.name),
        f"{prefix}_neighbors": str(neighbor_path.name),
        f"{prefix}_edge_type": str(edge_type_path.name),
        f"{prefix}_edge_timestamp": str(edge_time_path.name),
    }
```

这段代码做的是一个典型优化：

```text
naive 方案：每次训练时全图扫描 edge_index 找某节点的邻边
当前方案：提前把边按中心节点排序，并写成 CSR 风格数组
```

具体过程：

- `counts = np.bincount(centers, ...)`
  统计每个中心节点有多少条边

- `ptr = np.zeros(num_nodes + 1, ...)`
  初始化 CSR 指针数组

- `np.cumsum(counts, out=ptr[1:])`
  构造指针，使得：
  - 节点 `i` 的边区间是 `[ptr[i], ptr[i+1])`

- `order = np.argsort(centers, kind="stable")`
  按中心节点排序边

- `neighbor_ordered / edge_type_ordered / edge_timestamp_ordered`
  用同一个排序索引同步重排邻居、边类型和时间

然后把 4 个数组分别写盘：

- `ptr`
- `neighbors`
- `edge_type`
- `edge_timestamp`

这个函数会被调用两次：

- 一次写出边缓存
- 一次写入边缓存

所以后续 GNN 能很快做：

```python
start = graph.in_ptr[center]
end = graph.in_ptr[center + 1]
neighbors = graph.in_neighbors[start:end]
```

### 5.2 `_bincount_float(...)`

代码：

```python
def _bincount_float(indices: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(indices, weights=weights, minlength=size).astype(np.float32, copy=False)
```

这是一个很小的工具函数，作用是：

- 对浮点权重做 `np.bincount`
- 并统一转成 `float32`

它主要服务于邻域均值特征构建。

## 6. 邻域特征构建

### 6.1 `_build_neighbor_features(...)`

代码：

```python
def _build_neighbor_features(
    data: PhaseData,
    phase_dir: Path,
    indegree: np.ndarray,
    outdegree: np.ndarray,
    x: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[str, dict[str, dict[str, Any]]]:
    _, neighbor_groups = _group_definition()
    neighbor_spans = _allocate_group_spans(neighbor_groups)
    shape = (data.num_nodes, neighbor_spans["neighbor"]["end"])
    neighbor_path = phase_dir / "neighbor_features.npy"
    matrix = np.lib.format.open_memmap(
        neighbor_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    src = data.edge_index[:, 0]
    dst = data.edge_index[:, 1]
    in_den = np.maximum(indegree.astype(np.float32, copy=False), 1.0)
    out_den = np.maximum(outdegree.astype(np.float32, copy=False), 1.0)

    col = 0
    for reducer in ("mean", "max", "missing_ratio"):
        for feature_idx in range(RAW_FEATURE_COUNT):
            values = x[:, feature_idx] if reducer != "missing_ratio" else missing_mask[:, feature_idx]
            in_center_values = values[src]
            out_center_values = values[dst]
            if reducer == "mean" or reducer == "missing_ratio":
                in_result = _bincount_float(dst, in_center_values, data.num_nodes) / in_den
                out_result = _bincount_float(src, out_center_values, data.num_nodes) / out_den
            else:
                in_result = np.full(data.num_nodes, -np.inf, dtype=np.float32)
                out_result = np.full(data.num_nodes, -np.inf, dtype=np.float32)
                np.maximum.at(in_result, dst, in_center_values.astype(np.float32, copy=False))
                np.maximum.at(out_result, src, out_center_values.astype(np.float32, copy=False))
                in_result[indegree == 0] = 0.0
                out_result[outdegree == 0] = 0.0
            matrix[:, col] = in_result
            matrix[:, col + RAW_FEATURE_COUNT] = out_result
            col += 1
        col += RAW_FEATURE_COUNT
    del matrix
    return neighbor_path.name, neighbor_spans
```

这个函数负责构建 `M3` 专用的 1-hop 邻域聚合特征。

它的输出不是单独若干文件，而是一个完整矩阵 `neighbor_features.npy`，其中每一行对应一个节点，每一列是某种邻域统计。

### 它到底在算什么

对每个原始特征 `x0...x16`，它分别计算：

- 入邻居均值
- 出邻居均值
- 入邻居最大值
- 出邻居最大值
- 入邻居缺失比例
- 出邻居缺失比例

也就是：

```text
3 种 reducer × 2 个方向 × 17 维原始特征
```

### 为什么这样写

最朴素的做法是：

- 对每个节点
- 遍历它所有邻居
- 再逐列算均值/最大值/缺失率

这在百万级图上会非常慢。

当前实现的优化点是：

- 利用 `src/dst` 一次性把“边上的源特征”对齐
- 用 `np.bincount` 做方向均值
- 用 `np.maximum.at` 做方向最大值

这样更新是局部数组操作，不需要 Python 层对每个节点写大循环。

### 代码逐段解释

- `neighbor_groups = _group_definition()`
  先拿到邻域特征的列名定义。

- `neighbor_spans = _allocate_group_spans(neighbor_groups)`
  把邻域列名映射成列范围定义。

- `open_memmap(...)`
  直接在磁盘上创建一个 `neighbor_features.npy`，避免大矩阵常驻内存。

- `src = data.edge_index[:, 0]`
  所有边的源节点。

- `dst = data.edge_index[:, 1]`
  所有边的目标节点。

- `in_den / out_den`
  用来做均值分母，并用 `max(..., 1.0)` 防止零度节点除零。

接下来是双层循环：

- 外层 `reducer in ("mean", "max", "missing_ratio")`
- 内层 `feature_idx in range(17)`

`values` 的选择逻辑：

- 如果是 `mean` 或 `max`，就用原始 `x[:, feature_idx]`
- 如果是 `missing_ratio`，就用 `missing_mask[:, feature_idx]`

然后：

- `in_center_values = values[src]`
  表示“这条边的源节点值”，最后按 `dst` 聚合，得到每个节点的入邻居聚合值。

- `out_center_values = values[dst]`
  表示“这条边的目标节点值”，最后按 `src` 聚合，得到每个节点的出邻居聚合值。

对 `mean` 和 `missing_ratio`：

```python
in_result = _bincount_float(dst, in_center_values, data.num_nodes) / in_den
out_result = _bincount_float(src, out_center_values, data.num_nodes) / out_den
```

就是：

```text
先按中心节点累加
-> 再除以入度或出度
-> 得到均值
```

对 `max`：

```python
np.maximum.at(in_result, dst, in_center_values)
np.maximum.at(out_result, src, out_center_values)
```

就是：

```text
按中心节点逐位置取最大值
```

最后把结果写回两组列：

- `matrix[:, col] = in_result`
- `matrix[:, col + RAW_FEATURE_COUNT] = out_result`

这里的列组织方式很紧凑：

- 前 17 列是入方向
- 后 17 列是出方向

循环结束后：

- `del matrix`
  显式释放 memmap 句柄

- 返回：
  - 文件名
  - 邻域组的 span 定义

### 这段代码的一个重要现实含义

当前实现里，`mean/max` 是直接基于原始 `x` 算的，所以 `-1` 会直接参与聚合；而 `missing_ratio` 是单独用 `missing_mask` 算的。

也就是说当前策略是：

```text
保留原始 -1 数值
+ 额外告诉模型邻域缺失比例
```

这对 baseline 是可跑的，但如果后面你要做更精细的缺失感知版本，这里会是重点升级点。

## 7. 主函数：`build_phase_feature_artifacts(...)`

### 7.1 这段代码解决什么问题

它是整份文件最核心的函数，负责：

```text
给定一个 phase
-> 读取原始 npz
-> 计算全部核心特征
-> 可选计算邻域特征
-> 写图缓存
-> 写时间元数据
-> 写 manifest
```

你可以把它理解成：

```text
单 phase 的离线特征工厂
```

### 7.2 完整代码主干

```python
def build_phase_feature_artifacts(
    phase: str,
    outdir: Path = FEATURE_OUTPUT_ROOT,
    build_neighbor: bool = True,
) -> dict[str, Any]:
    data = load_phase(phase, repo_root=REPO_ROOT)
    phase_dir = ensure_dir(outdir / phase)
    core_groups, _ = _group_definition()
    core_spans = _allocate_group_spans(core_groups)
    core_path = phase_dir / "core_features.npy"
    core = np.lib.format.open_memmap(
        core_path,
        mode="w+",
        dtype=np.float32,
        shape=(data.num_nodes, core_spans["time"]["end"]),
    )
    ...
```

这段函数很长，下面按执行顺序拆开讲。

### 7.3 第一步：加载原始数据并为核心矩阵分配空间

```python
data = load_phase(phase, repo_root=REPO_ROOT)
phase_dir = ensure_dir(outdir / phase)
core_groups, _ = _group_definition()
core_spans = _allocate_group_spans(core_groups)
core_path = phase_dir / "core_features.npy"
core = np.lib.format.open_memmap(
    core_path,
    mode="w+",
    dtype=np.float32,
    shape=(data.num_nodes, core_spans["time"]["end"]),
)
```

这里做了几件事：

- 用 `load_phase(...)` 读原始 `npz`
- 为当前 phase 建立输出目录
- 根据特征组定义得到核心特征矩阵的总列数
- 用 memmap 在磁盘上创建 `core_features.npy`

为什么 `shape` 用的是 `core_spans["time"]["end"]`：

- `time` 是核心组定义里的最后一个组
- 它的 `end` 恰好就是“核心矩阵总列数”

### 7.4 第二步：准备所有原始统计量

```python
x = np.asarray(data.x, dtype=np.float32)
missing_mask = (x == -1).astype(np.float32, copy=False)
indegree, outdegree, total_degree = compute_degree_arrays(data)
temporal = compute_temporal_core(data)
first_active = temporal["first_active"].astype(np.int32, copy=False)
last_active = temporal["last_active"].astype(np.int32, copy=False)
active_span = temporal["active_span"].astype(np.int32, copy=False)
time_windows = _build_edge_time_windows(data.edge_timestamp)
node_time_bucket = _assign_node_time_bucket(first_active, time_windows)

src = data.edge_index[:, 0]
dst = data.edge_index[:, 1]
background_mask = np.isin(data.y, (2, 3))
```

这一段就是“特征原料准备区”。

它准备了后面所有模块会用到的基础量：

- `x`
  原始节点特征

- `missing_mask`
  缺失标记。这里 `-1` 没有被填补，而是保留原值并额外构建 mask。

- `indegree/outdegree/total_degree`
  结构统计基础量

- `first_active/last_active/active_span`
  时间统计基础量

- `time_windows/node_time_bucket`
  `M5` 和时间特征都会用到

- `src/dst`
  边数组拆开的源节点/目标节点索引

- `background_mask`
  标记哪些节点属于背景类 `2/3`

### 7.5 第三步：写入原始特征、缺失特征与强组合特征

```python
col = 0
core[:, col : col + RAW_FEATURE_COUNT] = x
col += RAW_FEATURE_COUNT

core[:, col : col + RAW_FEATURE_COUNT] = missing_mask
col += RAW_FEATURE_COUNT

core[:, col] = np.sum(missing_mask, axis=1, dtype=np.float32)
col += 1

for left, right in STRONG_PAIRS:
    core[:, col] = (x[:, left] + x[:, right]) / 2.0
    core[:, col + 1] = np.abs(x[:, left] - x[:, right])
    col += 2
```

这里是最基础的表格特征部分：

- 先写原始 `x`
- 再写 `missing_mask`
- 再写总缺失数 `missing_count`
- 然后为每个强组合对写两列：
  - 均值
  - 绝对差

注意当前实现中，组合特征是直接在原始 `x` 上算的，所以如果某列是 `-1`，它会直接参与组合运算。

### 7.6 第四步：写入结构方向特征

```python
indegree_f = indegree.astype(np.float32, copy=False)
outdegree_f = outdegree.astype(np.float32, copy=False)
total_degree_f = total_degree.astype(np.float32, copy=False)
core[:, col] = indegree_f
core[:, col + 1] = outdegree_f
core[:, col + 2] = total_degree_f
core[:, col + 3] = outdegree_f / (indegree_f + 1.0)
core[:, col + 4] = indegree_f / (outdegree_f + 1.0)
col += 5
```

这部分是图方向统计：

- 入度
- 出度
- 总度
- 出入度比
- 入出度比

这里统一加 `1.0`，是为了避免零度节点除零。

### 7.7 第五步：写入按边类型拆分的方向计数

```python
for edge_type in range(1, NUM_EDGE_TYPES + 1):
    mask_t = data.edge_type == edge_type
    core[:, col] = np.bincount(dst[mask_t], minlength=data.num_nodes).astype(np.float32, copy=False)
    core[:, col + 1] = np.bincount(src[mask_t], minlength=data.num_nodes).astype(np.float32, copy=False)
    col += 2
```

对 11 类边，每类都写两列：

- 该类型入边数
- 该类型出边数

这样做的好处是：

- 不只是知道“总入度/总出度”
- 还知道“哪种关系类型贡献了多少入边/出边”

### 7.8 第六步：写入背景桥接特征

```python
bg_out_mask = background_mask[dst]
bg_in_mask = background_mask[src]
bg_out_count = np.bincount(src[bg_out_mask], minlength=data.num_nodes).astype(np.float32, copy=False)
bg_in_count = np.bincount(dst[bg_in_mask], minlength=data.num_nodes).astype(np.float32, copy=False)
bg_total_count = bg_in_count + bg_out_count
core[:, col] = bg_in_count
core[:, col + 1] = bg_out_count
core[:, col + 2] = bg_total_count
core[:, col + 3] = bg_in_count / (indegree_f + 1.0)
core[:, col + 4] = bg_out_count / (outdegree_f + 1.0)
core[:, col + 5] = bg_total_count / (total_degree_f + 1.0)
core[:, col + 6] = (bg_total_count > 0).astype(np.float32, copy=False)
col += 7
```

这部分是当前项目很有针对性的特征设计。

它在回答：

```text
某个目标节点与背景节点 2/3 的接触强不强
```

写入的量包括：

- 背景入边数
- 背景出边数
- 背景总接触数
- 各种比例
- 是否接触过背景节点

然后还进一步按边类型细分背景接触：

```python
for edge_type in range(1, NUM_EDGE_TYPES + 1):
    mask_t = data.edge_type == edge_type
    mask_bg_in = mask_t & bg_in_mask
    mask_bg_out = mask_t & bg_out_mask
    core[:, col] = np.bincount(dst[mask_bg_in], minlength=data.num_nodes).astype(np.float32, copy=False)
    core[:, col + 1] = np.bincount(src[mask_bg_out], minlength=data.num_nodes).astype(np.float32, copy=False)
    col += 2
```

也就是说，不仅知道“是否接触背景节点”，还知道“通过哪类边接触背景节点”。

### 7.9 第七步：写入时间特征

```python
core[:, col] = first_active.astype(np.float32, copy=False)
core[:, col + 1] = last_active.astype(np.float32, copy=False)
core[:, col + 2] = active_span.astype(np.float32, copy=False)
col += 3

window_total_counts: list[np.ndarray] = []
for window in time_windows:
    if window["window_idx"] == NUM_TIME_WINDOWS:
        mask_w = (data.edge_timestamp >= window["start_day"]) & (
            data.edge_timestamp <= window["end_day"]
        )
    else:
        mask_w = (data.edge_timestamp >= window["start_day"]) & (
            data.edge_timestamp < window["end_day"] + 1
        )
    in_count = np.bincount(dst[mask_w], minlength=data.num_nodes).astype(np.float32, copy=False)
    out_count = np.bincount(src[mask_w], minlength=data.num_nodes).astype(np.float32, copy=False)
    total_count = in_count + out_count
    window_total_counts.append(total_count)
    core[:, col] = in_count
    core[:, col + 1] = out_count
    core[:, col + 2] = total_count
    col += 3
early_total = window_total_counts[0] + window_total_counts[1]
late_total = window_total_counts[2] + window_total_counts[3]
core[:, col] = (early_total + 1.0) / (late_total + 1.0)
col += 1
```

时间特征分两部分：

第一部分是节点级时间统计：

- 首次活跃
- 最后活跃
- 活跃跨度

第二部分是时间窗内的边计数：

- 每个时间窗的入边数
- 每个时间窗的出边数
- 每个时间窗的总边数

最后再补一个：

- `early_late_total_ratio`
  表示早期活跃与晚期活跃的相对强度

这里对最后一个时间窗做了单独判断，是为了保证边界闭区间覆盖完整。

### 7.10 第八步：可选构建邻域特征

```python
neighbor_file = None
neighbor_spans: dict[str, dict[str, Any]] = {}
if build_neighbor:
    neighbor_file, neighbor_spans = _build_neighbor_features(
        data=data,
        phase_dir=phase_dir,
        indegree=indegree,
        outdegree=outdegree,
        x=x,
        missing_mask=missing_mask,
    )
```

这一步很简单但很关键：

- 如果 `build_neighbor=False`
  只构建核心特征和图缓存，速度更快

- 如果 `build_neighbor=True`
  再额外构建 `M3` 需要的邻域特征

这就是为什么 `run_training.py build_features --skip-neighbor` 能做 smoke test 加速。

### 7.11 第九步：写出入边图缓存与时间元数据

```python
graph_meta = {}
graph_meta.update(
    _write_graph_arrays(
        phase_dir=phase_dir,
        prefix="out",
        centers=src,
        neighbors=dst,
        edge_type=data.edge_type,
        edge_timestamp=data.edge_timestamp,
        num_nodes=data.num_nodes,
    )
)
graph_meta.update(
    _write_graph_arrays(
        phase_dir=phase_dir,
        prefix="in",
        centers=dst,
        neighbors=src,
        edge_type=data.edge_type,
        edge_timestamp=data.edge_timestamp,
        num_nodes=data.num_nodes,
    )
)
graph_dir = ensure_dir(phase_dir / "graph")
first_active_file = graph_dir / "first_active.npy"
node_time_bucket_file = graph_dir / "node_time_bucket.npy"
np.save(first_active_file, first_active)
np.save(node_time_bucket_file, node_time_bucket)
```

这一段把 GNN 需要的结构缓存全部写盘：

- 一次写出边缓存
- 一次写入边缓存
- 再写 `first_active`
- 再写 `node_time_bucket`

设计原因：

- GNN 的 `sample_relation_subgraph(...)` 同时需要快速访问入边和出边
- 时间模型 `M5` 需要节点时间桶

所以 `features.py` 不只是特征构建器，它同时还是图缓存构建器。

### 7.12 第十步：写 manifest 并返回

```python
manifest = {
    "phase": phase,
    "num_nodes": data.num_nodes,
    "num_edges": data.num_edges,
    "core_file": core_path.name,
    "neighbor_file": neighbor_file,
    "core_groups": core_spans,
    "neighbor_groups": neighbor_spans,
    "graph_meta": {
        "dir": "graph",
        "files": graph_meta,
        "first_active_file": first_active_file.name,
        "node_time_bucket_file": node_time_bucket_file.name,
        "num_edge_types": NUM_EDGE_TYPES,
        "num_relations": NUM_EDGE_TYPES * 2,
        "max_day": int(data.edge_timestamp.max()),
        "time_windows": time_windows,
    },
}
write_json(phase_dir / "feature_manifest.json", manifest)
return manifest
```

manifest 是这一整套缓存系统的“目录说明书”。

它记录了：

- 这个 phase 的规模信息
- 核心特征文件名
- 邻域特征文件名
- 每个特征组的列范围和列名
- 图缓存文件位置
- 时间窗口和关系数等元数据

后面所有读取逻辑都依赖它：

- `FeatureStore` 用它找列范围
- `load_graph_cache(...)` 用它找图缓存文件

所以如果没有 `feature_manifest.json`，后面训练阶段几乎没法自动装配。

## 8. 公开接口：训练阶段如何消费这些产物

### 8.1 `build_feature_artifacts(...)`

代码：

```python
def build_feature_artifacts(
    phases: list[str],
    outdir: Path = FEATURE_OUTPUT_ROOT,
    build_neighbor: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in phases:
        summary[phase] = build_phase_feature_artifacts(
            phase=phase,
            outdir=outdir,
            build_neighbor=build_neighbor,
        )
    return summary
```

这个函数只是一个多 phase 包装器。  
它回答的问题是：

```text
如果我要一次构建 phase1 和 phase2，要怎么复用单 phase 逻辑
```

设计上保持很薄，避免把复杂逻辑写两遍。

### 8.2 `load_feature_manifest(...)`

代码：

```python
def load_feature_manifest(phase: str, outdir: Path = FEATURE_OUTPUT_ROOT) -> dict[str, Any]:
    return json.loads((outdir / phase / "feature_manifest.json").read_text(encoding="utf-8"))
```

这是一个极薄的读取器，但它是 `FeatureStore` 和 `load_graph_cache(...)` 的共同入口。

### 8.3 `load_graph_cache(...)`

代码：

```python
def load_graph_cache(phase: str, outdir: Path = FEATURE_OUTPUT_ROOT) -> GraphCache:
    manifest = load_feature_manifest(phase, outdir=outdir)
    graph_dir = outdir / phase / manifest["graph_meta"]["dir"]
    files = manifest["graph_meta"]["files"]
    return GraphCache(
        phase=phase,
        num_nodes=int(manifest["num_nodes"]),
        max_day=int(manifest["graph_meta"]["max_day"]),
        num_edge_types=int(manifest["graph_meta"]["num_edge_types"]),
        num_relations=int(manifest["graph_meta"]["num_relations"]),
        time_windows=list(manifest["graph_meta"]["time_windows"]),
        out_ptr=np.load(graph_dir / files["out_ptr"], mmap_mode="r"),
        out_neighbors=np.load(graph_dir / files["out_neighbors"], mmap_mode="r"),
        out_edge_type=np.load(graph_dir / files["out_edge_type"], mmap_mode="r"),
        out_edge_timestamp=np.load(graph_dir / files["out_edge_timestamp"], mmap_mode="r"),
        in_ptr=np.load(graph_dir / files["in_ptr"], mmap_mode="r"),
        in_neighbors=np.load(graph_dir / files["in_neighbors"], mmap_mode="r"),
        in_edge_type=np.load(graph_dir / files["in_edge_type"], mmap_mode="r"),
        in_edge_timestamp=np.load(graph_dir / files["in_edge_timestamp"], mmap_mode="r"),
        first_active=np.load(graph_dir / manifest["graph_meta"]["first_active_file"], mmap_mode="r"),
        node_time_bucket=np.load(
            graph_dir / manifest["graph_meta"]["node_time_bucket_file"],
            mmap_mode="r",
        ),
    )
```

这个函数把 manifest 中记录的路径真正打开，封装成一个 `GraphCache`。

关键点有两个：

- 全部用 `mmap_mode="r"`
  说明这些数组是只读内存映射，不会整文件复制进内存

- 它把磁盘文件重新组织成后续 GNN 最方便消费的字段结构

### 8.4 `default_feature_groups(...)`

代码：

```python
def default_feature_groups(model_name: str) -> list[str]:
    if model_name == "m1_tabular":
        return ["raw_x", "missing_mask", "missing_summary"]
    if model_name == "m2_hybrid":
        return [
            "raw_x",
            "missing_mask",
            "missing_summary",
            "strong_combo",
            "graph_direction",
            "edge_type",
            "background",
            "time",
        ]
    if model_name == "m3_neighbor":
        return default_feature_groups("m2_hybrid") + ["neighbor"]
    if model_name in {"m4_graphsage", "m5_temporal_graphsage"}:
        return default_feature_groups("m2_hybrid")
    raise KeyError(f"Unsupported model name: {model_name}")
```

这是训练框架里“模型名 -> 默认特征组”的总开关。

它有几个很重要的设计含义：

- `M1`
  只看表格缺失感知特征

- `M2`
  在 `M1` 上加入图统计、背景、时间特征，定义为 benchmark

- `M3`
  在 `M2` 上再加邻域聚合特征

- `M4/M5`
  输入节点特征默认沿用 `M2`
  图结构消息传递由 GNN 自己负责，不额外读取 `neighbor`

所以 `features.py` 不只是“计算特征”，它实际上还参与了整套实验协议定义。

## 9. 端到端执行总结

把整份文件浓缩成最短流程，就是：

```text
phase npz
-> load_phase
-> 计算 raw/missing/graph/time/background/core features
-> 可选计算 neighbor features
-> 写 core_features.npy / neighbor_features.npy
-> 写 in/out CSR graph cache
-> 写 first_active / node_time_bucket
-> 写 feature_manifest.json
-> 训练阶段通过 FeatureStore / GraphCache 读取
```

如果你在训练代码里看到：

- `FeatureStore("phase1", feature_groups, ...)`
- `load_graph_cache("phase1", ...)`

它们的上游全部都来自这份文件。

## 10. 这份实现的设计优点

### 10.1 把大图训练前移成离线预处理

好处是：

- 训练阶段更快
- 结构更清晰
- 不需要每次重复算同样的图统计

### 10.2 用 memmap 控制内存占用

大规模图最怕把所有特征和所有邻接同时搬进内存。  
这里的写法把：

- 核心特征
- 邻域特征
- 图缓存

都做成了磁盘数组 + 按需切片读取。

### 10.3 把“列定义”和“列取值”分离

`_group_definition(...)` 和 `_allocate_group_spans(...)` 专门定义 schema；  
`build_phase_feature_artifacts(...)` 专门负责算值。

这样训练阶段只关心“我要哪些组”，不关心“这些组具体在第几列”。

## 11. 读这份文件最容易犯的误区

### 11.1 误以为这里只有特征工程

实际上它同时负责：

- 节点特征缓存
- 邻域聚合特征
- GNN 图缓存
- 时间桶元数据
- 特征协议定义

### 11.2 误以为 `neighbor` 是 GNN 必需输入

不是。

- `M3` 会用 `neighbor`
- `M4/M5` 默认不使用 `neighbor`
- GNN 自己从 `GraphCache` 里采样图结构，再做消息传递

### 11.3 误以为 `-1` 被填补掉了

当前实现不是插补，而是：

- 原值保留
- 再加缺失 mask
- 再加缺失汇总

### 11.4 误以为 `node_time_bucket` 是动态连续时间表示

不是。

它只是一个按 `first_active` 划出来的离散时间桶，用于 `M5` 组织 batch。

## 12. 验证命令与平台说明

### 12.1 最直接的构建命令

在当前仓库根目录下：

```bash
python experiment/training/run_training.py build_features --phase both
```

如果你只想做更快的 smoke：

```bash
python experiment/training/run_training.py build_features --phase both --skip-neighbor
```

### 12.2 构建后应看到的产物

在 `experiment/outputs/training/features/phase1/` 和 `phase2/` 下应出现：

- `core_features.npy`
- `neighbor_features.npy`（如果没有 `--skip-neighbor`）
- `feature_manifest.json`
- `graph/` 目录

### 12.3 Windows 平台说明

当前实现大量使用：

- `numpy.memmap`
- `np.save / np.load(..., mmap_mode="r")`

这是正常的 Windows 可用写法。  
它不是 bug，也不是临时文件，而是这套大图缓存方案的核心设计。

## 13. 后续阅读顺序建议

如果你已经读懂这份文件，下一步建议按这个顺序继续：

1. [run_training.py](./run_training.py)
   看训练入口如何调用 `build_features`
2. [gbdt_models.py](./gbdt_models.py)
   看 `FeatureStore.take_rows(...)` 如何被树模型消费
3. [gnn_models.py](./gnn_models.py)
   看 `GraphCache` 如何被子图采样和消息传递消费

## 14. 一句话总结

`features.py` 的本质不是“若干特征小函数集合”，而是：

```text
把原始 DGraph npz 数据编译成
可被 LightGBM 和 GNN 高效消费的
节点特征缓存 + 邻域特征缓存 + 图结构缓存 + 元数据协议
```

如果你接下来要继续精读训练框架，最值得反复看懂的函数是：

- `build_phase_feature_artifacts(...)`
- `_build_neighbor_features(...)`
- `_write_graph_arrays(...)`
- `FeatureStore.take_rows(...)`
