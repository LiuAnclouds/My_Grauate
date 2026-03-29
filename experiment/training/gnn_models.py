from __future__ import annotations

import copy
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment.training.common import ensure_dir, resolve_device, safe_auc, set_global_seed, write_json
from experiment.training.features import FeatureStore, GraphCache, default_feature_groups

# 变为不可变数据类
@dataclass(frozen=True)
class GraphPhaseContext:
    phase: str # 数据阶段
    feature_store: FeatureStore # 节点读取器
    graph_cache: GraphCache # 图结构缓存
    labels: np.ndarray #  整个phase的节点标签数组

# 不可变数据类
@dataclass(frozen=True)
class SampledSubgraph:
    # 子图里包含的所有节点、全局节点ID
    node_ids: np.ndarray
    # 子图里每条边的源节点局部编号
    edge_src: np.ndarray
    # 子图里每条边的目标节点局部编号
    edge_dst: np.ndarray
    # 每条边对应关系的类型id
    rel_ids: np.ndarray
    # 每条边的时间戳
    edge_timestamp: np.ndarray
    # 种子节点在node_ids这个局部节点表中的位置
    target_local_idx: np.ndarray


def _sample_edge_indices(
    edge_timestamp: np.ndarray,  # 中心节点相连的边的时间
    fanout: int, # 当前中心节点这一跳上保留多少边
    rng: np.random.Generator, #Numpy 随机数生成器，保证采样可复现
    snapshot_end: int | None, # 事件模型快照时间上界, 当前采样/前向允许看到的最晚时间点，防止模型看到未来边
) -> np.ndarray:
    # 当前节点没有候选边
    if edge_timestamp.size == 0:
        return np.empty(0, dtype=np.int32)
    # 过滤节点
    if snapshot_end is not None:
        valid = np.flatnonzero(edge_timestamp <= snapshot_end)
    else:
        # 不进行过滤选所有节点
        valid = np.arange(edge_timestamp.size, dtype=np.int32)
    # 小于过滤条件不进行过滤
    if valid.size <= fanout:
        return valid.astype(np.int32, copy=False)
    choice = rng.choice(valid, size=fanout, replace=False)
    return np.sort(choice.astype(np.int32, copy=False))

# 为种子节点采样
def sample_relation_subgraph(
    graph: GraphCache,
    seed_nodes: np.ndarray, #种子节点
    fanouts: list[int],
    rng: np.random.Generator,
    snapshot_end: int | None = None,
) -> SampledSubgraph:
    # 将种子节点转换格式
    seeds = np.asarray(seed_nodes, dtype=np.int32)
    # 节点排序,按先后顺序排（无学习意义）
    ordered_nodes: list[int] = []
    # 节点去重（防止共同节点重复加入）
    seen_nodes: set[int] = set()
    # 新节点加入子图
    def add_nodes(nodes: np.ndarray) -> None:
        for node in nodes.tolist():
            # 没有重复
            if node not in seen_nodes:
                seen_nodes.add(node)
                ordered_nodes.append(int(node))

    add_nodes(seeds)
    # 第一跳扩展的根
    frontier = seeds
    edge_records: list[tuple[int, int, int, int]] = []
    # 每一跳节点的筛选数量不同
    for fanout in fanouts:
        # 如果上一跳没有节点可以扩充了
        if frontier.size == 0:
            break
        # 下一跳中心节点的集合
        next_frontier: list[np.ndarray] = []
        # 枚举当前hop的每个中心结点
        for center in frontier.tolist():
            # in_ptr里存的是当前节点的入边范围，i为起始的边索引，i+1为结束的边索引，在这之间的就是被包含的入边
            in_start = int(graph.in_ptr[center])
            in_end = int(graph.in_ptr[center + 1])
            # 根据要求限制筛选出符合条件的边
            in_choice = _sample_edge_indices(
                edge_timestamp=np.asarray(graph.in_edge_timestamp[in_start:in_end]),
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
            )
            # 如果确实采到了入边
            if in_choice.size:
                # 这些入边对应的节点
                in_neighbors = np.asarray(graph.in_neighbors[in_start:in_end])[in_choice]
                # 入边的类型
                in_type = np.asarray(graph.in_edge_type[in_start:in_end])[in_choice]
                # 入边的时间
                in_time = np.asarray(graph.in_edge_timestamp[in_start:in_end])[in_choice]
                # 这些邻居会成为下一跳的中心节点
                next_frontier.append(in_neighbors.astype(np.int32, copy=False))
                # 把一跳领居也加入子图节点集合
                add_nodes(in_neighbors.astype(np.int32, copy=False))
                #  把边的特征存入边记录器中
                edge_records.extend(
                    (
                        int(src),
                        int(center),
                        int(edge_type - 1),
                        int(edge_time),
                    )
                    for src, edge_type, edge_time in zip(
                        in_neighbors.tolist(),
                        in_type.tolist(),
                        in_time.tolist(),
                        strict=True,
                    )
                )
            # 出边的范围
            out_start = int(graph.out_ptr[center])
            out_end = int(graph.out_ptr[center + 1])
            # 对出边进行选择和筛选
            out_choice = _sample_edge_indices(
                edge_timestamp=np.asarray(graph.out_edge_timestamp[out_start:out_end]),
                fanout=fanout,
                rng=rng,
                snapshot_end=snapshot_end,
            )
            # 如果出边不为空
            if out_choice.size:
                # 获得出边节点
                out_neighbors = np.asarray(graph.out_neighbors[out_start:out_end])[out_choice]
                # 出边类型
                out_type = np.asarray(graph.out_edge_type[out_start:out_end])[out_choice]
                # 出边时间
                out_time = np.asarray(graph.out_edge_timestamp[out_start:out_end])[out_choice]
                # 作为下一跳中心节点
                next_frontier.append(out_neighbors.astype(np.int32, copy=False))
                # 加入节点
                add_nodes(out_neighbors.astype(np.int32, copy=False))
                # 对边进行保存
                edge_records.extend(
                    (
                        int(src),
                        int(center),
                        int(edge_type - 1 + graph.num_edge_types),
                        int(edge_time),
                    )
                    for src, edge_type, edge_time in zip(
                        out_neighbors.tolist(),
                        out_type.tolist(),
                        out_time.tolist(),
                        strict=True,
                    )
                )
        # 如果下一跳中心节点不为空
        if next_frontier:
            frontier = np.unique(np.concatenate(next_frontier)).astype(np.int32, copy=False)
        else:
            frontier = np.empty(0, dtype=np.int32)
    #子图中最终包含的全局节点id列表
    node_ids = np.asarray(ordered_nodes, dtype=np.int32)
    # 全局节点编号映射为子图局部编号
    global_to_local = {int(node): idx for idx, node in enumerate(node_ids.tolist())}
    #种子节点在子图局部编号里的位置
    target_local_idx = np.asarray([global_to_local[int(node)] for node in seeds.tolist()], dtype=np.int64)
    # 如果边特征不为空，拆分为数组方便训练
    if edge_records:
        edge_src = np.asarray([global_to_local[src] for src, _, _, _ in edge_records], dtype=np.int64)
        edge_dst = np.asarray([global_to_local[dst] for _, dst, _, _ in edge_records], dtype=np.int64)
        rel_ids = np.asarray([rel for _, _, rel, _ in edge_records], dtype=np.int64)
        edge_timestamp = np.asarray([ts for _, _, _, ts in edge_records], dtype=np.int64)
    # 否则就只用自身特征
    else:
        edge_src = np.empty(0, dtype=np.int64)
        edge_dst = np.empty(0, dtype=np.int64)
        rel_ids = np.empty(0, dtype=np.int64)
        edge_timestamp = np.empty(0, dtype=np.int64)
    return SampledSubgraph(
        node_ids=node_ids,
        edge_src=edge_src,
        edge_dst=edge_dst,
        rel_ids=rel_ids,
        edge_timestamp=edge_timestamp,
        target_local_idx=target_local_idx,
    )

# 轻量时间编码器
class TimeEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, relative_time: torch.Tensor) -> torch.Tensor:
        return self.net(relative_time)
# 关系图神经网络基础层架构
class RelationSAGELayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        rel_dim: int,
        time_dim: int = 0,
    ) -> None:
        super().__init__()
        #  关系编码, 每种关系用rel_dim个维度表示
        self.relation_embedding = nn.Embedding(num_relations, rel_dim)
        # 消息输入维度
        msg_in_dim = in_dim + rel_dim + time_dim
        # 消息编码
        self.msg_linear = nn.Linear(msg_in_dim, out_dim)
        # 自身特征编码
        self.self_linear = nn.Linear(in_dim, out_dim)
        # 邻居节点特征编码
        self.neigh_linear = nn.Linear(out_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor, # 子图节点特征
        edge_src: torch.Tensor, # 边的源节点
        edge_dst: torch.Tensor, # 边的目标节点
        rel_ids: torch.Tensor, # 边类型
        time_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 如果当前子图没有边，用自身特征
        if edge_src.numel() == 0:
            return F.relu(self.self_linear(x))
        # 关系类型编码
        relation = self.relation_embedding(rel_ids)
        # 消息的基础组成
        msg_parts = [x[edge_src], relation]
        if time_feature is not None:
            msg_parts.append(time_feature)
        # 维度拼接
        msg = self.msg_linear(torch.cat(msg_parts, dim=-1))
        # 为每个节点准备一个邻居消息累加槽
        agg = x.new_zeros((x.shape[0], msg.shape[1]))
        # 把所有消息按目标节点加进去，将第i条边汇聚的消息传给第i条边的目标节点，按行也就是按节点进行累加
        agg.index_add_(0, edge_dst, msg)
        # 准备每个节点收到多少条消息的计数器，看目标节点接收到了多少条消息特征，然后进行平均池化
        deg = x.new_zeros((x.shape[0], 1))
        deg.index_add_(
            0,
            edge_dst,
            torch.ones((edge_dst.shape[0], 1), device=x.device, dtype=x.dtype),
        )
        # 进行消息特征的平均池化
        agg = agg / deg.clamp_min(1.0)
        # 节点自身特征 + 消息聚合之后的特征
        out = self.self_linear(x) + self.neigh_linear(agg)
        return F.relu(out)

# 关系图网络

class RelationGraphSAGENetwork(nn.Module):
    def __init__(
        self,
        input_dim: int, # 输入维度
        hidden_dim: int, # 隐藏维度
        num_layers: int, # 层数
        num_relations: int, # 边关系条数
        rel_dim: int, #边类型维度
        dropout: float,
        temporal: bool, # 是否启用时序分组
    ) -> None:
        super().__init__()
        self.temporal = temporal
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.time_encoder = TimeEncoder(rel_dim) if temporal else None
        time_dim = rel_dim if temporal else 0
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = hidden_dim if layer_idx > 0 else hidden_dim
            self.layers.append(
                RelationSAGELayer(
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    num_relations=num_relations,
                    rel_dim=rel_dim,
                    time_dim=time_dim,
                )
            )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_relative_time: torch.Tensor | None,
        target_local_idx: torch.Tensor,
    ) -> torch.Tensor:
        # Relu
        h = F.relu(self.input_proj(x))
        h = self.dropout(h)
        time_feature = None
        if self.temporal and edge_relative_time is not None and edge_relative_time.numel():
            time_feature = self.time_encoder(edge_relative_time)
        for layer in self.layers:
            h = layer(h, edge_src, edge_dst, rel_ids, time_feature=time_feature)
            h = self.dropout(h)
        return self.classifier(h[target_local_idx]).squeeze(-1)

 # 训练基类
class BaseGraphSAGEExperiment:
    def __init__(
        self,
        model_name: str, # 实验名
        seed: int, # 随机种子
        input_dim: int, # 输入维度
        num_relations: int, # 关系数
        max_day: int, # 该实验使用的全局最大天数，用于时间归一化。
        feature_groups: list[str] | None = None, # 输入使用哪些特征组。
        hidden_dim: int = 128, # 隐藏层维度
        num_layers: int = 2, # 层数
        rel_dim: int = 32, # 边类型维度
        fanouts: list[int] | None = None, # 随机过滤
        batch_size: int = 1024,
        epochs: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.2,
        device: str | None = None,
        temporal: bool = False,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.feature_groups = feature_groups or default_feature_groups(model_name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rel_dim = rel_dim
        self.fanouts = fanouts or [15, 10]
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.max_day = max_day
        self.temporal = temporal
        self.device = torch.device(resolve_device(device))
        self.network = RelationGraphSAGENetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_relations=num_relations,
            rel_dim=rel_dim,
            dropout=dropout,
            temporal=temporal,
        ).to(self.device)

    def _iter_batches(
        self,
        context: GraphPhaseContext, # 图结构数据类型
        node_ids: np.ndarray, # 节点id
        training: bool, # 是否训练
        rng: np.random.Generator, # 随机数生产器
    ) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
        # 节点格式转换
        nodes = np.asarray(node_ids, dtype=np.int32)
        # 原始输入位置
        positions = np.arange(nodes.size, dtype=np.int32)
        # 是否启用时序
        if self.temporal:
            buckets = np.asarray(context.graph_cache.node_time_bucket[nodes], dtype=np.int8)
            batches: list[tuple[np.ndarray, np.ndarray, int | None]] = []
            # 按时间窗口遍历
            for bucket_idx, window in enumerate(context.graph_cache.time_windows):
                # 挑选出位于当前时间窗口下的节点
                bucket_nodes = nodes[buckets == bucket_idx]
                #保留对应节点的id
                bucket_positions = positions[buckets == bucket_idx]
                # 如果当前时间窗口没有节点为空
                if bucket_nodes.size == 0:
                    continue
                if training:
                    order = rng.permutation(bucket_nodes.size)
                    bucket_nodes = bucket_nodes[order]
                    bucket_positions = bucket_positions[order]
                for start in range(0, bucket_nodes.size, self.batch_size):
                    batches.append(
                        (
                            bucket_nodes[start : start + self.batch_size],
                            bucket_positions[start : start + self.batch_size],
                            int(window["end_day"]),
                        )
                    )
            return batches
        if training:
            order = rng.permutation(nodes.size)
            nodes = nodes[order]
            positions = positions[order]
        return [
            (
                nodes[start : start + self.batch_size],
                positions[start : start + self.batch_size],
                None,
            )
            for start in range(0, nodes.size, self.batch_size)
        ]

    def _tensorize_subgraph(
        self,
        context: GraphPhaseContext,
        subgraph: SampledSubgraph,
        snapshot_end: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        x_np = context.feature_store.take_rows(subgraph.node_ids)
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        edge_src = torch.as_tensor(subgraph.edge_src, dtype=torch.long, device=self.device)
        edge_dst = torch.as_tensor(subgraph.edge_dst, dtype=torch.long, device=self.device)
        rel_ids = torch.as_tensor(subgraph.rel_ids, dtype=torch.long, device=self.device)
        target_idx = torch.as_tensor(subgraph.target_local_idx, dtype=torch.long, device=self.device)
        edge_relative_time = None
        if self.temporal and subgraph.edge_timestamp.size:
            snapshot = snapshot_end if snapshot_end is not None else self.max_day
            relative_time = (snapshot - subgraph.edge_timestamp.astype(np.float32)) / max(self.max_day, 1)
            relative_time = np.clip(relative_time, 0.0, 1.0)
            edge_relative_time = torch.as_tensor(
                relative_time.reshape(-1, 1),
                dtype=torch.float32,
                device=self.device,
            )
        return x, edge_src, edge_dst, rel_ids, edge_relative_time, target_idx

    def fit(
        self,
        context: GraphPhaseContext,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
    ) -> dict[str, float]:
        set_global_seed(self.seed)
        train_ids = np.asarray(train_ids, dtype=np.int32)
        val_ids = np.asarray(val_ids, dtype=np.int32)
        train_labels = context.labels[train_ids].astype(np.float32, copy=False)
        val_labels = context.labels[val_ids].astype(np.int8, copy=False)

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        pos_count = float(np.sum(train_labels == 1))
        neg_count = float(np.sum(train_labels == 0))
        pos_weight = torch.tensor(
            [neg_count / max(pos_count, 1.0)],
            dtype=torch.float32,
            device=self.device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_state = None
        best_val_auc = -math.inf
        best_epoch = -1
        epoch_rng = np.random.default_rng(self.seed)

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            batch_losses: list[float] = []
            for batch_nodes, _, snapshot_end in self._iter_batches(
                context=context,
                node_ids=train_ids,
                training=True,
                rng=epoch_rng,
            ):
                subgraph = sample_relation_subgraph(
                    graph=context.graph_cache,
                    seed_nodes=batch_nodes,
                    fanouts=self.fanouts,
                    rng=epoch_rng,
                    snapshot_end=snapshot_end,
                )
                x, edge_src, edge_dst, rel_ids, edge_relative_time, target_idx = self._tensorize_subgraph(
                    context=context,
                    subgraph=subgraph,
                    snapshot_end=snapshot_end,
                )
                y_batch = torch.as_tensor(
                    context.labels[batch_nodes],
                    dtype=torch.float32,
                    device=self.device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(
                    x=x,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    rel_ids=rel_ids,
                    edge_relative_time=edge_relative_time,
                    target_local_idx=target_idx,
                )
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().item()))

            val_prob = self.predict_proba(context=context, node_ids=val_ids, batch_seed=self.seed + epoch)
            val_auc = safe_auc(val_labels, val_prob)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_state = copy.deepcopy(self.network.state_dict())
            print(
                f"[{self.model_name}] epoch={epoch} "
                f"train_loss={np.mean(batch_losses):.6f} val_auc={val_auc:.6f}"
            )

        if best_state is None:
            raise RuntimeError(f"{self.model_name}: failed to capture a best checkpoint.")
        self.network.load_state_dict(best_state)
        return {
            "val_auc": float(best_val_auc),
            "best_epoch": float(best_epoch),
        }

    @torch.no_grad()
    def predict_proba(
        self,
        context: GraphPhaseContext,
        node_ids: np.ndarray,
        batch_seed: int | None = None,
    ) -> np.ndarray:
        self.network.eval()
        node_ids = np.asarray(node_ids, dtype=np.int32)
        rng = np.random.default_rng(self.seed if batch_seed is None else batch_seed)
        probabilities = np.zeros(node_ids.shape[0], dtype=np.float32)
        for batch_nodes, batch_positions, snapshot_end in self._iter_batches(
            context=context,
            node_ids=node_ids,
            training=False,
            rng=rng,
        ):
            subgraph = sample_relation_subgraph(
                graph=context.graph_cache,
                seed_nodes=batch_nodes,
                fanouts=self.fanouts,
                rng=rng,
                snapshot_end=snapshot_end,
            )
            x, edge_src, edge_dst, rel_ids, edge_relative_time, target_idx = self._tensorize_subgraph(
                context=context,
                subgraph=subgraph,
                snapshot_end=snapshot_end,
            )
            logits = self.network(
                x=x,
                edge_src=edge_src,
                edge_dst=edge_dst,
                rel_ids=rel_ids,
                edge_relative_time=edge_relative_time,
                target_local_idx=target_idx,
            )
            batch_prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
            probabilities[batch_positions] = batch_prob
        return probabilities

    def save(self, run_dir: Path) -> None:
        ensure_dir(run_dir)
        torch.save(self.network.state_dict(), run_dir / "model.pt")
        write_json(
            run_dir / "model_meta.json",
            {
                "model_name": self.model_name,
                "seed": self.seed,
                "feature_groups": self.feature_groups,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "rel_dim": self.rel_dim,
                "fanouts": self.fanouts,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "dropout": self.dropout,
                "max_day": self.max_day,
                "temporal": self.temporal,
            },
        )

    @classmethod
    def load(
        cls,
        run_dir: Path,
        input_dim: int,
        num_relations: int,
        device: str | None = None,
    ) -> "BaseGraphSAGEExperiment":
        meta = json.loads((run_dir / "model_meta.json").read_text(encoding="utf-8"))
        instance = cls(
            model_name=meta["model_name"],
            seed=int(meta["seed"]),
            input_dim=input_dim,
            num_relations=num_relations,
            max_day=int(meta["max_day"]),
            feature_groups=list(meta["feature_groups"]),
            hidden_dim=int(meta["hidden_dim"]),
            num_layers=int(meta["num_layers"]),
            rel_dim=int(meta["rel_dim"]),
            fanouts=list(meta["fanouts"]),
            batch_size=int(meta["batch_size"]),
            epochs=int(meta["epochs"]),
            learning_rate=float(meta["learning_rate"]),
            weight_decay=float(meta["weight_decay"]),
            dropout=float(meta["dropout"]),
            device=device,
            temporal=bool(meta["temporal"]),
        )
        state_dict = torch.load(
            run_dir / "model.pt",
            map_location=instance.device,
            weights_only=True,
        )
        instance.network.load_state_dict(state_dict)
        return instance


class RelationGraphSAGEExperiment(BaseGraphSAGEExperiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["temporal"] = False
        super().__init__(*args, **kwargs)


class TemporalRelationGraphSAGEExperiment(BaseGraphSAGEExperiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["temporal"] = True
        super().__init__(*args, **kwargs)
