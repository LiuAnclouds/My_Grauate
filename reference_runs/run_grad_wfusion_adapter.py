from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = REPO_ROOT.parent / "reference" / "WWW25-Grad"
sys.path.insert(0, str(REF_ROOT))

from models.WeightedFusion import WeightFusion  # noqa: E402


DATASETS = {
    "xinye_dgraph": {
        "npz": REPO_ROOT / "data/raw/xinye_dgraph/phase1_gdata.npz",
        "analysis": REPO_ROOT / "outputs/analysis/xinye_dgraph",
    },
    "elliptic_transactions": {
        "npz": REPO_ROOT / "data/raw/elliptic_transactions/prepared/graph_gdata.npz",
        "analysis": REPO_ROOT / "outputs/analysis/elliptic_transactions",
    },
    "ellipticpp_transactions": {
        "npz": REPO_ROOT / "data/raw/ellipticpp_transactions/prepared/graph_gdata.npz",
        "analysis": REPO_ROOT / "outputs/analysis/ellipticpp_transactions",
    },
}


def _sample_ids(ids: np.ndarray, y: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if max_count <= 0 or ids.size <= max_count:
        return ids
    rng = np.random.default_rng(seed)
    pos = ids[y[ids] == 1]
    neg = ids[y[ids] == 0]
    pos_keep = min(pos.size, max(1, int(round(max_count * max(pos.size / max(ids.size, 1), 0.05)))))
    neg_keep = min(neg.size, max_count - pos_keep)
    parts = []
    if pos_keep:
        parts.append(rng.choice(pos, size=pos_keep, replace=False))
    if neg_keep:
        parts.append(rng.choice(neg, size=neg_keep, replace=False))
    sampled = np.concatenate(parts) if parts else rng.choice(ids, size=max_count, replace=False)
    rng.shuffle(sampled)
    return sampled.astype(np.int64, copy=False)


def _add_context_nodes(
    *,
    selected: np.ndarray,
    edge_index: np.ndarray,
    max_context: int,
    seed: int,
) -> np.ndarray:
    if max_context <= 0:
        return selected
    selected_mask = np.zeros(int(edge_index.max()) + 1, dtype=bool)
    selected_mask[selected] = True
    incident = selected_mask[edge_index[:, 0]] | selected_mask[edge_index[:, 1]]
    neighbors = np.unique(edge_index[incident].reshape(-1))
    neighbors = np.setdiff1d(neighbors, selected, assume_unique=False)
    if neighbors.size > max_context:
        rng = np.random.default_rng(seed)
        neighbors = rng.choice(neighbors, size=max_context, replace=False)
    return np.unique(np.concatenate([selected, neighbors.astype(np.int64, copy=False)])).astype(np.int64)


def _build_graph(dataset: str, train_cap: int, val_cap: int, context_cap: int, seed: int):
    spec = DATASETS[dataset]
    raw = np.load(spec["npz"], allow_pickle=False)
    y = np.asarray(raw["y"], dtype=np.int64).reshape(-1)
    train_ids = np.load(spec["analysis"] / "train_ids.npy").astype(np.int64)
    val_ids = np.load(spec["analysis"] / "val_ids.npy").astype(np.int64)
    train_ids = _sample_ids(train_ids, y, train_cap, seed)
    val_ids = _sample_ids(val_ids, y, val_cap, seed + 1)
    edge_index = np.asarray(raw["edge_index"], dtype=np.int64)
    selected = np.unique(np.concatenate([train_ids, val_ids])).astype(np.int64)
    selected = _add_context_nodes(
        selected=selected,
        edge_index=edge_index,
        max_context=context_cap,
        seed=seed + 2,
    )

    x_all = np.asarray(raw["x"], dtype=np.float32)
    x = x_all[selected].astype(np.float32, copy=True)
    x[~np.isfinite(x)] = -1.0
    local_train = np.isin(selected, train_ids)
    local_val = np.isin(selected, val_ids)
    train_values = x[local_train]
    med = np.nanmedian(np.where(train_values == -1.0, np.nan, train_values), axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    x = np.where(x == -1.0, med.reshape(1, -1), x)
    scaler = StandardScaler()
    scaler.fit(x[local_train])
    x = np.clip(scaler.transform(x).astype(np.float32), -8.0, 8.0)

    remap = np.full(y.shape[0], -1, dtype=np.int64)
    remap[selected] = np.arange(selected.size, dtype=np.int64)
    edge_mask = (remap[edge_index[:, 0]] >= 0) & (remap[edge_index[:, 1]] >= 0)
    src = remap[edge_index[edge_mask, 0]]
    dst = remap[edge_index[edge_mask, 1]]
    if src.size == 0:
        src = np.arange(selected.size, dtype=np.int64)
        dst = np.arange(selected.size, dtype=np.int64)

    graph = dgl.heterograph(
        {("node", "relation0", "node"): (torch.as_tensor(src), torch.as_tensor(dst))},
        num_nodes_dict={"node": int(selected.size)},
    )
    graph.nodes["node"].data["feature"] = torch.as_tensor(x, dtype=torch.float32)
    graph.nodes["node"].data["label"] = torch.as_tensor(y[selected], dtype=torch.long)
    graph.nodes["node"].data["train_mask"] = torch.as_tensor(local_train, dtype=torch.bool)
    graph.nodes["node"].data["val_mask"] = torch.as_tensor(local_val, dtype=torch.bool)
    graph.nodes["node"].data["test_mask"] = torch.as_tensor(local_val, dtype=torch.bool)
    return graph


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    graph = _build_graph(args.dataset, args.train_cap, args.val_cap, args.context_cap, args.seed)
    device = torch.device("cpu")
    wf_args = SimpleNamespace(
        WFusion_use_WFusion=True,
        WFusion_epochs=args.epochs,
        device=device,
    )
    model = WeightFusion(
        global_args=wf_args,
        in_feats=graph.nodes["node"].data["feature"].shape[1],
        h_feats=args.hidden_dim,
        num_classes=2,
        graph=graph,
        relations_idx=[0],
        device=device,
        d=args.order,
    ).to(device)
    features = graph.nodes["node"].data["feature"].to(device)
    labels = graph.nodes["node"].data["label"].to(device)
    train_mask = graph.nodes["node"].data["train_mask"].to(device)
    val_mask = graph.nodes["node"].data["val_mask"].to(device)
    pos = labels[train_mask].sum().item()
    neg = train_mask.sum().item() - pos
    class_weight = torch.tensor([1.0, max(float(neg) / max(float(pos), 1.0), 1.0)], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_ap = -1.0
    best_epoch = -1
    last_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(features, graph)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
        model.eval()
        with torch.no_grad():
            probs = model(features, graph).softmax(dim=1)[:, 1].cpu().numpy()
        y_val = labels[val_mask].cpu().numpy()
        val_score = probs[val_mask.cpu().numpy()]
        if np.unique(y_val).size < 2:
            auc = float("nan")
            ap = float("nan")
        else:
            auc = float(roc_auc_score(y_val, val_score))
            ap = float(average_precision_score(y_val, val_score))
        if auc > best_auc:
            best_auc = auc
            best_ap = ap
            best_epoch = epoch
        print(f"epoch={epoch} loss={last_loss:.6f} val_auc={auc:.6f} val_ap={ap:.6f} best={best_auc:.6f}@{best_epoch}")

    result = {
        "adapter": "WWW25-Grad original WeightedFusion on our induced graph, relation0 only",
        "dataset": args.dataset,
        "nodes": int(graph.num_nodes("node")),
        "edges": int(graph.num_edges(("node", "relation0", "node"))),
        "train_nodes": int(train_mask.sum().item()),
        "val_nodes": int(val_mask.sum().item()),
        "val_pos": int(labels[val_mask].sum().item()),
        "best_val_auc": best_auc,
        "best_val_ap": best_ap,
        "best_epoch": best_epoch,
        "last_loss": last_loss,
        "config": vars(args),
        "caveat": (
            "This does not include GuiDDPM generated relations. The original Grad "
            "pipeline requires training/sampling fixed-size adjacency subgraphs and "
            "then merging generated relation files."
        ),
    }
    outdir = REPO_ROOT / "reference_runs" / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"grad_wfusion_{args.dataset}_n{graph.num_nodes('node')}.json"
    outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="elliptic_transactions")
    parser.add_argument("--train-cap", type=int, default=2048)
    parser.add_argument("--val-cap", type=int, default=2048)
    parser.add_argument("--context-cap", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
