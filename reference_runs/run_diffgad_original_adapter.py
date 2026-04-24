from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_dense_adj


REPO_ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = REPO_ROOT.parent / "reference" / "DiffGAD"
sys.path.insert(0, str(REF_ROOT))

from auto_encoder import GraphAE  # noqa: E402
from diffusion_model import MLPDiffusion, Model  # noqa: E402


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
    if neg_keep < 1 and neg.size:
        neg_keep = 1
        pos_keep = min(pos_keep, max_count - 1)
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


def _build_induced_graph(dataset: str, train_cap: int, val_cap: int, context_cap: int, seed: int):
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
    x = scaler.transform(x).astype(np.float32)
    x = np.clip(x, -8.0, 8.0)

    remap = np.full(y.shape[0], -1, dtype=np.int64)
    remap[selected] = np.arange(selected.size, dtype=np.int64)
    edge_mask = (remap[edge_index[:, 0]] >= 0) & (remap[edge_index[:, 1]] >= 0)
    edge_local = np.stack(
        [remap[edge_index[edge_mask, 0]], remap[edge_index[edge_mask, 1]]],
        axis=0,
    )
    if edge_local.shape[1] == 0:
        edge_local = np.vstack([np.arange(selected.size), np.arange(selected.size)])

    return {
        "x": torch.as_tensor(x, dtype=torch.float32),
        "y": torch.as_tensor(y[selected], dtype=torch.long),
        "edge_index": torch.as_tensor(edge_local, dtype=torch.long),
        "train_mask": torch.as_tensor(local_train, dtype=torch.bool),
        "val_mask": torch.as_tensor(local_val, dtype=torch.bool),
        "global_ids": selected,
    }


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    graph = _build_induced_graph(
        args.dataset,
        args.train_cap,
        args.val_cap,
        args.context_cap,
        args.seed,
    )
    x = graph["x"]
    y = graph["y"]
    edge_index = graph["edge_index"]
    train_mask = graph["train_mask"]
    val_mask = graph["val_mask"]

    ae = GraphAE(
        in_dim=x.shape[1],
        hid_dim=args.hidden_dim,
        num_layers=args.ae_layers,
        dropout=args.dropout,
    )
    opt = torch.optim.Adam(ae.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
    dense_adj = to_dense_adj(edge_index, max_num_nodes=x.shape[0])[0]

    ae_losses: list[float] = []
    ae_val_auc = float("nan")
    ae_val_ap = float("nan")
    for _ in range(args.ae_epochs):
        ae.train()
        x_hat, s_hat, _ = ae(x, edge_index)
        score = ae.loss_func(x, x_hat, dense_adj, s_hat, args.ae_alpha)
        loss = score[train_mask].mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt.step()
        ae_losses.append(float(loss.detach().item()))
        with torch.no_grad():
            ae.eval()
            x_hat, s_hat, _ = ae(x, edge_index)
            score_eval = ae.loss_func(x, x_hat, dense_adj, s_hat, args.ae_alpha)
            labels = y[val_mask].cpu().numpy()
            val_score = score_eval[val_mask].detach().cpu().numpy()
            ae_val_auc = _safe_auc(labels, val_score)
            ae_val_ap = float(average_precision_score(labels, val_score))

    with torch.no_grad():
        embeddings = ae.encode(x, edge_index).detach()

    denoise_fn = MLPDiffusion(args.hidden_dim, args.diffusion_dim)
    dm = Model(denoise_fn=denoise_fn, hid_dim=args.hidden_dim)
    dm_opt = torch.optim.Adam(dm.parameters(), lr=args.diff_lr, weight_decay=args.weight_decay)
    diff_losses: list[float] = []
    for _ in range(args.diff_epochs):
        dm.train()
        loss, _, _ = dm(embeddings[train_mask])
        dm_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dm.parameters(), 1.0)
        dm_opt.step()
        diff_losses.append(float(loss.detach().item()))

    scores = []
    dm.eval()
    with torch.no_grad():
        for _ in range(args.score_samples):
            _, score, _ = dm(embeddings)
            scores.append(score.detach().cpu().numpy())
    diff_score = np.mean(np.stack(scores, axis=0), axis=0)
    labels = y[val_mask].cpu().numpy()
    val_score = diff_score[val_mask.cpu().numpy()]
    diff_val_auc = _safe_auc(labels, val_score)
    diff_val_ap = float(average_precision_score(labels, val_score))

    result = {
        "adapter": "DiffGAD original modules: GraphAE + diffusion_model.Model",
        "dataset": args.dataset,
        "nodes": int(x.shape[0]),
        "edges": int(edge_index.shape[1]),
        "train_nodes": int(train_mask.sum().item()),
        "val_nodes": int(val_mask.sum().item()),
        "val_pos": int(y[val_mask].sum().item()),
        "ae_val_auc": ae_val_auc,
        "ae_val_ap": ae_val_ap,
        "diffusion_val_auc": diff_val_auc,
        "diffusion_val_ap": diff_val_ap,
        "ae_last_loss": ae_losses[-1] if ae_losses else None,
        "diff_last_loss": diff_losses[-1] if diff_losses else None,
        "config": vars(args),
        "caveat": (
            "Runs an induced subgraph because the original DiffGAD GraphAE builds "
            "a dense N x N adjacency reconstruction."
        ),
    }
    outdir = REPO_ROOT / "reference_runs" / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"diffgad_{args.dataset}_n{int(x.shape[0])}.json"
    outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="elliptic_transactions")
    parser.add_argument("--train-cap", type=int, default=1024)
    parser.add_argument("--val-cap", type=int, default=1024)
    parser.add_argument("--context-cap", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-dim", type=int, default=128)
    parser.add_argument("--ae-layers", type=int, default=4)
    parser.add_argument("--ae-epochs", type=int, default=8)
    parser.add_argument("--diff-epochs", type=int, default=20)
    parser.add_argument("--score-samples", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ae-lr", type=float, default=0.005)
    parser.add_argument("--diff-lr", type=float, default=0.004)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ae-alpha", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
