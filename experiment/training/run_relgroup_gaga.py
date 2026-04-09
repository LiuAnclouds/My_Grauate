from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (
    MODEL_OUTPUT_ROOT,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    load_phase_arrays,
    save_prediction_npz,
    set_global_seed,
    write_json,
)
from experiment.training.features import FEATURE_OUTPUT_ROOT, load_graph_cache
from experiment.training.run_xgb_graphprop import _resolve_half_lives
from experiment.training.run_xgb_multiclass_bg import _binary_score_from_softprob, _build_sample_weight
from experiment.training.run_xgb_multiclass_bg_relgroup import (
    GROUP_NAMES,
    _cache_key as _relgroup_cache_key,
    _load_or_build_relgroup,
    _take_feature_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe GAGA-style relation-group transformer over cached relgroup tokens "
            "plus tabular center features."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "gaga_safe")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument("--feature-model", choices=("m2_hybrid", "m3_neighbor"), default="m3_neighbor")
    parser.add_argument("--extra-groups", nargs="*", default=())
    parser.add_argument("--cache-root", type=Path, default=MODEL_OUTPUT_ROOT / "_multiclass_bg_relgroup_cache")
    parser.add_argument("--directions", nargs="+", choices=("in", "out"), default=("out",))
    parser.add_argument("--edge-types", type=int, nargs="+", default=(1, 4, 5, 6, 10))
    parser.add_argument("--agg-half-life-days", type=float, nargs="+", default=(20.0, 90.0))
    parser.add_argument("--selected-raw-indices", type=int, nargs="+", default=(7, 12, 13))
    parser.add_argument("--selected-missing-indices", type=int, nargs="+", default=(0, 1, 6, 8, 9, 15, 16))
    parser.add_argument("--append-count-features", action="store_true")
    parser.add_argument("--include-future-background", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--agg-type", choices=("cat", "mean"), default="cat")
    parser.add_argument("--background-weight", type=float, default=0.5)
    parser.add_argument("--fraud-weight-scale", type=float, default=1.0)
    parser.add_argument("--time-weight-half-life-days", type=float, default=0.0)
    parser.add_argument("--time-weight-floor", type=float, default=0.25)
    return parser.parse_args()


@dataclass(frozen=True)
class RelgroupMeta:
    token_width: int
    num_groups: int
    num_relation_blocks: int
    half_life_indices: np.ndarray
    direction_indices: np.ndarray
    edge_type_indices: np.ndarray
    resolved_half_lives: list[float | None]
    directions: list[str]
    edge_types: list[int]


class MemmapDataset:
    def __init__(
        self,
        base: np.ndarray,
        tokens: np.ndarray,
        labels: np.ndarray,
        token_width: int,
        num_groups: int,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        self.base = np.asarray(base, dtype=np.float32)
        self.tokens = tokens
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        self.token_width = int(token_width)
        self.num_groups = int(num_groups)
        if self.tokens.shape[1] % (self.token_width * self.num_groups) != 0:
            raise ValueError("Relgroup token width does not divide flattened feature matrix.")
        self.num_relation_blocks = self.tokens.shape[1] // (self.token_width * self.num_groups)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def batch_iter(self, batch_size: int, shuffle: bool, seed: int):
        indices = np.arange(len(self), dtype=np.int64)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for start in range(0, indices.shape[0], int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            base = np.asarray(self.base[batch_idx], dtype=np.float32)
            token_flat = np.asarray(self.tokens[batch_idx], dtype=np.float32)
            token = token_flat.reshape(
                batch_idx.shape[0],
                self.num_relation_blocks,
                self.num_groups,
                self.token_width,
            )
            label = self.labels[batch_idx]
            if self.sample_weight is None:
                yield base, token, label, None
            else:
                yield base, token, label, np.asarray(self.sample_weight[batch_idx], dtype=np.float32)


def _resolve_device(device: str):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _build_relgroup_meta(args: argparse.Namespace, token_dim: int) -> RelgroupMeta:
    resolved_half_lives = _resolve_half_lives(list(args.agg_half_life_days))
    token_width = len(args.selected_raw_indices) + len(args.selected_missing_indices) + (
        1 if args.append_count_features else 0
    )
    num_groups = len(GROUP_NAMES)
    denom = token_width * num_groups
    if token_dim % denom != 0:
        raise ValueError(
            f"Relgroup feature dim={token_dim} is incompatible with token_width={token_width} and num_groups={num_groups}."
        )
    block_meta: list[tuple[int, int, int]] = []
    for half_idx, _ in enumerate(resolved_half_lives):
        for dir_idx, _ in enumerate(args.directions):
            for edge_idx, _ in enumerate(args.edge_types):
                block_meta.append((half_idx, dir_idx, edge_idx))
    num_relation_blocks = token_dim // denom
    if len(block_meta) != num_relation_blocks:
        raise ValueError(
            f"Expected {len(block_meta)} relation blocks from args but got {num_relation_blocks} from cached features."
        )
    return RelgroupMeta(
        token_width=token_width,
        num_groups=num_groups,
        num_relation_blocks=num_relation_blocks,
        half_life_indices=np.asarray([item[0] for item in block_meta], dtype=np.int64),
        direction_indices=np.asarray([item[1] for item in block_meta], dtype=np.int64),
        edge_type_indices=np.asarray([item[2] for item in block_meta], dtype=np.int64),
        resolved_half_lives=list(resolved_half_lives),
        directions=list(args.directions),
        edge_types=list(args.edge_types),
    )


def _build_datasets(args: argparse.Namespace):
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    graph = load_graph_cache("phase1", outdir=args.feature_dir)
    first_active = np.asarray(graph.first_active, dtype=np.int32)

    if args.include_future_background:
        train_mask = (
            ((first_active <= int(split.threshold_day)) & np.isin(phase1_y, (0, 1)))
            | np.isin(phase1_y, (2, 3))
        )
    else:
        train_mask = (first_active <= int(split.threshold_day)) & np.isin(phase1_y, (0, 1, 2, 3))

    historical_ids = np.flatnonzero(train_mask).astype(np.int32, copy=False)
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)

    base_phase1, base_feature_names = _take_feature_matrix(
        feature_dir=args.feature_dir,
        phase="phase1",
        feature_model=args.feature_model,
        extra_groups=list(args.extra_groups),
        split_ids={"train": historical_ids, "val": val_ids},
    )
    base_phase2, _ = _take_feature_matrix(
        feature_dir=args.feature_dir,
        phase="phase2",
        feature_model=args.feature_model,
        extra_groups=list(args.extra_groups),
        split_ids={"external": external_ids},
    )

    cache_dir = ensure_dir(args.cache_root / _relgroup_cache_key(args, threshold_day=int(split.threshold_day)))
    phase1_relgroup, phase2_relgroup, relgroup_feature_names = _load_or_build_relgroup(
        args=args,
        cache_dir=cache_dir,
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        phase1_ids={"train": historical_ids, "val": val_ids},
        phase2_ids={"external": external_ids},
    )
    rel_meta = _build_relgroup_meta(args, token_dim=len(relgroup_feature_names))

    y_train = phase1_y[historical_ids].astype(np.int64, copy=False)
    y_val = phase1_y[val_ids].astype(np.int64, copy=False)
    y_external = phase2_y[external_ids].astype(np.int64, copy=False)
    train_first_active = first_active[historical_ids].astype(np.int32, copy=False)
    sample_weight_payload = _build_sample_weight(
        y_train,
        args,
        train_first_active=train_first_active,
        threshold_day=int(split.threshold_day),
    )

    train_ds = MemmapDataset(
        base=base_phase1["train"],
        tokens=phase1_relgroup["train"],
        labels=y_train,
        sample_weight=np.asarray(sample_weight_payload["sample_weight"], dtype=np.float32),
        token_width=rel_meta.token_width,
        num_groups=rel_meta.num_groups,
    )
    val_ds = MemmapDataset(
        base=base_phase1["val"],
        tokens=phase1_relgroup["val"],
        labels=y_val,
        token_width=rel_meta.token_width,
        num_groups=rel_meta.num_groups,
    )
    external_ds = MemmapDataset(
        base=base_phase2["external"],
        tokens=phase2_relgroup["external"],
        labels=y_external,
        token_width=rel_meta.token_width,
        num_groups=rel_meta.num_groups,
    )
    return (
        train_ds,
        val_ds,
        external_ds,
        historical_ids,
        val_ids,
        external_ids,
        sample_weight_payload,
        len(base_feature_names),
        rel_meta,
        cache_dir,
    )


def _compute_auc(model, dataset: MemmapDataset, batch_size: int, device):
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for base_np, token_np, _, _ in dataset.batch_iter(batch_size=batch_size, shuffle=False, seed=0):
            base = torch.from_numpy(base_np).to(device, non_blocking=True)
            token = torch.from_numpy(token_np).to(device, non_blocking=True)
            logits = model(base, token)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    prob = np.concatenate(probs, axis=0).astype(np.float32, copy=False)
    score = _binary_score_from_softprob(prob)
    metrics = compute_binary_classification_metrics(dataset.labels.astype(np.int8, copy=False), score)
    return metrics, score


def train_main() -> None:
    args = parse_args()
    import torch
    import torch.nn as nn

    torch.set_float32_matmul_precision("high")
    set_global_seed(args.seed)
    device = _resolve_device(args.device)
    run_dir = ensure_dir(args.outdir / args.run_name)

    (
        train_ds,
        val_ds,
        external_ds,
        historical_ids,
        val_ids,
        external_ids,
        sample_weight_payload,
        base_dim,
        rel_meta,
        cache_dir,
    ) = _build_datasets(args)

    class RelgroupGAGA(nn.Module):
        def __init__(
            self,
            base_dim: int,
            rel_meta: RelgroupMeta,
            hidden_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float,
            agg_type: str,
        ) -> None:
            super().__init__()
            self.token_proj = nn.Sequential(
                nn.LayerNorm(rel_meta.token_width),
                nn.Linear(rel_meta.token_width, hidden_dim),
                nn.GELU(),
            )
            self.center_proj = nn.Sequential(
                nn.LayerNorm(base_dim),
                nn.Linear(base_dim, hidden_dim),
                nn.GELU(),
            )
            self.base_skip = nn.Sequential(
                nn.LayerNorm(base_dim),
                nn.Linear(base_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.group_embedding = nn.Embedding(rel_meta.num_groups + 1, hidden_dim)
            self.edge_type_embedding = nn.Embedding(len(rel_meta.edge_types), hidden_dim)
            self.direction_embedding = nn.Embedding(len(rel_meta.directions), hidden_dim)
            self.half_life_embedding = nn.Embedding(len(rel_meta.resolved_half_lives), hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout)
            self.num_groups = rel_meta.num_groups
            self.num_relation_blocks = rel_meta.num_relation_blocks
            self.agg_type = agg_type

            agg_dim = hidden_dim * rel_meta.num_relation_blocks if agg_type == "cat" else hidden_dim
            self.head = nn.Sequential(
                nn.LayerNorm(agg_dim + hidden_dim),
                nn.Linear(agg_dim + hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, 4),
            )

            self.register_buffer(
                "half_life_index",
                torch.from_numpy(rel_meta.half_life_indices.astype(np.int64, copy=False)),
                persistent=False,
            )
            self.register_buffer(
                "direction_index",
                torch.from_numpy(rel_meta.direction_indices.astype(np.int64, copy=False)),
                persistent=False,
            )
            self.register_buffer(
                "edge_type_index",
                torch.from_numpy(rel_meta.edge_type_indices.astype(np.int64, copy=False)),
                persistent=False,
            )
            self.register_buffer(
                "group_index",
                torch.arange(rel_meta.num_groups, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "center_group_index",
                torch.tensor([rel_meta.num_groups], dtype=torch.int64),
                persistent=False,
            )

        def forward(self, base_x, token_x):
            batch_size = base_x.shape[0]
            hidden = self.token_proj(token_x)
            rel_emb = (
                self.half_life_embedding(self.half_life_index)
                + self.direction_embedding(self.direction_index)
                + self.edge_type_embedding(self.edge_type_index)
            )
            rel_emb = rel_emb.unsqueeze(0).unsqueeze(2)

            group_emb = self.group_embedding(self.group_index).unsqueeze(0).unsqueeze(0)
            center_group_emb = self.group_embedding(self.center_group_index).view(1, 1, 1, -1)

            hidden = hidden + rel_emb + group_emb
            center = self.center_proj(base_x).unsqueeze(1).unsqueeze(2)
            center = center.expand(batch_size, self.num_relation_blocks, 1, -1)
            center = center + rel_emb + center_group_emb

            seq = torch.cat([center, hidden], dim=2)
            seq = seq.reshape(batch_size, self.num_relation_blocks * (self.num_groups + 1), -1)
            seq = self.dropout(seq)
            seq = self.encoder(seq)
            seq = seq.reshape(batch_size, self.num_relation_blocks, self.num_groups + 1, -1)

            center_out = seq[:, :, 0, :]
            if self.agg_type == "cat":
                relation_repr = center_out.reshape(batch_size, -1)
            else:
                relation_repr = center_out.mean(dim=1)
            fused = torch.cat([relation_repr, self.base_skip(base_x)], dim=1)
            return self.head(fused)

    model = RelgroupGAGA(
        base_dim=base_dim,
        rel_meta=rel_meta,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        agg_type=args.agg_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -math.inf
    best_epoch = -1
    patience_left = int(args.early_stop_patience)

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        steps = 0
        for step, (base_np, token_np, y_np, w_np) in enumerate(
            train_ds.batch_iter(batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch),
            start=1,
        ):
            base = torch.from_numpy(base_np).to(device, non_blocking=True)
            token = torch.from_numpy(token_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)
            weights = None if w_np is None else torch.from_numpy(w_np).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(base, token)
                per_sample = torch.nn.functional.cross_entropy(logits, y, reduction="none")
                if weights is not None:
                    loss = (per_sample * weights).mean()
                else:
                    loss = per_sample.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            steps = step

        val_metrics, _ = _compute_auc(model, val_ds, batch_size=args.eval_batch_size, device=device)
        print(
            f"[relgroup_gaga] epoch={epoch + 1} "
            f"train_loss={running_loss / max(steps, 1):.6f} "
            f"val_auc={val_metrics['auc']:.6f}"
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_left = int(args.early_stop_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    val_metrics, val_score = _compute_auc(model, val_ds, batch_size=args.eval_batch_size, device=device)
    external_metrics, external_score = _compute_auc(model, external_ds, batch_size=args.eval_batch_size, device=device)

    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, val_ds.labels.astype(np.int8), val_score)
    save_prediction_npz(
        run_dir / "phase2_external_predictions.npz",
        external_ids,
        external_ds.labels.astype(np.int8),
        external_score,
    )
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, run_dir / "model.pt")

    summary = {
        "model": "relgroup_gaga",
        "run_name": args.run_name,
        "seed": args.seed,
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "directions": list(args.directions),
        "edge_types": list(args.edge_types),
        "agg_half_life_days": [
            None if value is None else float(value) for value in rel_meta.resolved_half_lives
        ],
        "selected_raw_indices": list(args.selected_raw_indices),
        "selected_missing_indices": list(args.selected_missing_indices),
        "append_count_features": bool(args.append_count_features),
        "include_future_background": bool(args.include_future_background),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "hidden_dim": int(args.hidden_dim),
        "num_heads": int(args.num_heads),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "agg_type": args.agg_type,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "sample_weight": {
            "class_weight": sample_weight_payload["class_weight"],
            "time_weight": sample_weight_payload["time_weight"],
        },
        "base_dim": int(base_dim),
        "token_width": int(rel_meta.token_width),
        "num_groups": int(rel_meta.num_groups),
        "num_relation_blocks": int(rel_meta.num_relation_blocks),
        "historical_train_size": int(historical_ids.size),
        "cache_dir": str(cache_dir),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[relgroup_gaga] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f} "
        f"best_epoch={best_epoch}"
    )


if __name__ == "__main__":
    train_main()
