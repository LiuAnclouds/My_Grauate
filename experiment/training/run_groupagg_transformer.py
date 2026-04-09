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
from experiment.training.run_xgb_multiclass_bg import _binary_score_from_softprob, _build_sample_weight
from experiment.training.run_xgb_multiclass_bg_groupagg import (
    GROUP_NAMES,
    _cache_key as _groupagg_cache_key,
    _load_or_build_groupagg_features,
    _take_feature_matrix,
)
from experiment.training.run_xgb_graphprop import _resolve_half_lives
from experiment.training.features import FEATURE_OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight Transformer over cached groupagg tokens plus base tabular features.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "groupagg_transformer")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument("--feature-model", choices=("m2_hybrid", "m3_neighbor"), default="m3_neighbor")
    parser.add_argument("--extra-groups", nargs="*", default=())
    parser.add_argument("--cache-root", type=Path, default=MODEL_OUTPUT_ROOT / "_multiclass_bg_groupagg_cache")
    parser.add_argument(
        "--agg-blocks",
        nargs="+",
        choices=("in1", "out1", "in2", "out2", "bi1", "bi2"),
        default=("in1", "out1"),
    )
    parser.add_argument("--agg-half-life-days", type=float, nargs="+", default=(20.0, 90.0))
    parser.add_argument("--append-count-features", action="store_true")
    parser.add_argument("--include-future-background", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--background-weight", type=float, default=0.5)
    parser.add_argument("--fraud-weight-scale", type=float, default=1.0)
    parser.add_argument("--time-weight-half-life-days", type=float, default=0.0)
    parser.add_argument("--time-weight-floor", type=float, default=0.25)
    return parser.parse_args()


@dataclass(frozen=True)
class DatasetBundle:
    base: np.ndarray
    tokens: np.ndarray
    labels: np.ndarray
    sample_weight: np.ndarray | None = None


class MemmapDataset:
    def __init__(
        self,
        base: np.ndarray,
        tokens: np.ndarray,
        labels: np.ndarray,
        token_width: int,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        self.base = np.asarray(base, dtype=np.float32)
        self.tokens = tokens
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        self.token_width = int(token_width)
        if self.tokens.shape[1] % self.token_width != 0:
            raise ValueError("Token width does not divide groupagg feature dim.")
        self.num_tokens = self.tokens.shape[1] // self.token_width

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
            token = token_flat.reshape(batch_idx.shape[0], self.num_tokens, self.token_width)
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


def _build_datasets(args: argparse.Namespace):
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    phase1_first_active = np.asarray(
        load_phase_arrays("phase1", keys=("x",))["x"].shape[0] * [0],
        dtype=np.int32,
    )
    from experiment.training.features import load_graph_cache

    graph = load_graph_cache("phase1", outdir=args.feature_dir)
    phase1_first_active = np.asarray(graph.first_active, dtype=np.int32)

    if args.include_future_background:
        train_mask = (
            ((phase1_first_active <= int(split.threshold_day)) & np.isin(phase1_y, (0, 1)))
            | np.isin(phase1_y, (2, 3))
        )
    else:
        train_mask = (phase1_first_active <= int(split.threshold_day)) & np.isin(phase1_y, (0, 1, 2, 3))
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

    cache_dir = ensure_dir(args.cache_root / _groupagg_cache_key(args, threshold_day=int(split.threshold_day)))
    phase1_groupagg, phase2_groupagg, _ = _load_or_build_groupagg_features(
        args=args,
        cache_dir=cache_dir,
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        phase1_ids={"train": historical_ids, "val": val_ids},
        phase2_ids={"external": external_ids},
        agg_half_lives=_resolve_half_lives(list(args.agg_half_life_days)),
    )

    y_train = phase1_y[historical_ids].astype(np.int64, copy=False)
    y_val = phase1_y[val_ids].astype(np.int64, copy=False)
    y_external = phase2_y[external_ids].astype(np.int64, copy=False)
    train_first_active = phase1_first_active[historical_ids].astype(np.int32, copy=False)
    sample_weight_payload = _build_sample_weight(
        y_train,
        args,
        train_first_active=train_first_active,
        threshold_day=int(split.threshold_day),
    )

    token_width = phase1_groupagg["train"].shape[1] // (
        len(_resolve_half_lives(list(args.agg_half_life_days))) * len(args.agg_blocks) * len(GROUP_NAMES)
    )
    if token_width <= 0:
        raise ValueError("Invalid token width derived from groupagg cache.")

    train_ds = MemmapDataset(
        base=base_phase1["train"],
        tokens=phase1_groupagg["train"],
        labels=y_train,
        sample_weight=np.asarray(sample_weight_payload["sample_weight"], dtype=np.float32),
        token_width=token_width,
    )
    val_ds = MemmapDataset(
        base=base_phase1["val"],
        tokens=phase1_groupagg["val"],
        labels=y_val,
        sample_weight=None,
        token_width=token_width,
    )
    external_ds = MemmapDataset(
        base=base_phase2["external"],
        tokens=phase2_groupagg["external"],
        labels=y_external,
        sample_weight=None,
        token_width=token_width,
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

    set_global_seed(args.seed)
    device = _resolve_device(args.device)
    run_dir = ensure_dir(args.outdir / args.run_name)

    train_ds, val_ds, external_ds, historical_ids, val_ids, external_ids, sample_weight_payload, base_dim = _build_datasets(args)

    class GroupAggTransformer(nn.Module):
        def __init__(self, base_dim: int, token_width: int, num_tokens: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
            super().__init__()
            self.token_proj = nn.Linear(token_width, hidden_dim)
            self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.base_proj = nn.Sequential(
                nn.LayerNorm(base_dim),
                nn.Linear(base_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 4),
            )

        def forward(self, base_x, token_x):
            token_h = self.token_proj(token_x) + self.pos_emb
            token_h = self.encoder(token_h)
            token_mean = token_h.mean(dim=1)
            token_max = token_h.amax(dim=1)
            base_h = self.base_proj(base_x)
            fused = torch.cat([token_mean, token_max, base_h], dim=1)
            return self.head(fused)

    model = GroupAggTransformer(
        base_dim=base_dim,
        token_width=train_ds.token_width,
        num_tokens=train_ds.num_tokens,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
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
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            steps = step

        val_metrics, _ = _compute_auc(model, val_ds, batch_size=args.eval_batch_size, device=device)
        print(
            f"[groupagg_transformer] epoch={epoch + 1} "
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
        "model": "groupagg_transformer",
        "run_name": args.run_name,
        "seed": args.seed,
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "agg_blocks": list(args.agg_blocks),
        "agg_half_life_days": [None if value is None else float(value) for value in _resolve_half_lives(list(args.agg_half_life_days))],
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
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "sample_weight": {
            "class_weight": sample_weight_payload["class_weight"],
            "time_weight": sample_weight_payload["time_weight"],
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[groupagg_transformer] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f} "
        f"best_epoch={best_epoch}"
    )


if __name__ == "__main__":
    train_main()
