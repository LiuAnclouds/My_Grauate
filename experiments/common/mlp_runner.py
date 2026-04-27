from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dyrift.features.features import FeatureStore, build_hybrid_feature_normalizer, resolve_feature_groups
from dyrift.models.engine import GraphModelConfig
from dyrift.models.runtime import build_runtime
from dyrift.utils.common import (
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    resolve_device,
    set_global_seed,
    write_clean_epoch_metrics,
    write_json,
)

from .contracts import DatasetPlan, ExperimentConfig, resolve_dataset_output_roots


@dataclass(frozen=True)
class FeatureMatrices:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def run_mlp_dataset(
    *,
    config: ExperimentConfig,
    plan: DatasetPlan,
    dataset_dir: Path,
    seeds: list[int],
    device: str,
) -> Path:
    mlp_spec = config.runner_spec
    model_display_name = str(mlp_spec.get("model_display_name") or config.display_name)
    include_target_context = bool(mlp_spec.get("include_target_context", False))
    epochs = int(mlp_spec.get("epochs", plan.epochs))
    patience = int(mlp_spec.get("early_stop_patience", 5))
    min_delta = float(mlp_spec.get("early_stop_min_delta", 0.0))
    batch_size = int(mlp_spec.get("batch_size", plan.batch_size))
    hidden_dims = [int(value) for value in mlp_spec.get("hidden_dims", [256, 128])]
    dropout = float(mlp_spec.get("dropout", 0.25))
    learning_rate = float(mlp_spec.get("learning_rate", 1e-3))
    weight_decay = float(mlp_spec.get("weight_decay", 1e-4))
    resolved_device = resolve_device(device)

    matrices = _build_feature_matrices(
        plan=plan,
        include_target_context=include_target_context,
    )
    print(
        "[experiment:mlp] "
        f"experiment={config.experiment_name} "
        f"dataset={plan.dataset_name} "
        f"model={model_display_name} "
        f"features={matrices.train_x.shape[1]} "
        f"target_context={'on' if include_target_context and plan.target_context_groups else 'off'} "
        f"patience={patience} min_delta={min_delta:.6f}"
    )

    for seed in seeds:
        set_global_seed(int(seed))
        seed_dir = ensure_dir(dataset_dir / f"seed_{int(seed)}")
        model = TabularMLP(
            input_dim=matrices.train_x.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(resolved_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        pos_count = max(int(np.sum(matrices.train_y == 1)), 1)
        neg_count = max(int(np.sum(matrices.train_y == 0)), 1)
        pos_weight = torch.tensor(float(neg_count / pos_count), device=resolved_device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_x = torch.as_tensor(matrices.train_x, dtype=torch.float32, device=resolved_device)
        train_y = torch.as_tensor(matrices.train_y, dtype=torch.float32, device=resolved_device)
        val_x = torch.as_tensor(matrices.val_x, dtype=torch.float32, device=resolved_device)
        val_y = torch.as_tensor(matrices.val_y, dtype=torch.float32, device=resolved_device)
        rng = np.random.default_rng(int(seed))

        rows: list[dict[str, Any]] = []
        best_val_auc = -float("inf")
        early_stop_reference_auc = -float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, epochs + 1):
            model.train()
            permutation = rng.permutation(train_x.shape[0])
            for start in range(0, permutation.size, batch_size):
                batch_idx = torch.as_tensor(
                    permutation[start : start + batch_size],
                    dtype=torch.long,
                    device=resolved_device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x.index_select(0, batch_idx))
                loss = criterion(logits, train_y.index_select(0, batch_idx))
                loss.backward()
                optimizer.step()

            train_prob = _predict_probability(model, train_x)
            val_prob = _predict_probability(model, val_x)
            train_auc = compute_binary_classification_metrics(matrices.train_y, train_prob)["auc"]
            val_auc = compute_binary_classification_metrics(matrices.val_y, val_prob)["auc"]
            train_loss = _binary_log_loss(matrices.train_y, train_prob)
            val_loss = _binary_log_loss(matrices.val_y, val_prob)
            if val_auc > best_val_auc:
                best_val_auc = float(val_auc)
                best_epoch = int(epoch)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            if val_auc > early_stop_reference_auc + min_delta:
                early_stop_reference_auc = float(val_auc)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_auc": float(train_auc),
                "val_auc": float(val_auc),
                "best_epoch": int(best_epoch),
            }
            rows.append(row)
            print(
                f"[experiment:{config.experiment_name}] dataset={plan.dataset_name} "
                f"seed={seed} epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} train_auc={train_auc:.6f} "
                f"val_auc={val_auc:.6f} best_epoch={best_epoch}"
            )
            _write_csv(seed_dir / "epoch_metrics.csv", rows)

            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"[experiment:{config.experiment_name}] early_stop "
                    f"dataset={plan.dataset_name} seed={seed} epoch={epoch} "
                    f"patience={patience} min_delta={min_delta:.6f}"
                )
                break

        if best_state is None:
            raise RuntimeError(f"{config.experiment_name}: failed to capture a best MLP checkpoint.")
        last_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(best_state)
        torch.save(best_state, seed_dir / "best_model.pt")
        torch.save(last_state, seed_dir / "last_model.pt")
        torch.save(best_state, seed_dir / "model.pt")
        fit_metrics = {
            "val_auc": float(best_val_auc),
            "best_epoch": int(best_epoch),
            "trained_epochs": int(len(rows)),
            "early_stop_patience": int(patience),
            "early_stop_min_delta": float(min_delta),
            "early_stop_reference_auc": float(early_stop_reference_auc),
            "best_checkpoint": "best_model.pt",
            "last_checkpoint": "last_model.pt",
        }
        write_json(seed_dir / "fit_metrics.json", fit_metrics)
        write_json(
            seed_dir / "model_meta.json",
            {
                "model_name": "mlp",
                "model_display_name": model_display_name,
                "input_dim": int(matrices.train_x.shape[1]),
                "hidden_dims": hidden_dims,
                "dropout": float(dropout),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "early_stop_patience": int(patience),
                "early_stop_min_delta": float(min_delta),
                "best_epoch": int(best_epoch),
            },
        )

    return write_clean_epoch_metrics(
        dataset_dir / "epoch_metrics.csv",
        [dataset_dir / f"seed_{int(seed)}" / "epoch_metrics.csv" for seed in seeds],
    )


def _build_feature_matrices(*, plan: DatasetPlan, include_target_context: bool) -> FeatureMatrices:
    dataset_analysis_root, _ = resolve_dataset_output_roots(plan.dataset_name)
    split = load_experiment_split(analysis_root=dataset_analysis_root)
    train_ids = np.asarray(split.train_ids, dtype=np.int32)
    val_ids = np.asarray(split.val_ids, dtype=np.int32)

    feature_groups = resolve_feature_groups("dyrift_gnn", feature_profile=plan.feature_profile)
    normalizer_state = build_hybrid_feature_normalizer(
        phase=split.train_phase,
        selected_groups=feature_groups,
        train_ids=train_ids,
        outdir=plan.feature_dir,
    )
    train_store = FeatureStore(
        split.train_phase,
        feature_groups,
        outdir=plan.feature_dir,
        normalizer_state=normalizer_state,
    )
    val_store = FeatureStore(
        split.val_phase,
        feature_groups,
        outdir=plan.feature_dir,
        normalizer_state=normalizer_state,
    )
    train_x = train_store.take_rows(train_ids)
    val_x = val_store.take_rows(val_ids)

    if include_target_context and plan.target_context_groups:
        target_normalizer = build_hybrid_feature_normalizer(
            phase=split.train_phase,
            selected_groups=plan.target_context_groups,
            train_ids=train_ids,
            outdir=plan.feature_dir,
        )
        train_target_store = FeatureStore(
            split.train_phase,
            plan.target_context_groups,
            outdir=plan.feature_dir,
            normalizer_state=target_normalizer,
        )
        val_target_store = FeatureStore(
            split.val_phase,
            plan.target_context_groups,
            outdir=plan.feature_dir,
            normalizer_state=target_normalizer,
        )
        train_x = np.concatenate([train_x, train_target_store.take_rows(train_ids)], axis=1)
        val_x = np.concatenate([val_x, val_target_store.take_rows(val_ids)], axis=1)

    runtime = build_runtime(
        feature_dir=plan.feature_dir,
        model_name="dyrift_gnn",
        split=split,
        train_ids=train_ids,
        graph_config=GraphModelConfig(feature_norm="hybrid"),
        feature_profile=plan.feature_profile,
        target_context_groups=plan.target_context_groups,
    )
    train_y = np.asarray(runtime.phase1_context.labels[train_ids], dtype=np.int8)
    val_y = np.asarray(runtime.phase1_context.labels[val_ids], dtype=np.int8)
    train_mask = np.isin(train_y, (0, 1))
    val_mask = np.isin(val_y, (0, 1))
    return FeatureMatrices(
        train_x=np.asarray(train_x[train_mask], dtype=np.float32),
        train_y=np.asarray(train_y[train_mask], dtype=np.int8),
        val_x=np.asarray(val_x[val_mask], dtype=np.float32),
        val_y=np.asarray(val_y[val_mask], dtype=np.int8),
    )


@torch.no_grad()
def _predict_probability(model: nn.Module, x: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, x.shape[0], int(batch_size)):
        logits = model(x[start : start + int(batch_size)])
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def _binary_log_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)), dtype=np.float64))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "train_loss", "val_loss", "train_auc", "val_auc", "best_epoch"),
        )
        writer.writeheader()
        writer.writerows(rows)
