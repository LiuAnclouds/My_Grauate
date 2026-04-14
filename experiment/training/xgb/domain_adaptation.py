from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def add_domain_weight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--domain-weight-target-scope",
        choices=("none", "phase1_val"),
        default="none",
        help=(
            "Use unlabeled target-split features to build covariate-shift weights for "
            "historical training rows."
        ),
    )
    parser.add_argument(
        "--domain-weight-mix",
        type=float,
        default=0.0,
        help="0 disables domain weighting, 1 fully applies the learned domain odds weights.",
    )
    parser.add_argument(
        "--domain-weight-power",
        type=float,
        default=1.0,
        help="Exponent applied to the target-vs-source odds before clipping.",
    )
    parser.add_argument("--domain-weight-clip-min", type=float, default=0.25)
    parser.add_argument("--domain-weight-clip-max", type=float, default=6.0)
    parser.add_argument("--domain-n-estimators", type=int, default=1200)
    parser.add_argument("--domain-learning-rate", type=float, default=0.03)
    parser.add_argument("--domain-max-depth", type=int, default=6)
    parser.add_argument("--domain-min-child-weight", type=float, default=20.0)
    parser.add_argument("--domain-subsample", type=float, default=0.8)
    parser.add_argument("--domain-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--domain-gamma", type=float, default=0.0)
    parser.add_argument("--domain-reg-alpha", type=float, default=0.0)
    parser.add_argument("--domain-reg-lambda", type=float, default=1.0)
    parser.add_argument(
        "--domain-early-stopping-rounds",
        type=int,
        default=100,
        help="Early stopping rounds used by the domain classifier.",
    )
    parser.add_argument(
        "--domain-max-train-nodes",
        type=int,
        default=250000,
        help="Max source-domain rows sampled from historical train for domain fitting.",
    )
    parser.add_argument(
        "--domain-max-target-nodes",
        type=int,
        default=250000,
        help="Max target-domain rows sampled from the unlabeled target split for domain fitting.",
    )
    parser.add_argument(
        "--domain-predict-batch-size",
        type=int,
        default=100000,
        help="Batch size used when scoring all historical rows with the domain classifier.",
    )


def domain_weight_enabled(args: Any) -> bool:
    return (
        str(getattr(args, "domain_weight_target_scope", "none")) != "none"
        and float(getattr(args, "domain_weight_mix", 0.0)) > 0.0
    )


def _subsample_rows(values: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if limit is None or limit <= 0 or arr.shape[0] <= limit:
        return arr
    rng = np.random.default_rng(seed)
    choice = np.sort(rng.choice(arr.shape[0], size=int(limit), replace=False))
    return np.asarray(arr[choice], dtype=np.float32, copy=False)


def _predict_binary_in_batches(
    booster,
    features: np.ndarray,
    *,
    best_iteration: int,
    batch_size: int,
) -> np.ndarray:
    import xgboost as xgb

    x = np.asarray(features, dtype=np.float32)
    output = np.empty(x.shape[0], dtype=np.float32)
    effective_batch = max(int(batch_size), 1)
    for start in range(0, x.shape[0], effective_batch):
        end = min(start + effective_batch, x.shape[0])
        dmatrix = xgb.DMatrix(x[start:end])
        pred = booster.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1))
        output[start:end] = np.asarray(pred, dtype=np.float32).reshape(-1)
    return output


def build_domain_adaptation_weights(
    *,
    x_train: np.ndarray,
    x_target: np.ndarray,
    device: str,
    seed: int,
    mix: float,
    power: float,
    clip_min: float,
    clip_max: float,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    subsample: float,
    colsample_bytree: float,
    gamma: float,
    reg_alpha: float,
    reg_lambda: float,
    max_bin: int,
    early_stopping_rounds: int,
    max_train_nodes: int | None,
    max_target_nodes: int | None,
    predict_batch_size: int,
    target_scope: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import xgboost as xgb

    train_x = np.asarray(x_train, dtype=np.float32)
    target_x = np.asarray(x_target, dtype=np.float32)
    sampled_train = _subsample_rows(train_x, max_train_nodes, seed + 101)
    sampled_target = _subsample_rows(target_x, max_target_nodes, seed + 211)

    domain_x = np.concatenate([sampled_train, sampled_target], axis=0).astype(np.float32, copy=False)
    domain_y = np.concatenate(
        [
            np.zeros(sampled_train.shape[0], dtype=np.int8),
            np.ones(sampled_target.shape[0], dtype=np.int8),
        ],
        axis=0,
    )
    fit_x, holdout_x, fit_y, holdout_y = train_test_split(
        domain_x,
        domain_y,
        test_size=0.15,
        random_state=int(seed) + 17,
        stratify=domain_y,
    )

    dtrain = xgb.DMatrix(fit_x, label=fit_y)
    dholdout = xgb.DMatrix(holdout_x, label=holdout_y)
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": str(device),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "min_child_weight": float(min_child_weight),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "gamma": float(gamma),
            "reg_alpha": float(reg_alpha),
            "reg_lambda": float(reg_lambda),
            "max_bin": int(max_bin),
            "seed": int(seed) + 701,
        },
        dtrain=dtrain,
        num_boost_round=int(n_estimators),
        evals=[(dholdout, "validation")],
        maximize=True,
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", int(n_estimators) - 1))
    holdout_prob = booster.predict(dholdout, iteration_range=(0, best_iteration + 1))
    holdout_auc = float(roc_auc_score(holdout_y, np.asarray(holdout_prob, dtype=np.float32)))

    train_domain_prob = _predict_binary_in_batches(
        booster,
        train_x,
        best_iteration=best_iteration,
        batch_size=int(predict_batch_size),
    )
    odds = train_domain_prob / np.clip(1.0 - train_domain_prob, 1e-4, None)
    shifted = np.power(np.clip(odds, 1e-4, None), float(power)).astype(np.float32, copy=False)
    shifted = np.clip(shifted, float(clip_min), float(clip_max))
    weights = ((1.0 - float(mix)) + float(mix) * shifted).astype(np.float32, copy=False)
    mean_weight = float(np.mean(weights, dtype=np.float64))
    if mean_weight > 0.0:
        weights /= mean_weight

    payload = {
        "enabled": True,
        "target_scope": str(target_scope),
        "mix": float(mix),
        "power": float(power),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "domain_auc": holdout_auc,
        "best_iteration": best_iteration,
        "fit_train_rows": int(sampled_train.shape[0]),
        "fit_target_rows": int(sampled_target.shape[0]),
        "weight_summary": {
            "min": float(np.min(weights)),
            "p50": float(np.median(weights)),
            "p90": float(np.quantile(weights, 0.9)),
            "p99": float(np.quantile(weights, 0.99)),
            "max": float(np.max(weights)),
            "mean": float(np.mean(weights)),
        },
    }
    print(
        f"[domain-weight] scope={target_scope} auc={holdout_auc:.6f} "
        f"mix={float(mix):.3f} p50={payload['weight_summary']['p50']:.4f} "
        f"p90={payload['weight_summary']['p90']:.4f} max={payload['weight_summary']['max']:.4f}"
    )
    return weights, payload


def build_domain_adaptation_weights_from_args(
    args: Any,
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    disabled_payload = {
        "enabled": False,
        "target_scope": str(getattr(args, "domain_weight_target_scope", "none")),
        "mix": float(getattr(args, "domain_weight_mix", 0.0)),
        "power": float(getattr(args, "domain_weight_power", 1.0)),
    }
    if not domain_weight_enabled(args):
        return np.ones(np.asarray(x_train).shape[0], dtype=np.float32), disabled_payload

    target_scope = str(getattr(args, "domain_weight_target_scope", "none"))
    if target_scope != "phase1_val":
        raise ValueError(f"Unsupported domain_weight_target_scope={target_scope!r}")
    return build_domain_adaptation_weights(
        x_train=np.asarray(x_train, dtype=np.float32),
        x_target=np.asarray(x_val, dtype=np.float32),
        device=str(getattr(args, "device", "cuda")),
        seed=int(getattr(args, "seed", 42)),
        mix=float(getattr(args, "domain_weight_mix", 0.0)),
        power=float(getattr(args, "domain_weight_power", 1.0)),
        clip_min=float(getattr(args, "domain_weight_clip_min", 0.25)),
        clip_max=float(getattr(args, "domain_weight_clip_max", 6.0)),
        n_estimators=int(getattr(args, "domain_n_estimators", 1200)),
        learning_rate=float(getattr(args, "domain_learning_rate", 0.03)),
        max_depth=int(getattr(args, "domain_max_depth", 6)),
        min_child_weight=float(getattr(args, "domain_min_child_weight", 20.0)),
        subsample=float(getattr(args, "domain_subsample", 0.8)),
        colsample_bytree=float(getattr(args, "domain_colsample_bytree", 0.8)),
        gamma=float(getattr(args, "domain_gamma", 0.0)),
        reg_alpha=float(getattr(args, "domain_reg_alpha", 0.0)),
        reg_lambda=float(getattr(args, "domain_reg_lambda", 1.0)),
        max_bin=int(getattr(args, "max_bin", 256)),
        early_stopping_rounds=int(getattr(args, "domain_early_stopping_rounds", 100)),
        max_train_nodes=getattr(args, "domain_max_train_nodes", 250000),
        max_target_nodes=getattr(args, "domain_max_target_nodes", 250000),
        predict_batch_size=int(getattr(args, "domain_predict_batch_size", 100000)),
        target_scope=target_scope,
    )
