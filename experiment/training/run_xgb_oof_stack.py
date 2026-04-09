from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (  # noqa: E402
    BLEND_OUTPUT_ROOT,
    FEATURE_OUTPUT_ROOT,
    align_prediction_bundle,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    load_phase_arrays,
    load_prediction_npz,
    resolve_prediction_path,
    save_prediction_npz,
    write_json,
)
from experiment.training.features import FeatureStore, load_graph_cache, resolve_feature_groups  # noqa: E402
from experiment.training.run_xgb_multiclass_bg import (  # noqa: E402
    _build_sample_weight,
    _binary_score_from_softprob,
)
from experiment.training.run_xgb_multiclass_bg_graphprop import (  # noqa: E402
    _load_or_build_labelprop_features,
)
from experiment.training.run_xgb_graphprop import _load_or_build_cached_features  # noqa: E402


BASE_OOF_CACHE_ROOT = BLEND_OUTPUT_ROOT / "_base_oof_cache"


@dataclass(frozen=True)
class BasePredictionSet:
    run_dir: Path
    family: str
    train_mode: str
    train_node_ids: np.ndarray
    train_score: np.ndarray
    val_score: np.ndarray
    external_score: np.ndarray
    phase1_train_auc: float
    phase1_val_auc: float
    phase2_external_auc: float
    full_rounds: int | None


@dataclass(frozen=True)
class OOFFoldPlan:
    folds: list[tuple[np.ndarray, np.ndarray]]
    train_keep_mask: np.ndarray
    diagnostics: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a leakage-safe meta-model on phase1-train OOF base predictions, then "
            "evaluate on phase1_val and phase2_external."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Base run directories used to generate OOF meta-features.",
    )
    parser.add_argument("--outdir", type=Path, default=BLEND_OUTPUT_ROOT)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument(
        "--meta-model",
        choices=("xgboost", "logistic", "histgb"),
        default="logistic",
    )
    parser.add_argument("--include-rank-features", action="store_true")
    parser.add_argument(
        "--append-ensemble-stats",
        action="store_true",
        help="Append row-wise ensemble summary statistics computed from base probabilities.",
    )
    parser.add_argument(
        "--append-temporal-prototype-features",
        action="store_true",
        help=(
            "Append leakage-safe temporal prototype-memory features built from earlier "
            "phase1-train meta inputs in chronological order."
        ),
    )
    parser.add_argument(
        "--prototype-half-life-days",
        type=float,
        default=90.0,
        help="Half-life used by the temporal prototype memory on meta-input features.",
    )
    parser.add_argument(
        "--meta-context-model",
        choices=("none", "m2_hybrid", "m3_neighbor"),
        default="none",
        help="Optional node-level feature family appended to the meta stage.",
    )
    parser.add_argument(
        "--meta-context-extra-groups",
        nargs="*",
        default=(),
        help="Extra feature groups appended to --meta-context-model.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument(
        "--base-fold-strategy",
        choices=("stratified_random", "time_forward"),
        default="stratified_random",
        help="How to build OOF folds for the base learners used by the meta stage.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--base-max-estimators", type=int, default=1200)
    parser.add_argument("--base-early-stopping-rounds", type=int, default=300)
    parser.add_argument(
        "--base-round-agg",
        choices=("mean", "median", "max"),
        default="mean",
    )
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument(
        "--logistic-penalty",
        choices=("l2", "l1", "elasticnet"),
        default="l2",
    )
    parser.add_argument("--logistic-l1-ratio", type=float, default=0.5)
    parser.add_argument(
        "--logistic-standardize",
        action="store_true",
        help="Standardize meta features before logistic regression.",
    )
    parser.add_argument("--logistic-max-iter", type=int, default=5000)
    parser.add_argument("--logistic-tol", type=float, default=1e-4)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument(
        "--meta-time-weight-half-life-days",
        type=float,
        default=0.0,
        help="If > 0, upweight recent phase1-train nodes when fitting the meta model.",
    )
    parser.add_argument(
        "--meta-time-weight-floor",
        type=float,
        default=0.4,
        help="Minimum recency weight used by --meta-time-weight-half-life-days.",
    )
    return parser.parse_args()


def _load_summary(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _align_bundle_to_ids(bundle: dict[str, np.ndarray], target_ids: np.ndarray) -> dict[str, np.ndarray]:
    bundle_ids = np.asarray(bundle["node_ids"], dtype=np.int32)
    ref_ids = np.asarray(target_ids, dtype=np.int32)
    if not np.all(np.isin(ref_ids, bundle_ids)):
        missing = int(np.sum(~np.isin(ref_ids, bundle_ids)))
        raise AssertionError(f"Prediction bundle is missing {missing} requested node ids.")
    return align_prediction_bundle(bundle, ref_ids)


def _array_sha1(values: np.ndarray) -> str:
    arr = np.asarray(values)
    return hashlib.sha1(arr.view(np.uint8).tobytes()).hexdigest()


def _base_oof_cache_key(
    *,
    run_dir: Path,
    summary: dict[str, object],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    oof_plan: OOFFoldPlan,
    args: argparse.Namespace,
) -> str:
    payload = {
        "run_dir": str(run_dir.resolve()),
        "summary": summary,
        "train_ids_hash": _array_sha1(np.asarray(train_ids, dtype=np.int32)),
        "val_ids_hash": _array_sha1(np.asarray(val_ids, dtype=np.int32)),
        "external_ids_hash": _array_sha1(np.asarray(external_ids, dtype=np.int32)),
        "oof_plan": oof_plan.diagnostics,
        "base_args": {
            "n_splits": int(args.n_splits),
            "base_fold_strategy": str(args.base_fold_strategy),
            "base_max_estimators": int(args.base_max_estimators),
            "base_early_stopping_rounds": int(args.base_early_stopping_rounds),
            "base_round_agg": str(args.base_round_agg),
            "device": str(args.device),
            "random_state": int(args.random_state),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _load_cached_base_prediction_set(cache_dir: Path) -> BasePredictionSet | None:
    meta_path = cache_dir / "meta.json"
    pred_path = cache_dir / "predictions.npz"
    if not (meta_path.exists() and pred_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    payload = np.load(pred_path)
    return BasePredictionSet(
        run_dir=Path(str(meta["run_dir"])),
        family=str(meta["family"]),
        train_mode=str(meta["train_mode"]),
        train_node_ids=np.asarray(payload["train_node_ids"], dtype=np.int32),
        train_score=np.asarray(payload["train_score"], dtype=np.float32),
        val_score=np.asarray(payload["val_score"], dtype=np.float32),
        external_score=np.asarray(payload["external_score"], dtype=np.float32),
        phase1_train_auc=float(meta["phase1_train_auc"]),
        phase1_val_auc=float(meta["phase1_val_auc"]),
        phase2_external_auc=float(meta["phase2_external_auc"]),
        full_rounds=None if meta["full_rounds"] is None else int(meta["full_rounds"]),
    )


def _save_cached_base_prediction_set(cache_dir: Path, pred: BasePredictionSet) -> None:
    ensure_dir(cache_dir)
    write_json(
        cache_dir / "meta.json",
        {
            "run_dir": str(pred.run_dir),
            "family": pred.family,
            "train_mode": pred.train_mode,
            "phase1_train_auc": pred.phase1_train_auc,
            "phase1_val_auc": pred.phase1_val_auc,
            "phase2_external_auc": pred.phase2_external_auc,
            "full_rounds": pred.full_rounds,
        },
    )
    np.savez_compressed(
        cache_dir / "predictions.npz",
        train_node_ids=np.asarray(pred.train_node_ids, dtype=np.int32),
        train_score=np.asarray(pred.train_score, dtype=np.float32),
        val_score=np.asarray(pred.val_score, dtype=np.float32),
        external_score=np.asarray(pred.external_score, dtype=np.float32),
    )


def _rank_norm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    return ranks


def _historical_ids_from_summary(summary: dict[str, object], feature_dir: Path) -> np.ndarray:
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase1_graph = load_graph_cache("phase1", outdir=feature_dir)
    first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)
    threshold_day = int(summary.get("threshold_day", split.threshold_day))
    min_train_day = int(summary.get("min_train_first_active_day", 0))
    include_future_background = bool(summary.get("include_future_background", False))
    if include_future_background:
        train_mask = (
            ((first_active <= threshold_day) & (first_active >= min_train_day) & np.isin(phase1_y, (0, 1)))
            | ((first_active >= min_train_day) & np.isin(phase1_y, (2, 3)))
        )
    else:
        train_mask = (
            (first_active <= threshold_day)
            & (first_active >= min_train_day)
            & np.isin(phase1_y, (0, 1, 2, 3))
        )
    historical_ids = np.flatnonzero(train_mask).astype(np.int32, copy=False)
    expected_size = int(summary["historical_train_size"])
    if historical_ids.size != expected_size:
        raise AssertionError(
            f"historical train size mismatch: rebuilt={historical_ids.size} summary={expected_size}"
        )
    if not np.all(np.isin(np.asarray(split.train_ids, dtype=np.int32), historical_ids)):
        raise AssertionError("phase1 split train ids are not contained in rebuilt historical ids.")
    return historical_ids


def _row_positions(historical_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(historical_ids, target_ids)
    if np.any(positions >= historical_ids.size) or not np.array_equal(historical_ids[positions], target_ids):
        raise AssertionError("Failed to align target ids against historical cache rows.")
    return positions.astype(np.int64, copy=False)


def _build_oof_fold_plan(
    *,
    train_ids: np.ndarray,
    y_train: np.ndarray,
    train_first_active: np.ndarray,
    args: argparse.Namespace,
) -> OOFFoldPlan:
    strategy = str(args.base_fold_strategy)
    n_splits = int(args.n_splits)
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    if strategy == "stratified_random":
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(args.random_state),
        )
        folds = [
            (fit_idx.astype(np.int32, copy=False), hold_idx.astype(np.int32, copy=False))
            for fit_idx, hold_idx in skf.split(train_ids, y_train)
        ]
        return OOFFoldPlan(
            folds=folds,
            train_keep_mask=np.ones(train_ids.shape[0], dtype=bool),
            diagnostics={
                "strategy": strategy,
                "full_train_size": int(train_ids.size),
                "meta_train_size": int(train_ids.size),
                "warmup_size": 0,
                "warmup_pos": 0,
            },
        )

    if strategy != "time_forward":
        raise ValueError(f"Unsupported base fold strategy: {strategy}")

    order = np.lexsort((np.asarray(train_ids, dtype=np.int32), np.asarray(train_first_active, dtype=np.int32)))
    sorted_days = np.asarray(train_first_active[order], dtype=np.int32)
    sorted_labels = np.asarray(y_train[order], dtype=np.int8)
    unique_days, day_start_idx, day_counts = np.unique(
        sorted_days,
        return_index=True,
        return_counts=True,
    )
    day_pos = np.add.reduceat((sorted_labels == 1).astype(np.int32, copy=False), day_start_idx)
    cum_pos = np.cumsum(day_pos)
    total_pos = int(cum_pos[-1]) if cum_pos.size else 0
    if total_pos < n_splits + 1:
        raise ValueError(
            "time_forward OOF needs enough positives to populate one warmup block plus "
            f"{n_splits} holdout blocks, got positives={total_pos}"
        )
    if unique_days.size < n_splits + 1:
        raise ValueError(
            "time_forward OOF needs at least one activation day per chronological block, "
            f"got days={unique_days.size} for blocks={n_splits + 1}"
        )

    boundary_day_idx: list[int] = []
    prev_idx = -1
    for split_idx in range(1, n_splits + 1):
        target_pos = total_pos * split_idx / float(n_splits + 1)
        candidate_idx = int(np.searchsorted(cum_pos, target_pos, side="left"))
        candidate_idx = max(candidate_idx, prev_idx + 1)
        remaining_blocks = (n_splits + 1) - split_idx
        max_candidate_idx = unique_days.size - remaining_blocks
        candidate_idx = min(candidate_idx, max_candidate_idx)
        boundary_day_idx.append(candidate_idx)
        prev_idx = candidate_idx

    block_starts = [0]
    block_ends: list[int] = []
    for day_idx in boundary_day_idx:
        next_start = int(day_start_idx[day_idx + 1]) if day_idx + 1 < unique_days.size else int(order.size)
        block_ends.append(next_start)
        block_starts.append(next_start)
    block_ends.append(int(order.size))

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for block_idx in range(1, n_splits + 1):
        fit_idx = np.asarray(order[: block_starts[block_idx]], dtype=np.int32)
        hold_idx = np.asarray(order[block_starts[block_idx] : block_ends[block_idx]], dtype=np.int32)
        if fit_idx.size == 0 or hold_idx.size == 0:
            raise ValueError(
                "time_forward OOF produced an empty fit/hold block, "
                f"block_idx={block_idx} fit={fit_idx.size} hold={hold_idx.size}"
            )
        if np.sum(y_train[hold_idx] == 1) == 0:
            raise ValueError(
                "time_forward OOF produced a holdout block with zero positives, "
                f"block_idx={block_idx}"
            )
        folds.append((fit_idx, hold_idx))

    train_keep_mask = np.zeros(train_ids.shape[0], dtype=bool)
    for _, hold_idx in folds:
        train_keep_mask[hold_idx] = True

    warmup_idx = np.asarray(order[: block_starts[1]], dtype=np.int32)
    block_summaries = []
    for block_idx, (start, end) in enumerate(zip(block_starts, block_ends, strict=True)):
        block_train_idx = np.asarray(order[start:end], dtype=np.int32)
        block_summaries.append(
            {
                "block_idx": int(block_idx),
                "start_day": int(sorted_days[start]),
                "end_day": int(sorted_days[end - 1]),
                "size": int(block_train_idx.size),
                "positives": int(np.sum(y_train[block_train_idx] == 1)),
            }
        )

    return OOFFoldPlan(
        folds=folds,
        train_keep_mask=train_keep_mask,
        diagnostics={
            "strategy": strategy,
            "full_train_size": int(train_ids.size),
            "meta_train_size": int(np.sum(train_keep_mask)),
            "warmup_size": int(warmup_idx.size),
            "warmup_pos": int(np.sum(y_train[warmup_idx] == 1)),
            "block_summaries": block_summaries,
        },
    )


def _build_time_decay_sample_weight(
    *,
    train_first_active: np.ndarray,
    threshold_day: int,
    half_life_days: float,
    floor: float,
) -> np.ndarray | None:
    if float(half_life_days) <= 0.0:
        return None
    age = np.clip(
        float(threshold_day) - np.asarray(train_first_active, dtype=np.float32),
        0.0,
        None,
    )
    decay = np.power(np.float32(0.5), age / float(half_life_days)).astype(np.float32, copy=False)
    clipped_floor = float(np.clip(floor, 0.0, 1.0))
    sample_weight = (clipped_floor + (1.0 - clipped_floor) * decay).astype(np.float32, copy=False)
    mean_weight = float(np.mean(sample_weight, dtype=np.float64))
    if mean_weight > 0.0:
        sample_weight /= mean_weight
    return sample_weight


def _summary_to_weight_args(summary: dict[str, object], y_hist: np.ndarray | None = None) -> SimpleNamespace:
    counts = summary.get("historical_train_label_counts")
    if counts is None:
        if y_hist is None:
            raise ValueError("historical_train_label_counts missing and y_hist not provided.")
        counts = {
            str(label): int(np.sum(np.asarray(y_hist, dtype=np.int32) == label))
            for label in (0, 1, 2, 3)
        }
    class_weight = summary["class_weight"]
    count0 = float(counts["0"])
    count1 = float(counts["1"])
    count2 = float(counts["2"])
    time_weight = summary.get("time_weight", {})
    return SimpleNamespace(
        fraud_weight_scale=float(class_weight["1"]) * count1 / max(count0, 1.0),
        background_weight=float(class_weight["2"]) * count2 / max(count0, 1.0),
        time_weight_half_life_days=float(time_weight.get("half_life_days") or 0.0),
        time_weight_floor=float(time_weight.get("floor") or 0.25),
    )


def _multiclass_binary_auc_loss(labels: np.ndarray, predt: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    prob = np.asarray(predt, dtype=np.float32).reshape(labels.shape[0], 4)
    score = _binary_score_from_softprob(prob)
    return 1.0 - float(roc_auc_score(labels, score))


def _best_round_from_model(model: xgb.XGBClassifier) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return int(model.n_estimators)
    return int(best_iteration) + 1


def _aggregate_rounds(values: list[int], method: str) -> int:
    if not values:
        raise ValueError("Expected at least one best-iteration value.")
    arr = np.asarray(values, dtype=np.float32)
    if method == "mean":
        return max(int(round(float(np.mean(arr)))), 1)
    if method == "median":
        return max(int(round(float(np.median(arr)))), 1)
    if method == "max":
        return max(int(np.max(arr)), 1)
    raise ValueError(f"Unsupported round aggregation method: {method}")


def _make_base_xgb(
    summary: dict[str, object],
    *,
    n_estimators: int,
    device: str,
    random_state: int,
    eval_metric=None,
    early_stopping_rounds: int | None = None,
) -> xgb.XGBClassifier:
    params = summary["params"]
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        tree_method="hist",
        device=str(device),
        n_estimators=int(n_estimators),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        gamma=float(params["gamma"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        max_bin=int(params["max_bin"]),
        random_state=int(random_state),
        disable_default_eval_metric=1,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
    )


def _run_oof_multiclass_stack_base(
    *,
    run_dir: Path,
    summary: dict[str, object],
    x_hist: np.ndarray,
    x_val: np.ndarray,
    x_external: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    first_active: np.ndarray,
    historical_ids: np.ndarray,
    oof_plan: OOFFoldPlan,
    args: argparse.Namespace,
) -> BasePredictionSet:
    train_pos = _row_positions(historical_ids, train_ids)
    y_hist = phase1_y[historical_ids].astype(np.int32, copy=False)
    y_train = phase1_y[train_ids].astype(np.int8, copy=False)
    y_val = phase1_y[val_ids].astype(np.int8, copy=False)
    y_external = phase2_y[external_ids].astype(np.int8, copy=False)
    first_active_hist = first_active[historical_ids].astype(np.int32, copy=False)
    bg_pos = np.flatnonzero(np.isin(y_hist, (2, 3))).astype(np.int32, copy=False)
    weight_args = _summary_to_weight_args(summary, y_hist=y_hist)
    split = load_experiment_split()
    threshold_day = int(summary.get("threshold_day", split.threshold_day))

    train_oof = np.full(train_ids.shape[0], np.nan, dtype=np.float32)
    fold_rounds: list[int] = []
    for fold_idx, (fit_idx, hold_idx) in enumerate(oof_plan.folds, start=1):
        fit_hist_pos = np.concatenate([bg_pos, train_pos[fit_idx]]).astype(np.int32, copy=False)
        hold_hist_pos = train_pos[hold_idx].astype(np.int32, copy=False)
        y_fit = y_hist[fit_hist_pos].astype(np.int32, copy=False)
        weight_payload = _build_sample_weight(
            y_fit,
            weight_args,
            train_first_active=first_active_hist[fit_hist_pos],
            threshold_day=threshold_day,
        )
        model = _make_base_xgb(
            summary,
            n_estimators=int(args.base_max_estimators),
            device=str(args.device),
            random_state=int(args.random_state) + fold_idx,
            eval_metric=_multiclass_binary_auc_loss,
            early_stopping_rounds=int(args.base_early_stopping_rounds),
        )
        model.fit(
            x_hist[fit_hist_pos],
            y_fit,
            sample_weight=np.asarray(weight_payload["sample_weight"], dtype=np.float32),
            eval_set=[(x_hist[hold_hist_pos], y_hist[hold_hist_pos].astype(np.int32))],
            verbose=False,
        )
        fold_rounds.append(_best_round_from_model(model))
        hold_prob = model.predict_proba(x_hist[hold_hist_pos]).astype(np.float32, copy=False)
        train_oof[hold_idx] = _binary_score_from_softprob(hold_prob)
        hold_auc = float(roc_auc_score(y_hist[hold_hist_pos], train_oof[hold_idx]))
        print(
            f"[oof-base] run={run_dir.name} fold={fold_idx} "
            f"rounds={fold_rounds[-1]} hold_auc={hold_auc:.6f}",
            flush=True,
        )
        del model
        gc.collect()

    if not np.all(np.isfinite(train_oof[oof_plan.train_keep_mask])):
        raise AssertionError(f"{run_dir}: missing OOF predictions on the retained meta-train subset.")

    full_rounds = _aggregate_rounds(fold_rounds, str(args.base_round_agg))
    full_weight_payload = _build_sample_weight(
        y_hist,
        weight_args,
        train_first_active=first_active_hist,
        threshold_day=threshold_day,
    )
    full_model = _make_base_xgb(
        summary,
        n_estimators=full_rounds,
        device=str(args.device),
        random_state=int(args.random_state),
        eval_metric=None,
        early_stopping_rounds=None,
    )
    full_model.fit(
        x_hist,
        y_hist,
        sample_weight=np.asarray(full_weight_payload["sample_weight"], dtype=np.float32),
        verbose=False,
    )
    val_score = _binary_score_from_softprob(full_model.predict_proba(x_val).astype(np.float32, copy=False))
    external_score = _binary_score_from_softprob(
        full_model.predict_proba(x_external).astype(np.float32, copy=False)
    )
    train_node_ids = np.asarray(train_ids[oof_plan.train_keep_mask], dtype=np.int32, copy=False)
    train_score = np.asarray(train_oof[oof_plan.train_keep_mask], dtype=np.float32, copy=False)
    phase1_train_auc = float(roc_auc_score(phase1_y[train_node_ids].astype(np.int8, copy=False), train_score))
    phase1_val_auc = float(roc_auc_score(y_val, val_score))
    phase2_external_auc = float(roc_auc_score(y_external, external_score))
    print(
        f"[oof-base] run={run_dir.name} train_auc={phase1_train_auc:.6f} "
        f"val_auc={phase1_val_auc:.6f} external_auc={phase2_external_auc:.6f} "
        f"full_rounds={full_rounds} meta_train_size={train_node_ids.size}",
        flush=True,
    )
    del full_model
    gc.collect()
    return BasePredictionSet(
        run_dir=run_dir,
        family=str(summary["model"]),
        train_mode="phase1_train_oof",
        train_node_ids=train_node_ids,
        train_score=train_score,
        val_score=val_score,
        external_score=external_score,
        phase1_train_auc=phase1_train_auc,
        phase1_val_auc=phase1_val_auc,
        phase2_external_auc=phase2_external_auc,
        full_rounds=int(full_rounds),
    )


def _build_multiclass_bg_prediction_set(
    *,
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    first_active: np.ndarray,
    oof_plan: OOFFoldPlan,
    args: argparse.Namespace,
) -> BasePredictionSet:
    split = load_experiment_split()
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    historical_ids = _historical_ids_from_summary(summary, feature_dir=feature_dir)
    feature_groups = resolve_feature_groups(str(summary["feature_model"]), list(summary.get("extra_groups", [])))
    phase1_store = FeatureStore("phase1", feature_groups, outdir=feature_dir)
    phase2_store = FeatureStore("phase2", feature_groups, outdir=feature_dir)
    x_hist = phase1_store.take_rows(historical_ids).astype(np.float32, copy=False)
    x_val = phase1_store.take_rows(val_ids).astype(np.float32, copy=False)
    x_external = phase2_store.take_rows(external_ids).astype(np.float32, copy=False)
    return _run_oof_multiclass_stack_base(
        run_dir=run_dir,
        summary=summary,
        x_hist=x_hist,
        x_val=x_val,
        x_external=x_external,
        train_ids=train_ids,
        val_ids=val_ids,
        external_ids=external_ids,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        first_active=first_active,
        historical_ids=historical_ids,
        oof_plan=oof_plan,
        args=args,
    )


def _build_multiclass_bg_graphprop_prediction_set(
    *,
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    first_active: np.ndarray,
    oof_plan: OOFFoldPlan,
    args: argparse.Namespace,
) -> BasePredictionSet:
    split = load_experiment_split()
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    historical_ids = _historical_ids_from_summary(summary, feature_dir=args.feature_dir)
    train_pos = _row_positions(historical_ids, train_ids)
    cache_dir = Path(str(summary["cache_dir"]))
    phase1_train_path = cache_dir / "phase1_train.npy"
    phase1_val_path = cache_dir / "phase1_val.npy"
    phase2_external_path = cache_dir / "phase2_external.npy"
    if phase1_train_path.exists() and phase1_val_path.exists() and phase2_external_path.exists():
        x_hist_base = np.load(phase1_train_path, mmap_mode="r")
        x_val_base = np.load(phase1_val_path, mmap_mode="r")
        x_external_base = np.load(phase2_external_path, mmap_mode="r")
    else:
        graphprop_args = SimpleNamespace(
            feature_dir=feature_dir,
            base_model=str(summary["base_model"]),
            prop_model=str(summary["prop_model"]),
            prop_blocks=list(summary.get("prop_blocks", [])),
            extra_groups=list(summary.get("extra_groups", [])),
            base_extra_groups=(
                list(summary.get("extra_groups", []))
                if summary.get("base_extra_groups") is None
                else list(summary.get("base_extra_groups", []))
            ),
            prop_extra_groups=(
                list(summary.get("extra_groups", []))
                if summary.get("prop_extra_groups") is None
                else list(summary.get("prop_extra_groups", []))
            ),
            max_train_nodes=None,
            max_val_nodes=None,
            max_external_nodes=None,
        )
        half_lives = [None if value is None else float(value) for value in list(summary.get("prop_half_life_days", []))]
        phase1_base, phase2_base, _ = _load_or_build_cached_features(
            args=graphprop_args,
            cache_dir=cache_dir,
            phase1_ids={"train": historical_ids, "val": val_ids},
            phase2_ids={"external": external_ids},
            half_lives=half_lives,
        )
        x_hist_base = phase1_base["train"]
        x_val_base = phase1_base["val"]
        x_external_base = phase2_base["external"]

    hist_blocks = [x_hist_base]
    val_blocks = [x_val_base]
    external_blocks = [x_external_base]

    train_labelprop = cache_dir / "phase1_train_labelprop.npy"
    val_labelprop = cache_dir / "phase1_val_labelprop.npy"
    external_labelprop = cache_dir / "phase2_external_labelprop.npy"
    label_prop_blocks = list(summary.get("label_prop_blocks", []))
    if label_prop_blocks and not (train_labelprop.exists() and val_labelprop.exists() and external_labelprop.exists()):
        if bool(summary.get("append_best_label_context", False)) or bool(summary.get("append_best_groupagg", False)) or bool(summary.get("append_best_relgroup", False)):
            raise NotImplementedError(
                f"{run_dir}: cache rebuild fallback currently supports graphprop base and labelprop only."
            )
        labelprop_args = SimpleNamespace(
            feature_dir=feature_dir,
            label_prop_blocks=label_prop_blocks,
        )
        phase1_labelprop, phase2_labelprop, _ = _load_or_build_labelprop_features(
            args=labelprop_args,
            cache_dir=cache_dir,
            split=split,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            phase1_ids={"train": historical_ids, "val": val_ids},
            phase2_ids={"external": external_ids},
            label_prop_half_lives=[
                None if value is None else float(value)
                for value in list(summary.get("label_prop_half_life_days", summary.get("prop_half_life_days", [])))
            ],
        )
        hist_blocks.append(phase1_labelprop["train"])
        val_blocks.append(phase1_labelprop["val"])
        external_blocks.append(phase2_labelprop["external"])
    elif train_labelprop.exists() and val_labelprop.exists() and external_labelprop.exists():
        hist_blocks.append(np.load(train_labelprop, mmap_mode="r"))
        val_blocks.append(np.load(val_labelprop, mmap_mode="r"))
        external_blocks.append(np.load(external_labelprop, mmap_mode="r"))

    x_hist = (
        np.concatenate([np.asarray(block, dtype=np.float32) for block in hist_blocks], axis=1)
        if len(hist_blocks) > 1
        else np.asarray(hist_blocks[0], dtype=np.float32)
    )
    x_val = (
        np.concatenate([np.asarray(block, dtype=np.float32) for block in val_blocks], axis=1)
        if len(val_blocks) > 1
        else np.asarray(val_blocks[0], dtype=np.float32)
    )
    x_external = (
        np.concatenate([np.asarray(block, dtype=np.float32) for block in external_blocks], axis=1)
        if len(external_blocks) > 1
        else np.asarray(external_blocks[0], dtype=np.float32)
    )
    if not np.array_equal(historical_ids[train_pos], train_ids):
        raise AssertionError("graphprop cache alignment failed for train ids.")
    return _run_oof_multiclass_stack_base(
        run_dir=run_dir,
        summary=summary,
        x_hist=x_hist,
        x_val=x_val,
        x_external=x_external,
        train_ids=train_ids,
        val_ids=val_ids,
        external_ids=external_ids,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        first_active=first_active,
        historical_ids=historical_ids,
        oof_plan=oof_plan,
        args=args,
    )


def _build_multiclass_bg_relgroup_prediction_set(
    *,
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    first_active: np.ndarray,
    oof_plan: OOFFoldPlan,
    args: argparse.Namespace,
) -> BasePredictionSet:
    split = load_experiment_split()
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    historical_ids = _historical_ids_from_summary(summary, feature_dir=feature_dir)
    feature_groups = resolve_feature_groups(str(summary["feature_model"]), list(summary.get("extra_groups", [])))
    phase1_store = FeatureStore("phase1", feature_groups, outdir=feature_dir)
    phase2_store = FeatureStore("phase2", feature_groups, outdir=feature_dir)
    cache_dir = Path(str(summary["cache_dir"]))
    rel_hist = np.asarray(np.load(cache_dir / "phase1_train.npy", mmap_mode="r"), dtype=np.float32)
    rel_val = np.asarray(np.load(cache_dir / "phase1_val.npy", mmap_mode="r"), dtype=np.float32)
    rel_external = np.asarray(np.load(cache_dir / "phase2_external.npy", mmap_mode="r"), dtype=np.float32)
    x_hist = np.concatenate([phase1_store.take_rows(historical_ids), rel_hist], axis=1).astype(np.float32, copy=False)
    x_val = np.concatenate([phase1_store.take_rows(val_ids), rel_val], axis=1).astype(np.float32, copy=False)
    x_external = np.concatenate([phase2_store.take_rows(external_ids), rel_external], axis=1).astype(
        np.float32,
        copy=False,
    )
    return _run_oof_multiclass_stack_base(
        run_dir=run_dir,
        summary=summary,
        x_hist=x_hist,
        x_val=x_val,
        x_external=x_external,
        train_ids=train_ids,
        val_ids=val_ids,
        external_ids=external_ids,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        first_active=first_active,
        historical_ids=historical_ids,
        oof_plan=oof_plan,
        args=args,
    )


def _build_multiclass_bg_groupagg_prediction_set(
    *,
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    first_active: np.ndarray,
    oof_plan: OOFFoldPlan,
    args: argparse.Namespace,
) -> BasePredictionSet:
    split = load_experiment_split()
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    historical_ids = _historical_ids_from_summary(summary, feature_dir=feature_dir)
    feature_groups = resolve_feature_groups(str(summary["feature_model"]), list(summary.get("extra_groups", [])))
    phase1_store = FeatureStore("phase1", feature_groups, outdir=feature_dir)
    phase2_store = FeatureStore("phase2", feature_groups, outdir=feature_dir)
    cache_dir = Path(str(summary["cache_dir"]))
    group_hist = np.asarray(np.load(cache_dir / "phase1_train_groupagg.npy", mmap_mode="r"), dtype=np.float32)
    group_val = np.asarray(np.load(cache_dir / "phase1_val_groupagg.npy", mmap_mode="r"), dtype=np.float32)
    group_external = np.asarray(
        np.load(cache_dir / "phase2_external_groupagg.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    x_hist = np.concatenate([phase1_store.take_rows(historical_ids), group_hist], axis=1).astype(np.float32, copy=False)
    x_val = np.concatenate([phase1_store.take_rows(val_ids), group_val], axis=1).astype(np.float32, copy=False)
    x_external = np.concatenate([phase2_store.take_rows(external_ids), group_external], axis=1).astype(
        np.float32,
        copy=False,
    )
    return _run_oof_multiclass_stack_base(
        run_dir=run_dir,
        summary=summary,
        x_hist=x_hist,
        x_val=x_val,
        x_external=x_external,
        train_ids=train_ids,
        val_ids=val_ids,
        external_ids=external_ids,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        first_active=first_active,
        historical_ids=historical_ids,
        oof_plan=oof_plan,
        args=args,
    )


def _build_scoreprop_stage1_prediction_set(
    *,
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    oof_plan: OOFFoldPlan,
) -> BasePredictionSet:
    split = load_experiment_split()
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    historical_ids = _historical_ids_from_summary(summary, feature_dir=feature_dir)
    train_pos = _row_positions(historical_ids, train_ids)
    cache_dir = Path(str(summary["cache_dir"]))
    phase1_train_oof_prob = np.asarray(np.load(cache_dir / "phase1_train_oof_softprob.npy", mmap_mode="r"))
    phase1_full_prob = np.asarray(np.load(cache_dir / "phase1_full_softprob.npy", mmap_mode="r"))
    phase2_full_prob = np.asarray(np.load(cache_dir / "phase2_full_softprob.npy", mmap_mode="r"))
    train_score = _binary_score_from_softprob(np.asarray(phase1_train_oof_prob[train_pos], dtype=np.float32))
    val_score = _binary_score_from_softprob(np.asarray(phase1_full_prob[val_ids], dtype=np.float32))
    external_score = _binary_score_from_softprob(np.asarray(phase2_full_prob[external_ids], dtype=np.float32))
    train_node_ids = np.asarray(train_ids[oof_plan.train_keep_mask], dtype=np.int32, copy=False)
    train_score = np.asarray(train_score[oof_plan.train_keep_mask], dtype=np.float32, copy=False)
    y_train = phase1_y[train_node_ids].astype(np.int8, copy=False)
    y_val = phase1_y[val_ids].astype(np.int8, copy=False)
    y_external = phase2_y[external_ids].astype(np.int8, copy=False)
    phase1_train_auc = float(roc_auc_score(y_train, train_score))
    phase1_val_auc = float(roc_auc_score(y_val, val_score))
    phase2_external_auc = float(roc_auc_score(y_external, external_score))
    print(
        f"[oof-base] run={run_dir.name} train_auc={phase1_train_auc:.6f} "
        f"val_auc={phase1_val_auc:.6f} external_auc={phase2_external_auc:.6f} "
        f"mode=scoreprop_stage1",
        flush=True,
    )
    return BasePredictionSet(
        run_dir=run_dir,
        family=f"{summary['model']}_stage1",
        train_mode="phase1_train_oof_stage1_score",
        train_node_ids=train_node_ids,
        train_score=train_score,
        val_score=val_score,
        external_score=external_score,
        phase1_train_auc=phase1_train_auc,
        phase1_val_auc=phase1_val_auc,
        phase2_external_auc=phase2_external_auc,
        full_rounds=None,
    )


def _build_saved_oof_prediction_set(
    *,
    run_dir: Path,
    summary: dict[str, object],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
) -> BasePredictionSet:
    fit_scope = str(summary.get("fit_scope") or "")
    if fit_scope != "phase1_train_oof_only":
        raise ValueError(f"{run_dir}: only phase1_train_oof_only summaries can be reused safely, got {fit_scope!r}")

    train_bundle = _align_bundle_to_ids(
        load_prediction_npz(resolve_prediction_path(run_dir, "phase1_train")),
        train_ids,
    )
    val_bundle = _align_bundle_to_ids(
        load_prediction_npz(resolve_prediction_path(run_dir, "phase1_val")),
        val_ids,
    )
    external_bundle = _align_bundle_to_ids(
        load_prediction_npz(resolve_prediction_path(run_dir, "phase2_external")),
        external_ids,
    )
    expected_train_y = phase1_y[train_ids].astype(np.int8, copy=False)
    expected_val_y = phase1_y[val_ids].astype(np.int8, copy=False)
    expected_external_y = phase2_y[external_ids].astype(np.int8, copy=False)
    if not np.array_equal(np.asarray(train_bundle["y_true"], dtype=np.int8), expected_train_y):
        raise AssertionError(f"{run_dir}: phase1_train labels do not match current split ids.")
    if not np.array_equal(np.asarray(val_bundle["y_true"], dtype=np.int8), expected_val_y):
        raise AssertionError(f"{run_dir}: phase1_val labels do not match current split ids.")
    if not np.array_equal(np.asarray(external_bundle["y_true"], dtype=np.int8), expected_external_y):
        raise AssertionError(f"{run_dir}: phase2_external labels do not match current split ids.")

    train_score = np.asarray(train_bundle["probability"], dtype=np.float32, copy=False)
    val_score = np.asarray(val_bundle["probability"], dtype=np.float32, copy=False)
    external_score = np.asarray(external_bundle["probability"], dtype=np.float32, copy=False)
    phase1_train_auc = float(roc_auc_score(expected_train_y, train_score))
    phase1_val_auc = float(roc_auc_score(expected_val_y, val_score))
    phase2_external_auc = float(roc_auc_score(expected_external_y, external_score))
    print(
        f"[oof-base-saved] run={run_dir.name} train_auc={phase1_train_auc:.6f} "
        f"val_auc={phase1_val_auc:.6f} external_auc={phase2_external_auc:.6f} "
        f"method={summary.get('method', 'unknown')}",
        flush=True,
    )
    return BasePredictionSet(
        run_dir=run_dir,
        family=f"saved_oof_{summary.get('method', 'unknown')}",
        train_mode=fit_scope,
        train_node_ids=np.asarray(train_ids, dtype=np.int32, copy=False),
        train_score=train_score,
        val_score=val_score,
        external_score=external_score,
        phase1_train_auc=phase1_train_auc,
        phase1_val_auc=phase1_val_auc,
        phase2_external_auc=phase2_external_auc,
        full_rounds=None,
    )


def _score_base_run(
    *,
    run_dir: Path,
    feature_dir: Path,
    train_ids: np.ndarray,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    first_active: np.ndarray,
    oof_plan: OOFFoldPlan,
    args: argparse.Namespace,
) -> BasePredictionSet:
    summary = _load_summary(run_dir)
    fit_scope = str(summary.get("fit_scope") or "")
    split = load_experiment_split()
    meta_train_ids = np.asarray(train_ids[oof_plan.train_keep_mask], dtype=np.int32, copy=False)
    if fit_scope == "phase1_train_oof_only":
        cache_key = _base_oof_cache_key(
            run_dir=run_dir,
            summary=summary,
            train_ids=meta_train_ids,
            val_ids=np.asarray(split.val_ids, dtype=np.int32),
            external_ids=np.asarray(split.external_ids, dtype=np.int32),
            oof_plan=oof_plan,
            args=args,
        )
        cache_dir = BASE_OOF_CACHE_ROOT / cache_key
        cached = _load_cached_base_prediction_set(cache_dir)
        if cached is not None:
            print(
                f"[oof-base-cache] hit run={run_dir.name} key={cache_key[:12]} "
                f"val_auc={cached.phase1_val_auc:.6f} external_auc={cached.phase2_external_auc:.6f}",
                flush=True,
            )
            return cached
        pred = _build_saved_oof_prediction_set(
            run_dir=run_dir,
            summary=summary,
            train_ids=meta_train_ids,
            val_ids=np.asarray(split.val_ids, dtype=np.int32),
            external_ids=np.asarray(split.external_ids, dtype=np.int32),
            phase1_y=phase1_y,
            phase2_y=phase2_y,
        )
        _save_cached_base_prediction_set(cache_dir, pred)
        return pred
    cache_key = _base_oof_cache_key(
        run_dir=run_dir,
        summary=summary,
        train_ids=train_ids,
        val_ids=np.asarray(split.val_ids, dtype=np.int32),
        external_ids=np.asarray(split.external_ids, dtype=np.int32),
        oof_plan=oof_plan,
        args=args,
    )
    cache_dir = BASE_OOF_CACHE_ROOT / cache_key
    cached = _load_cached_base_prediction_set(cache_dir)
    if cached is not None:
        print(
            f"[oof-base-cache] hit run={run_dir.name} key={cache_key[:12]} "
            f"val_auc={cached.phase1_val_auc:.6f} external_auc={cached.phase2_external_auc:.6f}",
            flush=True,
        )
        return cached
    family = str(summary["model"])
    if family == "xgboost_gpu_multiclass_bg":
        pred = _build_multiclass_bg_prediction_set(
            run_dir=run_dir,
            summary=summary,
            feature_dir=feature_dir,
            train_ids=train_ids,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            first_active=first_active,
            oof_plan=oof_plan,
            args=args,
        )
        _save_cached_base_prediction_set(cache_dir, pred)
        return pred
    if family == "xgboost_gpu_multiclass_bg_graphprop":
        pred = _build_multiclass_bg_graphprop_prediction_set(
            run_dir=run_dir,
            summary=summary,
            feature_dir=feature_dir,
            train_ids=train_ids,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            first_active=first_active,
            oof_plan=oof_plan,
            args=args,
        )
        _save_cached_base_prediction_set(cache_dir, pred)
        return pred
    if family == "xgboost_gpu_multiclass_bg_relgroup":
        pred = _build_multiclass_bg_relgroup_prediction_set(
            run_dir=run_dir,
            summary=summary,
            feature_dir=feature_dir,
            train_ids=train_ids,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            first_active=first_active,
            oof_plan=oof_plan,
            args=args,
        )
        _save_cached_base_prediction_set(cache_dir, pred)
        return pred
    if family == "xgboost_gpu_multiclass_bg_groupagg":
        pred = _build_multiclass_bg_groupagg_prediction_set(
            run_dir=run_dir,
            summary=summary,
            feature_dir=feature_dir,
            train_ids=train_ids,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            first_active=first_active,
            oof_plan=oof_plan,
            args=args,
        )
        _save_cached_base_prediction_set(cache_dir, pred)
        return pred
    if family == "xgboost_gpu_multiclass_bg_scoreprop":
        pred = _build_scoreprop_stage1_prediction_set(
            run_dir=run_dir,
            summary=summary,
            feature_dir=feature_dir,
            train_ids=train_ids,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            oof_plan=oof_plan,
        )
        _save_cached_base_prediction_set(cache_dir, pred)
        return pred
    raise NotImplementedError(f"Unsupported base family for OOF stacker: {family}")


def _build_meta_features(
    prediction_sets: list[BasePredictionSet],
    include_rank_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    external_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    if not prediction_sets:
        raise ValueError("Expected at least one base prediction set.")
    train_node_ids = np.asarray(prediction_sets[0].train_node_ids, dtype=np.int32, copy=False)
    for idx, pred in enumerate(prediction_sets):
        if not np.array_equal(np.asarray(pred.train_node_ids, dtype=np.int32), train_node_ids):
            raise AssertionError("Base prediction sets produced different phase1-train node alignments.")
        train_prob = np.asarray(pred.train_score, dtype=np.float32)
        val_prob = np.asarray(pred.val_score, dtype=np.float32)
        external_prob = np.asarray(pred.external_score, dtype=np.float32)
        train_blocks.append(train_prob.reshape(-1, 1))
        val_blocks.append(val_prob.reshape(-1, 1))
        external_blocks.append(external_prob.reshape(-1, 1))
        feature_names.append(f"model_{idx}_prob")
        if include_rank_features:
            train_blocks.append(_rank_norm(train_prob).reshape(-1, 1))
            val_blocks.append(_rank_norm(val_prob).reshape(-1, 1))
            external_blocks.append(_rank_norm(external_prob).reshape(-1, 1))
            feature_names.append(f"model_{idx}_rank")
    return (
        np.concatenate(train_blocks, axis=1).astype(np.float32, copy=False),
        np.concatenate(val_blocks, axis=1).astype(np.float32, copy=False),
        np.concatenate(external_blocks, axis=1).astype(np.float32, copy=False),
        feature_names,
        train_node_ids,
    )


def _build_ensemble_stat_features(
    prediction_sets: list[BasePredictionSet],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_prob = np.column_stack([np.asarray(pred.train_score, dtype=np.float32) for pred in prediction_sets])
    val_prob = np.column_stack([np.asarray(pred.val_score, dtype=np.float32) for pred in prediction_sets])
    external_prob = np.column_stack([np.asarray(pred.external_score, dtype=np.float32) for pred in prediction_sets])

    def stats_block(matrix: np.ndarray) -> np.ndarray:
        ordered = np.sort(np.asarray(matrix, dtype=np.float32), axis=1)
        top1 = ordered[:, -1]
        top2 = ordered[:, -2] if ordered.shape[1] > 1 else ordered[:, -1]
        bottom = ordered[:, 0]
        return np.column_stack(
            [
                matrix.mean(axis=1),
                matrix.std(axis=1),
                np.median(matrix, axis=1),
                bottom,
                top1,
                top1 - bottom,
                top1 - top2,
            ]
        ).astype(np.float32, copy=False)

    feature_names = [
        "ensemble_prob_mean",
        "ensemble_prob_std",
        "ensemble_prob_median",
        "ensemble_prob_min",
        "ensemble_prob_max",
        "ensemble_prob_range",
        "ensemble_prob_top_gap",
    ]
    return stats_block(train_prob), stats_block(val_prob), stats_block(external_prob), feature_names


def _prototype_row_features(
    x: np.ndarray,
    *,
    pos_mean: np.ndarray | None,
    neg_mean: np.ndarray | None,
    pos_count: float,
    neg_count: float,
) -> np.ndarray:
    out = np.zeros(8, dtype=np.float32)
    x_norm = float(np.linalg.norm(x))
    if pos_mean is not None:
        pos_norm = float(np.linalg.norm(pos_mean))
        if x_norm > 0.0 and pos_norm > 0.0:
            out[0] = float(np.dot(x, pos_mean) / (x_norm * pos_norm))
        out[4] = float(np.linalg.norm(x - pos_mean))
    if neg_mean is not None:
        neg_norm = float(np.linalg.norm(neg_mean))
        if x_norm > 0.0 and neg_norm > 0.0:
            out[1] = float(np.dot(x, neg_mean) / (x_norm * neg_norm))
        out[5] = float(np.linalg.norm(x - neg_mean))
    out[2] = out[0] - out[1]
    out[3] = out[5] - out[4]
    out[6] = float(np.log1p(max(pos_count, 0.0)))
    out[7] = float(np.log1p(max(neg_count, 0.0)))
    return out


def _build_temporal_prototype_features(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_external: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    phase1_first_active: np.ndarray,
    phase2_first_active: np.ndarray,
    half_life_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_days = np.asarray(phase1_first_active[train_ids], dtype=np.int32)
    order = np.lexsort((np.asarray(train_ids, dtype=np.int32), train_days))
    train_block = np.zeros((x_train.shape[0], 8), dtype=np.float32)

    pos_sum = np.zeros(x_train.shape[1], dtype=np.float64)
    neg_sum = np.zeros(x_train.shape[1], dtype=np.float64)
    pos_count = 0.0
    neg_count = 0.0
    last_day: int | None = None
    use_decay = float(half_life_days) > 0.0

    for idx in order:
        day = int(train_days[idx])
        if use_decay and last_day is not None and day > last_day:
            decay = float(np.power(0.5, (day - last_day) / float(half_life_days)))
            pos_sum *= decay
            neg_sum *= decay
            pos_count *= decay
            neg_count *= decay
        pos_mean = None if pos_count <= 0.0 else (pos_sum / max(pos_count, 1e-6)).astype(np.float32, copy=False)
        neg_mean = None if neg_count <= 0.0 else (neg_sum / max(neg_count, 1e-6)).astype(np.float32, copy=False)
        train_block[idx] = _prototype_row_features(
            np.asarray(x_train[idx], dtype=np.float32),
            pos_mean=pos_mean,
            neg_mean=neg_mean,
            pos_count=pos_count,
            neg_count=neg_count,
        )
        if int(y_train[idx]) == 1:
            pos_sum += np.asarray(x_train[idx], dtype=np.float64)
            pos_count += 1.0
        else:
            neg_sum += np.asarray(x_train[idx], dtype=np.float64)
            neg_count += 1.0
        last_day = day

    pos_mean_final = None if pos_count <= 0.0 else (pos_sum / max(pos_count, 1e-6)).astype(np.float32, copy=False)
    neg_mean_final = None if neg_count <= 0.0 else (neg_sum / max(neg_count, 1e-6)).astype(np.float32, copy=False)
    last_train_day = int(np.max(train_days)) if train_days.size else 0

    def eval_block(x_eval: np.ndarray, eval_days: np.ndarray) -> np.ndarray:
        out = np.zeros((x_eval.shape[0], 8), dtype=np.float32)
        for row_idx in range(x_eval.shape[0]):
            eff_pos = pos_count
            eff_neg = neg_count
            if use_decay:
                gap = max(int(eval_days[row_idx]) - last_train_day, 0)
                if gap > 0:
                    decay = float(np.power(0.5, gap / float(half_life_days)))
                    eff_pos *= decay
                    eff_neg *= decay
            out[row_idx] = _prototype_row_features(
                np.asarray(x_eval[row_idx], dtype=np.float32),
                pos_mean=pos_mean_final,
                neg_mean=neg_mean_final,
                pos_count=eff_pos,
                neg_count=eff_neg,
            )
        return out

    val_block = eval_block(x_val, np.asarray(phase1_first_active[val_ids], dtype=np.int32))
    external_block = eval_block(x_external, np.asarray(phase2_first_active[external_ids], dtype=np.int32))
    feature_names = [
        "proto_cos_pos",
        "proto_cos_neg",
        "proto_cos_gap",
        "proto_dist_gap",
        "proto_l2_pos",
        "proto_l2_neg",
        "proto_log_pos_count",
        "proto_log_neg_count",
    ]
    return train_block, val_block, external_block, feature_names


def _load_meta_context_features(
    *,
    feature_dir: Path,
    context_model: str,
    context_extra_groups: list[str],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_groups = resolve_feature_groups(context_model, context_extra_groups)
    phase1_store = FeatureStore("phase1", feature_groups, outdir=feature_dir)
    phase2_store = FeatureStore("phase2", feature_groups, outdir=feature_dir)
    return (
        phase1_store.take_rows(train_ids).astype(np.float32, copy=False),
        phase1_store.take_rows(val_ids).astype(np.float32, copy=False),
        phase2_store.take_rows(external_ids).astype(np.float32, copy=False),
        [f"context__{name}" for name in phase1_store.feature_names],
    )


def _fit_meta_model(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
):
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos_weight = neg_count / max(pos_count, 1.0)
    if args.meta_model == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        penalty = str(args.logistic_penalty)
        solver = "lbfgs" if penalty == "l2" else "saga"
        estimator = LogisticRegression(
            max_iter=int(args.logistic_max_iter),
            C=float(args.logistic_c),
            class_weight="balanced",
            random_state=0,
            penalty=penalty,
            solver=solver,
            tol=float(args.logistic_tol),
            l1_ratio=(float(args.logistic_l1_ratio) if penalty == "elasticnet" else None),
        )
        if bool(args.logistic_standardize) or penalty != "l2":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("logistic", estimator),
                ]
            )
            model.fit(x_train, y_train, logistic__sample_weight=sample_weight)
        else:
            model = estimator
            model.fit(x_train, y_train, sample_weight=sample_weight)
        return model, {
            "scale_pos_weight": scale_pos_weight,
            "logistic_penalty": penalty,
            "logistic_l1_ratio": float(args.logistic_l1_ratio),
            "logistic_standardize": bool(args.logistic_standardize),
            "logistic_solver": solver,
            "logistic_max_iter": int(args.logistic_max_iter),
            "logistic_tol": float(args.logistic_tol),
        }

    if args.meta_model == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            max_leaf_nodes=int(args.max_leaf_nodes),
            l2_regularization=float(args.l2_regularization),
            max_iter=int(args.n_estimators),
            early_stopping=False,
            random_state=0,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return model, {
            "scale_pos_weight": scale_pos_weight,
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "l2_regularization": float(args.l2_regularization),
        }

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device=str(args.device),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        min_child_weight=float(args.min_child_weight),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        gamma=float(args.gamma),
        reg_alpha=float(args.reg_alpha),
        reg_lambda=float(args.reg_lambda),
        max_bin=int(args.max_bin),
        scale_pos_weight=float(scale_pos_weight),
        random_state=0,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
    return model, {"scale_pos_weight": scale_pos_weight}


def _predict_meta_model(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    phase1_graph = load_graph_cache("phase1", outdir=args.feature_dir)
    phase2_graph = None
    first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)
    train_ids = np.asarray(split.train_ids, dtype=np.int32)
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    y_train = phase1_y[train_ids].astype(np.int8, copy=False)
    y_val = phase1_y[val_ids].astype(np.int8, copy=False)
    y_external = phase2_y[external_ids].astype(np.int8, copy=False)
    oof_plan = _build_oof_fold_plan(
        train_ids=train_ids,
        y_train=y_train,
        train_first_active=first_active[train_ids].astype(np.int32, copy=False),
        args=args,
    )

    prediction_sets: list[BasePredictionSet] = []
    for run_dir in args.run_dirs:
        prediction_sets.append(
            _score_base_run(
            run_dir=run_dir,
            feature_dir=args.feature_dir,
            train_ids=train_ids,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            first_active=first_active,
            oof_plan=oof_plan,
            args=args,
        )
        )
        gc.collect()

    x_train, x_val, x_external, feature_names, meta_train_ids = _build_meta_features(
        prediction_sets,
        include_rank_features=bool(args.include_rank_features),
    )
    y_meta_train = phase1_y[meta_train_ids].astype(np.int8, copy=False)
    if args.append_ensemble_stats:
        stat_train, stat_val, stat_external, stat_feature_names = _build_ensemble_stat_features(prediction_sets)
        x_train = np.concatenate([x_train, stat_train], axis=1).astype(np.float32, copy=False)
        x_val = np.concatenate([x_val, stat_val], axis=1).astype(np.float32, copy=False)
        x_external = np.concatenate([x_external, stat_external], axis=1).astype(np.float32, copy=False)
        feature_names.extend(stat_feature_names)
    if args.append_temporal_prototype_features:
        if phase2_graph is None:
            phase2_graph = load_graph_cache("phase2", outdir=args.feature_dir)
        proto_train, proto_val, proto_external, proto_feature_names = _build_temporal_prototype_features(
            x_train=x_train,
            x_val=x_val,
            x_external=x_external,
            y_train=y_meta_train,
            train_ids=meta_train_ids,
            val_ids=val_ids,
            external_ids=external_ids,
            phase1_first_active=first_active,
            phase2_first_active=np.asarray(phase2_graph.first_active, dtype=np.int32),
            half_life_days=float(args.prototype_half_life_days),
        )
        x_train = np.concatenate([x_train, proto_train], axis=1).astype(np.float32, copy=False)
        x_val = np.concatenate([x_val, proto_val], axis=1).astype(np.float32, copy=False)
        x_external = np.concatenate([x_external, proto_external], axis=1).astype(np.float32, copy=False)
        feature_names.extend([f"prototype__{name}" for name in proto_feature_names])
    if args.meta_context_model != "none":
        ctx_train, ctx_val, ctx_external, ctx_feature_names = _load_meta_context_features(
            feature_dir=args.feature_dir,
            context_model=str(args.meta_context_model),
            context_extra_groups=list(args.meta_context_extra_groups),
            train_ids=meta_train_ids,
            val_ids=val_ids,
            external_ids=external_ids,
        )
        x_train = np.concatenate([x_train, ctx_train], axis=1).astype(np.float32, copy=False)
        x_val = np.concatenate([x_val, ctx_val], axis=1).astype(np.float32, copy=False)
        x_external = np.concatenate([x_external, ctx_external], axis=1).astype(np.float32, copy=False)
        feature_names.extend(ctx_feature_names)
    meta_sample_weight = _build_time_decay_sample_weight(
        train_first_active=first_active[meta_train_ids].astype(np.int32, copy=False),
        threshold_day=int(split.threshold_day),
        half_life_days=float(args.meta_time_weight_half_life_days),
        floor=float(args.meta_time_weight_floor),
    )
    model, meta_aux = _fit_meta_model(
        args=args,
        x_train=x_train,
        y_train=y_meta_train,
        sample_weight=meta_sample_weight,
    )
    train_prob = _predict_meta_model(model, x_train)
    val_prob = _predict_meta_model(model, x_val)
    external_prob = _predict_meta_model(model, x_external)

    train_metrics = compute_binary_classification_metrics(y_meta_train, train_prob)
    val_metrics = compute_binary_classification_metrics(y_val, val_prob)
    external_metrics = compute_binary_classification_metrics(y_external, external_prob)

    run_dir = ensure_dir(args.outdir / args.run_name)
    np.savez_compressed(
        run_dir / "phase1_train_meta_features.npz",
        node_ids=meta_train_ids,
        y_true=y_meta_train,
        features=x_train.astype(np.float32, copy=False),
    )
    np.savez_compressed(
        run_dir / "phase1_val_meta_features.npz",
        node_ids=val_ids,
        y_true=y_val,
        features=x_val.astype(np.float32, copy=False),
    )
    np.savez_compressed(
        run_dir / "phase2_external_meta_features.npz",
        node_ids=external_ids,
        y_true=y_external,
        features=x_external.astype(np.float32, copy=False),
    )
    save_prediction_npz(run_dir / "phase1_train_predictions.npz", meta_train_ids, y_meta_train, train_prob)
    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, y_val, val_prob)
    save_prediction_npz(run_dir / "phase2_external_predictions.npz", external_ids, y_external, external_prob)
    if args.meta_model == "xgboost":
        model.save_model(run_dir / "model.json")
    else:
        import pickle

        with (run_dir / "model.pkl").open("wb") as fp:
            pickle.dump(model, fp)

    summary = {
        "blend_name": args.run_name,
        "method": str(args.meta_model),
        "fit_scope": "phase1_train_oof_only",
        "warning": (
            "Meta-model is fitted only on phase1 train labels using out-of-fold base predictions. "
            "phase1_val and phase2_external are never used to fit the meta stage."
        ),
        "base_runs": [
            {
                "run_dir": str(pred.run_dir),
                "family": pred.family,
                "train_mode": pred.train_mode,
                "phase1_train_auc": pred.phase1_train_auc,
                "phase1_val_auc": pred.phase1_val_auc,
                "phase2_external_auc": pred.phase2_external_auc,
                "full_rounds": pred.full_rounds,
            }
            for pred in prediction_sets
        ],
        "n_splits": int(args.n_splits),
        "base_fold_strategy": str(args.base_fold_strategy),
        "base_max_estimators": int(args.base_max_estimators),
        "base_early_stopping_rounds": int(args.base_early_stopping_rounds),
        "base_round_agg": str(args.base_round_agg),
        "oof_plan": oof_plan.diagnostics,
        "include_rank_features": bool(args.include_rank_features),
        "append_ensemble_stats": bool(args.append_ensemble_stats),
        "append_temporal_prototype_features": bool(args.append_temporal_prototype_features),
        "prototype_half_life_days": float(args.prototype_half_life_days),
        "meta_context_model": None if args.meta_context_model == "none" else str(args.meta_context_model),
        "meta_context_extra_groups": list(args.meta_context_extra_groups),
        "feature_dim": int(x_train.shape[1]),
        "feature_names": feature_names,
        "meta_feature_paths": {
            "phase1_train": str(run_dir / "phase1_train_meta_features.npz"),
            "phase1_val": str(run_dir / "phase1_val_meta_features.npz"),
            "phase2_external": str(run_dir / "phase2_external_meta_features.npz"),
        },
        "phase1_train_size": int(meta_train_ids.size),
        "phase1_train_source_size": int(train_ids.size),
        "phase1_val_size": int(val_ids.size),
        "phase2_external_size": int(external_ids.size),
        "phase1_train_auc": train_metrics["auc"],
        "phase1_train_pr_auc": train_metrics["pr_auc"],
        "phase1_train_ap": train_metrics["ap"],
        "phase1_val_auc": val_metrics["auc"],
        "phase1_val_pr_auc": val_metrics["pr_auc"],
        "phase1_val_ap": val_metrics["ap"],
        "phase2_external_auc": external_metrics["auc"],
        "phase2_external_pr_auc": external_metrics["pr_auc"],
        "phase2_external_ap": external_metrics["ap"],
        "params": {
            "meta_model": str(args.meta_model),
            "device": str(args.device),
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "min_child_weight": float(args.min_child_weight),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "gamma": float(args.gamma),
            "reg_alpha": float(args.reg_alpha),
            "reg_lambda": float(args.reg_lambda),
            "max_bin": int(args.max_bin),
            "logistic_c": float(args.logistic_c),
            "logistic_penalty": str(args.logistic_penalty),
            "logistic_l1_ratio": float(args.logistic_l1_ratio),
            "logistic_standardize": bool(args.logistic_standardize),
            "logistic_max_iter": int(args.logistic_max_iter),
            "logistic_tol": float(args.logistic_tol),
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "l2_regularization": float(args.l2_regularization),
            "append_temporal_prototype_features": bool(args.append_temporal_prototype_features),
            "prototype_half_life_days": float(args.prototype_half_life_days),
            "meta_time_weight_half_life_days": float(args.meta_time_weight_half_life_days),
            "meta_time_weight_floor": float(args.meta_time_weight_floor),
            **meta_aux,
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[xgb_oof_stack] run={args.run_name} "
        f"train_auc={train_metrics['auc']:.6f} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f}"
    )


if __name__ == "__main__":
    main()
