from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (
    FEATURE_OUTPUT_ROOT,
    MODEL_OUTPUT_ROOT,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    load_phase_arrays,
    save_prediction_npz,
    set_global_seed,
    write_json,
)
from experiment.training.features import FeatureStore, load_graph_cache, resolve_feature_groups
from experiment.training.run_xgb_graphprop import _half_life_tag, _normalized_csr, _resolve_half_lives
from experiment.training.run_xgb_multiclass_bg import (
    _binary_score_from_softprob,
    _build_sample_weight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU XGBoost two-stage multiclass training with OOF score propagation. "
            "Stage 1 learns node-level soft probabilities, then stage 2 consumes "
            "local + graph-propagated stage-1 scores without leaking validation labels."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--feature-model",
        choices=("m2_hybrid", "m3_neighbor"),
        default="m3_neighbor",
    )
    parser.add_argument(
        "--extra-groups",
        nargs="*",
        default=(),
        help="Optional extra offline feature groups appended to the feature_model.",
    )
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "xgboost_gpu")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=MODEL_OUTPUT_ROOT / "_multiclass_bg_scoreprop_cache",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=300)
    parser.add_argument("--background-weight", type=float, default=0.5)
    parser.add_argument("--fraud-weight-scale", type=float, default=1.0)
    parser.add_argument(
        "--time-weight-half-life-days",
        type=float,
        default=0.0,
        help="If > 0, exponentially upweight recent historical nodes toward the split threshold.",
    )
    parser.add_argument(
        "--time-weight-floor",
        type=float,
        default=0.25,
        help="Minimum recency weight when --time-weight-half-life-days > 0.",
    )
    parser.add_argument(
        "--include-future-background",
        action="store_true",
        help=(
            "Include all phase1 background nodes (labels 2/3) regardless of activation day, "
            "while still restricting 0/1 supervision to the historical side of the split."
        ),
    )
    parser.add_argument(
        "--oof-folds",
        type=int,
        default=4,
        help="Number of phase1 historical folds used to create train-safe stage-1 predictions.",
    )
    parser.add_argument(
        "--prop-blocks",
        nargs="+",
        choices=("in1", "out1", "in2", "out2", "bi1", "bi2"),
        default=("in1", "out1", "in2", "out2"),
        help="Propagation blocks appended to the stage-2 feature matrix.",
    )
    parser.add_argument(
        "--prop-half-life-days",
        type=float,
        nargs="+",
        default=(20.0, 90.0),
        help="Optional edge-time half-lives used for score propagation.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=200000,
        help="Batch size used when scoring all phase1/phase2 nodes with the stage-1 model.",
    )
    parser.add_argument(
        "--include-raw-features",
        action="store_true",
        help="Append the raw feature_model block to stage-2 features in addition to local/propagated scores.",
    )
    return parser.parse_args()


def _cache_key(args: argparse.Namespace, threshold_day: int) -> str:
    payload = {
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "threshold_day": int(threshold_day),
        "background_weight": float(args.background_weight),
        "fraud_weight_scale": float(args.fraud_weight_scale),
        "time_weight_half_life_days": float(args.time_weight_half_life_days),
        "time_weight_floor": float(args.time_weight_floor),
        "include_future_background": bool(args.include_future_background),
        "oof_folds": int(args.oof_folds),
        "params": {
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
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "device": str(args.device),
            "seed": int(args.seed),
        },
        "feature_dir": str(args.feature_dir.resolve()),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _make_xgb_params(args: argparse.Namespace) -> dict[str, float | int | str]:
    return {
        "objective": "multi:softprob",
        "num_class": 4,
        "tree_method": "hist",
        "device": args.device,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "seed": args.seed,
        "disable_default_eval_metric": 1,
    }


def _build_feature_store(feature_dir: Path, phase: str, feature_model: str, extra_groups: list[str]) -> FeatureStore:
    return FeatureStore(
        phase,
        resolve_feature_groups(feature_model, extra_groups),
        outdir=feature_dir,
    )


def _take_rows(feature_store: FeatureStore, node_ids: np.ndarray) -> np.ndarray:
    return feature_store.take_rows(np.asarray(node_ids, dtype=np.int32)).astype(np.float32, copy=False)


def _predict_full_softprob(
    booster,
    feature_store: FeatureStore,
    predict_batch_size: int,
    best_iteration: int,
) -> np.ndarray:
    import xgboost as xgb

    num_nodes = int(feature_store.core.shape[0])
    output = np.empty((num_nodes, 4), dtype=np.float32)
    iterator = range(0, num_nodes, int(predict_batch_size))
    for start in tqdm(
        iterator,
        total=(num_nodes + int(predict_batch_size) - 1) // int(predict_batch_size),
        desc=f"stage1:{feature_store.phase}:predict",
        unit="batch",
        dynamic_ncols=True,
    ):
        end = min(start + int(predict_batch_size), num_nodes)
        node_ids = np.arange(start, end, dtype=np.int32)
        batch = _take_rows(feature_store, node_ids)
        dmatrix = xgb.DMatrix(batch)
        prob = booster.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1)).reshape(-1, 4)
        output[start:end] = np.asarray(prob, dtype=np.float32, copy=False)
    return output


def _score_feature_names() -> list[str]:
    return [
        "prob_normal",
        "prob_fraud",
        "prob_bg2",
        "prob_bg3",
        "prob_foreground",
        "prob_background",
        "fraud_given_foreground",
        "fraud_logit",
        "softmax_entropy",
        "bg2_minus_bg3",
        "fraud_minus_normal",
    ]


def _make_score_features(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    foreground = np.clip(prob[:, 0] + prob[:, 1], 1e-6, 1.0)
    background = np.clip(prob[:, 2] + prob[:, 3], 1e-6, 1.0)
    fraud_given_foreground = np.clip(prob[:, 1] / foreground, 1e-6, 1.0 - 1e-6)
    fraud_logit = np.log(fraud_given_foreground / (1.0 - fraud_given_foreground)).astype(np.float32, copy=False)
    entropy = (-(prob * np.log(np.clip(prob, 1e-6, 1.0)))).sum(axis=1).astype(np.float32, copy=False)
    bg2_minus_bg3 = (prob[:, 2] - prob[:, 3]).astype(np.float32, copy=False)
    fraud_minus_normal = (prob[:, 1] - prob[:, 0]).astype(np.float32, copy=False)
    return np.column_stack(
        [
            prob,
            foreground,
            background,
            fraud_given_foreground,
            fraud_logit,
            entropy,
            bg2_minus_bg3,
            fraud_minus_normal,
        ]
    ).astype(np.float32, copy=False)


def _multiclass_binary_auc_ignore_background(predt: np.ndarray, dmatrix) -> tuple[str, float]:
    labels = np.asarray(dmatrix.get_label(), dtype=np.int8)
    prob = np.asarray(predt, dtype=np.float32).reshape(labels.shape[0], 4)
    mask = np.isin(labels, (0, 1))
    if np.sum(mask) < 2 or np.unique(labels[mask]).size < 2:
        return "binary_auc", 0.5
    fg_prob = np.clip(prob[:, 0] + prob[:, 1], 1e-6, None)
    binary_score = prob[:, 1] / fg_prob
    return "binary_auc", float(roc_auc_score(labels[mask], binary_score[mask]))


def _load_or_build_stage1_probabilities(
    args: argparse.Namespace,
    cache_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_first_active: np.ndarray,
    threshold_day: int,
    x_val: np.ndarray,
    y_val: np.ndarray,
    phase1_store: FeatureStore,
    phase2_store: FeatureStore,
    historical_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    import xgboost as xgb

    phase1_train_oof_path = cache_dir / "phase1_train_oof_softprob.npy"
    phase1_full_path = cache_dir / "phase1_full_softprob.npy"
    phase2_full_path = cache_dir / "phase2_full_softprob.npy"
    meta_path = cache_dir / "stage1_meta.json"
    if phase1_train_oof_path.exists() and phase1_full_path.exists() and phase2_full_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (
            np.load(phase1_train_oof_path, mmap_mode="r"),
            np.load(phase1_full_path, mmap_mode="r"),
            np.load(phase2_full_path, mmap_mode="r"),
            int(meta["best_iteration"]),
        )

    ensure_dir(cache_dir)
    params = _make_xgb_params(args)
    oof_prob = np.zeros((x_train.shape[0], 4), dtype=np.float32)
    splitter = StratifiedKFold(
        n_splits=int(args.oof_folds),
        shuffle=True,
        random_state=int(args.seed),
    )
    base_args = argparse.Namespace(**vars(args))
    for fold_idx, (fit_idx, hold_idx) in enumerate(splitter.split(x_train, y_train), start=1):
        fold_y_train = y_train[fit_idx].astype(np.int32, copy=False)
        fold_first_active = train_first_active[fit_idx].astype(np.int32, copy=False)
        fold_weight_payload = _build_sample_weight(
            fold_y_train,
            base_args,
            train_first_active=fold_first_active,
            threshold_day=int(threshold_day),
        )
        dtrain = xgb.DMatrix(
            np.asarray(x_train[fit_idx], dtype=np.float32),
            label=fold_y_train,
            weight=np.asarray(fold_weight_payload["sample_weight"], dtype=np.float32),
        )
        dhold = xgb.DMatrix(
            np.asarray(x_train[hold_idx], dtype=np.float32),
            label=y_train[hold_idx].astype(np.int32, copy=False),
        )
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(args.n_estimators),
            evals=[(dhold, f"fold{fold_idx}_valid")],
            custom_metric=_multiclass_binary_auc_ignore_background,
            maximize=True,
            early_stopping_rounds=int(args.early_stopping_rounds),
            verbose_eval=50,
        )
        hold_prob = booster.predict(dhold, iteration_range=(0, int(booster.best_iteration) + 1)).reshape(-1, 4)
        oof_prob[hold_idx] = np.asarray(hold_prob, dtype=np.float32, copy=False)

    full_weight_payload = _build_sample_weight(
        y_train,
        base_args,
        train_first_active=train_first_active,
        threshold_day=int(threshold_day),
    )
    dtrain_full = xgb.DMatrix(
        x_train,
        label=y_train,
        weight=np.asarray(full_weight_payload["sample_weight"], dtype=np.float32),
    )
    dval = xgb.DMatrix(x_val, label=y_val)
    full_booster = xgb.train(
        params=params,
        dtrain=dtrain_full,
        num_boost_round=int(args.n_estimators),
        evals=[(dval, "validation")],
        custom_metric=_multiclass_binary_auc_ignore_background,
        maximize=True,
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=50,
    )
    best_iteration = int(full_booster.best_iteration)
    phase1_full_prob = _predict_full_softprob(
        full_booster,
        phase1_store,
        predict_batch_size=int(args.predict_batch_size),
        best_iteration=best_iteration,
    )
    phase2_full_prob = _predict_full_softprob(
        full_booster,
        phase2_store,
        predict_batch_size=int(args.predict_batch_size),
        best_iteration=best_iteration,
    )

    np.save(phase1_train_oof_path, np.asarray(oof_prob, dtype=np.float32))
    np.save(phase1_full_path, np.asarray(phase1_full_prob, dtype=np.float32))
    np.save(phase2_full_path, np.asarray(phase2_full_prob, dtype=np.float32))
    meta_path.write_text(
        json.dumps(
            {
                "best_iteration": best_iteration,
                "historical_train_size": int(historical_ids.size),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return (
        np.load(phase1_train_oof_path, mmap_mode="r"),
        np.load(phase1_full_path, mmap_mode="r"),
        np.load(phase2_full_path, mmap_mode="r"),
        best_iteration,
    )


def _append_feature_block(
    blocks: dict[str, list[np.ndarray]],
    feature_names: list[str],
    split_ids: dict[str, np.ndarray],
    full_matrix: np.ndarray,
    names: list[str],
    prefix: str,
) -> None:
    feature_names.extend(f"{prefix}__{name}" for name in names)
    for split_name, node_ids in split_ids.items():
        blocks[split_name].append(np.asarray(full_matrix[node_ids], dtype=np.float32, copy=False))


def _append_propagated_score_blocks(
    blocks: dict[str, list[np.ndarray]],
    feature_names: list[str],
    split_ids: dict[str, np.ndarray],
    score_matrix: np.ndarray,
    score_names: list[str],
    graph_cache,
    prop_blocks: list[str],
    prop_half_life_days: list[float | None],
) -> None:
    need_in = any(block in {"in1", "in2", "bi1", "bi2"} for block in prop_blocks)
    need_out = any(block in {"out1", "out2", "bi1", "bi2"} for block in prop_blocks)
    use_half_life_prefix = len(prop_half_life_days) > 1

    for half_life in prop_half_life_days:
        prefix_root = f"{_half_life_tag(half_life)}__" if use_half_life_prefix else ""
        a_in = None
        a_out = None
        in1 = None
        out1 = None
        if need_in:
            a_in = _normalized_csr(
                graph_cache.in_ptr,
                graph_cache.in_neighbors,
                graph_cache.num_nodes,
                timestamps=graph_cache.in_edge_timestamp,
                max_day=graph_cache.max_day,
                half_life_days=half_life,
            )
            in1 = (a_in @ score_matrix).astype(np.float32, copy=False)
            if "in1" in prop_blocks:
                _append_feature_block(blocks, feature_names, split_ids, in1, score_names, f"{prefix_root}in1")
        if need_out:
            a_out = _normalized_csr(
                graph_cache.out_ptr,
                graph_cache.out_neighbors,
                graph_cache.num_nodes,
                timestamps=graph_cache.out_edge_timestamp,
                max_day=graph_cache.max_day,
                half_life_days=half_life,
            )
            out1 = (a_out @ score_matrix).astype(np.float32, copy=False)
            if "out1" in prop_blocks:
                _append_feature_block(blocks, feature_names, split_ids, out1, score_names, f"{prefix_root}out1")
        if "bi1" in prop_blocks:
            if in1 is None or out1 is None:
                raise RuntimeError("bi1 propagation requires inbound and outbound first-hop features.")
            bi1 = ((in1 + out1) * 0.5).astype(np.float32, copy=False)
            _append_feature_block(blocks, feature_names, split_ids, bi1, score_names, f"{prefix_root}bi1")
        if "in2" in prop_blocks:
            if a_in is None or in1 is None:
                raise RuntimeError("in2 propagation requires inbound adjacency and first-hop features.")
            in2 = (a_in @ in1).astype(np.float32, copy=False)
            _append_feature_block(blocks, feature_names, split_ids, in2, score_names, f"{prefix_root}in2")
        if "out2" in prop_blocks:
            if a_out is None or out1 is None:
                raise RuntimeError("out2 propagation requires outbound adjacency and first-hop features.")
            out2 = (a_out @ out1).astype(np.float32, copy=False)
            _append_feature_block(blocks, feature_names, split_ids, out2, score_names, f"{prefix_root}out2")
        if "bi2" in prop_blocks:
            if a_in is None or a_out is None or in1 is None or out1 is None:
                raise RuntimeError("bi2 propagation requires inbound/outbound adjacency and first-hop features.")
            bi2 = (((a_in @ in1) + (a_out @ out1)) * 0.5).astype(np.float32, copy=False)
            _append_feature_block(blocks, feature_names, split_ids, bi2, score_names, f"{prefix_root}bi2")


def _write_feature_importance(booster, feature_names: list[str], path: Path) -> None:
    scores = booster.get_score(importance_type="gain")
    rows = []
    for idx, feature_name in enumerate(feature_names):
        rows.append({"feature_name": feature_name, "gain": float(scores.get(f"f{idx}", 0.0))})
    rows.sort(key=lambda row: row["gain"], reverse=True)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["feature_name", "gain"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    import xgboost as xgb

    set_global_seed(args.seed)
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    phase1_graph = load_graph_cache("phase1", outdir=args.feature_dir)
    phase2_graph = load_graph_cache("phase2", outdir=args.feature_dir)
    phase1_first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)

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

    phase1_store = _build_feature_store(args.feature_dir, "phase1", args.feature_model, list(args.extra_groups))
    phase2_store = _build_feature_store(args.feature_dir, "phase2", args.feature_model, list(args.extra_groups))

    x_train_raw = _take_rows(phase1_store, historical_ids)
    x_val_raw = _take_rows(phase1_store, val_ids)
    x_external_raw = _take_rows(phase2_store, external_ids)
    raw_feature_names = list(phase1_store.feature_names)

    y_train = phase1_y[historical_ids].astype(np.int32, copy=False)
    y_val = phase1_y[val_ids].astype(np.int32, copy=False)
    y_external = phase2_y[external_ids].astype(np.int32, copy=False)
    train_first_active = phase1_first_active[historical_ids].astype(np.int32, copy=False)

    cache_dir = ensure_dir(args.cache_root / _cache_key(args, threshold_day=int(split.threshold_day)))
    phase1_train_oof_prob, phase1_full_prob, phase2_full_prob, stage1_best_iteration = _load_or_build_stage1_probabilities(
        args=args,
        cache_dir=cache_dir,
        x_train=x_train_raw,
        y_train=y_train,
        train_first_active=train_first_active,
        threshold_day=int(split.threshold_day),
        x_val=x_val_raw,
        y_val=y_val,
        phase1_store=phase1_store,
        phase2_store=phase2_store,
        historical_ids=historical_ids,
    )

    if not args.include_raw_features:
        del x_train_raw, x_val_raw, x_external_raw

    phase1_train_safe_prob = np.asarray(phase1_full_prob, dtype=np.float32).copy()
    phase1_train_safe_prob[historical_ids] = np.asarray(phase1_train_oof_prob, dtype=np.float32)
    phase1_eval_prob = np.asarray(phase1_full_prob, dtype=np.float32)
    phase2_eval_prob = np.asarray(phase2_full_prob, dtype=np.float32)

    score_names = _score_feature_names()
    phase1_train_safe_score = _make_score_features(phase1_train_safe_prob)
    phase1_eval_score = _make_score_features(phase1_eval_prob)
    phase2_eval_score = _make_score_features(phase2_eval_prob)

    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    external_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    if args.include_raw_features:
        train_blocks.append(np.asarray(x_train_raw, dtype=np.float32, copy=False))
        val_blocks.append(np.asarray(x_val_raw, dtype=np.float32, copy=False))
        external_blocks.append(np.asarray(x_external_raw, dtype=np.float32, copy=False))
        feature_names.extend(f"raw__{name}" for name in raw_feature_names)

    train_blocks.append(np.asarray(phase1_train_safe_score[historical_ids], dtype=np.float32, copy=False))
    val_blocks.append(np.asarray(phase1_eval_score[val_ids], dtype=np.float32, copy=False))
    external_blocks.append(np.asarray(phase2_eval_score[external_ids], dtype=np.float32, copy=False))
    feature_names.extend(f"local__{name}" for name in score_names)

    split_blocks = {
        "train": train_blocks,
        "val": val_blocks,
        "external": external_blocks,
    }
    half_lives = _resolve_half_lives(list(args.prop_half_life_days))
    _append_propagated_score_blocks(
        blocks={"train": split_blocks["train"]},
        feature_names=feature_names,
        split_ids={"train": historical_ids},
        score_matrix=phase1_train_safe_score,
        score_names=score_names,
        graph_cache=phase1_graph,
        prop_blocks=list(args.prop_blocks),
        prop_half_life_days=half_lives,
    )
    _append_propagated_score_blocks(
        blocks={"val": split_blocks["val"]},
        feature_names=[],
        split_ids={"val": val_ids},
        score_matrix=phase1_eval_score,
        score_names=score_names,
        graph_cache=phase1_graph,
        prop_blocks=list(args.prop_blocks),
        prop_half_life_days=half_lives,
    )
    _append_propagated_score_blocks(
        blocks={"external": split_blocks["external"]},
        feature_names=[],
        split_ids={"external": external_ids},
        score_matrix=phase2_eval_score,
        score_names=score_names,
        graph_cache=phase2_graph,
        prop_blocks=list(args.prop_blocks),
        prop_half_life_days=half_lives,
    )

    x_train = np.concatenate(split_blocks["train"], axis=1).astype(np.float32, copy=False)
    x_val = np.concatenate(split_blocks["val"], axis=1).astype(np.float32, copy=False)
    x_external = np.concatenate(split_blocks["external"], axis=1).astype(np.float32, copy=False)

    sample_weight_payload = _build_sample_weight(
        y_train,
        args,
        train_first_active=train_first_active,
        threshold_day=int(split.threshold_day),
    )
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=np.asarray(sample_weight_payload["sample_weight"], dtype=np.float32))
    dval = xgb.DMatrix(x_val, label=y_val)
    dexternal = xgb.DMatrix(x_external, label=y_external)

    params = _make_xgb_params(args)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(args.n_estimators),
        evals=[(dval, "validation")],
        custom_metric=_multiclass_binary_auc_ignore_background,
        maximize=True,
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=50,
    )

    stage2_best_iteration = int(booster.best_iteration)
    val_prob = booster.predict(dval, iteration_range=(0, stage2_best_iteration + 1)).reshape(-1, 4)
    external_prob = booster.predict(dexternal, iteration_range=(0, stage2_best_iteration + 1)).reshape(-1, 4)
    val_score = _binary_score_from_softprob(val_prob)
    external_score = _binary_score_from_softprob(external_prob)
    val_metrics = compute_binary_classification_metrics(y_val, val_score)
    external_metrics = compute_binary_classification_metrics(y_external, external_score)

    run_dir = ensure_dir(args.outdir / args.run_name)
    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, y_val, val_score)
    save_prediction_npz(run_dir / "phase2_external_predictions.npz", external_ids, y_external, external_score)
    booster.save_model(run_dir / "model.json")
    _write_feature_importance(booster, feature_names, run_dir / "feature_importance.csv")

    summary = {
        "model": "xgboost_gpu_multiclass_bg_scoreprop",
        "run_name": args.run_name,
        "seed": args.seed,
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "include_raw_features": bool(args.include_raw_features),
        "include_future_background": bool(args.include_future_background),
        "oof_folds": int(args.oof_folds),
        "prop_blocks": list(args.prop_blocks),
        "prop_half_life_days": [None if value is None else float(value) for value in half_lives],
        "feature_dim": int(len(feature_names)),
        "threshold_day": int(split.threshold_day),
        "historical_train_size": int(historical_ids.size),
        "historical_train_label_counts": {str(label): int(np.sum(y_train == label)) for label in (0, 1, 2, 3)},
        "class_weight": sample_weight_payload["class_weight"],
        "time_weight": sample_weight_payload["time_weight"],
        "stage1_best_iteration": stage1_best_iteration,
        "stage2_best_iteration": stage2_best_iteration,
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "params": params,
        "cache_dir": str(cache_dir),
        "prediction_paths": {
            "phase1_val": str(run_dir / "phase1_val_predictions.npz"),
            "phase2_external": str(run_dir / "phase2_external_predictions.npz"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[xgboost_gpu_multiclass_bg_scoreprop] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f} "
        f"stage1_best_iteration={stage1_best_iteration} "
        f"stage2_best_iteration={stage2_best_iteration}"
    )


if __name__ == "__main__":
    main()
