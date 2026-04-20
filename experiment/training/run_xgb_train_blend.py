from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (  # noqa: E402
    BLEND_OUTPUT_ROOT,
    FEATURE_OUTPUT_ROOT,
    align_prediction_bundle,
    compute_binary_classification_metrics,
    ensure_dir,
    load_prediction_npz,
    load_experiment_split,
    load_phase_arrays,
    resolve_prediction_path,
    save_prediction_npz,
    write_json,
)
from experiment.training.features import FeatureStore, load_graph_cache, resolve_feature_groups  # noqa: E402
from experiment.training.run_xgb_multiclass_bg_scoreprop import (  # noqa: E402
    _append_propagated_score_blocks,
    _make_score_features,
    _score_feature_names,
)


@dataclass(frozen=True)
class BasePredictionSet:
    run_dir: Path
    family: str
    train_mode: str
    train_score: np.ndarray
    val_score: np.ndarray
    external_score: np.ndarray
    verify_val_max_abs_diff: float
    verify_external_max_abs_diff: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a leakage-safe phase1-train-only meta-model on saved base runs, "
            "then evaluate on phase1_val and phase2_external."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Base run directories with saved model artifacts.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BLEND_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--meta-model",
        choices=("xgboost", "logistic"),
        default="xgboost",
    )
    parser.add_argument(
        "--include-rank-features",
        action="store_true",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--verify-tol", type=float, default=1e-4)
    return parser.parse_args()


def _load_summary(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

def _rank_norm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    return ranks


def _binary_score_from_softprob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1, 4)
    foreground = np.clip(prob[:, 0] + prob[:, 1], 1e-6, None)
    return (prob[:, 1] / foreground).astype(np.float32, copy=False)


def _historical_ids_from_summary(summary: dict[str, object], feature_dir: Path) -> np.ndarray:
    split = load_experiment_split()
    historical_selection_mode = str(summary.get("historical_selection_mode", ""))
    expected_size = int(summary["historical_train_size"])
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase1_graph = load_graph_cache("phase1", outdir=feature_dir)
    first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)
    threshold_day = int(summary["threshold_day"])
    min_train_day = int(summary.get("min_train_first_active_day", 0))
    include_future_background = bool(summary.get("include_future_background", False))
    if historical_selection_mode in {
        "split_train_ids",
        "split_train_ids_recent_start",
    }:
        split_train_ids = np.asarray(split.train_ids, dtype=np.int32)
        if min_train_day > 0:
            split_train_ids = split_train_ids[first_active[split_train_ids] >= min_train_day].astype(
                np.int32,
                copy=False,
            )
        historical_ids = split_train_ids
        if historical_ids.size != expected_size:
            raise AssertionError(
                f"historical train size mismatch: rebuilt={historical_ids.size} summary={expected_size}"
            )
        overlap = np.intersect1d(historical_ids, np.asarray(split.val_ids, dtype=np.int32))
        if overlap.size:
            raise AssertionError(f"historical train ids overlap validation ids: overlap={overlap.size}")
        return historical_ids
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
    if historical_ids.size != expected_size:
        raise AssertionError(
            f"historical train size mismatch: rebuilt={historical_ids.size} summary={expected_size}"
        )
    overlap = np.intersect1d(historical_ids, np.asarray(split.val_ids, dtype=np.int32))
    if overlap.size:
        raise AssertionError(f"historical train ids overlap validation ids: overlap={overlap.size}")
    if not np.all(np.isin(split.train_ids, historical_ids)):
        raise AssertionError("phase1 split train ids are not contained in rebuilt historical ids.")
    return historical_ids


def _row_positions(historical_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(historical_ids, target_ids)
    if np.any(positions >= historical_ids.size) or not np.array_equal(historical_ids[positions], target_ids):
        raise AssertionError("Failed to align target ids against historical cache rows.")
    return positions.astype(np.int64, copy=False)


def _load_xgb_booster(run_dir: Path):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(run_dir / "model.json")
    booster.set_param({"device": "cpu"})
    return booster


def _predict_xgb_multiclass_score(
    booster,
    matrix: np.ndarray,
    best_iteration: int,
) -> np.ndarray:
    import xgboost as xgb

    dmatrix = xgb.DMatrix(np.asarray(matrix, dtype=np.float32))
    prob = booster.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1)).reshape(-1, 4)
    return _binary_score_from_softprob(prob)


def _load_saved_scores(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    val_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase1_val"))
    external_bundle = load_prediction_npz(resolve_prediction_path(run_dir, "phase2_external"))
    return val_bundle, external_bundle


def _verify_saved_predictions(
    run_dir: Path,
    expected_val_ids: np.ndarray,
    expected_external_ids: np.ndarray,
    val_score: np.ndarray,
    external_score: np.ndarray,
    tol: float,
) -> tuple[float, float]:
    saved_val, saved_external = _load_saved_scores(run_dir)
    if not np.array_equal(saved_val["node_ids"], expected_val_ids):
        saved_val = align_prediction_bundle(saved_val, expected_val_ids)
    if not np.array_equal(saved_external["node_ids"], expected_external_ids):
        saved_external = align_prediction_bundle(saved_external, expected_external_ids)
    if not np.array_equal(saved_val["node_ids"], expected_val_ids):
        raise AssertionError(f"{run_dir}: phase1_val node ids do not match recommended split.")
    if not np.array_equal(saved_external["node_ids"], expected_external_ids):
        raise AssertionError(f"{run_dir}: phase2_external node ids do not match recommended split.")
    val_diff = float(np.max(np.abs(np.asarray(val_score, dtype=np.float32) - saved_val["probability"])))
    external_diff = float(
        np.max(np.abs(np.asarray(external_score, dtype=np.float32) - saved_external["probability"]))
    )
    if val_diff > tol or external_diff > tol:
        raise AssertionError(
            f"{run_dir}: saved prediction verification failed "
            f"(val_diff={val_diff:.6g}, external_diff={external_diff:.6g}, tol={tol:.6g})"
        )
    return val_diff, external_diff


def _score_multiclass_bg_run(
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    verify_tol: float,
) -> BasePredictionSet:
    feature_model = str(summary["feature_model"])
    extra_groups = list(summary.get("extra_groups", []))
    phase1_store = FeatureStore("phase1", resolve_feature_groups(feature_model, extra_groups), outdir=feature_dir)
    phase2_store = FeatureStore("phase2", resolve_feature_groups(feature_model, extra_groups), outdir=feature_dir)
    x_train = phase1_store.take_rows(train_ids)
    x_val = phase1_store.take_rows(val_ids)
    x_external = phase2_store.take_rows(external_ids)
    booster = _load_xgb_booster(run_dir)
    best_iteration = int(summary["best_iteration"])
    train_score = _predict_xgb_multiclass_score(booster, x_train, best_iteration)
    val_score = _predict_xgb_multiclass_score(booster, x_val, best_iteration)
    external_score = _predict_xgb_multiclass_score(booster, x_external, best_iteration)
    val_diff, external_diff = _verify_saved_predictions(
        run_dir,
        expected_val_ids=val_ids,
        expected_external_ids=external_ids,
        val_score=val_score,
        external_score=external_score,
        tol=verify_tol,
    )
    return BasePredictionSet(
        run_dir=run_dir,
        family=str(summary["model"]),
        train_mode="in_sample_train_predict",
        train_score=train_score,
        val_score=val_score,
        external_score=external_score,
        verify_val_max_abs_diff=val_diff,
        verify_external_max_abs_diff=external_diff,
    )


def _score_multiclass_bg_graphprop_run(
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    verify_tol: float,
) -> BasePredictionSet:
    cache_dir = Path(str(summary["cache_dir"]))
    historical_ids = _historical_ids_from_summary(summary, feature_dir=feature_dir)
    train_pos = _row_positions(historical_ids, train_ids)

    x_train = np.load(cache_dir / "phase1_train.npy", mmap_mode="r")[train_pos]
    x_val = np.load(cache_dir / "phase1_val.npy", mmap_mode="r")
    x_external = np.load(cache_dir / "phase2_external.npy", mmap_mode="r")
    train_blocks = [np.asarray(x_train, dtype=np.float32)]
    val_blocks = [np.asarray(x_val, dtype=np.float32)]
    external_blocks = [np.asarray(x_external, dtype=np.float32)]

    train_labelprop = cache_dir / "phase1_train_labelprop.npy"
    val_labelprop = cache_dir / "phase1_val_labelprop.npy"
    external_labelprop = cache_dir / "phase2_external_labelprop.npy"
    if train_labelprop.exists() and val_labelprop.exists() and external_labelprop.exists():
        train_blocks.append(np.asarray(np.load(train_labelprop, mmap_mode="r")[train_pos], dtype=np.float32))
        val_blocks.append(np.asarray(np.load(val_labelprop, mmap_mode="r"), dtype=np.float32))
        external_blocks.append(np.asarray(np.load(external_labelprop, mmap_mode="r"), dtype=np.float32))

    booster = _load_xgb_booster(run_dir)
    best_iteration = int(summary["best_iteration"])
    train_score = _predict_xgb_multiclass_score(booster, np.concatenate(train_blocks, axis=1), best_iteration)
    val_score = _predict_xgb_multiclass_score(booster, np.concatenate(val_blocks, axis=1), best_iteration)
    external_score = _predict_xgb_multiclass_score(
        booster,
        np.concatenate(external_blocks, axis=1),
        best_iteration,
    )
    val_diff, external_diff = _verify_saved_predictions(
        run_dir,
        expected_val_ids=val_ids,
        expected_external_ids=external_ids,
        val_score=val_score,
        external_score=external_score,
        tol=verify_tol,
    )
    return BasePredictionSet(
        run_dir=run_dir,
        family=str(summary["model"]),
        train_mode="in_sample_train_predict",
        train_score=train_score,
        val_score=val_score,
        external_score=external_score,
        verify_val_max_abs_diff=val_diff,
        verify_external_max_abs_diff=external_diff,
    )


def _score_multiclass_bg_scoreprop_run(
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    verify_tol: float,
) -> BasePredictionSet:
    phase1_graph = load_graph_cache("phase1", outdir=feature_dir)
    phase2_graph = load_graph_cache("phase2", outdir=feature_dir)
    phase1_store = FeatureStore(
        "phase1",
        resolve_feature_groups(str(summary["feature_model"]), list(summary.get("extra_groups", []))),
        outdir=feature_dir,
    )
    phase2_store = FeatureStore(
        "phase2",
        resolve_feature_groups(str(summary["feature_model"]), list(summary.get("extra_groups", []))),
        outdir=feature_dir,
    )
    historical_ids = _historical_ids_from_summary(summary, feature_dir=feature_dir)
    cache_dir = Path(str(summary["cache_dir"]))
    phase1_train_oof_prob = np.asarray(np.load(cache_dir / "phase1_train_oof_softprob.npy", mmap_mode="r"))
    phase1_full_prob = np.asarray(np.load(cache_dir / "phase1_full_softprob.npy", mmap_mode="r"))
    phase2_full_prob = np.asarray(np.load(cache_dir / "phase2_full_softprob.npy", mmap_mode="r"))

    phase1_train_safe_prob = np.asarray(phase1_full_prob, dtype=np.float32).copy()
    phase1_train_safe_prob[historical_ids] = np.asarray(phase1_train_oof_prob, dtype=np.float32)
    phase1_train_safe_score = _make_score_features(phase1_train_safe_prob)
    phase1_eval_score = _make_score_features(phase1_full_prob)
    phase2_eval_score = _make_score_features(phase2_full_prob)
    score_names = _score_feature_names()

    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    external_blocks: list[np.ndarray] = []

    if bool(summary.get("include_raw_features", False)):
        train_blocks.append(phase1_store.take_rows(train_ids).astype(np.float32, copy=False))
        val_blocks.append(phase1_store.take_rows(val_ids).astype(np.float32, copy=False))
        external_blocks.append(phase2_store.take_rows(external_ids).astype(np.float32, copy=False))

    train_blocks.append(np.asarray(phase1_train_safe_score[train_ids], dtype=np.float32, copy=False))
    val_blocks.append(np.asarray(phase1_eval_score[val_ids], dtype=np.float32, copy=False))
    external_blocks.append(np.asarray(phase2_eval_score[external_ids], dtype=np.float32, copy=False))

    prop_blocks = list(summary.get("prop_blocks", []))
    prop_half_life_days = [
        None if value is None else float(value)
        for value in list(summary.get("prop_half_life_days", []))
    ]
    if prop_blocks:
        split_blocks = {
            "train": train_blocks,
            "val": val_blocks,
            "external": external_blocks,
        }
        _append_propagated_score_blocks(
            blocks={"train": split_blocks["train"]},
            feature_names=[],
            split_ids={"train": train_ids},
            score_matrix=phase1_train_safe_score,
            score_names=score_names,
            graph_cache=phase1_graph,
            prop_blocks=prop_blocks,
            prop_half_life_days=prop_half_life_days,
        )
        _append_propagated_score_blocks(
            blocks={"val": split_blocks["val"]},
            feature_names=[],
            split_ids={"val": val_ids},
            score_matrix=phase1_eval_score,
            score_names=score_names,
            graph_cache=phase1_graph,
            prop_blocks=prop_blocks,
            prop_half_life_days=prop_half_life_days,
        )
        _append_propagated_score_blocks(
            blocks={"external": split_blocks["external"]},
            feature_names=[],
            split_ids={"external": external_ids},
            score_matrix=phase2_eval_score,
            score_names=score_names,
            graph_cache=phase2_graph,
            prop_blocks=prop_blocks,
            prop_half_life_days=prop_half_life_days,
        )

    booster = _load_xgb_booster(run_dir)
    best_iteration = int(summary["stage2_best_iteration"])
    train_score = _predict_xgb_multiclass_score(booster, np.concatenate(train_blocks, axis=1), best_iteration)
    val_score = _predict_xgb_multiclass_score(booster, np.concatenate(val_blocks, axis=1), best_iteration)
    external_score = _predict_xgb_multiclass_score(
        booster,
        np.concatenate(external_blocks, axis=1),
        best_iteration,
    )
    val_diff, external_diff = _verify_saved_predictions(
        run_dir,
        expected_val_ids=val_ids,
        expected_external_ids=external_ids,
        val_score=val_score,
        external_score=external_score,
        tol=verify_tol,
    )
    return BasePredictionSet(
        run_dir=run_dir,
        family=str(summary["model"]),
        train_mode="oof_train_safe",
        train_score=train_score,
        val_score=val_score,
        external_score=external_score,
        verify_val_max_abs_diff=val_diff,
        verify_external_max_abs_diff=external_diff,
    )


def _score_catboost_run(
    run_dir: Path,
    summary: dict[str, object],
    feature_dir: Path,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    verify_tol: float,
) -> BasePredictionSet:
    from catboost import CatBoostClassifier

    feature_set = str(summary["feature_set"])
    feature_groups = resolve_feature_groups(feature_set, [])
    phase1_store = FeatureStore("phase1", feature_groups, outdir=feature_dir)
    phase2_store = FeatureStore("phase2", feature_groups, outdir=feature_dir)
    model = CatBoostClassifier()
    model.load_model(str(run_dir / "model.cbm"))
    train_score = model.predict_proba(phase1_store.take_rows(train_ids))[:, 1].astype(np.float32, copy=False)
    val_score = model.predict_proba(phase1_store.take_rows(val_ids))[:, 1].astype(np.float32, copy=False)
    external_score = model.predict_proba(phase2_store.take_rows(external_ids))[:, 1].astype(np.float32, copy=False)
    val_diff, external_diff = _verify_saved_predictions(
        run_dir,
        expected_val_ids=val_ids,
        expected_external_ids=external_ids,
        val_score=val_score,
        external_score=external_score,
        tol=verify_tol,
    )
    return BasePredictionSet(
        run_dir=run_dir,
        family=str(summary["model"]),
        train_mode="in_sample_train_predict",
        train_score=train_score,
        val_score=val_score,
        external_score=external_score,
        verify_val_max_abs_diff=val_diff,
        verify_external_max_abs_diff=external_diff,
    )


def _score_base_run(
    run_dir: Path,
    feature_dir: Path,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
    verify_tol: float,
) -> BasePredictionSet:
    summary = _load_summary(run_dir)
    family = str(summary["model"])
    if family == "xgboost_gpu_multiclass_bg":
        return _score_multiclass_bg_run(
            run_dir,
            summary,
            feature_dir,
            train_ids,
            val_ids,
            external_ids,
            verify_tol,
        )
    if family == "xgboost_gpu_multiclass_bg_graphprop":
        return _score_multiclass_bg_graphprop_run(
            run_dir,
            summary,
            feature_dir,
            train_ids,
            val_ids,
            external_ids,
            verify_tol,
        )
    if family == "xgboost_gpu_multiclass_bg_scoreprop":
        return _score_multiclass_bg_scoreprop_run(
            run_dir,
            summary,
            feature_dir,
            train_ids,
            val_ids,
            external_ids,
            verify_tol,
        )
    if family == "catboost_gpu":
        return _score_catboost_run(
            run_dir,
            summary,
            feature_dir,
            train_ids,
            val_ids,
            external_ids,
            verify_tol,
        )
    raise NotImplementedError(f"Unsupported base family for leakage-safe stacker: {family}")


def _build_meta_features(
    prediction_sets: list[BasePredictionSet],
    include_rank_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    external_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    for idx, pred in enumerate(prediction_sets):
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
    )


def _fit_meta_model(args: argparse.Namespace, x_train: np.ndarray, y_train: np.ndarray):
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos_weight = neg_count / max(pos_count, 1.0)
    if args.meta_model == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            max_iter=5000,
            C=float(args.logistic_c),
            class_weight="balanced",
            random_state=0,
        )
        model.fit(x_train, y_train)
        return model, {"scale_pos_weight": scale_pos_weight}

    import xgboost as xgb

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
    model.fit(x_train, y_train, verbose=False)
    return model, {"scale_pos_weight": scale_pos_weight}


def _predict_meta_model(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    train_ids = np.asarray(split.train_ids, dtype=np.int32)
    val_ids = np.asarray(split.val_ids, dtype=np.int32)
    external_ids = np.asarray(split.external_ids, dtype=np.int32)
    y_train = phase1_y[train_ids].astype(np.int8, copy=False)
    y_val = phase1_y[val_ids].astype(np.int8, copy=False)
    y_external = phase2_y[external_ids].astype(np.int8, copy=False)

    prediction_sets = [
        _score_base_run(
            run_dir=run_dir,
            feature_dir=args.feature_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            external_ids=external_ids,
            verify_tol=float(args.verify_tol),
        )
        for run_dir in args.run_dirs
    ]

    x_train, x_val, x_external, feature_names = _build_meta_features(
        prediction_sets,
        include_rank_features=bool(args.include_rank_features),
    )
    model, meta_aux = _fit_meta_model(args=args, x_train=x_train, y_train=y_train)
    train_prob = _predict_meta_model(model, x_train)
    val_prob = _predict_meta_model(model, x_val)
    external_prob = _predict_meta_model(model, x_external)

    train_metrics = compute_binary_classification_metrics(y_train, train_prob)
    val_metrics = compute_binary_classification_metrics(y_val, val_prob)
    external_metrics = compute_binary_classification_metrics(y_external, external_prob)

    run_dir = ensure_dir(args.outdir / args.run_name)
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
        "fit_scope": "phase1_train_only",
        "warning": (
            "Meta-model is fitted only on phase1 split train labels. "
            "phase1_val and phase2_external remain holdout for the meta stage."
        ),
        "base_runs": [
            {
                "run_dir": str(pred.run_dir),
                "family": pred.family,
                "train_mode": pred.train_mode,
                "verify_val_max_abs_diff": pred.verify_val_max_abs_diff,
                "verify_external_max_abs_diff": pred.verify_external_max_abs_diff,
            }
            for pred in prediction_sets
        ],
        "include_rank_features": bool(args.include_rank_features),
        "feature_dim": int(x_train.shape[1]),
        "feature_names": feature_names,
        "phase1_train_size": int(train_ids.size),
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
            "verify_tol": float(args.verify_tol),
            **meta_aux,
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[xgb_train_blend] run={args.run_name} "
        f"train_auc={train_metrics['auc']:.6f} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f}"
    )


if __name__ == "__main__":
    main()
