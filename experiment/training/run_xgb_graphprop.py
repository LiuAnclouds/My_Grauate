from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU XGBoost with graph-propagated tabular features.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Output run name under experiment/outputs/training/models/xgboost_gpu/.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_OUTPUT_ROOT,
        help="Directory containing built feature caches.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=MODEL_OUTPUT_ROOT / "xgboost_gpu",
        help="Output root for model artifacts.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=MODEL_OUTPUT_ROOT / "_graphprop_cache",
        help="Reusable cache root for sliced propagated features.",
    )
    parser.add_argument(
        "--base-model",
        choices=("m2_hybrid", "m3_neighbor"),
        default="m3_neighbor",
        help="Base feature block included directly in XGBoost.",
    )
    parser.add_argument(
        "--prop-model",
        choices=("m2_hybrid", "m3_neighbor"),
        default="m2_hybrid",
        help="Feature block used as source for graph propagation.",
    )
    parser.add_argument(
        "--prop-blocks",
        nargs="+",
        choices=("in1", "out1", "in2", "out2", "bi1", "bi2"),
        default=("in1", "out1", "in2", "out2"),
        help="Graph-propagation blocks to append.",
    )
    parser.add_argument(
        "--extra-groups",
        nargs="*",
        default=(),
        help=(
            "Optional extra offline feature groups appended to both base_model and prop_model. "
            "Useful for temporal_snapshot / temporal_recent / temporal_relation_recent."
        ),
    )
    parser.add_argument(
        "--base-extra-groups",
        nargs="*",
        default=None,
        help="Optional extra feature groups appended only to base_model. Defaults to --extra-groups.",
    )
    parser.add_argument(
        "--prop-extra-groups",
        nargs="*",
        default=None,
        help="Optional extra feature groups appended only to prop_model. Defaults to --extra-groups.",
    )
    parser.add_argument(
        "--prop-half-life-days",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional edge-time half-life(s) for decayed graph propagation. "
            "Pass one value for the original single-scale behavior or multiple values "
            "to append multi-scale propagated features."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=400)
    parser.add_argument("--max-train-nodes", type=int, default=None)
    parser.add_argument("--max-val-nodes", type=int, default=None)
    parser.add_argument("--max-external-nodes", type=int, default=None)
    return parser.parse_args()


def _slice_node_ids(node_ids: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or node_ids.size <= limit:
        return np.asarray(node_ids, dtype=np.int32)
    rng = np.random.default_rng(seed)
    choice = rng.choice(node_ids.size, size=limit, replace=False)
    return np.sort(node_ids[choice].astype(np.int32, copy=False))


def _cache_key(args: argparse.Namespace) -> str:
    payload = {
        "base_model": args.base_model,
        "prop_model": args.prop_model,
        "prop_blocks": list(args.prop_blocks),
        "extra_groups": list(args.extra_groups),
        "base_extra_groups": _resolve_model_extra_groups(args, "base"),
        "prop_extra_groups": _resolve_model_extra_groups(args, "prop"),
        "prop_half_life_days": _resolve_half_lives(args.prop_half_life_days),
        "feature_dir": str(args.feature_dir.resolve()),
        "max_train_nodes": args.max_train_nodes,
        "max_val_nodes": args.max_val_nodes,
        "max_external_nodes": args.max_external_nodes,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{args.base_model}__{args.prop_model}__{digest}"


def _resolve_half_lives(raw_values: list[float] | None) -> list[float | None]:
    if raw_values is None:
        return [None]
    resolved: list[float | None] = []
    seen: set[str] = set()
    for value in raw_values:
        half_life = float(value)
        if half_life <= 0.0:
            raise ValueError(f"prop half-life must be positive, got {half_life}")
        key = f"{half_life:.8f}"
        if key in seen:
            continue
        seen.add(key)
        resolved.append(half_life)
    return resolved or [None]


def _resolve_model_extra_groups(args: argparse.Namespace, prefix: str) -> list[str]:
    specific = getattr(args, f"{prefix}_extra_groups", None)
    if specific is None:
        return list(getattr(args, "extra_groups", ()))
    return list(specific)


def _half_life_tag(half_life_days: float | None) -> str:
    if half_life_days is None:
        return "raw"
    if float(half_life_days).is_integer():
        return f"decay{int(half_life_days)}"
    return f"decay{str(half_life_days).replace('.', 'p')}"


def _normalized_csr(
    ptr: np.ndarray,
    neighbors: np.ndarray,
    num_nodes: int,
    timestamps: np.ndarray | None = None,
    max_day: int | None = None,
    half_life_days: float | None = None,
) -> sp.csr_matrix:
    indptr = np.asarray(ptr, dtype=np.int64)
    indices = np.asarray(neighbors, dtype=np.int32)
    degree = np.diff(indptr).astype(np.int32, copy=False)
    if half_life_days is None:
        inv_degree = np.zeros(num_nodes, dtype=np.float32)
        mask = degree > 0
        inv_degree[mask] = 1.0 / degree[mask]
        data = np.repeat(inv_degree, degree).astype(np.float32, copy=False)
        return sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes), dtype=np.float32)

    if timestamps is None or max_day is None:
        raise ValueError("timestamps and max_day are required when prop half-life is enabled.")
    if half_life_days <= 0.0:
        raise ValueError(f"prop half-life must be positive, got {half_life_days}")

    edge_time = np.asarray(timestamps, dtype=np.float32)
    age = float(max_day) - edge_time
    data = np.power(np.float32(0.5), age / float(half_life_days)).astype(np.float32, copy=False)
    row_ids = np.repeat(np.arange(num_nodes, dtype=np.int32), degree)
    row_sums = np.bincount(row_ids, weights=data, minlength=num_nodes).astype(np.float32, copy=False)
    row_sums = np.maximum(row_sums, 1e-8)
    data = (data / row_sums[row_ids]).astype(np.float32, copy=False)
    return sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes), dtype=np.float32)


def _take_full_matrix(
    feature_dir: Path,
    phase: str,
    model_name: str,
    extra_groups: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    feature_store = FeatureStore(
        phase,
        resolve_feature_groups(model_name, extra_groups),
        outdir=feature_dir,
    )
    row_ids = np.arange(feature_store.core.shape[0], dtype=np.int32)
    matrix = feature_store.take_rows(row_ids).astype(np.float32, copy=False)
    return matrix, list(feature_store.feature_names)


def _append_slices(
    blocks: dict[str, list[np.ndarray]],
    name_blocks: list[str],
    full_matrix: np.ndarray,
    full_feature_names: list[str],
    split_ids: dict[str, np.ndarray],
    prefix: str,
) -> None:
    prefixed_names = [f"{prefix}__{name}" for name in full_feature_names]
    name_blocks.extend(prefixed_names)
    for split_name, node_ids in split_ids.items():
        blocks[split_name].append(full_matrix[node_ids].astype(np.float32, copy=False))


def _build_phase_feature_blocks(
    phase: str,
    feature_dir: Path,
    base_model: str,
    prop_model: str,
    prop_blocks: list[str],
    base_extra_groups: list[str],
    prop_extra_groups: list[str],
    prop_half_life_days: list[float | None],
    split_ids: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    graph_cache = load_graph_cache(phase, outdir=feature_dir)
    phase_blocks: dict[str, list[np.ndarray]] = {name: [] for name in split_ids}
    feature_names: list[str] = []
    half_lives = list(prop_half_life_days)
    use_half_life_prefix = len(half_lives) > 1

    with tqdm(
        total=3,
        desc=f"graphprop:{phase}:prepare",
        unit="step",
        dynamic_ncols=True,
    ) as prep_pbar:
        base_matrix, base_names = _take_full_matrix(
            feature_dir,
            phase,
            base_model,
            base_extra_groups,
        )
        prep_pbar.update(1)
        prop_matrix = base_matrix
        prop_names = base_names
        if prop_model != base_model or prop_extra_groups != base_extra_groups:
            prop_matrix, prop_names = _take_full_matrix(
                feature_dir,
                phase,
                prop_model,
                prop_extra_groups,
            )
        prep_pbar.update(1)
        prep_pbar.update(1)

    _append_slices(
        blocks=phase_blocks,
        name_blocks=feature_names,
        full_matrix=base_matrix,
        full_feature_names=base_names,
        split_ids=split_ids,
        prefix="base",
    )
    if prop_model == base_model:
        del base_matrix
    else:
        del base_matrix
        gc.collect()

    need_in = any(block in {"in1", "in2", "bi1", "bi2"} for block in prop_blocks)
    need_out = any(block in {"out1", "out2", "bi1", "bi2"} for block in prop_blocks)

    with tqdm(
        total=len(prop_blocks) * len(half_lives),
        desc=f"graphprop:{phase}:blocks",
        unit="block",
        dynamic_ncols=True,
    ) as block_pbar:
        for half_life in half_lives:
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
                in1 = (a_in @ prop_matrix).astype(np.float32, copy=False)
                if "in1" in prop_blocks:
                    _append_slices(
                        phase_blocks,
                        feature_names,
                        in1,
                        prop_names,
                        split_ids,
                        f"{prefix_root}in1",
                    )
                    block_pbar.update(1)
            if need_out:
                a_out = _normalized_csr(
                    graph_cache.out_ptr,
                    graph_cache.out_neighbors,
                    graph_cache.num_nodes,
                    timestamps=graph_cache.out_edge_timestamp,
                    max_day=graph_cache.max_day,
                    half_life_days=half_life,
                )
                out1 = (a_out @ prop_matrix).astype(np.float32, copy=False)
                if "out1" in prop_blocks:
                    _append_slices(
                        phase_blocks,
                        feature_names,
                        out1,
                        prop_names,
                        split_ids,
                        f"{prefix_root}out1",
                    )
                    block_pbar.update(1)
            if "bi1" in prop_blocks:
                if in1 is None or out1 is None:
                    raise RuntimeError("bi1 propagation requires inbound and outbound first-hop features.")
                bi1 = ((in1 + out1) * 0.5).astype(np.float32, copy=False)
                _append_slices(
                    phase_blocks,
                    feature_names,
                    bi1,
                    prop_names,
                    split_ids,
                    f"{prefix_root}bi1",
                )
                del bi1
                block_pbar.update(1)
            if "in2" in prop_blocks:
                if a_in is None or in1 is None:
                    raise RuntimeError("in2 propagation requires inbound adjacency and first-hop features.")
                in2 = (a_in @ in1).astype(np.float32, copy=False)
                _append_slices(
                    phase_blocks,
                    feature_names,
                    in2,
                    prop_names,
                    split_ids,
                    f"{prefix_root}in2",
                )
                del in2
                block_pbar.update(1)
            if "out2" in prop_blocks:
                if a_out is None or out1 is None:
                    raise RuntimeError("out2 propagation requires outbound adjacency and first-hop features.")
                out2 = (a_out @ out1).astype(np.float32, copy=False)
                _append_slices(
                    phase_blocks,
                    feature_names,
                    out2,
                    prop_names,
                    split_ids,
                    f"{prefix_root}out2",
                )
                del out2
                block_pbar.update(1)
            if "bi2" in prop_blocks:
                if a_in is None or a_out is None or in1 is None or out1 is None:
                    raise RuntimeError("bi2 propagation requires inbound/outbound adjacency and first-hop features.")
                bi2 = ((a_in @ in1) + (a_out @ out1)).astype(np.float32, copy=False) * 0.5
                _append_slices(
                    phase_blocks,
                    feature_names,
                    bi2,
                    prop_names,
                    split_ids,
                    f"{prefix_root}bi2",
                )
                del bi2
                block_pbar.update(1)
            del a_in, a_out, in1, out1
            gc.collect()

    del prop_matrix
    gc.collect()

    stacked_blocks = {
        split_name: np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        for split_name, parts in phase_blocks.items()
    }
    return stacked_blocks, feature_names


def _load_or_build_cached_features(
    args: argparse.Namespace,
    cache_dir: Path,
    phase1_ids: dict[str, np.ndarray],
    phase2_ids: dict[str, np.ndarray],
    half_lives: list[float | None],
    *,
    primary_phase: str = "phase1",
    external_phase: str = "phase2",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    feature_names_path = cache_dir / "feature_names.json"
    phase1_train_path = cache_dir / "phase1_train.npy"
    phase1_val_path = cache_dir / "phase1_val.npy"
    phase2_external_path = cache_dir / "phase2_external.npy"
    if (
        feature_names_path.exists()
        and phase1_train_path.exists()
        and phase1_val_path.exists()
        and phase2_external_path.exists()
    ):
        phase1 = {
            "train": np.load(phase1_train_path, mmap_mode="r"),
            "val": np.load(phase1_val_path, mmap_mode="r"),
        }
        phase2 = {"external": np.load(phase2_external_path, mmap_mode="r")}
        feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
        return phase1, phase2, list(feature_names)

    ensure_dir(cache_dir)
    phase1, feature_names = _build_phase_feature_blocks(
        phase=str(primary_phase),
        feature_dir=args.feature_dir,
        base_model=args.base_model,
        prop_model=args.prop_model,
        prop_blocks=list(args.prop_blocks),
        base_extra_groups=_resolve_model_extra_groups(args, "base"),
        prop_extra_groups=_resolve_model_extra_groups(args, "prop"),
        prop_half_life_days=half_lives,
        split_ids=phase1_ids,
    )
    phase2, phase2_feature_names = _build_phase_feature_blocks(
        phase=str(external_phase),
        feature_dir=args.feature_dir,
        base_model=args.base_model,
        prop_model=args.prop_model,
        prop_blocks=list(args.prop_blocks),
        base_extra_groups=_resolve_model_extra_groups(args, "base"),
        prop_extra_groups=_resolve_model_extra_groups(args, "prop"),
        prop_half_life_days=half_lives,
        split_ids=phase2_ids,
    )
    if phase2_feature_names != feature_names:
        raise AssertionError("Phase1/phase2 feature names are not aligned.")

    np.save(phase1_train_path, np.asarray(phase1["train"], dtype=np.float32))
    np.save(phase1_val_path, np.asarray(phase1["val"], dtype=np.float32))
    np.save(phase2_external_path, np.asarray(phase2["external"], dtype=np.float32))
    feature_names_path.write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    phase1 = {
        "train": np.load(phase1_train_path, mmap_mode="r"),
        "val": np.load(phase1_val_path, mmap_mode="r"),
    }
    phase2 = {"external": np.load(phase2_external_path, mmap_mode="r")}
    return phase1, phase2, feature_names


def _write_feature_importance(
    booster: Any,
    feature_names: list[str],
    path: Path,
) -> None:
    scores = booster.get_score(importance_type="gain")
    rows = []
    for idx, feature_name in enumerate(feature_names):
        gain = float(scores.get(f"f{idx}", 0.0))
        rows.append({"feature_name": feature_name, "gain": gain})
    rows.sort(key=lambda row: row["gain"], reverse=True)
    ensure_dir(path.parent)
    lines = ["feature_name,gain"]
    lines.extend(f"{row['feature_name']},{row['gain']:.10f}" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    import xgboost as xgb

    set_global_seed(args.seed)
    half_lives = _resolve_half_lives(args.prop_half_life_days)
    split = load_experiment_split()
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)

    train_ids = _slice_node_ids(split.train_ids, args.max_train_nodes, seed=args.seed + 11)
    val_ids = _slice_node_ids(split.val_ids, args.max_val_nodes, seed=args.seed + 17)
    external_ids = _slice_node_ids(split.external_ids, args.max_external_nodes, seed=args.seed + 29)
    cache_dir = ensure_dir(args.cache_root / _cache_key(args))
    run_dir = ensure_dir(args.outdir / args.run_name)

    phase1_mats, phase2_mats, feature_names = _load_or_build_cached_features(
        args=args,
        cache_dir=cache_dir,
        phase1_ids={"train": train_ids, "val": val_ids},
        phase2_ids={"external": external_ids},
        half_lives=half_lives,
    )

    y_train = phase1_y[train_ids]
    y_val = phase1_y[val_ids]
    y_external = phase2_y[external_ids]

    pos_count = float(np.sum(y_train == 1))
    neg_count = float(np.sum(y_train == 0))
    scale_pos_weight = neg_count / max(pos_count, 1.0)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": args.device,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": args.early_stopping_rounds,
        "random_state": args.seed,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        np.asarray(phase1_mats["train"], dtype=np.float32),
        y_train,
        eval_set=[(np.asarray(phase1_mats["val"], dtype=np.float32), y_val)],
        verbose=50,
    )

    val_prob = model.predict_proba(np.asarray(phase1_mats["val"], dtype=np.float32))[:, 1].astype(
        np.float32,
        copy=False,
    )
    external_prob = model.predict_proba(np.asarray(phase2_mats["external"], dtype=np.float32))[:, 1].astype(
        np.float32,
        copy=False,
    )
    val_metrics = compute_binary_classification_metrics(y_val, val_prob)
    external_metrics = compute_binary_classification_metrics(y_external, external_prob)

    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, y_val, val_prob)
    save_prediction_npz(
        run_dir / "phase2_external_predictions.npz",
        external_ids,
        y_external,
        external_prob,
    )
    model.save_model(run_dir / "model.json")
    _write_feature_importance(model.get_booster(), feature_names, run_dir / "feature_importance.csv")

    summary = {
        "model": "xgboost_gpu_graphprop",
        "run_name": args.run_name,
        "seed": args.seed,
        "base_model": args.base_model,
        "prop_model": args.prop_model,
        "prop_blocks": list(args.prop_blocks),
        "extra_groups": list(args.extra_groups),
        "base_extra_groups": _resolve_model_extra_groups(args, "base"),
        "prop_extra_groups": _resolve_model_extra_groups(args, "prop"),
        "prop_half_life_days": half_lives[0] if len(half_lives) == 1 else None,
        "prop_half_life_days_list": half_lives,
        "feature_dim": int(len(feature_names)),
        "train_size": int(train_ids.size),
        "val_size": int(val_ids.size),
        "external_size": int(external_ids.size),
        "best_iteration": int(getattr(model, "best_iteration", 0)),
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
        f"[xgboost_gpu_graphprop] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f} "
        f"best_iteration={summary['best_iteration']}"
    )


if __name__ == "__main__":
    main()
