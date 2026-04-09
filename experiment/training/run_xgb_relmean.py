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

from experiment.eda.data_loader import load_phase
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
from experiment.training.features import FeatureStore, FEATURE_OUTPUT_ROOT, resolve_feature_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU XGBoost with relation-specific neighbor mean features.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "xgboost_gpu")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=MODEL_OUTPUT_ROOT / "_relmean_cache",
    )
    parser.add_argument(
        "--base-model",
        choices=("m2_hybrid", "m3_neighbor"),
        default="m3_neighbor",
    )
    parser.add_argument(
        "--extra-groups",
        nargs="*",
        default=(),
        help="Optional extra offline feature groups appended to the base_model.",
    )
    parser.add_argument(
        "--edge-types",
        type=int,
        nargs="+",
        default=list(range(1, 12)),
        help="Relation types used for relation-specific neighbor means.",
    )
    parser.add_argument(
        "--include-missing-ratio",
        action="store_true",
        help="Also append relation-specific missing-ratio means for raw x features.",
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
        "extra_groups": list(args.extra_groups),
        "edge_types": list(args.edge_types),
        "include_missing_ratio": bool(args.include_missing_ratio),
        "feature_dir": str(args.feature_dir.resolve()),
        "max_train_nodes": args.max_train_nodes,
        "max_val_nodes": args.max_val_nodes,
        "max_external_nodes": args.max_external_nodes,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return f"{args.base_model}__{hashlib.sha1(raw).hexdigest()[:12]}"


def _take_base_matrix(
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
    split_blocks: dict[str, list[np.ndarray]],
    full_matrix: np.ndarray,
    split_ids: dict[str, np.ndarray],
) -> None:
    for split_name, node_ids in split_ids.items():
        split_blocks[split_name].append(full_matrix[node_ids].astype(np.float32, copy=False))


def _build_relation_csr(
    centers: np.ndarray,
    neighbors: np.ndarray,
    num_nodes: int,
) -> sp.csr_matrix:
    degree = np.bincount(centers, minlength=num_nodes).astype(np.int32, copy=False)
    inv_degree = np.zeros(num_nodes, dtype=np.float32)
    mask = degree > 0
    inv_degree[mask] = 1.0 / degree[mask]
    data = inv_degree[centers].astype(np.float32, copy=False)
    return sp.csr_matrix((data, (centers, neighbors)), shape=(num_nodes, num_nodes), dtype=np.float32)


def _build_phase_feature_blocks(
    phase: str,
    feature_dir: Path,
    base_model: str,
    extra_groups: list[str],
    edge_types: list[int],
    include_missing_ratio: bool,
    split_ids: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    data = load_phase(phase, repo_root=REPO_ROOT)
    base_matrix, base_names = _take_base_matrix(feature_dir, phase, base_model, extra_groups)
    split_blocks: dict[str, list[np.ndarray]] = {name: [] for name in split_ids}
    feature_names = [f"base__{name}" for name in base_names]
    _append_slices(split_blocks, base_matrix, split_ids)

    raw_x = np.asarray(data.x, dtype=np.float32)
    missing_mask = (raw_x == -1.0).astype(np.float32, copy=False)
    source_specs: list[tuple[str, np.ndarray, list[str]]] = [
        ("raw", raw_x, [f"x{i}" for i in range(raw_x.shape[1])]),
    ]
    if include_missing_ratio:
        source_specs.append(
            ("missing", missing_mask, [f"x{i}_is_neg1" for i in range(raw_x.shape[1])])
        )

    src = np.asarray(data.edge_index[:, 0], dtype=np.int32)
    dst = np.asarray(data.edge_index[:, 1], dtype=np.int32)
    edge_type = np.asarray(data.edge_type, dtype=np.int16)

    total_blocks = len(edge_types) * 2 * len(source_specs)
    with tqdm(
        total=total_blocks,
        desc=f"relmean:{phase}",
        unit="block",
        dynamic_ncols=True,
    ) as pbar:
        for rel_type in edge_types:
            mask = edge_type == rel_type
            src_t = src[mask]
            dst_t = dst[mask]
            if src_t.size == 0:
                for _ in range(2 * len(source_specs)):
                    pbar.update(1)
                continue
            a_out = _build_relation_csr(src_t, dst_t, data.num_nodes)
            a_in = _build_relation_csr(dst_t, src_t, data.num_nodes)
            for source_name, source_matrix, source_names in source_specs:
                out_mean = (a_out @ source_matrix).astype(np.float32, copy=False)
                feature_names.extend(
                    [f"out_t{rel_type}_{source_name}__{name}" for name in source_names]
                )
                _append_slices(split_blocks, out_mean, split_ids)
                del out_mean
                pbar.update(1)

                in_mean = (a_in @ source_matrix).astype(np.float32, copy=False)
                feature_names.extend(
                    [f"in_t{rel_type}_{source_name}__{name}" for name in source_names]
                )
                _append_slices(split_blocks, in_mean, split_ids)
                del in_mean
                pbar.update(1)
            del a_out, a_in
            gc.collect()

    del base_matrix, raw_x, missing_mask
    gc.collect()

    matrices = {
        split_name: np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        for split_name, parts in split_blocks.items()
    }
    return matrices, feature_names


def _load_or_build_cached_features(
    args: argparse.Namespace,
    cache_dir: Path,
    phase1_ids: dict[str, np.ndarray],
    phase2_ids: dict[str, np.ndarray],
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
        return (
            {
                "train": np.load(phase1_train_path, mmap_mode="r"),
                "val": np.load(phase1_val_path, mmap_mode="r"),
            },
            {
                "external": np.load(phase2_external_path, mmap_mode="r"),
            },
            list(json.loads(feature_names_path.read_text(encoding="utf-8"))),
        )

    ensure_dir(cache_dir)
    phase1, feature_names = _build_phase_feature_blocks(
        phase="phase1",
        feature_dir=args.feature_dir,
        base_model=args.base_model,
        extra_groups=list(args.extra_groups),
        edge_types=list(args.edge_types),
        include_missing_ratio=bool(args.include_missing_ratio),
        split_ids=phase1_ids,
    )
    phase2, phase2_feature_names = _build_phase_feature_blocks(
        phase="phase2",
        feature_dir=args.feature_dir,
        base_model=args.base_model,
        extra_groups=list(args.extra_groups),
        edge_types=list(args.edge_types),
        include_missing_ratio=bool(args.include_missing_ratio),
        split_ids=phase2_ids,
    )
    if feature_names != phase2_feature_names:
        raise AssertionError("Phase1/phase2 relation-mean features are not aligned.")

    np.save(phase1_train_path, np.asarray(phase1["train"], dtype=np.float32))
    np.save(phase1_val_path, np.asarray(phase1["val"], dtype=np.float32))
    np.save(phase2_external_path, np.asarray(phase2["external"], dtype=np.float32))
    feature_names_path.write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (
        {
            "train": np.load(phase1_train_path, mmap_mode="r"),
            "val": np.load(phase1_val_path, mmap_mode="r"),
        },
        {
            "external": np.load(phase2_external_path, mmap_mode="r"),
        },
        feature_names,
    )


def _write_feature_importance(booster: Any, feature_names: list[str], path: Path) -> None:
    scores = booster.get_score(importance_type="gain")
    rows = []
    for idx, feature_name in enumerate(feature_names):
        rows.append({"feature_name": feature_name, "gain": float(scores.get(f"f{idx}", 0.0))})
    rows.sort(key=lambda row: row["gain"], reverse=True)
    ensure_dir(path.parent)
    lines = ["feature_name,gain"]
    lines.extend(f"{row['feature_name']},{row['gain']:.10f}" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    import xgboost as xgb

    set_global_seed(args.seed)
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
        "model": "xgboost_gpu_relmean",
        "run_name": args.run_name,
        "seed": args.seed,
        "base_model": args.base_model,
        "extra_groups": list(args.extra_groups),
        "edge_types": list(args.edge_types),
        "include_missing_ratio": bool(args.include_missing_ratio),
        "feature_dim": int(len(feature_names)),
        "train_size": int(train_ids.size),
        "val_size": int(val_ids.size),
        "external_size": int(external_ids.size),
        "best_iteration": int(getattr(model, "best_iteration", 0)),
        "phase1_val_metrics": val_metrics,
        "phase2_external_metrics": external_metrics,
        "params": params,
        "cache_dir": str(cache_dir),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[xgboost_gpu_relmean] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f} "
        f"best_iteration={summary['best_iteration']}"
    )


if __name__ == "__main__":
    main()
