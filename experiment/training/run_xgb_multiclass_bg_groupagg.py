from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

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
from experiment.training.run_xgb_graphprop import _half_life_tag, _resolve_half_lives
from experiment.training.run_xgb_multiclass_bg import (
    _binary_score_from_softprob,
    _build_sample_weight,
    _multiclass_binary_auc,
)
from experiment.training.xgb.domain_adaptation import add_domain_weight_args
from experiment.training.xgb.multiclass_bg_runtime import (
    build_historical_multiclass_bg_split,
    train_multiclass_bg_xgb,
)


GROUP_NAMES = [
    "known_normal",
    "known_fraud",
    "known_bg2",
    "known_bg3",
    "unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU XGBoost multiclass training with leakage-safe known-label group "
            "neighbor aggregation inspired by GAGA-style grouping."
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
        default=MODEL_OUTPUT_ROOT / "_multiclass_bg_groupagg_cache",
    )
    parser.add_argument(
        "--agg-blocks",
        nargs="+",
        choices=("in1", "out1", "in2", "out2", "bi1", "bi2"),
        default=("in1", "out1"),
        help="Group-aggregation blocks appended to the feature matrix.",
    )
    parser.add_argument(
        "--agg-half-life-days",
        type=float,
        nargs="+",
        default=(20.0, 90.0),
        help="Optional edge-time half-lives used for weighted neighbor aggregation.",
    )
    parser.add_argument(
        "--append-count-features",
        action="store_true",
        help="Append per-group weighted neighbor counts alongside group mean features.",
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
    parser.add_argument("--min-train-first-active-day", type=int, default=0)
    add_domain_weight_args(parser)
    return parser.parse_args()


def _cache_key(args: argparse.Namespace, threshold_day: int) -> str:
    payload = {
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "agg_blocks": list(args.agg_blocks),
        "agg_half_life_days": _resolve_half_lives(list(args.agg_half_life_days)),
        "append_count_features": bool(args.append_count_features),
        "threshold_day": int(threshold_day),
        "min_train_first_active_day": int(args.min_train_first_active_day),
        "include_future_background": bool(args.include_future_background),
        "feature_dir": str(args.feature_dir.resolve()),
        "train_scope": "historical_multiclass_bg",
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _take_feature_matrix(
    feature_dir: Path,
    phase: str,
    feature_model: str,
    extra_groups: list[str],
    split_ids: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    store = FeatureStore(
        phase,
        resolve_feature_groups(feature_model, extra_groups),
        outdir=feature_dir,
    )
    feature_names = list(store.feature_names)
    mats = {
        split_name: store.take_rows(np.asarray(node_ids, dtype=np.int32)).astype(np.float32, copy=False)
        for split_name, node_ids in split_ids.items()
    }
    return mats, feature_names


def _weighted_csr(
    ptr: np.ndarray,
    neighbors: np.ndarray,
    num_nodes: int,
    timestamps: np.ndarray | None = None,
    max_day: int | None = None,
    half_life_days: float | None = None,
) -> sp.csr_matrix:
    indptr = np.asarray(ptr, dtype=np.int64)
    indices = np.asarray(neighbors, dtype=np.int32)
    if half_life_days is None:
        data = np.ones(indices.shape[0], dtype=np.float32)
        return sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes), dtype=np.float32)
    if timestamps is None or max_day is None:
        raise ValueError("timestamps and max_day are required when agg half-life is enabled.")
    edge_time = np.asarray(timestamps, dtype=np.float32)
    age = float(max_day) - edge_time
    data = np.power(np.float32(0.5), age / float(half_life_days)).astype(np.float32, copy=False)
    return sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes), dtype=np.float32)


def _build_known_group_masks(
    phase: str,
    labels: np.ndarray,
    split,
    num_nodes: int,
) -> dict[str, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int8)
    known_normal = np.zeros(num_nodes, dtype=np.float32)
    known_fraud = np.zeros(num_nodes, dtype=np.float32)
    known_bg2 = (labels_arr == 2).astype(np.float32, copy=False)
    known_bg3 = (labels_arr == 3).astype(np.float32, copy=False)
    if phase == "phase1":
        train_ids = np.asarray(split.train_ids, dtype=np.int32)
        train_labels = labels_arr[train_ids]
        known_normal[train_ids[train_labels == 0]] = 1.0
        known_fraud[train_ids[train_labels == 1]] = 1.0
    known_any = known_normal + known_fraud + known_bg2 + known_bg3
    unknown = (known_any == 0.0).astype(np.float32, copy=False)
    return {
        "known_normal": known_normal.astype(np.float32, copy=False),
        "known_fraud": known_fraud.astype(np.float32, copy=False),
        "known_bg2": known_bg2.astype(np.float32, copy=False),
        "known_bg3": known_bg3.astype(np.float32, copy=False),
        "unknown": unknown.astype(np.float32, copy=False),
    }


def _safe_mean(sum_matrix: np.ndarray, count_vector: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.asarray(count_vector, dtype=np.float32).reshape(-1, 1), 1e-6)
    return (np.asarray(sum_matrix, dtype=np.float32) / denom).astype(np.float32, copy=False)


def _append_split_block(
    out_blocks: dict[str, list[np.ndarray]],
    split_ids: dict[str, np.ndarray],
    full_matrix: np.ndarray,
) -> None:
    for split_name, node_ids in split_ids.items():
        out_blocks[split_name].append(np.asarray(full_matrix[node_ids], dtype=np.float32, copy=False))


def _expected_groupagg_dim(raw_feature_dim: int, agg_blocks: list[str], append_count_features: bool) -> int:
    per_group_dim = raw_feature_dim * 2 + (1 if append_count_features else 0)
    return len(agg_blocks) * len(GROUP_NAMES) * per_group_dim


def _build_groupagg_phase_blocks(
    phase: str,
    feature_dir: Path,
    split,
    labels: np.ndarray,
    split_ids: dict[str, np.ndarray],
    agg_blocks: list[str],
    agg_half_lives: list[float | None],
    append_count_features: bool,
    target_paths: dict[str, Path] | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if not agg_blocks:
        return {
            split_name: np.zeros((node_ids.shape[0], 0), dtype=np.float32)
            for split_name, node_ids in split_ids.items()
        }, []

    graph = load_graph_cache(phase, outdir=feature_dir)
    arrays = load_phase_arrays(phase, keys=("x",))
    raw_x = np.asarray(arrays["x"], dtype=np.float32)
    filled_x = np.where(raw_x == -1.0, 0.0, raw_x).astype(np.float32, copy=False)
    missing_mask = (raw_x == -1.0).astype(np.float32, copy=False)
    raw_feature_names = [f"x{i}" for i in range(raw_x.shape[1])]
    miss_feature_names = [f"x{i}_missing" for i in range(raw_x.shape[1])]
    total_dim = len(agg_half_lives) * _expected_groupagg_dim(
        raw_feature_dim=raw_x.shape[1],
        agg_blocks=agg_blocks,
        append_count_features=append_count_features,
    )

    group_masks = _build_known_group_masks(
        phase=phase,
        labels=labels,
        split=split,
        num_nodes=graph.num_nodes,
    )

    phase_blocks: dict[str, list[np.ndarray]] = {name: [] for name in split_ids}
    split_mmaps: dict[str, np.memmap] | None = None
    split_offsets: dict[str, int] | None = None
    if target_paths is not None:
        split_mmaps = {
            split_name: np.lib.format.open_memmap(
                target_paths[split_name],
                mode="w+",
                dtype=np.float32,
                shape=(node_ids.shape[0], total_dim),
            )
            for split_name, node_ids in split_ids.items()
        }
        split_offsets = {split_name: 0 for split_name in split_ids}

    def append_block(full_matrix: np.ndarray) -> None:
        if split_mmaps is None or split_offsets is None:
            _append_split_block(phase_blocks, split_ids, full_matrix)
            return
        width = full_matrix.shape[1]
        for split_name, node_ids in split_ids.items():
            offset = split_offsets[split_name]
            split_mmaps[split_name][:, offset : offset + width] = np.asarray(
                full_matrix[node_ids],
                dtype=np.float32,
                copy=False,
            )
            split_offsets[split_name] = offset + width
    feature_names: list[str] = []
    use_half_life_prefix = len(agg_half_lives) > 1
    need_in = any(block in {"in1", "in2", "bi1", "bi2"} for block in agg_blocks)
    need_out = any(block in {"out1", "out2", "bi1", "bi2"} for block in agg_blocks)

    for half_life in agg_half_lives:
        prefix_root = f"{_half_life_tag(half_life)}__" if use_half_life_prefix else ""
        a_in = None
        a_out = None
        if need_in:
            a_in = _weighted_csr(
                graph.in_ptr,
                graph.in_neighbors,
                graph.num_nodes,
                timestamps=graph.in_edge_timestamp,
                max_day=graph.max_day,
                half_life_days=half_life,
            )
        if need_out:
            a_out = _weighted_csr(
                graph.out_ptr,
                graph.out_neighbors,
                graph.num_nodes,
                timestamps=graph.out_edge_timestamp,
                max_day=graph.max_day,
                half_life_days=half_life,
            )

        in1_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        out1_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for group_name in GROUP_NAMES:
            group_mask = np.asarray(group_masks[group_name], dtype=np.float32)
            masked_raw = filled_x * group_mask.reshape(-1, 1)
            masked_missing = missing_mask * group_mask.reshape(-1, 1)
            if need_in and a_in is not None:
                in_count = np.asarray(a_in @ group_mask, dtype=np.float32)
                in_raw = _safe_mean(a_in @ masked_raw, in_count)
                in_missing = _safe_mean(a_in @ masked_missing, in_count)
                in1_cache[group_name] = (in_raw, in_missing, in_count)
                if "in1" in agg_blocks:
                    block = np.concatenate([in_raw, in_missing], axis=1).astype(np.float32, copy=False)
                    append_block(block)
                    feature_names.extend(
                        [f"groupagg__{prefix_root}in1__{group_name}__{name}" for name in raw_feature_names]
                        + [f"groupagg__{prefix_root}in1__{group_name}__{name}" for name in miss_feature_names]
                    )
                    if append_count_features:
                        count_block = np.log1p(in_count.reshape(-1, 1)).astype(np.float32, copy=False)
                        append_block(count_block)
                        feature_names.append(f"groupagg__{prefix_root}in1__{group_name}__logcount")
            if need_out and a_out is not None:
                out_count = np.asarray(a_out @ group_mask, dtype=np.float32)
                out_raw = _safe_mean(a_out @ masked_raw, out_count)
                out_missing = _safe_mean(a_out @ masked_missing, out_count)
                out1_cache[group_name] = (out_raw, out_missing, out_count)
                if "out1" in agg_blocks:
                    block = np.concatenate([out_raw, out_missing], axis=1).astype(np.float32, copy=False)
                    append_block(block)
                    feature_names.extend(
                        [f"groupagg__{prefix_root}out1__{group_name}__{name}" for name in raw_feature_names]
                        + [f"groupagg__{prefix_root}out1__{group_name}__{name}" for name in miss_feature_names]
                    )
                    if append_count_features:
                        count_block = np.log1p(out_count.reshape(-1, 1)).astype(np.float32, copy=False)
                        append_block(count_block)
                        feature_names.append(f"groupagg__{prefix_root}out1__{group_name}__logcount")
            del masked_raw, masked_missing
            gc.collect()

        if "bi1" in agg_blocks:
            for group_name in GROUP_NAMES:
                if group_name not in in1_cache or group_name not in out1_cache:
                    raise RuntimeError("bi1 requires both inbound and outbound one-hop group aggregations.")
                in_raw, in_missing, in_count = in1_cache[group_name]
                out_raw, out_missing, out_count = out1_cache[group_name]
                bi_raw = ((in_raw + out_raw) * 0.5).astype(np.float32, copy=False)
                bi_missing = ((in_missing + out_missing) * 0.5).astype(np.float32, copy=False)
                block = np.concatenate([bi_raw, bi_missing], axis=1).astype(np.float32, copy=False)
                append_block(block)
                feature_names.extend(
                    [f"groupagg__{prefix_root}bi1__{group_name}__{name}" for name in raw_feature_names]
                    + [f"groupagg__{prefix_root}bi1__{group_name}__{name}" for name in miss_feature_names]
                )
                if append_count_features:
                    bi_count = np.log1p((in_count + out_count).reshape(-1, 1) * 0.5).astype(np.float32, copy=False)
                    append_block(bi_count)
                    feature_names.append(f"groupagg__{prefix_root}bi1__{group_name}__logcount")
                del bi_raw, bi_missing
                gc.collect()

        if "in2" in agg_blocks:
            if a_in is None:
                raise RuntimeError("in2 requires inbound adjacency.")
            for group_name in GROUP_NAMES:
                in_raw, in_missing, in_count = in1_cache[group_name]
                in2_count = np.asarray(a_in @ in_count, dtype=np.float32)
                in2_raw = _safe_mean(a_in @ (in_raw * in_count.reshape(-1, 1)), in2_count)
                in2_missing = _safe_mean(a_in @ (in_missing * in_count.reshape(-1, 1)), in2_count)
                block = np.concatenate([in2_raw, in2_missing], axis=1).astype(np.float32, copy=False)
                append_block(block)
                feature_names.extend(
                    [f"groupagg__{prefix_root}in2__{group_name}__{name}" for name in raw_feature_names]
                    + [f"groupagg__{prefix_root}in2__{group_name}__{name}" for name in miss_feature_names]
                )
                if append_count_features:
                    count_block = np.log1p(in2_count.reshape(-1, 1)).astype(np.float32, copy=False)
                    append_block(count_block)
                    feature_names.append(f"groupagg__{prefix_root}in2__{group_name}__logcount")
                in1_cache[group_name] = (in2_raw, in2_missing, in2_count)
                gc.collect()

        if "out2" in agg_blocks:
            if a_out is None:
                raise RuntimeError("out2 requires outbound adjacency.")
            for group_name in GROUP_NAMES:
                out_raw, out_missing, out_count = out1_cache[group_name]
                out2_count = np.asarray(a_out @ out_count, dtype=np.float32)
                out2_raw = _safe_mean(a_out @ (out_raw * out_count.reshape(-1, 1)), out2_count)
                out2_missing = _safe_mean(a_out @ (out_missing * out_count.reshape(-1, 1)), out2_count)
                block = np.concatenate([out2_raw, out2_missing], axis=1).astype(np.float32, copy=False)
                append_block(block)
                feature_names.extend(
                    [f"groupagg__{prefix_root}out2__{group_name}__{name}" for name in raw_feature_names]
                    + [f"groupagg__{prefix_root}out2__{group_name}__{name}" for name in miss_feature_names]
                )
                if append_count_features:
                    count_block = np.log1p(out2_count.reshape(-1, 1)).astype(np.float32, copy=False)
                    append_block(count_block)
                    feature_names.append(f"groupagg__{prefix_root}out2__{group_name}__logcount")
                out1_cache[group_name] = (out2_raw, out2_missing, out2_count)
                gc.collect()

        if "bi2" in agg_blocks:
            for group_name in GROUP_NAMES:
                if group_name not in in1_cache or group_name not in out1_cache:
                    raise RuntimeError("bi2 requires both inbound and outbound two-hop group aggregations.")
                in_raw, in_missing, in_count = in1_cache[group_name]
                out_raw, out_missing, out_count = out1_cache[group_name]
                bi_raw = ((in_raw + out_raw) * 0.5).astype(np.float32, copy=False)
                bi_missing = ((in_missing + out_missing) * 0.5).astype(np.float32, copy=False)
                block = np.concatenate([bi_raw, bi_missing], axis=1).astype(np.float32, copy=False)
                append_block(block)
                feature_names.extend(
                    [f"groupagg__{prefix_root}bi2__{group_name}__{name}" for name in raw_feature_names]
                    + [f"groupagg__{prefix_root}bi2__{group_name}__{name}" for name in miss_feature_names]
                )
                if append_count_features:
                    bi_count = np.log1p((in_count + out_count).reshape(-1, 1) * 0.5).astype(np.float32, copy=False)
                    append_block(bi_count)
                    feature_names.append(f"groupagg__{prefix_root}bi2__{group_name}__logcount")
                del bi_raw, bi_missing
                gc.collect()

        del a_in, a_out, in1_cache, out1_cache
        gc.collect()

    if split_mmaps is not None and split_offsets is not None:
        for split_name, expected in split_offsets.items():
            if expected != total_dim:
                raise AssertionError(
                    f"Groupagg feature width mismatch for {phase}:{split_name}: expected {total_dim}, got {expected}"
                )
        for mmap in split_mmaps.values():
            mmap.flush()
        return {split_name: np.asarray(mmap) for split_name, mmap in split_mmaps.items()}, feature_names

    matrices = {
        split_name: np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        for split_name, parts in phase_blocks.items()
    }
    return matrices, feature_names


def _load_or_build_groupagg_features(
    args: argparse.Namespace,
    cache_dir: Path,
    split,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    phase1_ids: dict[str, np.ndarray],
    phase2_ids: dict[str, np.ndarray],
    agg_half_lives: list[float | None],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    feature_names_path = cache_dir / "groupagg_feature_names.json"
    phase1_train_path = cache_dir / "phase1_train_groupagg.npy"
    phase1_val_path = cache_dir / "phase1_val_groupagg.npy"
    phase2_external_path = cache_dir / "phase2_external_groupagg.npy"
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
    phase1, feature_names = _build_groupagg_phase_blocks(
        phase="phase1",
        feature_dir=args.feature_dir,
        split=split,
        labels=phase1_y,
        split_ids=phase1_ids,
        agg_blocks=list(args.agg_blocks),
        agg_half_lives=agg_half_lives,
        append_count_features=bool(args.append_count_features),
        target_paths={
            "train": phase1_train_path,
            "val": phase1_val_path,
        },
    )
    phase2, phase2_feature_names = _build_groupagg_phase_blocks(
        phase="phase2",
        feature_dir=args.feature_dir,
        split=split,
        labels=phase2_y,
        split_ids=phase2_ids,
        agg_blocks=list(args.agg_blocks),
        agg_half_lives=agg_half_lives,
        append_count_features=bool(args.append_count_features),
        target_paths={
            "external": phase2_external_path,
        },
    )
    if phase2_feature_names != feature_names:
        raise AssertionError("Phase1/phase2 group aggregation feature names are not aligned.")
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
    set_global_seed(args.seed)
    split = load_experiment_split()
    agg_half_lives = _resolve_half_lives(list(args.agg_half_life_days))
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    phase1_graph = load_graph_cache("phase1", outdir=args.feature_dir)
    first_active = np.asarray(phase1_graph.first_active, dtype=np.int32)
    split_data = build_historical_multiclass_bg_split(
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        first_active=first_active,
        include_future_background=bool(args.include_future_background),
        min_train_first_active_day=int(args.min_train_first_active_day),
    )

    base_phase1, base_feature_names = _take_feature_matrix(
        feature_dir=args.feature_dir,
        phase="phase1",
        feature_model=args.feature_model,
        extra_groups=list(args.extra_groups),
        split_ids={"train": split_data.historical_ids, "val": split_data.val_ids},
    )
    base_phase2, _ = _take_feature_matrix(
        feature_dir=args.feature_dir,
        phase="phase2",
        feature_model=args.feature_model,
        extra_groups=list(args.extra_groups),
        split_ids={"external": split_data.external_ids},
    )

    cache_dir = ensure_dir(args.cache_root / _cache_key(args, threshold_day=int(split.threshold_day)))
    phase1_groupagg, phase2_groupagg, groupagg_feature_names = _load_or_build_groupagg_features(
        args=args,
        cache_dir=cache_dir,
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        phase1_ids={"train": split_data.historical_ids, "val": split_data.val_ids},
        phase2_ids={"external": split_data.external_ids},
        agg_half_lives=agg_half_lives,
    )

    x_train = np.concatenate(
        [
            np.asarray(base_phase1["train"], dtype=np.float32),
            np.asarray(phase1_groupagg["train"], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    x_val = np.concatenate(
        [
            np.asarray(base_phase1["val"], dtype=np.float32),
            np.asarray(phase1_groupagg["val"], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    x_external = np.concatenate(
        [
            np.asarray(base_phase2["external"], dtype=np.float32),
            np.asarray(phase2_groupagg["external"], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    feature_names = list(base_feature_names) + list(groupagg_feature_names)

    run_dir = ensure_dir(args.outdir / args.run_name)
    train_multiclass_bg_xgb(
        args,
        split_data=split_data,
        x_train=x_train,
        x_val=x_val,
        x_external=x_external,
        feature_names=feature_names,
        run_dir=run_dir,
        model_name="xgboost_gpu_multiclass_bg_groupagg",
        summary_extra={
            "feature_model": args.feature_model,
            "extra_groups": list(args.extra_groups),
            "agg_blocks": list(args.agg_blocks),
            "agg_half_life_days": [None if value is None else float(value) for value in agg_half_lives],
            "append_count_features": bool(args.append_count_features),
            "include_future_background": bool(args.include_future_background),
            "cache_dir": str(cache_dir),
        },
    )


if __name__ == "__main__":
    main()
