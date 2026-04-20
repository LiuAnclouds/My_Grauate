from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

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


GROUP_NAMES = ["known_normal", "known_fraud", "known_bg2", "known_bg3", "unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU XGBoost with relation-specific known-label grouped neighbor feature means.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "xgboost_gpu")
    parser.add_argument("--cache-root", type=Path, default=MODEL_OUTPUT_ROOT / "_multiclass_bg_relgroup_cache")
    parser.add_argument("--feature-model", choices=("m2_hybrid", "m3_neighbor"), default="m3_neighbor")
    parser.add_argument("--extra-groups", nargs="*", default=())
    parser.add_argument("--directions", nargs="+", choices=("in", "out"), default=("out",))
    parser.add_argument("--edge-types", type=int, nargs="+", default=list(range(1, 12)))
    parser.add_argument("--agg-half-life-days", type=float, nargs="+", default=(20.0, 90.0))
    parser.add_argument("--selected-raw-indices", type=int, nargs="+", default=(7, 12, 13))
    parser.add_argument("--selected-missing-indices", type=int, nargs="+", default=(0, 1, 6, 8, 9, 15, 16))
    parser.add_argument("--append-count-features", action="store_true")
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
    parser.add_argument("--time-weight-half-life-days", type=float, default=0.0)
    parser.add_argument("--time-weight-floor", type=float, default=0.25)
    parser.add_argument("--include-future-background", action="store_true")
    parser.add_argument("--min-train-first-active-day", type=int, default=0)
    add_domain_weight_args(parser)
    return parser.parse_args()


def _cache_key(args: argparse.Namespace, threshold_day: int) -> str:
    payload = {
        "feature_model": args.feature_model,
        "extra_groups": list(args.extra_groups),
        "directions": list(args.directions),
        "edge_types": list(args.edge_types),
        "agg_half_life_days": _resolve_half_lives(list(args.agg_half_life_days)),
        "selected_raw_indices": list(args.selected_raw_indices),
        "selected_missing_indices": list(args.selected_missing_indices),
        "append_count_features": bool(args.append_count_features),
        "threshold_day": int(threshold_day),
        "min_train_first_active_day": int(args.min_train_first_active_day),
        "include_future_background": bool(args.include_future_background),
        "feature_dir": str(args.feature_dir.resolve()),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


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
    mats = {
        split_name: store.take_rows(np.asarray(node_ids, dtype=np.int32)).astype(np.float32, copy=False)
        for split_name, node_ids in split_ids.items()
    }
    return mats, list(store.feature_names)


def _build_group_masks(phase: str, labels: np.ndarray, split, num_nodes: int) -> dict[str, np.ndarray]:
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
    unknown = ((known_normal + known_fraud + known_bg2 + known_bg3) == 0.0).astype(np.float32, copy=False)
    return {
        "known_normal": known_normal,
        "known_fraud": known_fraud,
        "known_bg2": known_bg2,
        "known_bg3": known_bg3,
        "unknown": unknown,
    }


def _feature_names_for_combo(
    prefix: str,
    group_name: str,
    raw_idx: list[int],
    miss_idx: list[int],
    append_count: bool,
) -> list[str]:
    names = [f"{prefix}__{group_name}__x{idx}" for idx in raw_idx]
    names += [f"{prefix}__{group_name}__x{idx}_missing" for idx in miss_idx]
    if append_count:
        names.append(f"{prefix}__{group_name}__logcount")
    return names


def _build_phase_relgroup(
    phase: str,
    feature_dir: Path,
    labels: np.ndarray,
    split,
    split_ids: dict[str, np.ndarray],
    directions: list[str],
    edge_types: list[int],
    agg_half_lives: list[float | None],
    raw_idx: list[int],
    miss_idx: list[int],
    append_count_features: bool,
    target_paths: dict[str, Path],
) -> list[str]:
    graph = load_graph_cache(phase, outdir=feature_dir)
    x = np.asarray(load_phase_arrays(phase, keys=("x",))["x"], dtype=np.float32)
    filled_x = np.where(x == -1.0, 0.0, x).astype(np.float32, copy=False)
    missing_mask = (x == -1.0).astype(np.float32, copy=False)
    group_masks = _build_group_masks(phase, labels, split, graph.num_nodes)

    feat_per_combo = len(raw_idx) + len(miss_idx) + (1 if append_count_features else 0)
    total_dim = len(agg_half_lives) * len(directions) * len(edge_types) * len(GROUP_NAMES) * feat_per_combo
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
    feature_names: list[str] = []

    src_all = np.asarray(load_phase_arrays(phase, keys=("edge_index",))["edge_index"][:, 0], dtype=np.int32)
    dst_all = np.asarray(load_phase_arrays(phase, keys=("edge_index",))["edge_index"][:, 1], dtype=np.int32)
    type_all = np.asarray(load_phase_arrays(phase, keys=("edge_type",))["edge_type"], dtype=np.int16)
    ts_all = np.asarray(load_phase_arrays(phase, keys=("edge_timestamp",))["edge_timestamp"], dtype=np.int32)
    max_day = int(np.max(ts_all))

    def append_full(full_matrix: np.ndarray) -> None:
        width = full_matrix.shape[1]
        for split_name, node_ids in split_ids.items():
            start = split_offsets[split_name]
            split_mmaps[split_name][:, start : start + width] = np.asarray(full_matrix[node_ids], dtype=np.float32, copy=False)
            split_offsets[split_name] = start + width

    for half_life in agg_half_lives:
        if half_life is None:
            weight_all = np.ones(ts_all.shape[0], dtype=np.float32)
        else:
            age = float(max_day) - ts_all.astype(np.float32, copy=False)
            weight_all = np.power(np.float32(0.5), age / float(half_life)).astype(np.float32, copy=False)
        half_prefix = f"{_half_life_tag(half_life)}__" if len(agg_half_lives) > 1 else ""
        for direction in directions:
            if direction == "out":
                centers = src_all
                neighbors = dst_all
            else:
                centers = dst_all
                neighbors = src_all
            for rel_type in edge_types:
                rel_mask = type_all == int(rel_type)
                if not np.any(rel_mask):
                    for group_name in GROUP_NAMES:
                        feature_names.extend(
                            _feature_names_for_combo(
                                prefix=f"relgroup__{half_prefix}{direction}_t{rel_type}",
                                group_name=group_name,
                                raw_idx=raw_idx,
                                miss_idx=miss_idx,
                                append_count=append_count_features,
                            )
                        )
                        zeros = np.zeros((graph.num_nodes, feat_per_combo), dtype=np.float32)
                        append_full(zeros)
                    continue
                rel_centers = centers[rel_mask]
                rel_neighbors = neighbors[rel_mask]
                rel_weight = weight_all[rel_mask]
                for group_name in GROUP_NAMES:
                    group_mask = group_masks[group_name][rel_neighbors] > 0.5
                    if not np.any(group_mask):
                        block = np.zeros((graph.num_nodes, feat_per_combo), dtype=np.float32)
                    else:
                        c = rel_centers[group_mask]
                        n = rel_neighbors[group_mask]
                        w = rel_weight[group_mask]
                        count = np.bincount(c, weights=w, minlength=graph.num_nodes).astype(np.float32, copy=False)
                        cols = []
                        denom = np.maximum(count, 1e-6)
                        for idx in raw_idx:
                            weighted = w * filled_x[n, idx]
                            sums = np.bincount(c, weights=weighted, minlength=graph.num_nodes).astype(np.float32, copy=False)
                            cols.append((sums / denom).astype(np.float32, copy=False))
                        for idx in miss_idx:
                            weighted = w * missing_mask[n, idx]
                            sums = np.bincount(c, weights=weighted, minlength=graph.num_nodes).astype(np.float32, copy=False)
                            cols.append((sums / denom).astype(np.float32, copy=False))
                        if append_count_features:
                            cols.append(np.log1p(count).astype(np.float32, copy=False))
                        block = np.column_stack(cols).astype(np.float32, copy=False)
                    append_full(block)
                    feature_names.extend(
                        _feature_names_for_combo(
                            prefix=f"relgroup__{half_prefix}{direction}_t{rel_type}",
                            group_name=group_name,
                            raw_idx=raw_idx,
                            miss_idx=miss_idx,
                            append_count=append_count_features,
                        )
                    )
        del weight_all

    for split_name, offset in split_offsets.items():
        if offset != total_dim:
            raise AssertionError(f"Feature width mismatch for {phase}:{split_name}: {offset} vs {total_dim}")
    for mmap in split_mmaps.values():
        mmap.flush()
    return feature_names


def _load_or_build_relgroup(
    args: argparse.Namespace,
    cache_dir: Path,
    split,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    phase1_ids: dict[str, np.ndarray],
    phase2_ids: dict[str, np.ndarray],
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
        return (
            {"train": np.load(phase1_train_path, mmap_mode="r"), "val": np.load(phase1_val_path, mmap_mode="r")},
            {"external": np.load(phase2_external_path, mmap_mode="r")},
            list(json.loads(feature_names_path.read_text(encoding="utf-8"))),
        )

    ensure_dir(cache_dir)
    feature_names = _build_phase_relgroup(
        phase=str(primary_phase),
        feature_dir=args.feature_dir,
        labels=phase1_y,
        split=split,
        split_ids=phase1_ids,
        directions=list(args.directions),
        edge_types=list(args.edge_types),
        agg_half_lives=_resolve_half_lives(list(args.agg_half_life_days)),
        raw_idx=list(args.selected_raw_indices),
        miss_idx=list(args.selected_missing_indices),
        append_count_features=bool(args.append_count_features),
        target_paths={"train": phase1_train_path, "val": phase1_val_path},
    )
    phase2_has_rows = any(np.asarray(node_ids, dtype=np.int32).size for node_ids in phase2_ids.values())
    if phase2_has_rows:
        feature_names_2 = _build_phase_relgroup(
            phase=str(external_phase),
            feature_dir=args.feature_dir,
            labels=phase2_y,
            split=split,
            split_ids=phase2_ids,
            directions=list(args.directions),
            edge_types=list(args.edge_types),
            agg_half_lives=_resolve_half_lives(list(args.agg_half_life_days)),
            raw_idx=list(args.selected_raw_indices),
            miss_idx=list(args.selected_missing_indices),
            append_count_features=bool(args.append_count_features),
            target_paths={"external": phase2_external_path},
        )
        if feature_names_2 != feature_names:
            raise AssertionError("Phase1/phase2 relgroup feature names mismatch.")
    else:
        np.save(
            phase2_external_path,
            np.zeros((np.asarray(phase2_ids["external"], dtype=np.int32).size, len(feature_names)), dtype=np.float32),
        )
    feature_names_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    return (
        {"train": np.load(phase1_train_path, mmap_mode="r"), "val": np.load(phase1_val_path, mmap_mode="r")},
        {"external": np.load(phase2_external_path, mmap_mode="r")},
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
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    graph = load_graph_cache("phase1", outdir=args.feature_dir)
    first_active = np.asarray(graph.first_active, dtype=np.int32)
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
    phase1_relgroup, phase2_relgroup, relgroup_feature_names = _load_or_build_relgroup(
        args=args,
        cache_dir=cache_dir,
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        phase1_ids={"train": split_data.historical_ids, "val": split_data.val_ids},
        phase2_ids={"external": split_data.external_ids},
    )

    x_train = np.concatenate([np.asarray(base_phase1["train"], dtype=np.float32), np.asarray(phase1_relgroup["train"], dtype=np.float32)], axis=1).astype(np.float32, copy=False)
    x_val = np.concatenate([np.asarray(base_phase1["val"], dtype=np.float32), np.asarray(phase1_relgroup["val"], dtype=np.float32)], axis=1).astype(np.float32, copy=False)
    x_external = np.concatenate([np.asarray(base_phase2["external"], dtype=np.float32), np.asarray(phase2_relgroup["external"], dtype=np.float32)], axis=1).astype(np.float32, copy=False)
    feature_names = list(base_feature_names) + list(relgroup_feature_names)

    run_dir = ensure_dir(args.outdir / args.run_name)
    train_multiclass_bg_xgb(
        args,
        split_data=split_data,
        x_train=x_train,
        x_val=x_val,
        x_external=x_external,
        feature_names=feature_names,
        run_dir=run_dir,
        model_name="xgboost_gpu_multiclass_bg_relgroup",
        summary_extra={
            "feature_model": args.feature_model,
            "extra_groups": list(args.extra_groups),
            "directions": list(args.directions),
            "edge_types": list(args.edge_types),
            "agg_half_life_days": [
                None if value is None else float(value)
                for value in _resolve_half_lives(list(args.agg_half_life_days))
            ],
            "selected_raw_indices": list(args.selected_raw_indices),
            "selected_missing_indices": list(args.selected_missing_indices),
            "append_count_features": bool(args.append_count_features),
            "include_future_background": bool(args.include_future_background),
            "cache_dir": str(cache_dir),
        },
    )


if __name__ == "__main__":
    main()
