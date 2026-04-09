from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
from experiment.training.features import load_graph_cache
from experiment.training.run_xgb_graphprop import (
    _append_slices,
    _half_life_tag,
    _load_or_build_cached_features,
    _normalized_csr,
    _resolve_half_lives,
)
from experiment.training.run_xgb_multiclass_bg_groupagg import (
    _cache_key as _groupagg_cache_key,
    _load_or_build_groupagg_features,
)
from experiment.training.run_xgb_multiclass_bg_relgroup import (
    _cache_key as _relgroup_cache_key,
    _load_or_build_relgroup,
)
from experiment.training.xgb.label_context import (
    BEST_LABEL_CONTEXT_ANCHORS,
    BEST_LABEL_CONTEXT_RELATION_ANCHORS,
    BEST_LABEL_CONTEXT_PRESET_VERSION,
    KNOWN_LABEL_ANCHOR_NAMES,
    build_known_label_anchor_matrix,
    load_or_build_temporal_label_context_features,
)
from experiment.training.xgb_utils import (
    binary_score_from_softprob,
    build_multiclass_bg_sample_weight,
    multiclass_binary_auc,
    write_feature_importance_csv,
)
from experiment.training.xgb.multiclass_bg_runtime import (
    build_historical_multiclass_bg_split,
    train_multiclass_bg_xgb,
)

BEST_GROUPAGG_HALF_LIVES = (20.0, 90.0)
BEST_GROUPAGG_BLOCKS = ("in1", "out1")
BEST_RELGROUP_HALF_LIVES = (20.0, 90.0)
BEST_RELGROUP_DIRECTIONS = ("out",)
BEST_RELGROUP_EDGE_TYPES = (1, 4, 5, 6, 10)
BEST_RELGROUP_RAW_INDICES = (7, 12, 13)
BEST_RELGROUP_MISSING_INDICES = (0, 1, 6, 8, 9, 15, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU XGBoost multiclass training with historical background supervision on graphprop features.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "xgboost_gpu")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=MODEL_OUTPUT_ROOT / "_multiclass_bg_graphprop_cache",
    )
    parser.add_argument(
        "--base-model",
        choices=("m2_hybrid", "m3_neighbor"),
        default="m3_neighbor",
    )
    parser.add_argument(
        "--prop-model",
        choices=("m2_hybrid", "m3_neighbor"),
        default="m2_hybrid",
    )
    parser.add_argument(
        "--prop-blocks",
        nargs="*",
        choices=("in1", "out1", "in2", "out2", "bi1", "bi2"),
        default=("in1", "out1"),
        help=(
            "Optional graph-propagation blocks appended on top of base features. "
            "Pass no values after --prop-blocks to disable regular graph propagation "
            "and keep only base + optional label propagation features."
        ),
    )
    parser.add_argument(
        "--extra-groups",
        nargs="*",
        default=(),
        help="Optional extra offline feature groups appended to both base and propagated features.",
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
        default=(20.0,),
    )
    parser.add_argument(
        "--label-prop-blocks",
        nargs="+",
        choices=("in1", "out1", "in2", "out2", "bi1", "bi2"),
        default=(),
        help=(
            "Optional propagation blocks computed from leakage-safe known-label anchors "
            "(phase1 historical train labels plus background labels available in both phases)."
        ),
    )
    parser.add_argument(
        "--label-prop-half-life-days",
        type=float,
        nargs="+",
        default=None,
        help="Optional half-life(s) used only for known-label propagation. Defaults to --prop-half-life-days.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-child-weight", type=float, default=6.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=300)
    parser.add_argument("--background-weight", type=float, default=0.25)
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
        "--append-best-groupagg",
        action="store_true",
        help=(
            "Append the best historical GAGA-style grouped aggregation preset "
            "(m3 + temporal_bucket_norm + in1/out1 + half-lives 20/90)."
        ),
    )
    parser.add_argument(
        "--append-best-relgroup",
        action="store_true",
        help=(
            "Append the best historical relation-group preset "
            "(out directions, edge types 1/4/5/6/10, half-lives 20/90)."
        ),
    )
    parser.add_argument(
        "--append-best-label-context",
        action="store_true",
        help=(
            "Append leakage-safe temporal label-context features built only from "
            "phase1 train 0/1 labels plus globally visible background 2/3 labels."
        ),
    )
    parser.add_argument(
        "--label-context-anchors",
        nargs="+",
        choices=KNOWN_LABEL_ANCHOR_NAMES,
        default=BEST_LABEL_CONTEXT_ANCHORS,
        help="Anchor set used by the temporal label-context block.",
    )
    parser.add_argument(
        "--label-context-relation-anchors",
        nargs="+",
        choices=KNOWN_LABEL_ANCHOR_NAMES,
        default=BEST_LABEL_CONTEXT_RELATION_ANCHORS,
        help="Anchor set used by the relation-sliced part of the temporal label-context block.",
    )
    return parser.parse_args()


def _cache_key(args: argparse.Namespace, threshold_day: int) -> str:
    base_extra_groups = list(args.extra_groups) if args.base_extra_groups is None else list(args.base_extra_groups)
    prop_extra_groups = list(args.extra_groups) if args.prop_extra_groups is None else list(args.prop_extra_groups)
    payload = {
        "base_model": args.base_model,
        "prop_model": args.prop_model,
        "prop_blocks": list(args.prop_blocks),
        "extra_groups": list(args.extra_groups),
        "base_extra_groups": base_extra_groups,
        "prop_extra_groups": prop_extra_groups,
        "prop_half_life_days": _resolve_half_lives(list(args.prop_half_life_days)),
        "label_prop_blocks": list(args.label_prop_blocks),
        "label_prop_half_life_days": (
            _resolve_half_lives(list(args.label_prop_half_life_days))
            if args.label_prop_half_life_days is not None
            else None
        ),
        "threshold_day": int(threshold_day),
        "include_future_background": bool(args.include_future_background),
        "append_best_label_context": bool(args.append_best_label_context),
        "label_context_anchors": list(args.label_context_anchors),
        "label_context_relation_anchors": list(args.label_context_relation_anchors),
        "label_context_version": BEST_LABEL_CONTEXT_PRESET_VERSION,
        "feature_dir": str(args.feature_dir.resolve()),
        "train_scope": "historical_multiclass_bg",
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _multiclass_binary_auc(predt: np.ndarray, dmatrix) -> tuple[str, float]:
    return multiclass_binary_auc(predt, dmatrix)


def _binary_score_from_softprob(prob: np.ndarray) -> np.ndarray:
    return binary_score_from_softprob(prob)


def _build_sample_weight(
    y_train: np.ndarray,
    args: argparse.Namespace,
    train_first_active: np.ndarray | None = None,
    threshold_day: int | None = None,
) -> dict[str, float | dict[str, float]]:
    return build_multiclass_bg_sample_weight(
        y_train,
        fraud_weight_scale=float(args.fraud_weight_scale),
        background_weight=float(args.background_weight),
        time_weight_half_life_days=float(args.time_weight_half_life_days),
        time_weight_floor=float(args.time_weight_floor),
        train_first_active=train_first_active,
        threshold_day=threshold_day,
    )


def _write_feature_importance(booster, feature_names: list[str], path: Path) -> None:
    write_feature_importance_csv(booster, feature_names, path)


def _resolved_label_prop_half_lives(args: argparse.Namespace) -> list[float | None]:
    if args.label_prop_half_life_days is not None:
        return _resolve_half_lives(list(args.label_prop_half_life_days))
    return _resolve_half_lives(list(args.prop_half_life_days))


def _preset_extra_groups(extra_groups: list[str]) -> list[str]:
    groups = list(extra_groups)
    if "temporal_bucket_norm" not in groups:
        groups.append("temporal_bucket_norm")
    return groups


def _load_best_groupagg_preset(
    feature_dir: Path,
    split,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    phase1_ids: dict[str, np.ndarray],
    phase2_ids: dict[str, np.ndarray],
    extra_groups: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    preset_args = SimpleNamespace(
        feature_dir=feature_dir,
        feature_model="m3_neighbor",
        extra_groups=_preset_extra_groups(extra_groups),
        agg_blocks=list(BEST_GROUPAGG_BLOCKS),
        agg_half_life_days=list(BEST_GROUPAGG_HALF_LIVES),
        append_count_features=True,
        include_future_background=False,
        cache_root=MODEL_OUTPUT_ROOT / "_multiclass_bg_groupagg_cache",
    )
    cache_dir = ensure_dir(
        preset_args.cache_root / _groupagg_cache_key(preset_args, threshold_day=int(split.threshold_day))
    )
    return _load_or_build_groupagg_features(
        args=preset_args,
        cache_dir=cache_dir,
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        phase1_ids=phase1_ids,
        phase2_ids=phase2_ids,
        agg_half_lives=_resolve_half_lives(list(BEST_GROUPAGG_HALF_LIVES)),
    )


def _load_best_relgroup_preset(
    feature_dir: Path,
    split,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    phase1_ids: dict[str, np.ndarray],
    phase2_ids: dict[str, np.ndarray],
    extra_groups: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    preset_args = SimpleNamespace(
        feature_dir=feature_dir,
        feature_model="m3_neighbor",
        extra_groups=_preset_extra_groups(extra_groups),
        directions=list(BEST_RELGROUP_DIRECTIONS),
        edge_types=list(BEST_RELGROUP_EDGE_TYPES),
        agg_half_life_days=list(BEST_RELGROUP_HALF_LIVES),
        selected_raw_indices=list(BEST_RELGROUP_RAW_INDICES),
        selected_missing_indices=list(BEST_RELGROUP_MISSING_INDICES),
        append_count_features=True,
        include_future_background=False,
        cache_root=MODEL_OUTPUT_ROOT / "_multiclass_bg_relgroup_cache",
    )
    cache_dir = ensure_dir(
        preset_args.cache_root / _relgroup_cache_key(preset_args, threshold_day=int(split.threshold_day))
    )
    return _load_or_build_relgroup(
        args=preset_args,
        cache_dir=cache_dir,
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        phase1_ids=phase1_ids,
        phase2_ids=phase2_ids,
    )

def _empty_split_mats(split_ids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        split_name: np.zeros((node_ids.shape[0], 0), dtype=np.float32)
        for split_name, node_ids in split_ids.items()
    }


def _build_phase_labelprop_blocks(
    phase: str,
    feature_dir: Path,
    labels: np.ndarray,
    split,
    split_ids: dict[str, np.ndarray],
    label_prop_blocks: list[str],
    label_prop_half_lives: list[float | None],
) -> tuple[dict[str, np.ndarray], list[str]]:
    if not label_prop_blocks:
        return _empty_split_mats(split_ids), []

    graph_cache = load_graph_cache(phase, outdir=feature_dir)
    anchor_matrix = _build_known_label_anchor_matrix(
        phase=phase,
        labels=labels,
        split=split,
        num_nodes=graph_cache.num_nodes,
    )
    phase_blocks: dict[str, list[np.ndarray]] = {name: [] for name in split_ids}
    feature_names: list[str] = []
    need_in = any(block in {"in1", "in2", "bi1", "bi2"} for block in label_prop_blocks)
    need_out = any(block in {"out1", "out2", "bi1", "bi2"} for block in label_prop_blocks)
    use_half_life_prefix = len(label_prop_half_lives) > 1

    for half_life in label_prop_half_lives:
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
            in1 = (a_in @ anchor_matrix).astype(np.float32, copy=False)
            if "in1" in label_prop_blocks:
                _append_slices(
                    phase_blocks,
                    feature_names,
                    in1,
                    KNOWN_LABEL_ANCHOR_NAMES,
                    split_ids,
                    f"label__{prefix_root}in1",
                )
        if need_out:
            a_out = _normalized_csr(
                graph_cache.out_ptr,
                graph_cache.out_neighbors,
                graph_cache.num_nodes,
                timestamps=graph_cache.out_edge_timestamp,
                max_day=graph_cache.max_day,
                half_life_days=half_life,
            )
            out1 = (a_out @ anchor_matrix).astype(np.float32, copy=False)
            if "out1" in label_prop_blocks:
                _append_slices(
                    phase_blocks,
                    feature_names,
                    out1,
                    KNOWN_LABEL_ANCHOR_NAMES,
                    split_ids,
                    f"label__{prefix_root}out1",
                )
        if "bi1" in label_prop_blocks:
            if in1 is None or out1 is None:
                raise RuntimeError("label bi1 propagation requires inbound and outbound first-hop features.")
            bi1 = ((in1 + out1) * 0.5).astype(np.float32, copy=False)
            _append_slices(
                phase_blocks,
                feature_names,
                bi1,
                KNOWN_LABEL_ANCHOR_NAMES,
                split_ids,
                f"label__{prefix_root}bi1",
            )
            del bi1
        if "in2" in label_prop_blocks:
            if a_in is None or in1 is None:
                raise RuntimeError("label in2 propagation requires inbound adjacency and first-hop features.")
            in2 = (a_in @ in1).astype(np.float32, copy=False)
            _append_slices(
                phase_blocks,
                feature_names,
                in2,
                KNOWN_LABEL_ANCHOR_NAMES,
                split_ids,
                f"label__{prefix_root}in2",
            )
            del in2
        if "out2" in label_prop_blocks:
            if a_out is None or out1 is None:
                raise RuntimeError("label out2 propagation requires outbound adjacency and first-hop features.")
            out2 = (a_out @ out1).astype(np.float32, copy=False)
            _append_slices(
                phase_blocks,
                feature_names,
                out2,
                KNOWN_LABEL_ANCHOR_NAMES,
                split_ids,
                f"label__{prefix_root}out2",
            )
            del out2
        if "bi2" in label_prop_blocks:
            if a_in is None or a_out is None or in1 is None or out1 is None:
                raise RuntimeError("label bi2 propagation requires inbound/outbound adjacency and first-hop features.")
            bi2 = (((a_in @ in1) + (a_out @ out1)) * 0.5).astype(np.float32, copy=False)
            _append_slices(
                phase_blocks,
                feature_names,
                bi2,
                KNOWN_LABEL_ANCHOR_NAMES,
                split_ids,
                f"label__{prefix_root}bi2",
            )
            del bi2
        del a_in, a_out, in1, out1
        gc.collect()

    del anchor_matrix
    gc.collect()
    matrices = {
        split_name: np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        for split_name, parts in phase_blocks.items()
    }
    return matrices, feature_names


def _load_or_build_labelprop_features(
    args: argparse.Namespace,
    cache_dir: Path,
    split,
    phase1_y: np.ndarray,
    phase2_y: np.ndarray,
    phase1_ids: dict[str, np.ndarray],
    phase2_ids: dict[str, np.ndarray],
    label_prop_half_lives: list[float | None],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    if not args.label_prop_blocks:
        return _empty_split_mats(phase1_ids), _empty_split_mats(phase2_ids), []

    feature_names_path = cache_dir / "labelprop_feature_names.json"
    phase1_train_path = cache_dir / "phase1_train_labelprop.npy"
    phase1_val_path = cache_dir / "phase1_val_labelprop.npy"
    phase2_external_path = cache_dir / "phase2_external_labelprop.npy"
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
    phase1, feature_names = _build_phase_labelprop_blocks(
        phase="phase1",
        feature_dir=args.feature_dir,
        labels=phase1_y,
        split=split,
        split_ids=phase1_ids,
        label_prop_blocks=list(args.label_prop_blocks),
        label_prop_half_lives=label_prop_half_lives,
    )
    phase2, phase2_feature_names = _build_phase_labelprop_blocks(
        phase="phase2",
        feature_dir=args.feature_dir,
        labels=phase2_y,
        split=split,
        split_ids=phase2_ids,
        label_prop_blocks=list(args.label_prop_blocks),
        label_prop_half_lives=label_prop_half_lives,
    )
    if phase2_feature_names != feature_names:
        raise AssertionError("Phase1/phase2 known-label propagation feature names are not aligned.")

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


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    split = load_experiment_split()
    half_lives = _resolve_half_lives(list(args.prop_half_life_days))
    label_prop_half_lives = _resolved_label_prop_half_lives(args)
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
    )

    graphprop_args = SimpleNamespace(
        feature_dir=args.feature_dir,
        base_model=args.base_model,
        prop_model=args.prop_model,
        prop_blocks=list(args.prop_blocks),
        extra_groups=list(args.extra_groups),
        base_extra_groups=(
            list(args.extra_groups) if args.base_extra_groups is None else list(args.base_extra_groups)
        ),
        prop_extra_groups=(
            list(args.extra_groups) if args.prop_extra_groups is None else list(args.prop_extra_groups)
        ),
        max_train_nodes=None,
        max_val_nodes=None,
        max_external_nodes=None,
    )
    cache_dir = ensure_dir(args.cache_root / _cache_key(args, threshold_day=int(split.threshold_day)))
    phase1_mats, phase2_mats, feature_names = _load_or_build_cached_features(
        args=graphprop_args,
        cache_dir=cache_dir,
        phase1_ids={"train": split_data.historical_ids, "val": split_data.val_ids},
        phase2_ids={"external": split_data.external_ids},
        half_lives=half_lives,
    )
    phase1_labelprop, phase2_labelprop, labelprop_feature_names = _load_or_build_labelprop_features(
        args=args,
        cache_dir=cache_dir,
        split=split,
        phase1_y=phase1_y,
        phase2_y=phase2_y,
        phase1_ids={"train": split_data.historical_ids, "val": split_data.val_ids},
        phase2_ids={"external": split_data.external_ids},
        label_prop_half_lives=label_prop_half_lives,
    )
    phase1_label_context = _empty_split_mats({"train": split_data.historical_ids, "val": split_data.val_ids})
    phase2_label_context = _empty_split_mats({"external": split_data.external_ids})
    label_context_feature_names: list[str] = []
    if args.append_best_label_context:
        phase1_label_context, phase2_label_context, label_context_feature_names = (
            load_or_build_temporal_label_context_features(
                cache_dir=cache_dir,
                feature_dir=args.feature_dir,
                split=split,
                phase1_y=phase1_y,
                phase2_y=phase2_y,
                phase1_ids={"train": split_data.historical_ids, "val": split_data.val_ids},
                phase2_ids={"external": split_data.external_ids},
                anchor_names=tuple(args.label_context_anchors),
                relation_anchor_names=tuple(args.label_context_relation_anchors),
            )
        )

    x_train = np.asarray(phase1_mats["train"], dtype=np.float32)
    x_val = np.asarray(phase1_mats["val"], dtype=np.float32)
    x_external = np.asarray(phase2_mats["external"], dtype=np.float32)
    if labelprop_feature_names:
        x_train = np.concatenate(
            [x_train, np.asarray(phase1_labelprop["train"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        x_val = np.concatenate(
            [x_val, np.asarray(phase1_labelprop["val"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        x_external = np.concatenate(
            [x_external, np.asarray(phase2_labelprop["external"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        feature_names = list(feature_names) + list(labelprop_feature_names)
    if label_context_feature_names:
        x_train = np.concatenate(
            [x_train, np.asarray(phase1_label_context["train"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        x_val = np.concatenate(
            [x_val, np.asarray(phase1_label_context["val"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        x_external = np.concatenate(
            [x_external, np.asarray(phase2_label_context["external"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        feature_names = list(feature_names) + list(label_context_feature_names)
    if args.append_best_groupagg:
        phase1_groupagg, phase2_groupagg, groupagg_feature_names = _load_best_groupagg_preset(
            feature_dir=args.feature_dir,
            split=split,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            phase1_ids={"train": split_data.historical_ids, "val": split_data.val_ids},
            phase2_ids={"external": split_data.external_ids},
            extra_groups=list(args.extra_groups),
        )
        x_train = np.concatenate(
            [x_train, np.asarray(phase1_groupagg["train"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        x_val = np.concatenate(
            [x_val, np.asarray(phase1_groupagg["val"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        x_external = np.concatenate(
            [x_external, np.asarray(phase2_groupagg["external"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        feature_names = list(feature_names) + list(groupagg_feature_names)
    if args.append_best_relgroup:
        phase1_relgroup, phase2_relgroup, relgroup_feature_names = _load_best_relgroup_preset(
            feature_dir=args.feature_dir,
            split=split,
            phase1_y=phase1_y,
            phase2_y=phase2_y,
            phase1_ids={"train": split_data.historical_ids, "val": split_data.val_ids},
            phase2_ids={"external": split_data.external_ids},
            extra_groups=list(args.extra_groups),
        )
        x_train = np.concatenate(
            [x_train, np.asarray(phase1_relgroup["train"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        x_val = np.concatenate(
            [x_val, np.asarray(phase1_relgroup["val"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        x_external = np.concatenate(
            [x_external, np.asarray(phase2_relgroup["external"], dtype=np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        feature_names = list(feature_names) + list(relgroup_feature_names)
    run_dir = ensure_dir(args.outdir / args.run_name)
    train_multiclass_bg_xgb(
        args,
        split_data=split_data,
        x_train=x_train,
        x_val=x_val,
        x_external=x_external,
        feature_names=feature_names,
        run_dir=run_dir,
        model_name="xgboost_gpu_multiclass_bg_graphprop",
        summary_extra={
            "base_model": args.base_model,
            "prop_model": args.prop_model,
            "prop_blocks": list(args.prop_blocks),
            "label_prop_blocks": list(args.label_prop_blocks),
            "extra_groups": list(args.extra_groups),
            "base_extra_groups": (
                list(args.extra_groups) if args.base_extra_groups is None else list(args.base_extra_groups)
            ),
            "prop_extra_groups": (
                list(args.extra_groups) if args.prop_extra_groups is None else list(args.prop_extra_groups)
            ),
            "include_future_background": bool(args.include_future_background),
            "prop_half_life_days": [None if value is None else float(value) for value in half_lives],
            "label_prop_half_life_days": [
                None if value is None else float(value) for value in label_prop_half_lives
            ],
            "append_best_groupagg": bool(args.append_best_groupagg),
            "append_best_relgroup": bool(args.append_best_relgroup),
            "append_best_label_context": bool(args.append_best_label_context),
            "label_context_anchors": list(args.label_context_anchors),
            "label_context_relation_anchors": list(args.label_context_relation_anchors),
            "label_context_version": BEST_LABEL_CONTEXT_PRESET_VERSION,
            "cache_dir": str(cache_dir),
        },
    )


if __name__ == "__main__":
    main()
