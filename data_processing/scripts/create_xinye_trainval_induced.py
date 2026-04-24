from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.core.contracts import PreparedGraphContract, save_prepared_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a XinYe train/val induced graph with node ids remapped to a compact "
            "range. The original XinYe raw files are read-only inputs."
        )
    )
    parser.add_argument(
        "--source-npz",
        type=Path,
        default=REPO_ROOT / "data/raw/xinye_dgraph/phase1_gdata.npz",
        help="Original XinYe phase1 npz.",
    )
    parser.add_argument(
        "--source-analysis",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/xinye_dgraph",
        help="Analysis directory containing train_ids.npy, val_ids.npy, recommended_split.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root where data/raw and outputs/analysis will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="xinye_dgraph_trainval",
        help="Output dataset name under data/raw and outputs/analysis.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=0,
        help=(
            "Number of one-hop non-train/val context nodes to add. Use -1 to keep "
            "all one-hop context nodes."
        ),
    )
    parser.add_argument(
        "--context-anchor",
        choices=("base", "train"),
        default="base",
        help="Use train+val anchors or train-only anchors when collecting context nodes.",
    )
    parser.add_argument(
        "--context-time-filter",
        choices=("any", "past", "near60", "near120"),
        default="any",
        help="Filter anchor-context edges by timestamp before selecting context nodes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated npz and split files.",
    )
    return parser.parse_args()


def _load_flat(path: Path) -> np.ndarray:
    return np.load(path).astype(np.int64, copy=False).reshape(-1)


def _positive_count(y: np.ndarray, ids: np.ndarray) -> int:
    return int(np.sum(np.asarray(y[ids], dtype=np.int32) == 1))


def _select_context_nodes(
    *,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_timestamp: np.ndarray,
    train_original: np.ndarray,
    val_original: np.ndarray,
    threshold_day: int,
    context_limit: int,
    context_anchor: str,
    context_time_filter: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if context_limit == 0:
        return np.empty(0, dtype=np.int64), {
            "context_selection": "none",
            "candidate_count": 0,
            "selected_count": 0,
        }

    num_nodes = int(y.shape[0])
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    base_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_original] = True
    val_mask[val_original] = True
    base_mask[train_original] = True
    base_mask[val_original] = True
    anchor_mask = train_mask if context_anchor == "train" else base_mask

    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    if context_time_filter == "past":
        time_mask = edge_timestamp <= int(threshold_day)
    elif context_time_filter == "near60":
        time_mask = np.abs(edge_timestamp.astype(np.int64) - int(threshold_day)) <= 60
    elif context_time_filter == "near120":
        time_mask = np.abs(edge_timestamp.astype(np.int64) - int(threshold_day)) <= 120
    else:
        time_mask = np.ones(edge_timestamp.shape[0], dtype=bool)
    src_anchor = anchor_mask[src]
    dst_anchor = anchor_mask[dst]
    src_to_context = src_anchor & ~base_mask[dst] & time_mask
    dst_to_context = dst_anchor & ~base_mask[src] & time_mask
    if not np.any(src_to_context | dst_to_context):
        return np.empty(0, dtype=np.int64), {
            "context_selection": "one_hop_ranked",
            "context_anchor": context_anchor,
            "context_time_filter": context_time_filter,
            "candidate_count": 0,
            "selected_count": 0,
        }

    context_nodes = np.concatenate([dst[src_to_context], src[dst_to_context]]).astype(
        np.int64,
        copy=False,
    )
    anchor_nodes = np.concatenate([src[src_to_context], dst[dst_to_context]]).astype(
        np.int64,
        copy=False,
    )
    context_timestamps = np.concatenate(
        [edge_timestamp[src_to_context], edge_timestamp[dst_to_context]]
    ).astype(np.int32, copy=False)

    base_hits = np.bincount(context_nodes, minlength=num_nodes).astype(np.float32, copy=False)
    train_hits = np.bincount(
        context_nodes[train_mask[anchor_nodes]],
        minlength=num_nodes,
    ).astype(np.float32, copy=False)
    train_pos_hits = np.bincount(
        context_nodes[train_mask[anchor_nodes] & (y[anchor_nodes] == 1)],
        minlength=num_nodes,
    ).astype(np.float32, copy=False)
    near_threshold_hits = np.bincount(
        context_nodes[np.abs(context_timestamps.astype(np.int64) - int(threshold_day)) <= 60],
        minlength=num_nodes,
    ).astype(np.float32, copy=False)
    full_degree = np.bincount(edge_index.reshape(-1), minlength=num_nodes).astype(
        np.float32,
        copy=False,
    )

    candidates = np.flatnonzero(base_hits > 0).astype(np.int64, copy=False)
    score = (
        base_hits[candidates]
        + 3.0 * train_hits[candidates]
        + 12.0 * train_pos_hits[candidates]
        + 0.75 * near_threshold_hits[candidates]
    ) / np.log1p(np.maximum(full_degree[candidates], 1.0))
    if context_limit < 0 or candidates.size <= context_limit:
        selected = candidates
    else:
        keep = np.argpartition(score, -int(context_limit))[-int(context_limit):]
        selected = candidates[keep]
    selected = np.sort(selected.astype(np.int64, copy=False))

    stats = {
        "context_selection": "one_hop_ranked",
        "context_anchor": context_anchor,
        "context_time_filter": context_time_filter,
        "context_limit": int(context_limit),
        "candidate_count": int(candidates.size),
        "selected_count": int(selected.size),
        "selected_mean_base_hits": float(np.mean(base_hits[selected])) if selected.size else 0.0,
        "selected_mean_train_hits": float(np.mean(train_hits[selected])) if selected.size else 0.0,
        "selected_mean_train_pos_hits": (
            float(np.mean(train_pos_hits[selected])) if selected.size else 0.0
        ),
        "selected_mean_full_degree": float(np.mean(full_degree[selected])) if selected.size else 0.0,
        "score_min": float(np.min(score[keep])) if context_limit > 0 and candidates.size > context_limit else float(np.min(score)) if score.size else 0.0,
        "score_max": float(np.max(score)) if score.size else 0.0,
    }
    return selected, stats


def _split_payload(
    *,
    source_summary: dict[str, object],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    y: np.ndarray,
    context_stats: dict[str, object],
) -> dict[str, object]:
    threshold_day = int(source_summary.get("threshold_day", -1))
    train_pos = _positive_count(y, train_ids)
    val_pos = _positive_count(y, val_ids)
    return {
        "split_style": "single_graph",
        "train_phase": "graph",
        "val_phase": "graph",
        "external_phase": None,
        "threshold_day": threshold_day,
        "source_dataset": "xinye_dgraph",
        "source_split": "train_ids.npy + val_ids.npy",
        "context": context_stats,
        "train_split": {
            "threshold_day": threshold_day,
            "size": int(train_ids.size),
            "positive_count": train_pos,
            "positive_rate": float(train_pos / max(int(train_ids.size), 1)),
            "id_path": "train_ids.npy",
        },
        "val_split": {
            "threshold_day": threshold_day,
            "size": int(val_ids.size),
            "positive_count": val_pos,
            "positive_rate": float(val_pos / max(int(val_ids.size), 1)),
            "id_path": "val_ids.npy",
        },
        "unlabeled_pool": {
            "size": 0,
            "id_path": "unlabeled_ids.npy",
        },
    }


def main() -> None:
    args = parse_args()
    output_raw = args.output_root / "data/raw" / args.dataset_name / "graph_gdata.npz"
    output_analysis = args.output_root / "outputs/analysis" / args.dataset_name
    summary_path = output_analysis / "recommended_split.json"
    if not args.force and output_raw.exists() and summary_path.exists():
        raise FileExistsError(
            f"{args.dataset_name} already exists. Pass --force to regenerate it."
        )

    raw = np.load(args.source_npz, allow_pickle=False)
    x = np.asarray(raw["x"], dtype=np.float32)
    y = np.asarray(raw["y"], dtype=np.int32).reshape(-1)
    edge_index = np.asarray(raw["edge_index"], dtype=np.int64)
    edge_type = np.asarray(raw["edge_type"], dtype=np.int16).reshape(-1)
    edge_timestamp = np.asarray(raw["edge_timestamp"], dtype=np.int32).reshape(-1)
    train_original = _load_flat(args.source_analysis / "train_ids.npy")
    val_original = _load_flat(args.source_analysis / "val_ids.npy")
    source_summary = json.loads(
        (args.source_analysis / "recommended_split.json").read_text(encoding="utf-8")
    )
    context_original, context_stats = _select_context_nodes(
        y=y,
        edge_index=edge_index,
        edge_timestamp=edge_timestamp,
        train_original=train_original,
        val_original=val_original,
        threshold_day=int(source_summary.get("threshold_day", -1)),
        context_limit=int(args.context_limit),
        context_anchor=str(args.context_anchor),
        context_time_filter=str(args.context_time_filter),
    )
    selected_original = np.unique(
        np.concatenate([train_original, val_original, context_original])
    ).astype(
        np.int64,
        copy=False,
    )

    remap = np.full(y.shape[0], -1, dtype=np.int32)
    remap[selected_original] = np.arange(selected_original.size, dtype=np.int32)
    train_ids = remap[train_original].astype(np.int32, copy=False)
    val_ids = remap[val_original].astype(np.int32, copy=False)
    if np.any(train_ids < 0) or np.any(val_ids < 0):
        raise RuntimeError("Failed to remap all train/val ids.")

    src_local = remap[edge_index[:, 0]]
    dst_local = remap[edge_index[:, 1]]
    internal_edge_mask = (src_local >= 0) & (dst_local >= 0)
    induced_edges = np.column_stack(
        [src_local[internal_edge_mask], dst_local[internal_edge_mask]]
    ).astype(np.int32, copy=False)

    remapped_y = y[selected_original].astype(np.int32, copy=True)
    is_trainval_selected = np.zeros(selected_original.shape[0], dtype=bool)
    is_trainval_selected[train_ids] = True
    is_trainval_selected[val_ids] = True
    remapped_y[~is_trainval_selected] = -100

    contract = PreparedGraphContract(
        x=x[selected_original],
        y=remapped_y,
        edge_index=induced_edges,
        edge_type=edge_type[internal_edge_mask],
        edge_timestamp=edge_timestamp[internal_edge_mask],
        train_mask=train_ids,
        test_mask=np.empty(0, dtype=np.int32),
    )
    save_prepared_graph(output_raw, contract)

    output_analysis.mkdir(parents=True, exist_ok=True)
    np.save(output_analysis / "train_ids.npy", train_ids)
    np.save(output_analysis / "val_ids.npy", val_ids)
    np.save(output_analysis / "unlabeled_ids.npy", np.empty(0, dtype=np.int32))
    np.save(output_analysis / "original_node_ids.npy", selected_original.astype(np.int32))
    np.save(output_analysis / "context_ids.npy", remap[context_original].astype(np.int32, copy=False))
    summary_path.write_text(
        json.dumps(
            _split_payload(
                source_summary=source_summary,
                train_ids=train_ids,
                val_ids=val_ids,
                y=contract.y,
                context_stats=context_stats,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    degree = np.bincount(induced_edges.reshape(-1), minlength=selected_original.size)
    stats = {
        "dataset": args.dataset_name,
        "source_npz": str(args.source_npz),
        "num_nodes": int(selected_original.size),
        "num_edges": int(induced_edges.shape[0]),
        "train_nodes": int(train_ids.size),
        "val_nodes": int(val_ids.size),
        "context_nodes": int(context_original.size),
        **context_stats,
        "isolated_nodes": int(np.sum(degree == 0)),
        "isolated_ratio": float(np.mean(degree == 0)),
        "degree_leq_1_ratio": float(np.mean(degree <= 1)),
        "train_positive_count": _positive_count(contract.y, train_ids),
        "val_positive_count": _positive_count(contract.y, val_ids),
        "dense_float32_adj_tib": float(
            selected_original.size * selected_original.size * 4 / (1024**4)
        ),
    }
    (output_analysis / "dataset_summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
