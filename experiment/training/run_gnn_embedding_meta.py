from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (  # noqa: E402
    BLEND_OUTPUT_ROOT,
    FEATURE_OUTPUT_ROOT,
    compute_binary_classification_metrics,
    ensure_dir,
    load_experiment_split,
    load_phase_arrays,
    save_prediction_npz,
    set_global_seed,
    slice_node_ids,
    write_json,
)
from experiment.training.features import (  # noqa: E402
    FeatureStore,
    build_hybrid_feature_normalizer,
    resolve_feature_groups,
)
from experiment.training.gnn_models import GraphModelConfig  # noqa: E402
from experiment.training.graph_runtime import (  # noqa: E402
    build_graph_label_artifacts,
    make_graph_contexts,
    resolve_graph_experiment_class,
)


DEFAULT_CONTEXT_EXTRA_GROUPS = [
    "temporal_snapshot",
    "temporal_recent",
    "temporal_relation_recent",
    "temporal_bucket_norm",
    "graph_stats",
    "neighbor_similarity",
    "activation_early",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a second-stage meta model on saved GNN embeddings and optional "
            "tabular context features without touching validation/external labels."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--gnn-model-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more saved graph model seed directories, e.g. .../seed_42",
    )
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_OUTPUT_ROOT)
    parser.add_argument("--outdir", type=Path, default=BLEND_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--context-model",
        choices=("none", "m2_hybrid", "m3_neighbor"),
        default="m3_neighbor",
    )
    parser.add_argument(
        "--context-extra-groups",
        nargs="*",
        default=DEFAULT_CONTEXT_EXTRA_GROUPS,
        help="Extra groups appended to --context-model when context features are enabled.",
    )
    parser.add_argument(
        "--context-feature-norm",
        choices=("none", "hybrid"),
        default="none",
        help="Optional feature normalization for the tabular context block.",
    )
    parser.add_argument("--append-gnn-prob", action="store_true")
    parser.add_argument(
        "--meta-model",
        choices=("xgboost", "logistic", "histgb"),
        default="xgboost",
    )
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
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--time-weight-half-life-days", type=float, default=0.0)
    parser.add_argument("--time-weight-floor", type=float, default=0.25)
    parser.add_argument("--max-train-nodes", type=int, default=None)
    parser.add_argument("--max-val-nodes", type=int, default=None)
    parser.add_argument("--max-external-nodes", type=int, default=None)
    parser.add_argument("--save-meta-features", action="store_true")
    return parser.parse_args()


def _fit_meta_model(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
):
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos_weight = neg_count / max(pos_count, 1.0)

    if args.meta_model == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            max_iter=5000,
            C=float(args.logistic_c),
            class_weight="balanced",
            random_state=int(args.seed),
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return model, {"scale_pos_weight": scale_pos_weight}

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
            random_state=int(args.seed),
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
        random_state=int(args.seed),
    )
    model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
    return model, {"scale_pos_weight": scale_pos_weight}


def _predict_meta_model(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(np.float32, copy=False)


def _build_context_blocks(
    *,
    args: argparse.Namespace,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    external_ids: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, object]]:
    if args.context_model == "none":
        return None, None, None, {
            "context_model": None,
            "context_extra_groups": [],
            "context_feature_norm": "none",
            "context_feature_dim": 0,
        }

    feature_groups = resolve_feature_groups(str(args.context_model), list(args.context_extra_groups))
    normalizer_state = None
    if args.context_feature_norm == "hybrid":
        normalizer_state = build_hybrid_feature_normalizer(
            phase="phase1",
            selected_groups=feature_groups,
            train_ids=train_ids,
            outdir=args.feature_dir,
        )
    phase1_store = FeatureStore(
        "phase1",
        feature_groups,
        outdir=args.feature_dir,
        normalizer_state=normalizer_state,
    )
    phase2_store = FeatureStore(
        "phase2",
        feature_groups,
        outdir=args.feature_dir,
        normalizer_state=normalizer_state,
    )
    train_block = phase1_store.take_rows(train_ids).astype(np.float32, copy=False)
    val_block = phase1_store.take_rows(val_ids).astype(np.float32, copy=False)
    external_block = phase2_store.take_rows(external_ids).astype(np.float32, copy=False)
    return train_block, val_block, external_block, {
        "context_model": str(args.context_model),
        "context_extra_groups": list(args.context_extra_groups),
        "context_feature_norm": str(args.context_feature_norm),
        "context_feature_dim": int(train_block.shape[1]),
    }


def _build_time_weight(
    *,
    threshold_day: int,
    first_active: np.ndarray,
    train_ids: np.ndarray,
    half_life_days: float,
    floor: float,
) -> np.ndarray | None:
    if half_life_days <= 0.0:
        return None
    train_first_active = np.asarray(first_active[np.asarray(train_ids, dtype=np.int32)], dtype=np.float32)
    age = np.clip(float(threshold_day) - train_first_active, 0.0, None)
    decay = np.power(np.float32(0.5), age / float(half_life_days)).astype(np.float32, copy=False)
    floor = float(np.clip(floor, 0.0, 1.0))
    weight = (floor + (1.0 - floor) * decay).astype(np.float32, copy=False)
    mean_weight = float(np.mean(weight, dtype=np.float64))
    if mean_weight > 0.0:
        weight = weight / mean_weight
    return weight


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    split = load_experiment_split()
    train_ids = slice_node_ids(np.asarray(split.train_ids, dtype=np.int32), args.max_train_nodes, args.seed)
    val_ids = slice_node_ids(np.asarray(split.val_ids, dtype=np.int32), args.max_val_nodes, args.seed + 1)
    external_ids = slice_node_ids(
        np.asarray(split.external_ids, dtype=np.int32),
        args.max_external_nodes,
        args.seed + 2,
    )
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    y_train = phase1_y[train_ids].astype(np.int8, copy=False)
    y_val = phase1_y[val_ids].astype(np.int8, copy=False)
    y_external = phase2_y[external_ids].astype(np.int8, copy=False)

    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    external_blocks: list[np.ndarray] = []
    feature_breakdown: list[dict[str, object]] = []
    gnn_model_summaries: list[dict[str, object]] = []
    phase1_graph_for_weight = None

    for model_idx, gnn_model_dir in enumerate(args.gnn_model_dir):
        meta = json.loads((gnn_model_dir / "model_meta.json").read_text(encoding="utf-8"))
        model_name = str(meta["model_name"])
        selected_groups = list(meta["feature_groups"])
        graph_config = GraphModelConfig.from_dict(meta.get("graph_model_config"))
        label_artifacts = build_graph_label_artifacts(
            feature_dir=args.feature_dir,
            split_train_ids=np.asarray(split.train_ids, dtype=np.int32),
            threshold_day=int(split.threshold_day),
            known_label_feature=graph_config.known_label_feature,
            include_historical_background_negatives=graph_config.include_historical_background_negatives,
        )
        feature_normalizer_state = meta.get("feature_normalizer_state")
        phase1_context, phase2_context = make_graph_contexts(
            feature_dir=args.feature_dir,
            model_name=model_name,
            selected_groups=selected_groups,
            feature_normalizer_state=feature_normalizer_state,
            phase1_known_label_codes=label_artifacts["phase1_known_label_codes"],
            phase2_known_label_codes=label_artifacts["phase2_known_label_codes"],
            phase1_reference_day=int(split.threshold_day),
            phase2_reference_day=None,
            phase1_historical_background_ids=label_artifacts["phase1_historical_background_ids"],
        )
        if phase1_graph_for_weight is None:
            phase1_graph_for_weight = phase1_context.graph_cache

        state_dict = torch.load(
            gnn_model_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_input_dim = int(state_dict["input_proj.weight"].shape[1])
        base_feature_dim = int(phase1_context.feature_store.input_dim)
        inferred_label_feature_dim = max(checkpoint_input_dim - base_feature_dim, 0)
        if graph_config.known_label_feature:
            object.__setattr__(graph_config, "known_label_feature_dim", inferred_label_feature_dim)
        num_relations = phase1_context.graph_cache.num_relations
        experiment_cls = resolve_graph_experiment_class(model_name)
        experiment = experiment_cls.load(
            gnn_model_dir,
            input_dim=checkpoint_input_dim,
            num_relations=num_relations,
            device=args.device,
        )
        if graph_config.known_label_feature:
            object.__setattr__(
                experiment.graph_config,
                "known_label_feature_dim",
                inferred_label_feature_dim,
            )

        print(
            f"[gnn_embedding_meta] extracting from {gnn_model_dir} "
            f"model={model_name} train={train_ids.size} val={val_ids.size} external={external_ids.size}",
            flush=True,
        )
        train_prob, train_emb = experiment.predict_proba_and_embeddings(
            phase1_context,
            train_ids,
            batch_seed=args.seed + 1000 + model_idx * 100,
            progress_desc=f"{model_name}:meta{model_idx}:phase1_train",
        )
        val_prob, val_emb = experiment.predict_proba_and_embeddings(
            phase1_context,
            val_ids,
            batch_seed=args.seed + 2000 + model_idx * 100,
            progress_desc=f"{model_name}:meta{model_idx}:phase1_val",
        )
        external_prob, external_emb = experiment.predict_proba_and_embeddings(
            phase2_context,
            external_ids,
            batch_seed=args.seed + 3000 + model_idx * 100,
            progress_desc=f"{model_name}:meta{model_idx}:phase2_external",
        )

        train_blocks.append(train_emb)
        val_blocks.append(val_emb)
        external_blocks.append(external_emb)
        feature_breakdown.append(
            {
                "name": f"gnn_embedding_{model_idx}",
                "dim": int(train_emb.shape[1]),
                "source": str(gnn_model_dir),
            }
        )
        if args.append_gnn_prob:
            train_blocks.append(train_prob.reshape(-1, 1))
            val_blocks.append(val_prob.reshape(-1, 1))
            external_blocks.append(external_prob.reshape(-1, 1))
            feature_breakdown.append(
                {
                    "name": f"gnn_probability_{model_idx}",
                    "dim": 1,
                    "source": str(gnn_model_dir),
                }
            )

        gnn_model_summaries.append(
            {
                "model_dir": str(gnn_model_dir),
                "model_name": model_name,
                "selected_feature_groups": selected_groups,
                "graph_config": graph_config.to_dict(),
            }
        )

    ctx_train, ctx_val, ctx_external, context_meta = _build_context_blocks(
        args=args,
        train_ids=train_ids,
        val_ids=val_ids,
        external_ids=external_ids,
    )
    if ctx_train is not None and ctx_val is not None and ctx_external is not None:
        train_blocks.append(ctx_train)
        val_blocks.append(ctx_val)
        external_blocks.append(ctx_external)
        feature_breakdown.append(
            {
                "name": "context_features",
                "dim": int(ctx_train.shape[1]),
                "model": context_meta["context_model"],
            }
        )

    x_train = np.concatenate(train_blocks, axis=1).astype(np.float32, copy=False)
    x_val = np.concatenate(val_blocks, axis=1).astype(np.float32, copy=False)
    x_external = np.concatenate(external_blocks, axis=1).astype(np.float32, copy=False)
    train_sample_weight = _build_time_weight(
        threshold_day=int(split.threshold_day),
        first_active=np.asarray(phase1_graph_for_weight.first_active, dtype=np.int32),
        train_ids=train_ids,
        half_life_days=float(args.time_weight_half_life_days),
        floor=float(args.time_weight_floor),
    )

    print(
        f"[gnn_embedding_meta] fitting meta_model={args.meta_model} feature_dim={x_train.shape[1]}",
        flush=True,
    )
    model, meta_aux = _fit_meta_model(
        args=args,
        x_train=x_train,
        y_train=y_train,
        sample_weight=train_sample_weight,
    )
    train_meta_prob = _predict_meta_model(model, x_train)
    val_meta_prob = _predict_meta_model(model, x_val)
    external_meta_prob = _predict_meta_model(model, x_external)

    train_metrics = compute_binary_classification_metrics(y_train, train_meta_prob)
    val_metrics = compute_binary_classification_metrics(y_val, val_meta_prob)
    external_metrics = compute_binary_classification_metrics(y_external, external_meta_prob)

    run_dir = ensure_dir(args.outdir / args.run_name)
    save_prediction_npz(run_dir / "phase1_train_predictions.npz", train_ids, y_train, train_meta_prob)
    save_prediction_npz(run_dir / "phase1_val_predictions.npz", val_ids, y_val, val_meta_prob)
    save_prediction_npz(
        run_dir / "phase2_external_predictions.npz",
        external_ids,
        y_external,
        external_meta_prob,
    )
    if args.save_meta_features:
        np.savez_compressed(
            run_dir / "phase1_train_meta_features.npz",
            node_ids=train_ids,
            y_true=y_train,
            features=x_train,
        )
        np.savez_compressed(
            run_dir / "phase1_val_meta_features.npz",
            node_ids=val_ids,
            y_true=y_val,
            features=x_val,
        )
        np.savez_compressed(
            run_dir / "phase2_external_meta_features.npz",
            node_ids=external_ids,
            y_true=y_external,
            features=x_external,
        )

    if args.meta_model == "xgboost":
        model.save_model(run_dir / "model.json")
    else:
        with (run_dir / "model.pkl").open("wb") as fp:
            pickle.dump(model, fp)

    summary = {
        "blend_name": args.run_name,
        "method": str(args.meta_model),
        "fit_scope": "phase1_train_only",
        "warning": (
            "The second-stage model is fitted only on phase1 train labels. "
            "phase1_val and phase2_external are used strictly for evaluation."
        ),
        "gnn_model_dirs": [str(path) for path in args.gnn_model_dir],
        "gnn_models": gnn_model_summaries,
        "gnn_model_dir": str(args.gnn_model_dir[0]),
        "gnn_model_name": str(gnn_model_summaries[0]["model_name"]),
        "gnn_selected_feature_groups": list(gnn_model_summaries[0]["selected_feature_groups"]),
        "gnn_graph_config": dict(gnn_model_summaries[0]["graph_config"]),
        "append_gnn_prob": bool(args.append_gnn_prob),
        "feature_breakdown": feature_breakdown,
        "feature_dim": int(x_train.shape[1]),
        **context_meta,
        "time_weight": {
            "enabled": train_sample_weight is not None,
            "half_life_days": None if train_sample_weight is None else float(args.time_weight_half_life_days),
            "floor": None if train_sample_weight is None else float(args.time_weight_floor),
        },
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
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "l2_regularization": float(args.l2_regularization),
            "time_weight_half_life_days": float(args.time_weight_half_life_days),
            "time_weight_floor": float(args.time_weight_floor),
            **meta_aux,
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[gnn_embedding_meta] run={args.run_name} "
        f"train_auc={train_metrics['auc']:.6f} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
