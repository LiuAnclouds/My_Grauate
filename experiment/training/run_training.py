from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (
    BLEND_OUTPUT_ROOT,
    FEATURE_OUTPUT_ROOT,
    MODEL_OUTPUT_ROOT,
    ensure_dir,
    load_experiment_split,
    load_phase_arrays,
    save_prediction_npz,
    safe_auc,
    set_global_seed,
    slice_node_ids,
    write_json,
)
from experiment.training.features import (
    FeatureStore,
    build_feature_artifacts,
    build_hybrid_feature_normalizer,
    default_feature_groups,
    load_graph_cache,
)
from experiment.training.gbdt_models import LightGBMExperiment
from experiment.training.gnn_models import (
    GraphPhaseContext,
    GraphModelConfig,
    RelationGraphSAGEExperiment,
    TemporalRelationGraphSAGEExperiment,
)


DEFAULT_SEEDS = [42, 52, 62]
PROMOTION_DELTA_MIN = 0.003
PROMOTION_EXTERNAL_DROP_MAX = 0.001


def _path_repr(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training CLI for the DGraph anti-fraud benchmark framework."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build_features",
        help="Build offline feature caches and graph caches.",
    )
    build_parser.add_argument(
        "--phase",
        default="both",
        choices=("phase1", "phase2", "both"),
        help="Which phase to build.",
    )
    build_parser.add_argument(
        "--outdir",
        type=Path,
        default=FEATURE_OUTPUT_ROOT,
        help="Output directory for feature caches.",
    )
    build_parser.add_argument(
        "--skip-neighbor",
        action="store_true",
        help="Skip 1-hop neighbor aggregation features for a faster build.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train one model family and produce validation/external predictions.",
    )
    train_parser.add_argument(
        "--model",
        required=True,
        choices=(
            "m1_tabular",
            "m2_hybrid",
            "m3_neighbor",
            "m4_graphsage",
            "m5_temporal_graphsage",
        ),
        help="Model family to train.",
    )
    train_parser.add_argument(
        "--run-name",
        default="default",
        help="Subdirectory name for this experiment run.",
    )
    train_parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_OUTPUT_ROOT,
        help="Feature cache directory from build_features.",
    )
    train_parser.add_argument(
        "--outdir",
        type=Path,
        default=MODEL_OUTPUT_ROOT,
        help="Output directory for model checkpoints and predictions.",
    )
    train_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds used for the 3-seed evaluation rule.",
    )
    train_parser.add_argument(
        "--max-train-nodes",
        type=int,
        default=None,
        help="Optional node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--max-val-nodes",
        type=int,
        default=None,
        help="Optional validation node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--max-external-nodes",
        type=int,
        default=None,
        help="Optional phase2 external node cap for smoke tests.",
    )
    train_parser.add_argument(
        "--device",
        default=None,
        help="Torch device for graph models, e.g. cuda or cpu.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Epochs for graph models.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for graph models.",
    )
    train_parser.add_argument(
        "--fanouts",
        type=int,
        nargs="+",
        default=[15, 10],
        help="Neighbor fanouts per layer for graph models.",
    )
    train_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for graph models.",
    )
    train_parser.add_argument(
        "--rel-dim",
        type=int,
        default=32,
        help="Relation embedding dimension for graph models.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for graph models.",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for graph models.",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout for graph models.",
    )
    train_parser.add_argument(
        "--feature-norm",
        choices=("none", "hybrid"),
        default="none",
        help="Feature normalization recipe for graph models.",
    )
    train_parser.add_argument(
        "--norm",
        choices=("none", "layer", "batch"),
        default="none",
        help="Hidden-state normalization for modern graph blocks.",
    )
    train_parser.add_argument(
        "--residual",
        action="store_true",
        help="Enable residual connections in modern graph blocks.",
    )
    train_parser.add_argument(
        "--ffn",
        action="store_true",
        help="Enable FFN layers in modern graph blocks.",
    )
    train_parser.add_argument(
        "--jk",
        choices=("last", "sum"),
        default="last",
        help="Jumping knowledge mode for graph models.",
    )
    train_parser.add_argument(
        "--edge-encoder",
        choices=("basic", "gated"),
        default="basic",
        help="Edge-aware message encoder for modern graph blocks.",
    )
    train_parser.add_argument(
        "--subgraph-head",
        choices=("none", "meanmax"),
        default="none",
        help="Subgraph fusion head for graph models.",
    )
    train_parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Gradient clipping threshold; 0 disables clipping.",
    )
    train_parser.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default="none",
        help="Learning-rate scheduler for graph models.",
    )
    train_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stopping patience; 0 disables early stopping.",
    )

    blend_parser = subparsers.add_parser(
        "blend",
        help="Blend multiple model runs using validation probabilities.",
    )
    blend_parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Model run directories that contain averaged prediction files.",
    )
    blend_parser.add_argument(
        "--outdir",
        type=Path,
        default=BLEND_OUTPUT_ROOT,
        help="Output directory for blending artifacts.",
    )
    blend_parser.add_argument(
        "--name",
        default="m6_blend",
        help="Name for the blending output directory.",
    )
    return parser.parse_args()


def _model_run_dir(outdir: Path, model_name: str, run_name: str) -> Path:
    return ensure_dir(outdir / model_name / run_name)


def _load_labels_for_splits(split) -> tuple[np.ndarray, np.ndarray]:
    phase1 = load_phase_arrays("phase1", keys=("y",))
    phase2 = load_phase_arrays("phase2", keys=("y",))
    return (
        np.asarray(phase1["y"], dtype=np.int8),
        np.asarray(phase2["y"], dtype=np.int8),
    )


def _prepare_split_ids(args: argparse.Namespace):
    split = load_experiment_split()
    train_ids = slice_node_ids(split.train_ids, args.max_train_nodes, seed=11)
    val_ids = slice_node_ids(split.val_ids, args.max_val_nodes, seed=17)
    external_ids = slice_node_ids(split.external_ids, args.max_external_nodes, seed=29)
    return split, train_ids, val_ids, external_ids


def _build_graph_model_config(args: argparse.Namespace) -> GraphModelConfig:
    return GraphModelConfig(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        feature_norm=str(args.feature_norm),
        norm=str(args.norm),
        residual=bool(args.residual),
        ffn=bool(args.ffn),
        jk=str(args.jk),
        edge_encoder=str(args.edge_encoder),
        subgraph_head=str(args.subgraph_head),
        grad_clip=float(args.grad_clip),
        scheduler=str(args.scheduler),
        early_stop_patience=int(args.early_stop_patience),
    )


def _save_average_predictions(
    run_dir: Path,
    split_name: str,
    node_ids: np.ndarray,
    labels: np.ndarray,
    predictions: list[np.ndarray],
) -> Path:
    mean_pred = np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float32, copy=False)
    path = run_dir / f"{split_name}_avg_predictions.npz"
    save_prediction_npz(path, node_ids=node_ids, y_true=labels, probabilities=mean_pred)
    return path


def _benchmark_summary_path(outdir: Path, run_name: str) -> Path:
    return outdir / "m2_hybrid" / run_name / "summary.json"


def _build_promotion_decision(
    model_name: str,
    run_name: str,
    outdir: Path,
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    if model_name in {"m1_tabular", "m2_hybrid"}:
        return {
            "benchmark_reference": None,
            "promoted": model_name == "m2_hybrid",
            "reason": "m2_hybrid defines the internal benchmark" if model_name == "m2_hybrid" else "tabular baseline only",
        }

    benchmark_path = _benchmark_summary_path(outdir, run_name)
    if not benchmark_path.exists():
        return {
            "benchmark_reference": _path_repr(benchmark_path),
            "promoted": False,
            "reason": "benchmark summary not found",
        }
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    val_delta = float(summary_payload["phase1_val_auc_mean"] - benchmark["phase1_val_auc_mean"])
    external_delta = float(summary_payload["phase2_external_auc_mean"] - benchmark["phase2_external_auc_mean"])
    promoted = val_delta >= PROMOTION_DELTA_MIN and external_delta >= -PROMOTION_EXTERNAL_DROP_MAX
    return {
        "benchmark_reference": _path_repr(benchmark_path),
        "val_delta_vs_m2": val_delta,
        "external_delta_vs_m2": external_delta,
        "promoted": promoted,
        "rule": {
            "min_val_delta": PROMOTION_DELTA_MIN,
            "max_external_drop": PROMOTION_EXTERNAL_DROP_MAX,
        },
    }


def run_build_features(args: argparse.Namespace) -> None:
    phases = ["phase1", "phase2"] if args.phase == "both" else [args.phase]
    summary = build_feature_artifacts(
        phases=phases,
        outdir=args.outdir,
        build_neighbor=not args.skip_neighbor,
    )
    write_json(args.outdir / "build_summary.json", summary)
    print(f"Feature build finished: {args.outdir}")


def run_train_lightgbm(args: argparse.Namespace) -> None:
    with tqdm(
        total=3,
        desc=f"prepare:{args.model}",
        unit="step",
        dynamic_ncols=True,
    ) as prep_pbar:
        split, train_ids, val_ids, external_ids = _prepare_split_ids(args)
        prep_pbar.update(1)
        phase1_y, phase2_y = _load_labels_for_splits(split)
        feature_groups = default_feature_groups(args.model)
        prep_pbar.update(1)
        phase1_store = FeatureStore("phase1", feature_groups, outdir=args.feature_dir)
        phase2_store = FeatureStore("phase2", feature_groups, outdir=args.feature_dir)
        prep_pbar.update(1)

    run_dir = _model_run_dir(args.outdir, args.model, args.run_name)
    val_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    with tqdm(
        args.seeds,
        desc=f"train:{args.model}:seeds",
        unit="seed",
        dynamic_ncols=True,
    ) as seed_pbar:
        for seed in seed_pbar:
            set_global_seed(seed)
            seed_dir = ensure_dir(run_dir / f"seed_{seed}")
            model = LightGBMExperiment(
                model_name=args.model,
                seed=seed,
                feature_groups=feature_groups,
            )
            fit_metrics = model.fit(
                train_store=phase1_store,
                train_ids=train_ids,
                train_labels=phase1_y[train_ids],
                val_ids=val_ids,
                val_labels=phase1_y[val_ids],
            )
            val_prob = model.predict_proba(phase1_store, val_ids)
            external_prob = model.predict_proba(phase2_store, external_ids)
            val_auc = safe_auc(phase1_y[val_ids], val_prob)
            external_auc = safe_auc(phase2_y[external_ids], external_prob)
            model.save(seed_dir, feature_names=phase1_store.feature_names)
            save_prediction_npz(seed_dir / "phase1_val_predictions.npz", val_ids, phase1_y[val_ids], val_prob)
            save_prediction_npz(
                seed_dir / "phase2_external_predictions.npz",
                external_ids,
                phase2_y[external_ids],
                external_prob,
            )
            val_predictions.append(val_prob)
            external_predictions.append(external_prob)
            metrics.append(
                {
                    "seed": seed,
                    "val_auc": val_auc,
                    "external_auc": external_auc,
                    "best_iteration": fit_metrics["best_iteration"],
                }
            )
            seed_pbar.set_postfix(
                val_auc=f"{val_auc:.4f}",
                external_auc=f"{external_auc:.4f}",
                refresh=False,
            )
            tqdm.write(
                f"[{args.model}] seed={seed} phase1_val_auc={val_auc:.6f} "
                f"phase2_external_auc={external_auc:.6f}"
            )

    val_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase1_val",
        node_ids=val_ids,
        labels=phase1_y[val_ids],
        predictions=val_predictions,
    )
    external_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase2_external",
        node_ids=external_ids,
        labels=phase2_y[external_ids],
        predictions=external_predictions,
    )

    summary = {
        "model_name": args.model,
        "run_name": args.run_name,
        "feature_groups": feature_groups,
        "seeds": list(args.seeds),
        "phase1_train_size": int(train_ids.size),
        "phase1_val_size": int(val_ids.size),
        "phase2_external_size": int(external_ids.size),
        "phase1_val_auc_mean": float(np.mean([row["val_auc"] for row in metrics])),
        "phase1_val_auc_std": float(np.std([row["val_auc"] for row in metrics])),
        "phase2_external_auc_mean": float(np.mean([row["external_auc"] for row in metrics])),
        "phase2_external_auc_std": float(np.std([row["external_auc"] for row in metrics])),
        "seed_metrics": metrics,
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "phase2_external_avg_predictions": _path_repr(external_avg_path),
    }
    summary["promotion_decision"] = _build_promotion_decision(
        model_name=args.model,
        run_name=args.run_name,
        outdir=args.outdir,
        summary_payload=summary,
    )
    write_json(run_dir / "summary.json", summary)
    print(f"Training finished: {run_dir}")


def _make_graph_contexts(
    feature_dir: Path,
    model_name: str,
    feature_normalizer_state=None,
) -> tuple[GraphPhaseContext, GraphPhaseContext]:
    feature_groups = default_feature_groups(model_name)
    phase1_store = FeatureStore(
        "phase1",
        feature_groups,
        outdir=feature_dir,
        normalizer_state=feature_normalizer_state,
    )
    phase2_store = FeatureStore(
        "phase2",
        feature_groups,
        outdir=feature_dir,
        normalizer_state=feature_normalizer_state,
    )
    phase1_graph = load_graph_cache("phase1", outdir=feature_dir)
    phase2_graph = load_graph_cache("phase2", outdir=feature_dir)
    phase1_y = np.asarray(load_phase_arrays("phase1", keys=("y",))["y"], dtype=np.int8)
    phase2_y = np.asarray(load_phase_arrays("phase2", keys=("y",))["y"], dtype=np.int8)
    return (
        GraphPhaseContext("phase1", phase1_store, phase1_graph, phase1_y),
        GraphPhaseContext("phase2", phase2_store, phase2_graph, phase2_y),
    )


def run_train_graph(args: argparse.Namespace) -> None:
    with tqdm(
        total=3,
        desc=f"prepare:{args.model}",
        unit="step",
        dynamic_ncols=True,
    ) as prep_pbar:
        split, train_ids, val_ids, external_ids = _prepare_split_ids(args)
        feature_groups = default_feature_groups(args.model)
        graph_config = _build_graph_model_config(args)
        prep_pbar.update(1)
        feature_normalizer_state = None
        if graph_config.feature_norm == "hybrid":
            feature_normalizer_state = build_hybrid_feature_normalizer(
                phase="phase1",
                selected_groups=feature_groups,
                train_ids=train_ids,
                outdir=args.feature_dir,
            )
        prep_pbar.update(1)

        phase1_context, phase2_context = _make_graph_contexts(
            args.feature_dir,
            args.model,
            feature_normalizer_state=feature_normalizer_state,
        )
        prep_pbar.update(1)
    run_dir = _model_run_dir(args.outdir, args.model, args.run_name)
    input_dim = phase1_context.feature_store.input_dim
    num_relations = phase1_context.graph_cache.num_relations
    global_max_day = max(phase1_context.graph_cache.max_day, phase2_context.graph_cache.max_day)
    val_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    experiment_cls = (
        TemporalRelationGraphSAGEExperiment if args.model == "m5_temporal_graphsage" else RelationGraphSAGEExperiment
    )

    with tqdm(
        args.seeds,
        desc=f"train:{args.model}:seeds",
        unit="seed",
        dynamic_ncols=True,
    ) as seed_pbar:
        for seed in seed_pbar:
            set_global_seed(seed)
            seed_dir = ensure_dir(run_dir / f"seed_{seed}")
            experiment = experiment_cls(
                model_name=args.model,
                seed=seed,
                input_dim=input_dim,
                num_relations=num_relations,
                max_day=global_max_day,
                feature_groups=phase1_context.feature_store.selected_groups,
                hidden_dim=args.hidden_dim,
                num_layers=len(args.fanouts),
                rel_dim=args.rel_dim,
                fanouts=list(args.fanouts),
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=args.device,
                graph_config=graph_config,
                feature_normalizer_state=feature_normalizer_state,
            )
            fit_metrics = experiment.fit(
                context=phase1_context,
                train_ids=train_ids,
                val_ids=val_ids,
            )
            val_prob = experiment.predict_proba(
                phase1_context,
                val_ids,
                batch_seed=seed + 1000,
                progress_desc=f"{args.model}:seed{seed}:phase1_val",
            )
            external_prob = experiment.predict_proba(
                phase2_context,
                external_ids,
                batch_seed=seed + 2000,
                progress_desc=f"{args.model}:seed{seed}:phase2_external",
            )
            val_auc = safe_auc(phase1_context.labels[val_ids], val_prob)
            external_auc = safe_auc(phase2_context.labels[external_ids], external_prob)
            experiment.save(seed_dir)
            save_prediction_npz(
                seed_dir / "phase1_val_predictions.npz",
                val_ids,
                phase1_context.labels[val_ids],
                val_prob,
            )
            save_prediction_npz(
                seed_dir / "phase2_external_predictions.npz",
                external_ids,
                phase2_context.labels[external_ids],
                external_prob,
            )
            val_predictions.append(val_prob)
            external_predictions.append(external_prob)
            metrics.append(
                {
                    "seed": seed,
                    "val_auc": val_auc,
                    "external_auc": external_auc,
                    "best_epoch": fit_metrics["best_epoch"],
                }
            )
            seed_pbar.set_postfix(
                val_auc=f"{val_auc:.4f}",
                external_auc=f"{external_auc:.4f}",
                refresh=False,
            )
            tqdm.write(
                f"[{args.model}] seed={seed} phase1_val_auc={val_auc:.6f} "
                f"phase2_external_auc={external_auc:.6f}"
            )

    val_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase1_val",
        node_ids=val_ids,
        labels=phase1_context.labels[val_ids],
        predictions=val_predictions,
    )
    external_avg_path = _save_average_predictions(
        run_dir=run_dir,
        split_name="phase2_external",
        node_ids=external_ids,
        labels=phase2_context.labels[external_ids],
        predictions=external_predictions,
    )
    summary = {
        "model_name": args.model,
        "run_name": args.run_name,
        "feature_groups": phase1_context.feature_store.selected_groups,
        "seeds": list(args.seeds),
        "phase1_train_size": int(train_ids.size),
        "phase1_val_size": int(val_ids.size),
        "phase2_external_size": int(external_ids.size),
        "phase1_val_auc_mean": float(np.mean([row["val_auc"] for row in metrics])),
        "phase1_val_auc_std": float(np.std([row["val_auc"] for row in metrics])),
        "phase2_external_auc_mean": float(np.mean([row["external_auc"] for row in metrics])),
        "phase2_external_auc_std": float(np.std([row["external_auc"] for row in metrics])),
        "seed_metrics": metrics,
        "phase1_val_avg_predictions": _path_repr(val_avg_path),
        "phase2_external_avg_predictions": _path_repr(external_avg_path),
        "graph_config": {
            "input_dim": input_dim,
            "num_relations": num_relations,
            "hidden_dim": args.hidden_dim,
            "rel_dim": args.rel_dim,
            "fanouts": list(args.fanouts),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": args.device,
            **graph_config.to_dict(),
        },
    }
    summary["promotion_decision"] = _build_promotion_decision(
        model_name=args.model,
        run_name=args.run_name,
        outdir=args.outdir,
        summary_payload=summary,
    )
    write_json(run_dir / "summary.json", summary)
    print(f"Training finished: {run_dir}")


def _load_prediction_bundle(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "node_ids": np.asarray(data["node_ids"], dtype=np.int32),
        "y_true": np.asarray(data["y_true"], dtype=np.int8),
        "probability": np.asarray(data["probability"], dtype=np.float32),
    }


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def run_blend(args: argparse.Namespace) -> None:
    run_dir = ensure_dir(args.outdir / args.name)
    val_bundles = []
    external_bundles = []
    model_names = []
    for model_run_dir in args.run_dirs:
        val_path = model_run_dir / "phase1_val_avg_predictions.npz"
        external_path = model_run_dir / "phase2_external_avg_predictions.npz"
        if not val_path.exists() or not external_path.exists():
            raise FileNotFoundError(
                f"Missing averaged prediction files in {model_run_dir}. Expected "
                f"{val_path.name} and {external_path.name}."
            )
        val_bundles.append(_load_prediction_bundle(val_path))
        external_bundles.append(_load_prediction_bundle(external_path))
        model_names.append(model_run_dir.parent.name)

    base_val = val_bundles[0]
    base_external = external_bundles[0]
    if any(not np.array_equal(bundle["node_ids"], base_val["node_ids"]) for bundle in val_bundles[1:]):
        raise AssertionError("Validation node ids are not aligned across model runs.")
    if any(not np.array_equal(bundle["node_ids"], base_external["node_ids"]) for bundle in external_bundles[1:]):
        raise AssertionError("External node ids are not aligned across model runs.")

    val_matrix = np.column_stack([_logit(bundle["probability"]) for bundle in val_bundles])
    external_matrix = np.column_stack([_logit(bundle["probability"]) for bundle in external_bundles])
    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=0,
    )
    logistic.fit(val_matrix, base_val["y_true"])
    val_prob = logistic.predict_proba(val_matrix)[:, 1].astype(np.float32, copy=False)
    external_prob = logistic.predict_proba(external_matrix)[:, 1].astype(np.float32, copy=False)

    val_auc = safe_auc(base_val["y_true"], val_prob)
    external_auc = safe_auc(base_external["y_true"], external_prob)
    save_prediction_npz(
        run_dir / "phase1_val_blend_predictions.npz",
        base_val["node_ids"],
        base_val["y_true"],
        val_prob,
    )
    save_prediction_npz(
        run_dir / "phase2_external_blend_predictions.npz",
        base_external["node_ids"],
        base_external["y_true"],
        external_prob,
    )
    summary = {
        "blend_name": args.name,
        "model_runs": [_path_repr(path) for path in args.run_dirs],
        "model_names": model_names,
        "coefficients": logistic.coef_.reshape(-1).astype(float).tolist(),
        "intercept": float(logistic.intercept_[0]),
        "phase1_val_auc": val_auc,
        "phase2_external_auc": external_auc,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Blend finished: {run_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "build_features":
        run_build_features(args)
        return
    if args.command == "train":
        if args.model in {"m1_tabular", "m2_hybrid", "m3_neighbor"}:
            run_train_lightgbm(args)
            return
        run_train_graph(args)
        return
    if args.command == "blend":
        run_blend(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
