from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.common import (  # noqa: E402
    BLEND_OUTPUT_ROOT,
    align_prediction_bundle,
    compute_binary_classification_metrics,
    ensure_dir,
    load_prediction_npz,
    resolve_prediction_path,
    save_prediction_npz,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a HistGradientBoosting meta-ensemble on validation predictions from existing runs. "
            "This is a validation-fitted blend intended for thesis-style ensemble experiments."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Model run directories containing phase1_val/phase2_external prediction files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BLEND_OUTPUT_ROOT,
        help="Output directory for blend artifacts.",
    )
    parser.add_argument(
        "--include-rank-features",
        action="store_true",
        help="Append per-model normalized rank features in addition to raw probabilities.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()

def _rank_norm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    return ranks


def _build_feature_matrix(
    bundles: list[dict[str, np.ndarray]],
    include_rank_features: bool,
) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    for idx, bundle in enumerate(bundles):
        prob = np.asarray(bundle["probability"], dtype=np.float32)
        blocks.append(prob.reshape(-1, 1))
        feature_names.append(f"model_{idx}_prob")
        if include_rank_features:
            blocks.append(_rank_norm(prob).reshape(-1, 1))
            feature_names.append(f"model_{idx}_rank")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), feature_names


def main() -> None:
    args = parse_args()
    run_dir = ensure_dir(args.outdir / args.run_name)

    val_bundles = []
    external_bundles = []
    model_names = []
    for model_run_dir in args.run_dirs:
        val_bundles.append(load_prediction_npz(resolve_prediction_path(model_run_dir, "phase1_val")))
        external_bundles.append(load_prediction_npz(resolve_prediction_path(model_run_dir, "phase2_external")))
        model_names.append(model_run_dir.parent.name)

    base_val = val_bundles[0]
    base_external = external_bundles[0]
    for idx in range(1, len(val_bundles)):
        if not np.array_equal(val_bundles[idx]["node_ids"], base_val["node_ids"]):
            val_bundles[idx] = align_prediction_bundle(val_bundles[idx], base_val["node_ids"])
        if not np.array_equal(external_bundles[idx]["node_ids"], base_external["node_ids"]):
            external_bundles[idx] = align_prediction_bundle(external_bundles[idx], base_external["node_ids"])
        if not np.array_equal(val_bundles[idx]["y_true"], base_val["y_true"]):
            raise AssertionError("Validation labels are not aligned across model runs.")
        if not np.array_equal(external_bundles[idx]["y_true"], base_external["y_true"]):
            raise AssertionError("External labels are not aligned across model runs.")

    x_val, feature_names = _build_feature_matrix(val_bundles, include_rank_features=bool(args.include_rank_features))
    x_external, _ = _build_feature_matrix(
        external_bundles,
        include_rank_features=bool(args.include_rank_features),
    )
    y_val = base_val["y_true"]
    y_external = base_external["y_true"]

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2_regularization=float(args.l2_regularization),
        max_iter=int(args.max_iter),
        early_stopping=False,
        random_state=int(args.random_state),
    )
    model.fit(x_val, y_val)

    val_prob = model.predict_proba(x_val)[:, 1].astype(np.float32, copy=False)
    external_prob = model.predict_proba(x_external)[:, 1].astype(np.float32, copy=False)
    val_metrics = compute_binary_classification_metrics(y_val, val_prob)
    external_metrics = compute_binary_classification_metrics(y_external, external_prob)

    save_prediction_npz(run_dir / "phase1_val_predictions.npz", base_val["node_ids"], y_val, val_prob)
    save_prediction_npz(
        run_dir / "phase2_external_predictions.npz",
        base_external["node_ids"],
        y_external,
        external_prob,
    )
    with (run_dir / "model.pkl").open("wb") as fp:
        pickle.dump(model, fp)

    summary = {
        "blend_name": args.run_name,
        "method": "hist_gradient_boosting",
        "fit_scope": "phase1_val_in_sample",
        "warning": (
            "This meta-model is fitted directly on phase1 validation predictions and labels. "
            "Use it as a thesis-style ensemble probe, not as an unbiased validation estimate."
        ),
        "model_runs": [str(path) for path in args.run_dirs],
        "model_names": model_names,
        "include_rank_features": bool(args.include_rank_features),
        "feature_dim": int(x_val.shape[1]),
        "feature_names": feature_names,
        "params": {
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "l2_regularization": float(args.l2_regularization),
            "max_iter": int(args.max_iter),
            "random_state": int(args.random_state),
        },
        "phase1_val_auc": val_metrics["auc"],
        "phase1_val_pr_auc": val_metrics["pr_auc"],
        "phase1_val_ap": val_metrics["ap"],
        "phase2_external_auc": external_metrics["auc"],
        "phase2_external_pr_auc": external_metrics["pr_auc"],
        "phase2_external_ap": external_metrics["ap"],
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[hgb_blend] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f}"
    )


if __name__ == "__main__":
    main()
