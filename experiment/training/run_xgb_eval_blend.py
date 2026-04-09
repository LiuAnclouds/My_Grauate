from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


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
            "Fit a GPU XGBoost meta-ensemble on saved prediction files. "
            "Supports in-sample fitting on phase1_val, phase2_external, or both splits."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directories containing phase1_val/phase2_external prediction files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BLEND_OUTPUT_ROOT,
        help="Output directory for blend artifacts.",
    )
    parser.add_argument(
        "--fit-scope",
        choices=("phase1_val_in_sample", "phase2_external_in_sample", "combined_eval_in_sample"),
        default="combined_eval_in_sample",
        help="Which labeled split(s) to fit the meta-model on.",
    )
    parser.add_argument(
        "--include-rank-features",
        action="store_true",
        help="Append per-model normalized rank features in addition to raw probabilities.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-child-weight", type=float, default=4.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()

def _align_bundle(bundle: dict[str, np.ndarray], ref_node_ids: np.ndarray) -> dict[str, np.ndarray]:
    return align_prediction_bundle(bundle, ref_node_ids)


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
    x = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    return x, feature_names


def _build_fit_data(
    fit_scope: str,
    x_val: np.ndarray,
    x_external: np.ndarray,
    y_val: np.ndarray,
    y_external: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if fit_scope == "phase1_val_in_sample":
        return x_val, y_val
    if fit_scope == "phase2_external_in_sample":
        return x_external, y_external
    if fit_scope == "combined_eval_in_sample":
        return (
            np.concatenate([x_val, x_external], axis=0).astype(np.float32, copy=False),
            np.concatenate([y_val, y_external], axis=0).astype(np.int8, copy=False),
        )
    raise ValueError(f"Unsupported fit_scope={fit_scope}")


def _warning_for_scope(fit_scope: str) -> str:
    if fit_scope == "phase1_val_in_sample":
        return (
            "This meta-model is fitted directly on phase1 validation predictions and labels. "
            "Use it as a post-hoc ensemble probe, not as an unbiased validation estimate."
        )
    if fit_scope == "phase2_external_in_sample":
        return (
            "This meta-model is fitted directly on phase2 external predictions and labels. "
            "The reported external metrics are therefore in-sample."
        )
    return (
        "This meta-model is fitted directly on the union of phase1 validation and phase2 external "
        "predictions/labels. Both reported split metrics are therefore in-sample."
    )


def _write_feature_importance(model, feature_names: list[str], path: Path) -> None:
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    rows = []
    for idx, feature_name in enumerate(feature_names):
        rows.append(
            {
                "feature_name": feature_name,
                "gain": float(scores.get(f"f{idx}", 0.0)),
            }
        )
    rows.sort(key=lambda row: row["gain"], reverse=True)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["feature_name", "gain"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    import xgboost as xgb

    run_dir = ensure_dir(args.outdir / args.run_name)

    val_bundles = []
    external_bundles = []
    model_names = []
    base_val_node_ids = None
    base_external_node_ids = None
    base_val_labels = None
    base_external_labels = None

    for model_run_dir in args.run_dirs:
        val_bundle = load_prediction_npz(resolve_prediction_path(model_run_dir, "phase1_val"))
        external_bundle = load_prediction_npz(resolve_prediction_path(model_run_dir, "phase2_external"))

        if base_val_node_ids is None:
            base_val_node_ids = np.asarray(val_bundle["node_ids"], dtype=np.int64)
            base_external_node_ids = np.asarray(external_bundle["node_ids"], dtype=np.int64)
            base_val_labels = np.asarray(val_bundle["y_true"], dtype=np.int8)
            base_external_labels = np.asarray(external_bundle["y_true"], dtype=np.int8)

        aligned_val = _align_bundle(val_bundle, base_val_node_ids)
        aligned_external = _align_bundle(external_bundle, base_external_node_ids)
        if not np.array_equal(aligned_val["y_true"], base_val_labels):
            raise AssertionError(f"{model_run_dir}: phase1_val labels are not aligned.")
        if not np.array_equal(aligned_external["y_true"], base_external_labels):
            raise AssertionError(f"{model_run_dir}: phase2_external labels are not aligned.")

        val_bundles.append(aligned_val)
        external_bundles.append(aligned_external)
        model_names.append(model_run_dir.name)

    x_val, feature_names = _build_feature_matrix(
        val_bundles,
        include_rank_features=bool(args.include_rank_features),
    )
    x_external, _ = _build_feature_matrix(
        external_bundles,
        include_rank_features=bool(args.include_rank_features),
    )
    y_val = np.asarray(base_val_labels, dtype=np.int8)
    y_external = np.asarray(base_external_labels, dtype=np.int8)
    x_fit, y_fit = _build_fit_data(
        fit_scope=str(args.fit_scope),
        x_val=x_val,
        x_external=x_external,
        y_val=y_val,
        y_external=y_external,
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
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
        random_state=int(args.random_state),
        eval_metric="auc",
    )
    model.fit(x_fit, y_fit, verbose=False)

    val_prob = model.predict_proba(x_val)[:, 1].astype(np.float32, copy=False)
    external_prob = model.predict_proba(x_external)[:, 1].astype(np.float32, copy=False)
    val_metrics = compute_binary_classification_metrics(y_val, val_prob)
    external_metrics = compute_binary_classification_metrics(y_external, external_prob)

    save_prediction_npz(
        run_dir / "phase1_val_predictions.npz",
        node_ids=np.asarray(base_val_node_ids, dtype=np.int32),
        y_true=y_val,
        probabilities=val_prob,
    )
    save_prediction_npz(
        run_dir / "phase2_external_predictions.npz",
        node_ids=np.asarray(base_external_node_ids, dtype=np.int32),
        y_true=y_external,
        probabilities=external_prob,
    )
    model.save_model(run_dir / "model.json")
    _write_feature_importance(model, feature_names, run_dir / "feature_importance.csv")

    summary = {
        "blend_name": args.run_name,
        "method": "xgboost_gpu",
        "fit_scope": str(args.fit_scope),
        "warning": _warning_for_scope(str(args.fit_scope)),
        "model_runs": [str(path) for path in args.run_dirs],
        "model_names": model_names,
        "include_rank_features": bool(args.include_rank_features),
        "feature_dim": int(x_val.shape[1]),
        "feature_names": feature_names,
        "fit_rows": int(x_fit.shape[0]),
        "params": {
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
        f"[xgb_eval_blend] run={args.run_name} "
        f"val_auc={val_metrics['auc']:.6f} "
        f"external_auc={external_metrics['auc']:.6f}"
    )


if __name__ == "__main__":
    main()
