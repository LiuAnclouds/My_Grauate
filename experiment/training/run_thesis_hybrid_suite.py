from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import DATASET_ENV_VAR, get_dataset_spec
from experiment.training.thesis_contract import (
    OFFICIAL_HYBRID_BASE_MODEL,
    OFFICIAL_BACKBONE_MODEL,
    OFFICIAL_DATASETS,
    OFFICIAL_HYBRID_BASE_RUN_NAME_TEMPLATE,
    OFFICIAL_HYBRID_BLEND_ALPHA,
    OFFICIAL_HYBRID_GRAPHPROP_BACKGROUND_WEIGHT,
    OFFICIAL_HYBRID_GRAPHPROP_COLSAMPLE_BYTREE,
    OFFICIAL_HYBRID_GRAPHPROP_GAMMA,
    OFFICIAL_HYBRID_GRAPHPROP_LEARNING_RATE,
    OFFICIAL_HYBRID_GRAPHPROP_MAX_DEPTH,
    OFFICIAL_HYBRID_GRAPHPROP_MIN_CHILD_WEIGHT,
    OFFICIAL_HYBRID_GRAPHPROP_N_ESTIMATORS,
    OFFICIAL_HYBRID_GRAPHPROP_PROP_HALF_LIFE_DAYS,
    OFFICIAL_HYBRID_GRAPHPROP_SUBSAMPLE,
    OFFICIAL_HYBRID_RUN_NAME_TEMPLATE,
    OFFICIAL_HYBRID_SECONDARY_MODEL,
    OFFICIAL_HYBRID_SECONDARY_RUN_NAME_TEMPLATE,
    TRANSFORMER_BACKBONE_MODEL,
)

DEFAULT_DATASETS = OFFICIAL_DATASETS

DATASET_SHORT_NAMES = {
    "xinye_dgraph": "xy",
    "elliptic_transactions": "et",
    "ellipticpp_transactions": "epp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the clean GNN-primary thesis hybrid decision layer on multiple datasets."
    )
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--base-model",
        choices=(OFFICIAL_BACKBONE_MODEL, TRANSFORMER_BACKBONE_MODEL),
        default=OFFICIAL_HYBRID_BASE_MODEL,
        help=(
            "Saved thesis GNN backbone family used as the base branch of the hybrid. "
            "Defaults to the recommended teacher-guided `m8_utgt` backbone."
        ),
    )
    parser.add_argument(
        "--base-run-name-template",
        default=OFFICIAL_HYBRID_BASE_RUN_NAME_TEMPLATE,
        help="Per-dataset saved GNN backbone run name template. The default reproduces the recommended teacher-guided UTGT suite.",
    )
    parser.add_argument(
        "--run-name-template",
        default=OFFICIAL_HYBRID_RUN_NAME_TEMPLATE,
        help="Per-dataset hybrid run name template.",
    )
    parser.add_argument(
        "--secondary-run-name-template",
        default=OFFICIAL_HYBRID_SECONDARY_RUN_NAME_TEMPLATE,
        help="Per-dataset graphprop residual run name template.",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=OFFICIAL_HYBRID_BLEND_ALPHA,
        help="Secondary logit weight alpha. Keep `alpha < 0.5` to preserve a strict GNN-primary hybrid.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _dataset_training_root(dataset_name: str) -> Path:
    spec = get_dataset_spec(dataset_name)
    outputs_root = REPO_ROOT / "experiment" / "outputs"
    if spec.uses_legacy_output_layout:
        return outputs_root / "training"
    return outputs_root / spec.output_namespace / "training"


def _dataset_blend_root(dataset_name: str) -> Path:
    return _dataset_training_root(dataset_name) / "blends"


def _base_run_name(args: argparse.Namespace, dataset_name: str) -> str:
    dataset_short = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    return str(args.base_run_name_template).format(
        suite_name=args.suite_name,
        dataset=dataset_name,
        dataset_short=dataset_short,
    )


def _hybrid_run_name(args: argparse.Namespace, dataset_name: str) -> str:
    dataset_short = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    return str(args.run_name_template).format(
        suite_name=args.suite_name,
        dataset=dataset_name,
        dataset_short=dataset_short,
    )


def _secondary_run_name(args: argparse.Namespace, dataset_name: str) -> str:
    dataset_short = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    return str(args.secondary_run_name_template).format(
        suite_name=args.suite_name,
        dataset=dataset_name,
        dataset_short=dataset_short,
    )


def _command_preview(command: list[str], dataset_name: str) -> str:
    return f"{DATASET_ENV_VAR}={shlex.quote(dataset_name)} " + shlex.join(command)


def _run_command(command: list[str], dataset_name: str, *, dry_run: bool) -> None:
    preview = _command_preview(command, dataset_name)
    print(preview)
    if dry_run:
        return
    env = os.environ.copy()
    env[DATASET_ENV_VAR] = dataset_name
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _build_command(
    *,
    base_run_dir: Path,
    secondary_run_dir: Path,
    run_name: str,
    blend_alpha: float,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / "run_thesis_hybrid_blend.py"),
        "--base-run-dir",
        str(base_run_dir),
        "--secondary-run-dir",
        str(secondary_run_dir),
        "--run-name",
        run_name,
        "--blend-alpha",
        str(blend_alpha),
    ]


def _load_summary(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_suite_summary(*, suite_name: str, payload: dict[str, Any]) -> Path:
    suite_dir = REPO_ROOT / "experiment" / "outputs" / "thesis_suite" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        dataset_short = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
        base_run_name = _base_run_name(args, dataset_name)
        secondary_run_name = _secondary_run_name(args, dataset_name)
        hybrid_run_name = _hybrid_run_name(args, dataset_name)
        base_run_dir = _dataset_training_root(dataset_name) / "models" / args.base_model / base_run_name
        secondary_run_dir = _dataset_training_root(dataset_name) / "models" / "xgboost_gpu" / secondary_run_name
        summary_path = _dataset_blend_root(dataset_name) / hybrid_run_name / "summary.json"
        secondary_summary_path = secondary_run_dir / "summary.json"
        if not (args.skip_existing and secondary_summary_path.exists()):
            secondary_command = [
                sys.executable,
                str(REPO_ROOT / "experiment" / "training" / "run_thesis_graphprop_secondary.py"),
                "--run-name",
                secondary_run_name,
            ]
            _run_command(secondary_command, dataset_name, dry_run=args.dry_run)
            if args.dry_run:
                continue
        if args.skip_existing and summary_path.exists():
            summary = _load_summary(summary_path)
        else:
            command = _build_command(
                base_run_dir=base_run_dir,
                secondary_run_dir=secondary_run_dir,
                run_name=hybrid_run_name,
                blend_alpha=float(args.blend_alpha),
            )
            _run_command(command, dataset_name, dry_run=args.dry_run)
            if args.dry_run:
                continue
            summary = _load_summary(summary_path)

        results.append(
            {
                "dataset": dataset_name,
                "dataset_short": dataset_short,
                "base_run_dir": str(base_run_dir.relative_to(REPO_ROOT)),
                "secondary_run_dir": str(secondary_run_dir.relative_to(REPO_ROOT)),
                "run_name": hybrid_run_name,
                "summary_path": str(summary_path.relative_to(REPO_ROOT)),
                "gnn_val_auc": summary.get("gnn_val_auc"),
                "secondary_val_auc": summary.get("secondary_val_auc"),
                "val_auc_mean": summary.get("val_auc_mean"),
            }
        )

    if args.dry_run:
        return

    payload = {
        "suite_name": args.suite_name,
        "family": "thesis_gnn_primary_hybrid",
        "base_model": str(args.base_model),
        "blend_alpha": float(args.blend_alpha),
        "base_run_name_template": str(args.base_run_name_template),
        "dataset_isolation": True,
        "cross_dataset_training": False,
        "same_architecture_across_datasets": True,
        "same_secondary_hyperparameters_across_datasets": True,
        "split_guardrails": [
            "Every dataset loads only its own saved v4 backbone run.",
            "Every dataset fits its own graphprop residual branch on phase1_train only.",
            "No cross-dataset prediction bundle or feature cache is reused.",
        ],
        "secondary_model": {
            "family": OFFICIAL_HYBRID_SECONDARY_MODEL,
            "n_estimators": int(OFFICIAL_HYBRID_GRAPHPROP_N_ESTIMATORS),
            "learning_rate": float(OFFICIAL_HYBRID_GRAPHPROP_LEARNING_RATE),
            "max_depth": int(OFFICIAL_HYBRID_GRAPHPROP_MAX_DEPTH),
            "min_child_weight": float(OFFICIAL_HYBRID_GRAPHPROP_MIN_CHILD_WEIGHT),
            "subsample": float(OFFICIAL_HYBRID_GRAPHPROP_SUBSAMPLE),
            "colsample_bytree": float(OFFICIAL_HYBRID_GRAPHPROP_COLSAMPLE_BYTREE),
            "gamma": float(OFFICIAL_HYBRID_GRAPHPROP_GAMMA),
            "background_weight": float(OFFICIAL_HYBRID_GRAPHPROP_BACKGROUND_WEIGHT),
            "prop_half_life_days": [float(value) for value in OFFICIAL_HYBRID_GRAPHPROP_PROP_HALF_LIFE_DAYS],
        },
        "results": results,
    }
    summary_path = _write_suite_summary(suite_name=args.suite_name, payload=payload)
    print(f"Hybrid suite finished: {summary_path}")


if __name__ == "__main__":
    main()
