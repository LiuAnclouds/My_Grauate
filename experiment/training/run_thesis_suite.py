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
    OFFICIAL_BACKBONE_FEATURE_PROFILE,
    OFFICIAL_BACKBONE_MODEL,
    OFFICIAL_BACKBONE_PRESET,
    OFFICIAL_DATASETS,
    OFFICIAL_MAINLINE_BATCH_SIZE,
    OFFICIAL_MAINLINE_FANOUTS,
    OFFICIAL_MAINLINE_HIDDEN_DIM,
    OFFICIAL_MAINLINE_REL_DIM,
    OFFICIAL_SUITE_EPOCHS,
    OFFICIAL_SUITE_SEEDS,
)

DEFAULT_DATASETS = OFFICIAL_DATASETS

DATASET_SHORT_NAMES = {
    "xinye_dgraph": "xy",
    "elliptic_transactions": "et",
    "ellipticpp_transactions": "epp",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official thesis mainline on multiple datasets with one unified experiment contract."
        )
    )
    parser.add_argument(
        "--suite-name",
        required=True,
        help="Logical suite name used for the aggregate summary directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset registry keys to run in sequence.",
    )
    parser.add_argument(
        "--model",
        choices=("m5_temporal_graphsage", "m7_utpm"),
        default=OFFICIAL_BACKBONE_MODEL,
        help="Unified thesis-mainline model family.",
    )
    parser.add_argument(
        "--preset",
        default=OFFICIAL_BACKBONE_PRESET,
        help="Preset passed through to run_thesis_mainline.py.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=("utpm_unified", "utpm_shift_compact", "utpm_shift_enhanced"),
        default=OFFICIAL_BACKBONE_FEATURE_PROFILE,
        help="Shared feature contract for all datasets in the suite.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device passed through to the mainline trainer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=OFFICIAL_SUITE_EPOCHS,
        help="Epoch count used for every dataset in this suite.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=OFFICIAL_MAINLINE_BATCH_SIZE,
        help="Batch size used for every dataset in this suite.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=OFFICIAL_MAINLINE_HIDDEN_DIM,
        help="Hidden dimension used for every dataset in this suite.",
    )
    parser.add_argument(
        "--rel-dim",
        type=int,
        default=OFFICIAL_MAINLINE_REL_DIM,
        help="Relation embedding dimension used for every dataset in this suite.",
    )
    parser.add_argument(
        "--fanouts",
        nargs="+",
        type=int,
        default=list(OFFICIAL_MAINLINE_FANOUTS),
        help="Neighbor fanouts used for every dataset in this suite.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(OFFICIAL_SUITE_SEEDS),
        help="Seeds used for every dataset in this suite.",
    )
    parser.add_argument(
        "--graph-config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable low-level GraphModelConfig override forwarded to the mainline trainer.",
    )
    parser.add_argument(
        "--run-name-template",
        default="{suite_name}_{dataset_short}",
        help=(
            "Per-dataset run-name template. Available fields: "
            "{suite_name}, {dataset}, {dataset_short}, {model}, {preset}, {epochs}."
        ),
    )
    parser.add_argument(
        "--build-features",
        action="store_true",
        help="Build feature caches for each dataset before training.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset when the target summary.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def _dataset_training_root(dataset_name: str) -> Path:
    spec = get_dataset_spec(dataset_name)
    outputs_root = REPO_ROOT / "experiment" / "outputs"
    if spec.uses_legacy_output_layout:
        return outputs_root / "training"
    return outputs_root / spec.output_namespace / "training"


def _run_name_for_dataset(args: argparse.Namespace, dataset_name: str) -> str:
    dataset_short = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    return str(args.run_name_template).format(
        suite_name=args.suite_name,
        dataset=dataset_name,
        dataset_short=dataset_short,
        model=args.model,
        preset=args.preset,
        epochs=args.epochs,
    )


def _command_preview(command: list[str], dataset_name: str) -> str:
    return f"{DATASET_ENV_VAR}={shlex.quote(dataset_name)} " + shlex.join(command)


def _build_feature_command() -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / "run_thesis_mainline.py"),
        "build_features",
        "--phase",
        "both",
    ]


def _build_train_command(
    *,
    args: argparse.Namespace,
    run_name: str,
) -> list[str]:
    command: list[str] = [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / "run_thesis_mainline.py"),
        "train",
        "--model",
        args.model,
        "--preset",
        args.preset,
        "--feature-profile",
        args.feature_profile,
        "--run-name",
        run_name,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--rel-dim",
        str(args.rel_dim),
        "--fanouts",
        *[str(v) for v in args.fanouts],
        "--seeds",
        *[str(v) for v in args.seeds],
    ]
    for override in args.graph_config_override:
        command.extend(["--graph-config-override", str(override)])
    return command


def _run_command(command: list[str], dataset_name: str, *, dry_run: bool) -> None:
    preview = _command_preview(command, dataset_name)
    print(preview)
    if dry_run:
        return
    env = os.environ.copy()
    env[DATASET_ENV_VAR] = dataset_name
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _load_run_summary(dataset_name: str, model_name: str, run_name: str) -> tuple[Path, dict[str, Any]]:
    summary_path = _dataset_training_root(dataset_name) / "models" / model_name / run_name / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary_path, payload


def _write_suite_summary(
    *,
    suite_name: str,
    payload: dict[str, Any],
) -> Path:
    suite_dir = REPO_ROOT / "experiment" / "outputs" / "thesis_suite" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Thesis Suite: {suite_name}",
        "",
        "| Dataset | Run | Val AUC | Test AUC | External AUC | Summary |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["results"]:
        lines.append(
            "| {dataset} | {run_name} | {val_auc} | {test_auc} | {external_auc} | {summary_path} |".format(
                dataset=row["dataset"],
                run_name=row["run_name"],
                val_auc=_format_metric(row.get("val_auc_mean")),
                test_auc=_format_metric(row.get("test_auc_mean")),
                external_auc=_format_metric(row.get("external_auc_mean")),
                summary_path=row["summary_path"],
            )
        )
    (suite_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    if str(args.model) == OFFICIAL_BACKBONE_MODEL and str(args.preset) != OFFICIAL_BACKBONE_PRESET:
        raise ValueError(
            "The official thesis suite is locked to the unified m7 v4 backbone. "
            "Use `run_thesis_mainline.py` for ad hoc ablations."
        )
    if str(args.model) == "m5_temporal_graphsage" and str(args.preset) != "unified_baseline":
        raise ValueError(
            "The official thesis suite is locked to the unified m5 baseline preset `unified_baseline`."
        )

    results: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        _ = get_dataset_spec(dataset_name)
        run_name = _run_name_for_dataset(args, dataset_name)
        summary_target = _dataset_training_root(dataset_name) / "models" / args.model / run_name / "summary.json"
        if args.skip_existing and summary_target.exists():
            summary_path, summary_payload = _load_run_summary(dataset_name, args.model, run_name)
        else:
            if args.build_features:
                _run_command(_build_feature_command(), dataset_name, dry_run=args.dry_run)
            train_command = _build_train_command(args=args, run_name=run_name)
            _run_command(train_command, dataset_name, dry_run=args.dry_run)
            if args.dry_run:
                continue
            summary_path, summary_payload = _load_run_summary(dataset_name, args.model, run_name)

        results.append(
            {
                "dataset": dataset_name,
                "run_name": run_name,
                "summary_path": str(summary_path.relative_to(REPO_ROOT)),
                "val_auc_mean": summary_payload.get("val_auc_mean"),
                "test_auc_mean": summary_payload.get("test_auc_mean"),
                "external_auc_mean": summary_payload.get("external_auc_mean"),
            }
        )

    if args.dry_run:
        return

    payload = {
        "suite_name": args.suite_name,
        "model": args.model,
        "preset": args.preset,
        "feature_profile": args.feature_profile,
        "dataset_isolation": True,
        "cross_dataset_training": False,
        "same_architecture_across_datasets": True,
        "split_guardrails": [
            "Each dataset is trained separately under its own dataset env and output namespace.",
            "All datasets share the same model family, feature contract, and training protocol.",
            "No dataset contributes labels, predictions, or normalization statistics to another dataset.",
        ],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "rel_dim": int(args.rel_dim),
        "fanouts": [int(v) for v in args.fanouts],
        "seeds": [int(v) for v in args.seeds],
        "results": results,
    }
    summary_path = _write_suite_summary(suite_name=args.suite_name, payload=payload)
    print(f"Suite finished: {summary_path}")


if __name__ == "__main__":
    main()
