from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import DATASET_ENV_VAR, get_dataset_spec
from experiment.training.core.hparams import (
    DatasetHparams,
    load_hparam_profile,
    resolve_dataset_hparams,
)
from experiment.training.core.spec import (
    DYRIFT_GNN_MODEL,
    DYRIFT_MODEL_DISPLAY_NAME,
    DYRIFT_MODEL_SHORT_NAME,
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
    TRGT_BACKBONE_DISPLAY_NAME,
    TRGT_BACKBONE_SHORT_NAME,
    TRANSFORMER_BACKBONE_MODEL,
    TRANSFORMER_BACKBONE_DEPLOY_PRESET,
    TRANSFORMER_BACKBONE_PRESET,
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
            "Run the unified thesis GNN mainline on multiple datasets with one shared experiment contract."
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
        choices=("m5_temporal_graphsage", "m7_utpm", DYRIFT_GNN_MODEL),
        default=TRANSFORMER_BACKBONE_MODEL,
        help=(
            "Unified thesis-mainline model family. "
            "`m7_utpm` is the legacy stable backbone; "
            f"`{DYRIFT_GNN_MODEL}` is {DYRIFT_MODEL_SHORT_NAME} with the {TRGT_BACKBONE_SHORT_NAME} backbone."
        ),
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Preset passed through to runners/mainline.py. "
            "Defaults: `m5_temporal_graphsage` -> `unified_baseline`, "
            f"`{OFFICIAL_BACKBONE_MODEL}` -> `{OFFICIAL_BACKBONE_PRESET}`, "
            f"`{TRANSFORMER_BACKBONE_MODEL}` -> `{TRANSFORMER_BACKBONE_PRESET}`. "
            f"The deployable pure-GNN preset is `{TRANSFORMER_BACKBONE_DEPLOY_PRESET}`."
        ),
    )
    parser.add_argument(
        "--feature-profile",
        choices=("utpm_unified", "utpm_shift_compact", "utpm_shift_enhanced"),
        default=OFFICIAL_BACKBONE_FEATURE_PROFILE,
        help="Shared feature contract for all datasets in the suite.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help=(
            "Optional dataset-scoped feature root passed through to build_features/train. "
            "Use this to keep alternative unified feature caches separate from the default thesis cache."
        ),
    )
    parser.add_argument(
        "--feature-subdir",
        default=None,
        help=(
            "Optional feature subdirectory name created under each dataset's training root, "
            "for example `features_ap64`. This is the safest way to keep tri-dataset feature variants isolated."
        ),
    )
    parser.add_argument(
        "--dataset-hparams",
        type=Path,
        default=None,
        help=(
            "Optional JSON profile that keeps the architecture fixed but allows dataset-local "
            "hyperparameter overrides such as attr_proj_dim, hidden_dim, fanouts, or blend weights."
        ),
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
        "--learning-rate",
        type=float,
        default=None,
        help="Optional learning-rate override applied to every dataset unless the profile overrides it.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Optional weight-decay override applied to every dataset unless the profile overrides it.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional dropout override applied to every dataset unless the profile overrides it.",
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
        "--target-context-groups",
        nargs="*",
        default=None,
        help=(
            "Optional explicit target-context feature groups forwarded to the mainline trainer. "
            "Pass `none` to disable the internal target-context feature branch."
        ),
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


def _run_name_for_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    settings: DatasetHparams,
) -> str:
    dataset_short = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    template = settings.run_name_template or str(args.run_name_template)
    return str(template).format(
        suite_name=args.suite_name,
        dataset=dataset_name,
        dataset_short=dataset_short,
        model=args.model,
        preset=args.preset,
        epochs=settings.epochs,
    )


def _feature_dir_for_dataset(dataset_name: str, settings: DatasetHparams) -> Path | None:
    if settings.feature_subdir:
        return _dataset_training_root(dataset_name) / str(settings.feature_subdir)
    return settings.feature_dir


def _command_preview(command: list[str], dataset_name: str, *, extra_env: dict[str, str] | None = None) -> str:
    env_assignments = [f"{DATASET_ENV_VAR}={shlex.quote(dataset_name)}"]
    for key in sorted((extra_env or {}).keys()):
        env_assignments.append(f"{key}={shlex.quote(str(extra_env[key]))}")
    return " ".join(env_assignments) + " " + shlex.join(command)


def _build_feature_command(*, feature_dir: Path | None) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / "runners" / "mainline.py"),
        "build_features",
        "--phase",
        "both",
    ]
    if feature_dir is not None:
        command.extend(["--outdir", str(feature_dir)])
    return command


def _build_train_command(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    settings: DatasetHparams,
    run_name: str,
) -> list[str]:
    feature_dir = _feature_dir_for_dataset(dataset_name, settings)
    command: list[str] = [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / "runners" / "mainline.py"),
        "train",
        "--model",
        args.model,
        "--preset",
        args.preset,
        "--feature-profile",
        settings.feature_profile,
        "--run-name",
        run_name,
        "--device",
        args.device,
        "--epochs",
        str(settings.epochs),
        "--batch-size",
        str(settings.batch_size),
        "--hidden-dim",
        str(settings.hidden_dim),
        "--rel-dim",
        str(settings.rel_dim),
        "--fanouts",
        *[str(v) for v in settings.fanouts],
        "--seeds",
        *[str(v) for v in args.seeds],
    ]
    if feature_dir is not None:
        command.extend(["--feature-dir", str(feature_dir)])
    if settings.learning_rate is not None:
        command.extend(["--learning-rate", str(settings.learning_rate)])
    if settings.weight_decay is not None:
        command.extend(["--weight-decay", str(settings.weight_decay)])
    if settings.dropout is not None:
        command.extend(["--dropout", str(settings.dropout)])
    if settings.target_context_groups is not None:
        command.extend(["--target-context-groups", *[str(v) for v in settings.target_context_groups]])
    for override in settings.graph_config_overrides:
        command.extend(["--graph-config-override", str(override)])
    return command


def _run_command(
    command: list[str],
    dataset_name: str,
    *,
    extra_env: dict[str, str] | None,
    dry_run: bool,
) -> None:
    preview = _command_preview(command, dataset_name, extra_env=extra_env)
    print(preview)
    if dry_run:
        return
    env = os.environ.copy()
    env[DATASET_ENV_VAR] = dataset_name
    for key, value in (extra_env or {}).items():
        env[str(key)] = str(value)
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
    profile = load_hparam_profile(args.dataset_hparams)
    if args.preset is None:
        if str(args.model) == "m5_temporal_graphsage":
            args.preset = "unified_baseline"
        elif str(args.model) == OFFICIAL_BACKBONE_MODEL:
            args.preset = OFFICIAL_BACKBONE_PRESET
        elif str(args.model) == TRANSFORMER_BACKBONE_MODEL:
            args.preset = TRANSFORMER_BACKBONE_PRESET
        else:
            raise ValueError(f"Unsupported thesis suite model: {args.model}")

    if str(args.model) == OFFICIAL_BACKBONE_MODEL and str(args.preset) != OFFICIAL_BACKBONE_PRESET:
        raise ValueError(
            "The legacy m7 thesis suite is locked to the unified m7 v4 backbone. "
            "Use `runners/mainline.py` for ad hoc ablations."
        )
    if str(args.model) == "m5_temporal_graphsage" and str(args.preset) != "unified_baseline":
        raise ValueError(
            "The official thesis suite is locked to the unified m5 baseline preset `unified_baseline`."
        )
    if (
        str(args.model) == TRANSFORMER_BACKBONE_MODEL
        and str(args.preset)
        not in {
            TRANSFORMER_BACKBONE_PRESET,
            TRANSFORMER_BACKBONE_DEPLOY_PRESET,
        }
    ):
        raise ValueError(
            "The DyRIFT-GNN/TRGT thesis suite is locked to the official presets: "
            f"`{TRANSFORMER_BACKBONE_PRESET}`, "
            f"`{TRANSFORMER_BACKBONE_DEPLOY_PRESET}`."
        )
    results: list[dict[str, Any]] = []
    resolved_hparams_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset_name in args.datasets:
        _ = get_dataset_spec(dataset_name)
        dataset_hparams = resolve_dataset_hparams(
            args=args,
            dataset_name=dataset_name,
            profile=profile,
        )
        resolved_hparams_by_dataset[dataset_name] = dataset_hparams.to_summary_payload()
        run_name = _run_name_for_dataset(args, dataset_name, dataset_hparams)
        summary_target = _dataset_training_root(dataset_name) / "models" / args.model / run_name / "summary.json"
        if args.skip_existing and summary_target.exists():
            summary_path, summary_payload = _load_run_summary(dataset_name, args.model, run_name)
        else:
            if args.build_features:
                _run_command(
                    _build_feature_command(feature_dir=_feature_dir_for_dataset(dataset_name, dataset_hparams)),
                    dataset_name,
                    extra_env=dataset_hparams.feature_env,
                    dry_run=args.dry_run,
                )
            train_command = _build_train_command(
                args=args,
                dataset_name=dataset_name,
                settings=dataset_hparams,
                run_name=run_name,
            )
            _run_command(
                train_command,
                dataset_name,
                extra_env=dataset_hparams.feature_env,
                dry_run=args.dry_run,
            )
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
                "hparams": dataset_hparams.to_summary_payload(),
            }
        )

    if args.dry_run:
        return

    payload = {
        "suite_name": args.suite_name,
        "model": args.model,
        "method_display_name": (
            DYRIFT_MODEL_DISPLAY_NAME if str(args.model) == TRANSFORMER_BACKBONE_MODEL else args.model
        ),
        "method_short_name": (
            DYRIFT_MODEL_SHORT_NAME if str(args.model) == TRANSFORMER_BACKBONE_MODEL else args.model
        ),
        "backbone_display_name": (
            TRGT_BACKBONE_DISPLAY_NAME if str(args.model) == TRANSFORMER_BACKBONE_MODEL else args.model
        ),
        "backbone_short_name": (
            TRGT_BACKBONE_SHORT_NAME if str(args.model) == TRANSFORMER_BACKBONE_MODEL else args.model
        ),
        "preset": args.preset,
        "feature_profile": args.feature_profile,
        "feature_dir": None if args.feature_dir is None else str(args.feature_dir),
        "feature_subdir": args.feature_subdir,
        "dataset_hparams_path": None if profile is None else str(profile.path.relative_to(REPO_ROOT)),
        "feature_env_overrides": {"GRADPROJ_UTPM_ATTR_PROJ_DIM": os.environ.get("GRADPROJ_UTPM_ATTR_PROJ_DIM")},
        "graph_config_overrides": [str(v) for v in args.graph_config_override],
        "run_name_template": args.run_name_template,
        "target_context_groups": None if args.target_context_groups is None else [str(v) for v in args.target_context_groups],
        "dataset_isolation": True,
        "cross_dataset_training": False,
        "same_architecture_across_datasets": True,
        "split_guardrails": [
            "Each dataset is trained separately under its own dataset env and output namespace.",
            "All datasets share the same model family, feature contract family, and single-GNN deployment path.",
            "Dataset-local tuning is limited to feature capacity, model capacity, optimization, and low-level regularization hyperparameters.",
            "No dataset contributes labels, predictions, or normalization statistics to another dataset.",
        ],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "rel_dim": int(args.rel_dim),
        "fanouts": [int(v) for v in args.fanouts],
        "learning_rate": None if args.learning_rate is None else float(args.learning_rate),
        "weight_decay": None if args.weight_decay is None else float(args.weight_decay),
        "dropout": None if args.dropout is None else float(args.dropout),
        "seeds": [int(v) for v in args.seeds],
        "dataset_hparams": resolved_hparams_by_dataset,
        "results": results,
    }
    summary_path = _write_suite_summary(suite_name=args.suite_name, payload=payload)
    print(f"Suite finished: {summary_path}")


if __name__ == "__main__":
    main()
