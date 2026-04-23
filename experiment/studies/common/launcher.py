from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiment.datasets.core.registry import DATASET_ENV_VAR
from experiment.utils.common import ensure_dir, read_json, write_json

from .contracts import DatasetPlan, REPO_ROOT, StudyConfig, load_study_config, resolve_dataset_plan


def run_study(study_dir: Path) -> None:
    args = _parse_args()
    study = load_study_config(study_dir)
    seeds = [int(seed) for seed in (args.seeds or study.seeds)]

    if args.dataset:
        plan = resolve_dataset_plan(study, args.dataset)
        os.environ[DATASET_ENV_VAR] = plan.dataset_name
        for key, value in plan.feature_env.items():
            os.environ[key] = str(value)
        dataset_dir = ensure_dir(study.dataset_output_dir(plan.dataset_name))
        if args.skip_existing and (dataset_dir / "summary.json").exists():
            print(f"[study] skip existing dataset result: {dataset_dir / 'summary.json'}")
            return
        if args.build_features:
            _build_features(study_dir=study_dir, plan=plan, dry_run=args.dry_run)
            if args.dry_run:
                return
        if args.dry_run:
            _print_dataset_worker_preview(study=study, plan=plan, dataset_dir=dataset_dir, device=args.device, seeds=seeds)
            return
        _run_single_dataset(
            study=study,
            plan=plan,
            dataset_dir=dataset_dir,
            seeds=seeds,
            device=args.device,
        )
        return

    for dataset_name in study.datasets:
        plan = resolve_dataset_plan(study, dataset_name)
        command = [
            sys.executable,
            str(study_dir / "run.py"),
            "--dataset",
            plan.dataset_name,
            "--device",
            args.device,
        ]
        if args.build_features:
            command.append("--build-features")
        if args.skip_existing:
            command.append("--skip-existing")
        if args.dry_run:
            command.append("--dry-run")
        if seeds:
            command.extend(["--seeds", *[str(seed) for seed in seeds]])
        preview = _preview_command(command, plan)
        print(preview)
        if args.dry_run:
            continue
        env = os.environ.copy()
        env[DATASET_ENV_VAR] = plan.dataset_name
        for key, value in plan.feature_env.items():
            env[key] = str(value)
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    if args.dry_run:
        return
    _aggregate_study(study=study, seeds=seeds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated comparison or ablation study without modifying the production mainline.",
    )
    parser.add_argument("--dataset", default=None, help="Run a single dataset worker in the current process.")
    parser.add_argument("--device", default="cuda", help="Torch device for graph-based studies.")
    parser.add_argument("--build-features", action="store_true", help="Build dataset feature caches before the run.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a dataset if its summary already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed override.")
    return parser.parse_args()


def _run_single_dataset(
    *,
    study: StudyConfig,
    plan: DatasetPlan,
    dataset_dir: Path,
    seeds: list[int],
    device: str,
) -> None:
    if study.runner == "graph":
        from .graph_runner import run_graph_dataset

        summary_path = run_graph_dataset(
            study=study,
            plan=plan,
            dataset_dir=dataset_dir,
            seeds=seeds,
            device=device,
        )
    elif study.runner == "xgboost":
        from .xgboost_runner import run_xgboost_dataset

        summary_path = run_xgboost_dataset(
            study=study,
            plan=plan,
            dataset_dir=dataset_dir,
            seeds=seeds,
        )
    else:
        raise ValueError(f"Unsupported study runner: {study.runner}")
    print(f"[study] dataset summary written to {summary_path}")


def _build_features(*, study_dir: Path, plan: DatasetPlan, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiment" / "mainline.py"),
        "build_features",
        "--phase",
        "both",
        "--outdir",
        str(plan.feature_dir),
    ]
    preview = _preview_command(command, plan)
    print(preview)
    if dry_run:
        return
    env = os.environ.copy()
    env[DATASET_ENV_VAR] = plan.dataset_name
    for key, value in plan.feature_env.items():
        env[key] = str(value)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _preview_command(command: list[str], plan: DatasetPlan) -> str:
    env_assignments = [f"{DATASET_ENV_VAR}={shlex.quote(plan.dataset_name)}"]
    for key, value in sorted(plan.feature_env.items()):
        env_assignments.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(env_assignments) + " " + shlex.join(command)


def _print_dataset_worker_preview(
    *,
    study: StudyConfig,
    plan: DatasetPlan,
    dataset_dir: Path,
    device: str,
    seeds: list[int],
) -> None:
    print(
        f"[study:dry-run] study={study.study_name} dataset={plan.dataset_name} "
        f"runner={study.runner} output={dataset_dir} device={device} seeds={seeds}"
    )


def _aggregate_study(*, study: StudyConfig, seeds: list[int]) -> None:
    results: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []

    for dataset_name in study.datasets:
        dataset_dir = study.dataset_output_dir(dataset_name)
        summary_path = dataset_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing dataset summary: {summary_path}")
        summary = read_json(summary_path)
        results.append(
            {
                "dataset": summary["dataset"],
                "dataset_display_name": summary["dataset_display_name"],
                "display_name": summary["display_name"],
                "runner": summary["runner"],
                "val_auc_mean": summary.get("val_auc_mean"),
                "val_auc_std": summary.get("val_auc_std"),
                "test_auc_mean": summary.get("test_auc_mean"),
                "external_auc_mean": summary.get("external_auc_mean"),
                "best_epoch_mean": summary.get("best_epoch_mean"),
                "trained_epochs_mean": summary.get("trained_epochs_mean"),
                "summary_path": _path_repr(summary_path),
            }
        )
        seed_rows.extend(_read_csv_rows(dataset_dir / "seed_overview.csv"))
        epoch_rows.extend(_read_csv_rows(dataset_dir / "epoch_metrics_merged.csv"))

    ensure_dir(study.output_root)
    auc_summary_path = study.output_root / "auc_summary.csv"
    seed_overview_path = study.output_root / "seed_overview.csv"
    epoch_metrics_path = study.output_root / "epoch_metrics_all.csv"

    _write_csv(auc_summary_path, results)
    _write_csv(seed_overview_path, seed_rows)
    _write_csv(epoch_metrics_path, epoch_rows)

    summary_payload = {
        "study_name": study.study_name,
        "display_name": study.display_name,
        "study_type": study.study_type,
        "runner": study.runner,
        "description": study.description,
        "datasets": list(study.datasets),
        "dataset_profile_path": _path_repr(study.dataset_profile_path),
        "config_path": _path_repr(study.config_path),
        "output_root": _path_repr(study.output_root),
        "seeds": [int(seed) for seed in seeds],
        "auc_summary_path": _path_repr(auc_summary_path),
        "seed_overview_path": _path_repr(seed_overview_path),
        "epoch_metrics_path": _path_repr(epoch_metrics_path),
        "results": results,
    }
    write_json(study.output_root / "summary.json", summary_payload)
    _write_markdown_summary(study=study, results=results)
    print(f"[study] aggregate summary written to {study.output_root / 'summary.json'}")


def _write_markdown_summary(*, study: StudyConfig, results: list[dict[str, Any]]) -> None:
    lines = [
        f"# {study.display_name}",
        "",
        f"- Study name: `{study.study_name}`",
        f"- Study type: `{study.study_type}`",
        f"- Runner: `{study.runner}`",
        "",
        "| Dataset | Val AUC | Test AUC | External AUC | Trained Epochs | Summary |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            "| {dataset_display_name} | {val_auc} | {test_auc} | {external_auc} | {trained_epochs} | {summary_path} |".format(
                dataset_display_name=row["dataset_display_name"],
                val_auc=_format_metric(row.get("val_auc_mean")),
                test_auc=_format_metric(row.get("test_auc_mean")),
                external_auc=_format_metric(row.get("external_auc_mean")),
                trained_epochs=_format_metric(row.get("trained_epochs_mean")),
                summary_path=row["summary_path"],
            )
        )
    (study.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _path_repr(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _format_metric(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.6f}"
