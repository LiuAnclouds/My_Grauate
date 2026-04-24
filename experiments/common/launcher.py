from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from data_processing.core.registry import DATASET_ENV_VAR
from utils.common import ensure_dir

from .contracts import DatasetPlan, ExperimentConfig, REPO_ROOT, load_experiment_config, resolve_dataset_plan


def run_experiment(experiment_dir: Path) -> None:
    args = _parse_args()
    if not args.dataset:
        raise SystemExit("Please pass --dataset. Experiments are run one dataset at a time.")

    experiment = load_experiment_config(experiment_dir)
    plan = resolve_dataset_plan(experiment, args.dataset)
    seeds = [int(seed) for seed in (args.seeds or experiment.seeds)]

    os.environ[DATASET_ENV_VAR] = plan.dataset_name
    for key, value in plan.feature_env.items():
        os.environ[key] = str(value)

    dataset_dir = ensure_dir(experiment.dataset_output_dir(plan.dataset_name))
    if args.skip_existing and (dataset_dir / "epoch_metrics.csv").exists():
        print(f"[experiment] skip existing result: {dataset_dir / 'epoch_metrics.csv'}")
        return
    if args.build_features:
        _build_features(experiment_dir=experiment_dir, plan=plan, dry_run=args.dry_run)
        if args.dry_run:
            return
    if args.dry_run:
        _print_dataset_worker_preview(
            experiment=experiment,
            plan=plan,
            dataset_dir=dataset_dir,
            device=args.device,
            seeds=seeds,
        )
        return
    _run_single_dataset(
        experiment=experiment,
        plan=plan,
        dataset_dir=dataset_dir,
        seeds=seeds,
        device=args.device,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one comparison, ablation, or progressive experiment for one dataset.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name to run.")
    parser.add_argument("--device", default="cuda", help="Torch device for graph experiments.")
    parser.add_argument("--build-features", action="store_true", help="Build dataset feature caches before the run.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if epoch_metrics.csv already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run without training.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed override.")
    return parser.parse_args()


def _run_single_dataset(
    *,
    experiment: ExperimentConfig,
    plan: DatasetPlan,
    dataset_dir: Path,
    seeds: list[int],
    device: str,
) -> None:
    if experiment.runner == "graph":
        from .graph_runner import run_graph_dataset

        run_graph_dataset(
            config=experiment,
            plan=plan,
            dataset_dir=dataset_dir,
            seeds=seeds,
            device=device,
        )
    elif experiment.runner == "xgboost":
        from .xgboost_runner import run_xgboost_dataset

        run_xgboost_dataset(
            config=experiment,
            plan=plan,
            dataset_dir=dataset_dir,
            seeds=seeds,
        )
    else:
        raise ValueError(f"Unsupported experiment runner: {experiment.runner}")
    print(f"[experiment] result written to {dataset_dir / 'epoch_metrics.csv'}")


def _build_features(*, experiment_dir: Path, plan: DatasetPlan, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
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
    experiment: ExperimentConfig,
    plan: DatasetPlan,
    dataset_dir: Path,
    device: str,
    seeds: list[int],
) -> None:
    print(
        f"[experiment:dry-run] experiment={experiment.experiment_name} "
        f"dataset={plan.dataset_name} runner={experiment.runner} "
        f"model={experiment.model_name} output={dataset_dir} device={device} seeds={seeds}"
    )
