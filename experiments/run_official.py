from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dyrift.data_processing.core.registry import DATASET_ENV_VAR
from dyrift.models.spec import DYRIFT_GNN_MODEL, OFFICIAL_DATASETS, OFFICIAL_FULL_EXPERIMENT_NAME


COMPARISON_RUNNERS = (
    "linear_same_feature",
    "temporal_graphsage_reference",
    "tgat_style_reference",
)
ABLATION_RUNNERS = (
    "without_target_context_bridge",
    "without_drift_expert",
    "without_prototype_memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official DyRIFT-TGAT full, comparison, and ablation experiment matrix.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "full", "comparisons", "ablations"),
        default="all",
        help="Which part of the official experiment matrix to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(OFFICIAL_DATASETS),
        choices=tuple(OFFICIAL_DATASETS),
        help="Datasets to run.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for graph models.")
    parser.add_argument("--build-features", action="store_true", help="Build feature caches before experiment runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with an existing epoch_metrics.csv.")
    parser.add_argument("--skip-recovery", action="store_true", help="Do not refresh docs/generated result CSV files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = _resolve_stages(args.stage)
    for dataset_name in args.datasets:
        if "full" in stages:
            _run_full_model(dataset_name=dataset_name, device=args.device, skip_existing=args.skip_existing, dry_run=args.dry_run)
        if "comparisons" in stages:
            for experiment_name in COMPARISON_RUNNERS:
                _run_experiment(
                    experiment_group="comparisons",
                    experiment_name=experiment_name,
                    dataset_name=dataset_name,
                    device=args.device,
                    build_features=args.build_features,
                    skip_existing=args.skip_existing,
                    dry_run=args.dry_run,
                )
        if "ablations" in stages:
            for experiment_name in ABLATION_RUNNERS:
                _run_experiment(
                    experiment_group="ablations",
                    experiment_name=experiment_name,
                    dataset_name=dataset_name,
                    device=args.device,
                    build_features=args.build_features,
                    skip_existing=args.skip_existing,
                    dry_run=args.dry_run,
                )

    if not args.skip_recovery:
        _run_command([sys.executable, str(PROJECT_ROOT / "experiments" / "reporting" / "recover_results.py")], dry_run=args.dry_run)
    return 0


def _resolve_stages(stage: str) -> set[str]:
    if stage == "all":
        return {"full", "comparisons", "ablations"}
    return {stage}


def _run_full_model(*, dataset_name: str, device: str, skip_existing: bool, dry_run: bool) -> None:
    output_path = PROJECT_ROOT / "outputs" / "train" / OFFICIAL_FULL_EXPERIMENT_NAME / DYRIFT_GNN_MODEL / dataset_name / "epoch_metrics.csv"
    if skip_existing and output_path.exists():
        print(f"[official] skip existing full model: {output_path}")
        return
    parameter_file = PROJECT_ROOT / "configs" / "private" / f"{dataset_name}.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "train",
        "--parameter-file",
        str(parameter_file),
        "--device",
        str(device),
    ]
    _run_command(command, dataset_name=dataset_name, dry_run=dry_run)


def _run_experiment(
    *,
    experiment_group: str,
    experiment_name: str,
    dataset_name: str,
    device: str,
    build_features: bool,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / experiment_group / experiment_name / "run.py"),
        "--dataset",
        str(dataset_name),
        "--device",
        str(device),
    ]
    if build_features:
        command.append("--build-features")
    if skip_existing:
        command.append("--skip-existing")
    if dry_run:
        command.append("--dry-run")
    _run_command(command, dataset_name=dataset_name, dry_run=dry_run)


def _run_command(command: list[str], *, dataset_name: str | None = None, dry_run: bool) -> None:
    env = os.environ.copy()
    if dataset_name is not None:
        env[DATASET_ENV_VAR] = dataset_name
    preview = " ".join(_quote(part) for part in command)
    if dataset_name is not None:
        preview = f"{DATASET_ENV_VAR}={dataset_name} {preview}"
    print(f"[official] {preview}")
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _quote(value: str) -> str:
    text = str(value)
    if not text or any(ch.isspace() for ch in text):
        return f'"{text}"'
    return text


if __name__ == "__main__":
    raise SystemExit(main())
