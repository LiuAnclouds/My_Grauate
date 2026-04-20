from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import DATASET_ENV_VAR, DEFAULT_DATASET_NAME
from experiment.training.recipes import get_graph_recipe, list_recipe_names

OFFICIAL_THESIS_RECIPE_NAMES = ("baseline_m5_unified", "thesis_m7_utpm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reusable dataset-aware dynamic-graph training recipes."
    )
    parser.add_argument(
        "command",
        choices=("show", "build_features", "train", "build_and_train"),
        help="Whether to print or execute the selected recipe.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help="Dataset registry key, e.g. xinye_dgraph / elliptic_transactions / ellipticpp_transactions.",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        choices=list_recipe_names(),
        help="Named training recipe.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Required for train/build_and_train. Used as the run directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--train-extra-args",
        nargs=argparse.REMAINDER,
        default=(),
        help="Additional arguments appended to the underlying train command.",
    )
    return parser.parse_args()


def _base_command(subcommand: str, entrypoint: str) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / entrypoint),
        subcommand,
    ]


def _render_shell(command: list[str], dataset_name: str) -> str:
    return f"{DATASET_ENV_VAR}={shlex.quote(dataset_name)} " + shlex.join(command)


def _execute(command: list[str], dataset_name: str, dry_run: bool) -> None:
    shell_preview = _render_shell(command, dataset_name)
    print(shell_preview)
    if dry_run:
        return
    env = os.environ.copy()
    env[DATASET_ENV_VAR] = dataset_name
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    if args.recipe in OFFICIAL_THESIS_RECIPE_NAMES:
        raise ValueError(
            f"`{args.recipe}` is now an official thesis-only recipe. "
            "Use `experiment/training/run_thesis_recipe.py` instead of the legacy `run_recipe.py` wrapper."
        )
    recipe = get_graph_recipe(args.recipe, args.dataset)
    train_command = _base_command("train", recipe.entrypoint) + ["--run-name", args.run_name or recipe.name] + list(
        recipe.train_args
    ) + list(args.train_extra_args)
    build_command = _base_command("build_features", recipe.entrypoint) + list(recipe.build_args)

    if args.command == "show":
        print(f"recipe={recipe.name}")
        print(f"description={recipe.description}")
        print(_render_shell(build_command, args.dataset))
        print(_render_shell(train_command, args.dataset))
        return

    if args.command == "build_features":
        _execute(build_command, args.dataset, args.dry_run)
        return

    if args.command == "train":
        _execute(train_command, args.dataset, args.dry_run)
        return

    if args.command == "build_and_train":
        _execute(build_command, args.dataset, args.dry_run)
        _execute(train_command, args.dataset, args.dry_run)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
