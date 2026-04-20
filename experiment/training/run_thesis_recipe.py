from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import DATASET_ENV_VAR, DEFAULT_DATASET_NAME
from experiment.training.thesis_contract import (
    OFFICIAL_BACKBONE_MODEL,
    OFFICIAL_BACKBONE_PRESET,
    OFFICIAL_BASELINE_MODEL,
    OFFICIAL_BASELINE_PRESET,
    OFFICIAL_MAINLINE_BATCH_SIZE,
    OFFICIAL_MAINLINE_FANOUTS,
    OFFICIAL_MAINLINE_HIDDEN_DIM,
    OFFICIAL_MAINLINE_REL_DIM,
    OFFICIAL_SUITE_EPOCHS,
)


@dataclass(frozen=True)
class ThesisRecipe:
    name: str
    description: str
    build_args: tuple[str, ...]
    train_args: tuple[str, ...]


def _shared_train_args(*, model_name: str, preset_name: str, epochs: int) -> tuple[str, ...]:
    return (
        "--model",
        model_name,
        "--preset",
        preset_name,
        "--device",
        "cuda",
        "--seeds",
        "42",
        "52",
        "62",
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(OFFICIAL_MAINLINE_BATCH_SIZE),
        "--hidden-dim",
        str(OFFICIAL_MAINLINE_HIDDEN_DIM),
        "--rel-dim",
        str(OFFICIAL_MAINLINE_REL_DIM),
        "--fanouts",
        *[str(v) for v in OFFICIAL_MAINLINE_FANOUTS],
    )


def _thesis_recipe(name: str) -> ThesisRecipe:
    if name == "baseline_m5_unified":
        return ThesisRecipe(
            name=name,
            description=(
                "Official unified baseline: one shared UTPM feature schema plus temporal GraphSAGE, "
                "without the thesis auxiliary regularizer."
            ),
            build_args=("--phase", "both"),
            train_args=_shared_train_args(
                model_name=OFFICIAL_BASELINE_MODEL,
                preset_name=OFFICIAL_BASELINE_PRESET,
                epochs=16,
            ),
        )
    if name == "thesis_m7_utpm":
        return ThesisRecipe(
            name=name,
            description=(
                "Official thesis mainline: unified UTPM feature schema, temporal prototype memory, "
                "drift-residual adaptation, and pseudo-contrastive test-pool regularization in one shared "
                "dynamic-GNN path."
            ),
            build_args=("--phase", "both"),
            train_args=_shared_train_args(
                model_name=OFFICIAL_BACKBONE_MODEL,
                preset_name=OFFICIAL_BACKBONE_PRESET,
                epochs=OFFICIAL_SUITE_EPOCHS,
            ),
        )
    supported = ", ".join(_thesis_recipe_names())
    raise KeyError(f"Unsupported thesis recipe `{name}`. Supported: {supported}")


def _thesis_recipe_names() -> tuple[str, ...]:
    return ("baseline_m5_unified", "thesis_m7_utpm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official thesis-only dynamic-graph recipes without touching the legacy recipe registry."
    )
    parser.add_argument(
        "command",
        choices=("show", "build_features", "train", "build_and_train"),
        help="Whether to print or execute the selected thesis recipe.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help="Dataset registry key, e.g. xinye_dgraph / elliptic_transactions / ellipticpp_transactions.",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        choices=_thesis_recipe_names(),
        help="Named official thesis recipe.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name override. Defaults to the recipe name.",
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
        help="Additional arguments appended to the underlying thesis mainline train command.",
    )
    return parser.parse_args()


def _base_command(subcommand: str) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / "run_thesis_mainline.py"),
        subcommand,
    ]


def _render_shell(command: list[str], dataset_name: str) -> str:
    return f"{DATASET_ENV_VAR}={shlex.quote(dataset_name)} " + shlex.join(command)


def _execute(command: list[str], dataset_name: str, dry_run: bool) -> None:
    preview = _render_shell(command, dataset_name)
    print(preview)
    if dry_run:
        return
    env = os.environ.copy()
    env[DATASET_ENV_VAR] = dataset_name
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    recipe = _thesis_recipe(args.recipe)
    build_command = _base_command("build_features") + list(recipe.build_args)
    train_command = _base_command("train") + ["--run-name", args.run_name or recipe.name] + list(
        recipe.train_args
    ) + list(args.train_extra_args)

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
