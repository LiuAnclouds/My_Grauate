from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import get_active_dataset_spec
from experiment.training.common import MODEL_OUTPUT_ROOT, load_experiment_split
from experiment.training.thesis_contract import (
    OFFICIAL_HYBRID_GRAPHPROP_BACKGROUND_WEIGHT,
    OFFICIAL_HYBRID_GRAPHPROP_COLSAMPLE_BYTREE,
    OFFICIAL_HYBRID_GRAPHPROP_DEVICE,
    OFFICIAL_HYBRID_GRAPHPROP_EXTRA_GROUPS,
    OFFICIAL_HYBRID_GRAPHPROP_GAMMA,
    OFFICIAL_HYBRID_GRAPHPROP_LEARNING_RATE,
    OFFICIAL_HYBRID_GRAPHPROP_MAX_BIN,
    OFFICIAL_HYBRID_GRAPHPROP_MAX_DEPTH,
    OFFICIAL_HYBRID_GRAPHPROP_MIN_CHILD_WEIGHT,
    OFFICIAL_HYBRID_GRAPHPROP_N_ESTIMATORS,
    OFFICIAL_HYBRID_GRAPHPROP_PROP_HALF_LIFE_DAYS,
    OFFICIAL_HYBRID_GRAPHPROP_RANDOM_STATE,
    OFFICIAL_HYBRID_GRAPHPROP_REG_ALPHA,
    OFFICIAL_HYBRID_GRAPHPROP_REG_LAMBDA,
    OFFICIAL_HYBRID_GRAPHPROP_SUBSAMPLE,
    OFFICIAL_HYBRID_RECENT_START_RATIO,
)


ACTIVE_DATASET_SPEC = get_active_dataset_spec()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official thesis graphprop secondary branch under one shared cross-dataset contract."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outdir", type=Path, default=MODEL_OUTPUT_ROOT / "xgboost_gpu")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _derived_min_train_day(threshold_day: int) -> int:
    threshold = int(max(threshold_day, 0))
    if threshold <= 0:
        return 0
    cutoff = int(round(float(threshold) * float(OFFICIAL_HYBRID_RECENT_START_RATIO)))
    return min(max(cutoff, 0), max(threshold - 1, 0))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.outdir) / args.run_name
    summary_path = run_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"Skip existing graphprop secondary: {summary_path}")
        return

    split = load_experiment_split()
    min_train_day = _derived_min_train_day(int(split.threshold_day))
    command = [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / "run_xgb_multiclass_bg_graphprop.py"),
        "--run-name",
        args.run_name,
        "--extra-groups",
        *list(OFFICIAL_HYBRID_GRAPHPROP_EXTRA_GROUPS),
        "--prop-half-life-days",
        *[str(value) for value in OFFICIAL_HYBRID_GRAPHPROP_PROP_HALF_LIFE_DAYS],
        "--background-weight",
        str(OFFICIAL_HYBRID_GRAPHPROP_BACKGROUND_WEIGHT),
        "--min-train-first-active-day",
        str(min_train_day),
        "--learning-rate",
        str(OFFICIAL_HYBRID_GRAPHPROP_LEARNING_RATE),
        "--n-estimators",
        str(OFFICIAL_HYBRID_GRAPHPROP_N_ESTIMATORS),
        "--max-depth",
        str(OFFICIAL_HYBRID_GRAPHPROP_MAX_DEPTH),
        "--min-child-weight",
        str(OFFICIAL_HYBRID_GRAPHPROP_MIN_CHILD_WEIGHT),
        "--subsample",
        str(OFFICIAL_HYBRID_GRAPHPROP_SUBSAMPLE),
        "--colsample-bytree",
        str(OFFICIAL_HYBRID_GRAPHPROP_COLSAMPLE_BYTREE),
        "--gamma",
        str(OFFICIAL_HYBRID_GRAPHPROP_GAMMA),
        "--reg-alpha",
        str(OFFICIAL_HYBRID_GRAPHPROP_REG_ALPHA),
        "--reg-lambda",
        str(OFFICIAL_HYBRID_GRAPHPROP_REG_LAMBDA),
        "--max-bin",
        str(OFFICIAL_HYBRID_GRAPHPROP_MAX_BIN),
        "--seed",
        str(OFFICIAL_HYBRID_GRAPHPROP_RANDOM_STATE),
        "--device",
        str(OFFICIAL_HYBRID_GRAPHPROP_DEVICE),
    ]
    print(
        "[thesis_graphprop_secondary] "
        f"dataset={ACTIVE_DATASET_SPEC.name} "
        f"threshold_day={int(split.threshold_day)} "
        f"recent_start_ratio={float(OFFICIAL_HYBRID_RECENT_START_RATIO):.6f} "
        f"derived_min_train_day={min_train_day}"
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        "[thesis_graphprop_secondary] "
        f"run={args.run_name} val_auc={float(payload['phase1_val_metrics']['auc']):.6f}"
    )


if __name__ == "__main__":
    main()
