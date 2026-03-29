from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.eda.analysis import ALL_ANALYSES, run_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible EDA for the XinYe DGraph anti-fraud dataset."
    )
    parser.add_argument(
        "--phase",
        default="both",
        choices=("phase1", "phase2", "both"),
        help="Dataset phase to analyze.",
    )
    parser.add_argument(
        "--analysis",
        nargs="+",
        default=["all"],
        choices=(*ALL_ANALYSES, "all"),
        help="EDA modules to run.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "experiment" / "outputs" / "eda",
        help="Output directory for tables, plots and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phases = ["phase1", "phase2"] if args.phase == "both" else [args.phase]
    summary = run_eda(phases=phases, analyses=list(args.analysis), outdir=args.outdir)
    print(f"EDA finished. Summary written to: {args.outdir / 'dataset_summary.json'}")
    print(f"Analyzed phases: {', '.join(summary.get('phases', {}).keys())}")


if __name__ == "__main__":
    main()
