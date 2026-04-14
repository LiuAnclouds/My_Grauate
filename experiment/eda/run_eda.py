from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import get_active_dataset_spec, resolve_output_roots
from experiment.eda.analysis import ALL_ANALYSES, run_eda


EDA_OUTPUT_ROOT, _ = resolve_output_roots(REPO_ROOT)


def parse_args() -> argparse.Namespace:
    dataset_spec = get_active_dataset_spec()
    artifact_choices = (*dataset_spec.phase_filenames.keys(), "both")
    default_phase = "both" if len(dataset_spec.default_artifacts) > 1 else dataset_spec.default_artifacts[0]
    parser = argparse.ArgumentParser(
        description=f"Run reproducible EDA for the active dataset: {dataset_spec.display_name}."
    )
    parser.add_argument(
        "--phase",
        default=default_phase,
        choices=artifact_choices,
        help="Dataset artifact to analyze. `both` expands to the dataset default artifact set.",
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
        default=EDA_OUTPUT_ROOT,
        help="Output directory for tables, plots and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_spec = get_active_dataset_spec()
    phases = list(dataset_spec.default_artifacts) if args.phase == "both" else [args.phase]
    summary = run_eda(phases=phases, analyses=list(args.analysis), outdir=args.outdir)
    print(f"EDA finished. Summary written to: {args.outdir / 'dataset_summary.json'}")
    print(f"Analyzed phases: {', '.join(summary.get('phases', {}).keys())}")


if __name__ == "__main__":
    main()
