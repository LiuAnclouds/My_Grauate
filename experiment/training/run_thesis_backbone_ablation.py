from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.training.thesis_contract import (
    OFFICIAL_BACKBONE_FEATURE_PROFILE,
    OFFICIAL_BACKBONE_MODEL,
    OFFICIAL_BACKBONE_PRESET,
    OFFICIAL_HYBRID_BLEND_ALPHA,
    OFFICIAL_SUITE_EPOCHS,
    OFFICIAL_SUITE_SEEDS,
)


THESIS_SUITE_ROOT = REPO_ROOT / "experiment" / "outputs" / "thesis_suite"
THESIS_ABLATION_ROOT = REPO_ROOT / "experiment" / "outputs" / "thesis_ablation"
OFFICIAL_HYBRID_SUITE = "thesis_m7_v4_graphpropblend082"


@dataclass(frozen=True)
class BackboneAblation:
    setting_id: str
    suite_name: str
    label: str
    description: str
    overrides: tuple[str, ...]
    run_required: bool = True


BACKBONE_REFERENCE = BackboneAblation(
    setting_id="official_backbone",
    suite_name="thesis_m7_v4_unified_e8",
    label="Official Backbone",
    description="Unified pure m7_utpm backbone without decision-layer correction.",
    overrides=(),
    run_required=True,
)


ABLATIONS: tuple[BackboneAblation, ...] = (
    BACKBONE_REFERENCE,
    BackboneAblation(
        setting_id="ablate_no_prototype",
        suite_name="thesis_m7_v4_ablate_noprototype",
        label="No Prototype Memory",
        description="Disable prototype loss and prototype memory blending inside the GNN backbone.",
        overrides=(
            "prototype_multiclass_num_classes=0",
            "prototype_loss_weight=0.0",
            "prototype_loss_weight_schedule=none",
            "prototype_loss_min_weight=0.0",
            "prototype_neighbor_blend=0.0",
            "prototype_global_blend=0.0",
            "prototype_consistency_weight=0.0",
            "prototype_separation_weight=0.0",
        ),
    ),
    BackboneAblation(
        setting_id="ablate_no_pseudocontrast",
        suite_name="thesis_m7_v4_ablate_nopseudocontrast",
        label="No Pseudo-Contrastive Mining",
        description="Disable temporal pseudo-contrastive mining inside the GNN backbone.",
        overrides=(
            "pseudo_contrastive_weight=0.0",
        ),
    ),
    BackboneAblation(
        setting_id="ablate_no_drift_residual",
        suite_name="thesis_m7_v4_ablate_nodriftresidual",
        label="No Drift Residual Context",
        description="Disable target-context residual fusion and time-adapter drift correction.",
        overrides=(
            "target_context_fusion=none",
            "target_time_adapter_strength=0.0",
            "target_time_expert_entropy_weight=0.0",
            "normal_bucket_align_weight=0.0",
            "normal_bucket_adv_weight=0.0",
            "context_residual_scale=0.0",
            "context_residual_clip=0.0",
            "context_residual_budget=0.0",
            "context_residual_budget_weight=0.0",
            "context_residual_budget_schedule=none",
            "context_residual_budget_min_weight=0.0",
            "context_residual_budget_release_epochs=0",
            "context_residual_budget_release_delay_epochs=0",
        ),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _path_repr(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _macro_val_auc(results: list[dict[str, Any]]) -> float | None:
    values = [float(row["val_auc_mean"]) for row in results if row.get("val_auc_mean") is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _suite_summary_path(suite_name: str) -> Path:
    return THESIS_SUITE_ROOT / suite_name / "summary.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_suite_command(ablation: BackboneAblation, args: argparse.Namespace) -> list[str]:
    command: list[str] = [
        sys.executable,
        str(REPO_ROOT / "experiment" / "training" / "run_thesis_suite.py"),
        "--suite-name",
        ablation.suite_name,
        "--model",
        OFFICIAL_BACKBONE_MODEL,
        "--preset",
        OFFICIAL_BACKBONE_PRESET,
        "--feature-profile",
        OFFICIAL_BACKBONE_FEATURE_PROFILE,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    if args.build_features:
        command.append("--build-features")
    if args.skip_existing:
        command.append("--skip-existing")
    for override in ablation.overrides:
        command.extend(["--graph-config-override", override])
    return command


def _run_command(command: list[str], *, dry_run: bool) -> None:
    preview = shlex.join(command)
    print(preview)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _load_hybrid_reference() -> dict[str, Any]:
    payload = _load_json(_suite_summary_path(OFFICIAL_HYBRID_SUITE))
    rows: list[dict[str, Any]] = []
    for row in payload.get("results", []):
        rows.append(
            {
                "dataset": row["dataset"],
                "setting_id": "official_gnn_primary_blend",
                "setting_label": "Official GNN-primary Blend",
                "group": "decision_reference",
                "val_auc_mean": row.get("val_auc_mean"),
                "gnn_val_auc": row.get("gnn_val_auc"),
                "secondary_val_auc": row.get("secondary_val_auc"),
                "suite_name": OFFICIAL_HYBRID_SUITE,
                "summary_path": row.get("summary_path"),
                "description": (
                    "Current official thesis result with GNN-primary fixed logit fusion "
                    f"(alpha={float(payload.get('blend_alpha', OFFICIAL_HYBRID_BLEND_ALPHA)):.2f})."
                ),
                "graph_config_overrides": [],
            }
        )
    return {
        "suite_name": OFFICIAL_HYBRID_SUITE,
        "setting_id": "official_gnn_primary_blend",
        "label": "Official GNN-primary Blend",
        "macro_val_auc": _macro_val_auc(rows),
        "results": rows,
    }


def _write_results_files(
    *,
    out_dir: Path,
    manifest: dict[str, Any],
    setting_payloads: list[dict[str, Any]],
    hybrid_reference: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []

    backbone_reference = next(payload for payload in setting_payloads if payload["setting_id"] == BACKBONE_REFERENCE.setting_id)
    reference_by_dataset = {
        row["dataset"]: row["val_auc_mean"]
        for row in backbone_reference["results"]
        if row.get("val_auc_mean") is not None
    }

    for payload in setting_payloads:
        macro_rows.append(
            {
                "group": "backbone_ablation",
                "setting_id": payload["setting_id"],
                "setting_label": payload["label"],
                "suite_name": payload["suite_name"],
                "macro_val_auc": payload["macro_val_auc"],
                "description": payload["description"],
            }
        )
        for row in payload["results"]:
            ref_auc = reference_by_dataset.get(row["dataset"])
            current_auc = row.get("val_auc_mean")
            delta = None if ref_auc is None or current_auc is None else float(current_auc) - float(ref_auc)
            long_rows.append(
                {
                    "group": "backbone_ablation",
                    "setting_id": payload["setting_id"],
                    "setting_label": payload["label"],
                    "dataset": row["dataset"],
                    "suite_name": payload["suite_name"],
                    "val_auc_mean": current_auc,
                    "delta_vs_official_backbone": delta,
                    "summary_path": row["summary_path"],
                    "graph_config_overrides": json.dumps(payload["graph_config_overrides"], ensure_ascii=False),
                    "description": payload["description"],
                }
            )

    macro_rows.append(
        {
            "group": "decision_reference",
            "setting_id": hybrid_reference["setting_id"],
            "setting_label": hybrid_reference["label"],
            "suite_name": hybrid_reference["suite_name"],
            "macro_val_auc": hybrid_reference["macro_val_auc"],
            "description": "Reference official hybrid result for comparison with pure-backbone ablations.",
        }
    )
    for row in hybrid_reference["results"]:
        ref_auc = reference_by_dataset.get(row["dataset"])
        current_auc = row.get("val_auc_mean")
        delta = None if ref_auc is None or current_auc is None else float(current_auc) - float(ref_auc)
        long_rows.append(
            {
                "group": "decision_reference",
                "setting_id": hybrid_reference["setting_id"],
                "setting_label": hybrid_reference["label"],
                "dataset": row["dataset"],
                "suite_name": hybrid_reference["suite_name"],
                "val_auc_mean": current_auc,
                "delta_vs_official_backbone": delta,
                "summary_path": row["summary_path"],
                "graph_config_overrides": "[]",
                "description": row["description"],
            }
        )

    payload = {
        "manifest": manifest,
        "backbone_ablations": setting_payloads,
        "decision_reference": hybrid_reference,
        "long_rows": long_rows,
        "macro_rows": macro_rows,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "results_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "setting_id",
                "setting_label",
                "dataset",
                "suite_name",
                "val_auc_mean",
                "delta_vs_official_backbone",
                "summary_path",
                "graph_config_overrides",
                "description",
            ],
        )
        writer.writeheader()
        writer.writerows(long_rows)

    with (out_dir / "results_macro.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "setting_id",
                "setting_label",
                "suite_name",
                "macro_val_auc",
                "description",
            ],
        )
        writer.writeheader()
        writer.writerows(macro_rows)

    dataset_order = [row["dataset"] for row in backbone_reference["results"]]
    lines = [
        "# Thesis Backbone Ablation Report",
        "",
        f"- Created at: `{manifest['finished_at']}`",
        f"- Output root: `{_path_repr(out_dir)}`",
        "",
        "## Backbone Ablation Table",
        "",
        "| Setting | " + " | ".join(dataset_order) + " | Macro Val AUC |",
        "| --- | " + " | ".join(["---:"] * len(dataset_order)) + " | ---: |",
    ]
    for payload in setting_payloads:
        per_dataset = {row["dataset"]: row["val_auc_mean"] for row in payload["results"]}
        lines.append(
            "| {label} | {dataset_metrics} | {macro_auc} |".format(
                label=payload["label"],
                dataset_metrics=" | ".join(
                    _format_metric(per_dataset.get(dataset_name))
                    for dataset_name in dataset_order
                ),
                macro_auc=_format_metric(payload["macro_val_auc"]),
            )
        )

    lines.extend(
        [
            "",
            "## Decision Reference",
            "",
            "| Setting | " + " | ".join(dataset_order) + " | Macro Val AUC |",
            "| --- | " + " | ".join(["---:"] * len(dataset_order)) + " | ---: |",
        ]
    )
    hybrid_dataset = {row["dataset"]: row["val_auc_mean"] for row in hybrid_reference["results"]}
    lines.append(
        "| {label} | {dataset_metrics} | {macro_auc} |".format(
            label=hybrid_reference["label"],
            dataset_metrics=" | ".join(
                _format_metric(hybrid_dataset.get(dataset_name))
                for dataset_name in dataset_order
            ),
            macro_auc=_format_metric(hybrid_reference["macro_val_auc"]),
        )
    )

    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    for command_payload in manifest["commands"]:
        lines.append(f"- `{command_payload['setting_id']}`: `{command_payload['preview']}`")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tri-dataset m7 backbone ablations and write plotting-friendly result artifacts."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device passed through to run_thesis_suite.py.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=OFFICIAL_SUITE_EPOCHS,
        help="Epochs used for every backbone-ablation suite.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(OFFICIAL_SUITE_SEEDS),
        help="Seeds used for every backbone-ablation suite.",
    )
    parser.add_argument(
        "--build-features",
        action="store_true",
        help="Build features before training each suite when needed.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume mode. Skip already-finished per-dataset runs inside each suite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--output-tag",
        default="thesis_m7_v4_backbone_module_ablation",
        help="Subdirectory name under experiment/outputs/thesis_ablation/ for aggregated artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = THESIS_ABLATION_ROOT / args.output_tag
    started_at = _utc_now()

    commands: list[dict[str, Any]] = []
    setting_payloads: list[dict[str, Any]] = []

    for ablation in ABLATIONS:
        command = _build_suite_command(ablation, args)
        commands.append(
            {
                "setting_id": ablation.setting_id,
                "suite_name": ablation.suite_name,
                "preview": shlex.join(command),
                "graph_config_overrides": list(ablation.overrides),
            }
        )
        _run_command(command, dry_run=args.dry_run)
        if args.dry_run:
            continue

        suite_summary_path = _suite_summary_path(ablation.suite_name)
        suite_payload = _load_json(suite_summary_path)
        results = []
        for row in suite_payload.get("results", []):
            results.append(
                {
                    "dataset": row["dataset"],
                    "val_auc_mean": row.get("val_auc_mean"),
                    "test_auc_mean": row.get("test_auc_mean"),
                    "external_auc_mean": row.get("external_auc_mean"),
                    "summary_path": row["summary_path"],
                }
            )
        setting_payloads.append(
            {
                "setting_id": ablation.setting_id,
                "suite_name": ablation.suite_name,
                "label": ablation.label,
                "description": ablation.description,
                "graph_config_overrides": list(ablation.overrides),
                "suite_summary_path": _path_repr(suite_summary_path),
                "macro_val_auc": _macro_val_auc(results),
                "results": results,
            }
        )

    if args.dry_run:
        return

    hybrid_reference = _load_hybrid_reference()
    manifest = {
        "started_at": started_at,
        "finished_at": _utc_now(),
        "output_dir": _path_repr(output_dir),
        "device": args.device,
        "epochs": int(args.epochs),
        "seeds": [int(seed) for seed in args.seeds],
        "commands": commands,
        "official_hybrid_reference_suite": OFFICIAL_HYBRID_SUITE,
    }
    _write_results_files(
        out_dir=output_dir,
        manifest=manifest,
        setting_payloads=setting_payloads,
        hybrid_reference=hybrid_reference,
    )
    print(f"Ablation report ready: {_path_repr(output_dir / 'report.md')}")


if __name__ == "__main__":
    main()
