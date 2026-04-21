from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.registry import get_dataset_spec
from experiment.training.thesis_contract import OFFICIAL_HYBRID_SUITE_NAME


def _dataset_output_roots(dataset_name: str) -> tuple[Path, Path]:
    spec = get_dataset_spec(dataset_name)
    outputs_root = REPO_ROOT / "experiment" / "outputs"
    if spec.uses_legacy_output_layout:
        return outputs_root / "eda", outputs_root / "training"
    dataset_root = outputs_root / spec.output_namespace
    return dataset_root / "eda", dataset_root / "training"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_split_ids(eda_root: Path) -> dict[str, np.ndarray]:
    payload = _read_json(eda_root / "recommended_split.json")

    def _load(name: str) -> np.ndarray:
        node_info = payload.get(name)
        if node_info is None:
            return np.empty(0, dtype=np.int32)
        rel_path = node_info.get("id_path")
        if not rel_path:
            return np.empty(0, dtype=np.int32)
        return np.asarray(np.load(eda_root / rel_path), dtype=np.int32)

    return {
        "train": _load("train_split"),
        "val": _load("val_split"),
        "test_pool": _load("test_pool") if "test_pool" in payload else _load("unlabeled_pool"),
        "external": _load("external_eval"),
    }


def _load_prediction_ids(run_dir: Path, split_name: str) -> np.ndarray:
    candidates = (
        run_dir / f"{split_name}_avg_predictions.npz",
        run_dir / f"{split_name}_predictions.npz",
        run_dir / f"{split_name}_blend_predictions.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            data = np.load(candidate)
            key = "node_ids" if "node_ids" in data.files else "ids"
            return np.asarray(data[key], dtype=np.int32)
    raise FileNotFoundError(f"{run_dir}: missing prediction bundle for {split_name}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _pairwise_disjoint(split_ids: dict[str, np.ndarray]) -> dict[str, int]:
    names = ["train", "val", "test_pool", "external"]
    overlaps: dict[str, int] = {}
    for idx, left_name in enumerate(names):
        left = np.asarray(split_ids[left_name], dtype=np.int32)
        for right_name in names[idx + 1 :]:
            right = np.asarray(split_ids[right_name], dtype=np.int32)
            overlap = int(np.intersect1d(left, right).size)
            overlaps[f"{left_name}__{right_name}"] = overlap
    return overlaps


def _audit_dataset(
    *,
    dataset_name: str,
    base_run_dir: Path,
    secondary_run_dir: Path,
    hybrid_run_dir: Path,
) -> dict[str, Any]:
    eda_root, training_root = _dataset_output_roots(dataset_name)
    split_ids = _load_split_ids(eda_root)
    overlaps = _pairwise_disjoint(split_ids)
    for pair_name, overlap in overlaps.items():
        _assert(overlap == 0, f"{dataset_name}: split overlap detected for {pair_name}: {overlap}")

    _assert(str(base_run_dir).startswith(str(training_root)), f"{dataset_name}: base_run_dir escaped dataset root")
    _assert(
        str(secondary_run_dir).startswith(str(training_root)),
        f"{dataset_name}: secondary_run_dir escaped dataset root",
    )
    _assert(str(hybrid_run_dir).startswith(str(training_root)), f"{dataset_name}: hybrid_run_dir escaped dataset root")

    base_train_ids = _load_prediction_ids(base_run_dir, "phase1_train")
    base_val_ids = _load_prediction_ids(base_run_dir, "phase1_val")
    secondary_train_ids = _load_prediction_ids(secondary_run_dir, "phase1_train")
    secondary_val_ids = _load_prediction_ids(secondary_run_dir, "phase1_val")
    hybrid_val_ids = _load_prediction_ids(hybrid_run_dir, "phase1_val")

    _assert(
        np.array_equal(base_train_ids, split_ids["train"]),
        f"{dataset_name}: backbone phase1_train ids do not match recommended split train ids",
    )
    _assert(
        np.array_equal(base_val_ids, split_ids["val"]),
        f"{dataset_name}: backbone phase1_val ids do not match recommended split val ids",
    )
    _assert(
        np.array_equal(secondary_val_ids, split_ids["val"]),
        f"{dataset_name}: secondary phase1_val ids do not match recommended split val ids",
    )
    _assert(
        np.array_equal(hybrid_val_ids, split_ids["val"]),
        f"{dataset_name}: hybrid phase1_val ids do not match recommended split val ids",
    )
    _assert(
        np.intersect1d(secondary_train_ids, split_ids["val"]).size == 0,
        f"{dataset_name}: secondary train ids overlap validation ids",
    )
    _assert(
        np.intersect1d(secondary_train_ids, split_ids["test_pool"]).size == 0,
        f"{dataset_name}: secondary train ids overlap test_pool ids",
    )
    _assert(
        np.intersect1d(secondary_train_ids, split_ids["external"]).size == 0,
        f"{dataset_name}: secondary train ids overlap external ids",
    )
    _assert(
        np.all(np.isin(secondary_train_ids, split_ids["train"])),
        f"{dataset_name}: secondary train ids are not a subset of split-train ids",
    )

    secondary_summary = _read_json(secondary_run_dir / "summary.json")
    hybrid_summary = _read_json(hybrid_run_dir / "summary.json")

    return {
        "dataset": dataset_name,
        "eda_root": str(eda_root.relative_to(REPO_ROOT)),
        "training_root": str(training_root.relative_to(REPO_ROOT)),
        "split_sizes": {name: int(np.asarray(ids).size) for name, ids in split_ids.items()},
        "split_overlaps": overlaps,
        "base_run_dir": str(base_run_dir.relative_to(REPO_ROOT)),
        "secondary_run_dir": str(secondary_run_dir.relative_to(REPO_ROOT)),
        "hybrid_run_dir": str(hybrid_run_dir.relative_to(REPO_ROOT)),
        "secondary_train_size": int(secondary_train_ids.size),
        "secondary_selection_mode": str(secondary_summary.get("historical_selection_mode")),
        "secondary_min_train_first_active_day": int(secondary_summary.get("min_train_first_active_day", 0)),
        "checks": {
            "split_pairwise_disjoint": True,
            "base_train_matches_split_train": True,
            "base_val_matches_split_val": True,
            "secondary_val_matches_split_val": True,
            "hybrid_val_matches_split_val": True,
            "secondary_train_subset_of_split_train": True,
            "secondary_train_has_no_val_overlap": True,
            "secondary_train_has_no_test_pool_overlap": True,
            "secondary_train_has_no_external_overlap": True,
            "dataset_scoped_paths_only": True,
            "suite_declares_dataset_isolation": bool(hybrid_summary.get("dataset_isolation", False)),
            "suite_declares_no_cross_dataset_training": not bool(hybrid_summary.get("cross_dataset_training", True)),
        },
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Thesis Leakage Audit",
        "",
        f"- Suite: `{payload['suite_name']}`",
        "- Scope: direct train/val/test/external overlap and cross-dataset isolation.",
        "- Conclusion: no hard leakage was detected in the audited thesis suite.",
        "",
        "| Dataset | Train | Val | Test Pool | External | Secondary Train | Selection Mode | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["datasets"]:
        split_sizes = row["split_sizes"]
        lines.append(
            "| {dataset} | {train} | {val} | {test_pool} | {external} | {secondary_train} | {mode} | pass |".format(
                dataset=row["dataset"],
                train=split_sizes["train"],
                val=split_sizes["val"],
                test_pool=split_sizes["test_pool"],
                external=split_sizes["external"],
                secondary_train=row["secondary_train_size"],
                mode=row["secondary_selection_mode"],
            )
        )
    lines.extend(
        [
            "",
            "## Hard-Leakage Checklist",
            "",
            "- `train`, `val`, `test_pool`, and `external` id sets are pairwise disjoint for every dataset.",
            "- Every secondary training row is a subset of the dataset's `phase1_train` split.",
            "- Secondary and hybrid validation bundles exactly match the official validation ids.",
            "- Run directories stay inside dataset-scoped output namespaces; no cross-dataset prediction path is reused.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a thesis suite for hard leakage.")
    parser.add_argument(
        "--suite-summary",
        type=Path,
        default=REPO_ROOT / "experiment" / "outputs" / "thesis_suite" / OFFICIAL_HYBRID_SUITE_NAME / "summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_summary_path = Path(args.suite_summary).resolve()
    suite_payload = _read_json(suite_summary_path)
    dataset_rows: list[dict[str, Any]] = []
    for row in suite_payload["results"]:
        dataset_rows.append(
            _audit_dataset(
                dataset_name=str(row["dataset"]),
                base_run_dir=(REPO_ROOT / str(row["base_run_dir"])).resolve(),
                secondary_run_dir=(REPO_ROOT / str(row["secondary_run_dir"])).resolve(),
                hybrid_run_dir=(REPO_ROOT / str(row["summary_path"])).resolve().parent,
            )
        )

    report_payload = {
        "suite_name": str(suite_payload["suite_name"]),
        "hard_leakage_detected": False,
        "datasets": dataset_rows,
    }
    out_dir = suite_summary_path.parent
    json_path = out_dir / "leakage_audit.json"
    md_path = out_dir / "leakage_audit.md"
    json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown_report(report_payload), encoding="utf-8")
    print(f"Leakage audit written: {json_path}")
    print(f"Leakage audit written: {md_path}")


if __name__ == "__main__":
    main()
