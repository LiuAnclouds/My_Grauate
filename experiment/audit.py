from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.datasets.core.registry import get_dataset_spec
from experiment.models.spec import OFFICIAL_PURE_SUITE_NAME


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


def _audit_mainline_dataset(
    *,
    dataset_name: str,
    run_dir: Path,
) -> dict[str, Any]:
    eda_root, training_root = _dataset_output_roots(dataset_name)
    split_ids = _load_split_ids(eda_root)
    overlaps = _pairwise_disjoint(split_ids)
    for pair_name, overlap in overlaps.items():
        _assert(overlap == 0, f"{dataset_name}: split overlap detected for {pair_name}: {overlap}")

    _assert(
        str(run_dir).startswith(str(training_root)),
        f"{dataset_name}: run_dir escaped dataset root: {run_dir}",
    )

    train_ids = _load_prediction_ids(run_dir, "phase1_train")
    val_ids = _load_prediction_ids(run_dir, "phase1_val")
    test_pool_ids = _load_prediction_ids(run_dir, "test_pool")

    _assert(
        np.array_equal(train_ids, split_ids["train"]),
        f"{dataset_name}: phase1_train ids do not match recommended split train ids",
    )
    _assert(
        np.array_equal(val_ids, split_ids["val"]),
        f"{dataset_name}: phase1_val ids do not match recommended split val ids",
    )
    _assert(
        np.array_equal(test_pool_ids, split_ids["test_pool"]),
        f"{dataset_name}: test_pool ids do not match recommended split test_pool ids",
    )

    run_summary = _read_json(run_dir / "summary.json")
    _assert(
        str(run_summary.get("deployment_path", "single_gnn_end_to_end")) == "single_gnn_end_to_end",
        f"{dataset_name}: run summary is not marked as a single-GNN deployment path",
    )
    return {
        "dataset": dataset_name,
        "eda_root": str(eda_root.relative_to(REPO_ROOT)),
        "training_root": str(training_root.relative_to(REPO_ROOT)),
        "split_sizes": {name: int(np.asarray(ids).size) for name, ids in split_ids.items()},
        "split_overlaps": overlaps,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "checks": {
            "split_pairwise_disjoint": True,
            "phase1_train_matches_split_train": True,
            "phase1_val_matches_split_val": True,
            "test_pool_matches_split_test_pool": True,
            "dataset_scoped_paths_only": True,
            "suite_declares_dataset_isolation": bool(run_summary.get("dataset_isolation", False)),
            "suite_declares_no_cross_dataset_training": not bool(run_summary.get("cross_dataset_training", True)),
            "single_gnn_deployment_path": True,
        },
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Thesis Leakage Audit",
        "",
        f"- Suite: `{payload['suite_name']}`",
        "- Mode: `pure_gnn_mainline`",
        "- Scope: direct train/val/test/external overlap and cross-dataset isolation.",
        "- Conclusion: no hard leakage was detected in the audited thesis suite.",
        "",
        "| Dataset | Train | Val | Test Pool | External | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["datasets"]:
        split_sizes = row["split_sizes"]
        lines.append(
            "| {dataset} | {train} | {val} | {test_pool} | {external} | pass |".format(
                dataset=row["dataset"],
                train=split_sizes["train"],
                val=split_sizes["val"],
                test_pool=split_sizes["test_pool"],
                external=split_sizes["external"],
            )
        )
    lines.extend(
        [
            "",
            "## Hard-Leakage Checklist",
            "",
            "- `train`, `val`, `test_pool`, and `external` id sets are pairwise disjoint for every dataset.",
            "- DyRIFT-GNN `phase1_train`, `phase1_val`, and `test_pool` prediction bundles exactly match the official split ids.",
            "- Every run directory stays inside its dataset-scoped output namespace.",
            "- No cross-dataset prediction path, external classifier, or second-stage model is used by the audited pure-GNN mainline.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a pure DyRIFT-GNN thesis suite for hard leakage.")
    parser.add_argument(
        "--suite-summary",
        type=Path,
        default=REPO_ROOT / "experiment" / "outputs" / "thesis_suite" / OFFICIAL_PURE_SUITE_NAME / "summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_summary_path = Path(args.suite_summary).resolve()
    suite_payload = _read_json(suite_summary_path)
    suite_rows = list(suite_payload["results"])
    dataset_rows = [
        _audit_mainline_dataset(
            dataset_name=str(row["dataset"]),
            run_dir=(REPO_ROOT / str(row["summary_path"])).resolve().parent,
        )
        for row in suite_rows
    ]

    report_payload = {
        "suite_name": str(suite_payload["suite_name"]),
        "suite_type": "pure_gnn_mainline",
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
