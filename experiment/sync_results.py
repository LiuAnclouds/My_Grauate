from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "docs" / "results"


DATASETS: tuple[tuple[str, str, str], ...] = (
    ("xinye_dgraph", "XinYe DGraph", "xinye_auc"),
    ("elliptic_transactions", "Elliptic Transactions", "et_auc"),
    ("ellipticpp_transactions", "Elliptic++ Transactions", "epp_auc"),
)


@dataclass(frozen=True)
class ResultSpec:
    group: str
    study: str
    setting: str
    dataset: str
    summary_path: str
    source: str = "study"
    snapshot: bool = True
    manifest: bool = True

    @property
    def dataset_display_name(self) -> str:
        return dataset_display_name(self.dataset)


@dataclass(frozen=True)
class LoadedResult:
    spec: ResultSpec
    payload: dict[str, Any]
    val_auc: float
    train_auc: float | None
    phase2_train_auc: float | None
    phase2_holdout_auc: float | None


def dataset_display_name(dataset: str) -> str:
    for dataset_key, display_name, _column in DATASETS:
        if dataset == dataset_key:
            return display_name
    raise KeyError(f"Unknown dataset key: {dataset}")


def tri_specs(
    *,
    group: str,
    study: str,
    setting: str,
    path_template: str,
    source: str = "study",
    snapshot: bool = True,
    manifest: bool = True,
) -> list[ResultSpec]:
    return [
        ResultSpec(
            group=group,
            study=study,
            setting=setting,
            dataset=dataset,
            summary_path=path_template.format(dataset=dataset),
            source=source,
            snapshot=snapshot,
            manifest=manifest,
        )
        for dataset, _display_name, _column in DATASETS
    ]


MAINLINE_SPECS: list[ResultSpec] = [
    ResultSpec(
        group="main",
        study="dyrift_full_model_accepted",
        setting="Full DyRIFT-GNN",
        dataset="xinye_dgraph",
        summary_path="experiment/outputs/training/models/dyrift_gnn/full_xinye_repro_v1/summary.json",
        source="accepted_mainline",
    ),
    ResultSpec(
        group="main",
        study="dyrift_full_model_accepted",
        setting="Full DyRIFT-GNN",
        dataset="elliptic_transactions",
        summary_path=(
            "experiment/outputs/elliptic_transactions/training/models/dyrift_gnn/"
            "probe_et_dyrift_pure_compact_ctx3_h4_delaypc_timew_hl20_f035_v1/summary.json"
        ),
        source="accepted_mainline",
    ),
    ResultSpec(
        group="main",
        study="dyrift_full_model_accepted",
        setting="Full DyRIFT-GNN",
        dataset="ellipticpp_transactions",
        summary_path=(
            "experiment/outputs/ellipticpp_transactions/training/models/dyrift_gnn/"
            "probe_epp_dyrift_pure_ap96_mixed120_timew_hl20_f035_coldctx_v1/summary.json"
        ),
        source="accepted_mainline",
    ),
]


COMPARISON_SPECS: list[ResultSpec] = [
    *tri_specs(
        group="comparisons",
        study="plain_trgt_backbone",
        setting="Plain TRGT Backbone",
        path_template="experiment/outputs/studies/comparisons/plain_trgt_backbone/{dataset}/summary.json",
    ),
    *tri_specs(
        group="comparisons",
        study="tgat_style_reference",
        setting="TGAT-style Reference",
        path_template="experiment/outputs/studies/comparisons/tgat_style_reference/{dataset}/summary.json",
    ),
    *tri_specs(
        group="comparisons",
        study="temporal_graphsage_reference",
        setting="Temporal GraphSAGE Reference",
        path_template="experiment/outputs/studies/comparisons/temporal_graphsage_reference/{dataset}/summary.json",
    ),
    *tri_specs(
        group="comparisons",
        study="xgboost_same_input",
        setting="XGBoost Same Input",
        path_template="experiment/outputs/studies/comparisons/xgboost_same_input/{dataset}/summary.json",
    ),
]


ABLATION_SPECS: list[ResultSpec] = [
    *tri_specs(
        group="ablations",
        study="without_target_context_bridge",
        setting="w/o Target-Context Bridge",
        path_template="experiment/outputs/studies/ablations/without_target_context_bridge/{dataset}/summary.json",
    ),
    *tri_specs(
        group="ablations",
        study="without_drift_expert",
        setting="w/o Drift Expert",
        path_template="experiment/outputs/studies/ablations/without_drift_expert/{dataset}/summary.json",
    ),
    *tri_specs(
        group="ablations",
        study="without_prototype_memory",
        setting="w/o Prototype Memory",
        path_template="experiment/outputs/studies/ablations/without_prototype_memory/{dataset}/summary.json",
    ),
    *tri_specs(
        group="ablations",
        study="without_pseudo_contrastive",
        setting="w/o Pseudo-Contrastive Temporal Mining",
        path_template="experiment/outputs/studies/ablations/without_pseudo_contrastive/{dataset}/summary.json",
    ),
]


PROGRESSIVE_SPECS: list[ResultSpec] = [
    *tri_specs(
        group="progressive",
        study="trgt_bridge",
        setting="TRGT + Bridge",
        path_template="experiment/outputs/studies/progressive/trgt_bridge/{dataset}/summary.json",
    ),
    *tri_specs(
        group="progressive",
        study="trgt_bridge_drift",
        setting="TRGT + Bridge + Drift Expert",
        path_template="experiment/outputs/studies/progressive/trgt_bridge_drift/{dataset}/summary.json",
    ),
    *tri_specs(
        group="progressive",
        study="trgt_bridge_drift_prototype",
        setting="TRGT + Bridge + Drift Expert + Prototype Memory",
        path_template="experiment/outputs/studies/progressive/trgt_bridge_drift_prototype/{dataset}/summary.json",
    ),
    *tri_specs(
        group="progressive",
        study="trgt_bridge_drift_prototype_pseudocontrastive",
        setting="TRGT + Bridge + Drift Expert + Prototype Memory + Pseudo-Contrastive",
        path_template=(
            "experiment/outputs/studies/progressive/"
            "trgt_bridge_drift_prototype_pseudocontrastive/{dataset}/summary.json"
        ),
    ),
]


SUPPLEMENTARY_SPECS: list[ResultSpec] = [
    ResultSpec(
        group="supplementary",
        study="xinye_phase12_joint_train_phase1_val",
        setting="XinYe Joint Train on Phase1+Phase2 with Phase1 Validation",
        dataset="xinye_dgraph",
        summary_path=(
            "experiment/outputs/studies/supplementary/"
            "xinye_phase12_joint_train_phase1_val/xinye_dgraph/summary.json"
        ),
    ),
    ResultSpec(
        group="supplementary",
        study="xinye_phase12_phase_aware_balanced",
        setting="XinYe Phase-Aware Balanced Joint Training",
        dataset="xinye_dgraph",
        summary_path=(
            "experiment/outputs/studies/supplementary/"
            "xinye_phase12_phase_aware_balanced/xinye_dgraph/summary.json"
        ),
        snapshot=False,
    ),
    ResultSpec(
        group="supplementary",
        study="xinye_phase12_phase_aware_dualval",
        setting="XinYe Phase-Aware DualVal Joint Training",
        dataset="xinye_dgraph",
        summary_path=(
            "experiment/outputs/studies/supplementary/"
            "xinye_phase12_phase_aware_dualval/xinye_dgraph/summary.json"
        ),
        snapshot=False,
    ),
]


ALL_RESULT_SPECS: list[ResultSpec] = [
    *MAINLINE_SPECS,
    *COMPARISON_SPECS,
    *ABLATION_SPECS,
    *PROGRESSIVE_SPECS,
    *SUPPLEMENTARY_SPECS,
]


PRESENTATION_NOTES: dict[tuple[str, str], str] = {
    ("mainline", "Full DyRIFT-GNN"): "accepted pure-GNN deployment route",
    ("comparison", "Plain TRGT Backbone"): "backbone only",
    ("comparison", "TGAT-style Reference"): "temporal-attention GNN reference",
    ("comparison", "Temporal GraphSAGE Reference"): "temporal neighbor-aggregation GNN reference",
    (
        "comparison",
        "XGBoost Same Input",
    ): "supplementary non-GNN same-input reference; excluded from GNN deployment claim",
    ("ablation", "w/o Target-Context Bridge"): "subtractive ablation",
    ("ablation", "w/o Drift Expert"): "subtractive ablation",
    ("ablation", "w/o Prototype Memory"): "subtractive ablation",
    ("ablation", "w/o Pseudo-Contrastive Temporal Mining"): "subtractive ablation",
    ("progressive", "Plain TRGT Backbone"): "progressive start",
    ("progressive", "TRGT + Bridge"): "progressive method-building",
    ("progressive", "TRGT + Bridge + Drift Expert"): "progressive method-building",
    ("progressive", "TRGT + Bridge + Drift Expert + Prototype Memory"): "progressive method-building",
    (
        "progressive",
        "TRGT + Bridge + Drift Expert + Prototype Memory + Pseudo-Contrastive",
    ): "progressive method-building",
    ("progressive", "Full DyRIFT-GNN"): "accepted full model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize tracked result tables from saved experiment summary artifacts."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generated output against tracked docs/results files without writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded = load_all_results(ALL_RESULT_SPECS)
    outputs = build_outputs(loaded)
    validate_policy_files()
    if args.check:
        mismatches = compare_outputs(outputs)
        if mismatches:
            for mismatch in mismatches:
                print(mismatch, file=sys.stderr)
            return 1
        print("result tables are in sync")
        return 0
    write_outputs(outputs)
    print(f"wrote {len(outputs)} result files under docs/results")
    return 0


def load_all_results(specs: Iterable[ResultSpec]) -> dict[tuple[str, str, str], LoadedResult]:
    loaded: dict[tuple[str, str, str], LoadedResult] = {}
    for spec in specs:
        summary_path = REPO_ROOT / spec.summary_path
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary artifact: {spec.summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        val_auc = first_numeric(payload, ("val_auc_mean", "phase1_val_auc_mean"), spec.summary_path)
        loaded[(spec.group, spec.setting, spec.dataset)] = LoadedResult(
            spec=spec,
            payload=payload,
            val_auc=val_auc,
            train_auc=first_optional_numeric(payload, ("train_auc_mean", "phase1_train_auc_mean")),
            phase2_train_auc=optional_float(payload.get("phase2_train_auc_mean")),
            phase2_holdout_auc=optional_float(payload.get("phase2_holdout_auc_mean")),
        )
    return loaded


def build_outputs(loaded: dict[tuple[str, str, str], LoadedResult]) -> dict[Path, str]:
    outputs: dict[Path, str] = {}
    snapshot_rows = build_snapshot_rows(loaded)
    outputs[RESULTS_DIR / "thesis_study_results_snapshot.json"] = json_text(snapshot_rows)
    outputs[RESULTS_DIR / "accepted_mainline_summary.json"] = json_text(
        build_accepted_mainline_summary(loaded)
    )
    outputs[RESULTS_DIR / "thesis_dyrift_gnn_trgt_deploy_pure_v1_auc.csv"] = csv_text(
        [
            "group",
            "setting",
            "dataset",
            "dataset_key",
            "val_auc",
            "summary_path",
        ],
        build_mainline_auc_rows(loaded),
    )
    outputs[RESULTS_DIR / "comparison_auc.csv"] = csv_text(
        ["setting", "dataset", "dataset_key", "val_auc", "source"],
        [
            *build_flat_auc_rows(MAINLINE_SPECS, loaded, include_source=True),
            *build_flat_auc_rows(COMPARISON_SPECS, loaded, include_source=True),
        ],
    )
    outputs[RESULTS_DIR / "ablation_auc.csv"] = csv_text(
        ["setting", "dataset", "dataset_key", "val_auc"],
        [
            *build_flat_auc_rows(MAINLINE_SPECS, loaded, include_source=False),
            *build_flat_auc_rows(ABLATION_SPECS, loaded, include_source=False),
        ],
    )
    outputs[RESULTS_DIR / "progressive_auc.csv"] = csv_text(
        ["setting", "dataset", "dataset_key", "val_auc"],
        build_progressive_auc_rows(loaded),
    )
    outputs[RESULTS_DIR / "supplementary_auc.csv"] = csv_text(
        [
            "group",
            "setting",
            "study_name",
            "dataset",
            "dataset_key",
            "val_auc",
            "summary_path",
        ],
        build_supplementary_rows(loaded),
    )
    outputs[RESULTS_DIR / "presentation_auc_percent.csv"] = csv_text(
        [
            "table",
            "setting",
            "xinye_auc",
            "et_auc",
            "epp_auc",
            "macro_auc",
            "delta_vs_full",
            "note",
        ],
        build_presentation_rows(loaded),
    )
    outputs[RESULTS_DIR / "epoch_log_manifest.csv"] = csv_text(
        [
            "group",
            "setting",
            "dataset",
            "dataset_key",
            "best_epoch",
            "trained_epochs",
            "epoch_metrics_path",
            "train_log_path",
            "curve_path",
            "summary_path",
        ],
        build_epoch_manifest_rows(loaded),
    )
    outputs[RESULTS_DIR / "training_policy_summary.json"] = json_text(build_training_policy_summary())
    return outputs


def build_snapshot_rows(loaded: dict[tuple[str, str, str], LoadedResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ALL_RESULT_SPECS:
        if not spec.snapshot:
            continue
        result = loaded[(spec.group, spec.setting, spec.dataset)]
        rows.append(
            {
                "group": spec.group,
                "study": spec.study,
                "dataset": spec.dataset,
                "dataset_display_name": spec.dataset_display_name,
                "val_auc": result.val_auc,
                "summary_path": spec.summary_path,
            }
        )
    return rows


def build_accepted_mainline_summary(
    loaded: dict[tuple[str, str, str], LoadedResult]
) -> dict[str, Any]:
    return {
        "suite_name": "dyrift_gnn_accepted_mainline",
        "model": "dyrift_gnn",
        "method_display_name": "Dynamic Risk-Informed Fraud Graph Neural Network",
        "method_short_name": "DyRIFT-GNN",
        "backbone_display_name": "Temporal-Relational Graph Transformer",
        "backbone_short_name": "TRGT",
        "deployment_path": "single_gnn_end_to_end",
        "dataset_isolation": True,
        "cross_dataset_training": False,
        "same_architecture_across_datasets": True,
        "maintained_rerun_policy": {
            "max_epochs": 70,
            "min_early_stop_epoch": 30,
            "policy_path": "experiment/configs/training_policy.json",
            "note": "Existing accepted artifacts keep their observed epoch logs; future reruns use this policy.",
        },
        "results": [
            {
                "dataset": spec.dataset,
                "summary_path": spec.summary_path,
                "val_auc_mean": loaded[(spec.group, spec.setting, spec.dataset)].val_auc,
            }
            for spec in MAINLINE_SPECS
        ],
    }


def build_mainline_auc_rows(
    loaded: dict[tuple[str, str, str], LoadedResult]
) -> list[dict[str, Any]]:
    rows = [
        {
            "group": "mainline",
            "setting": spec.setting,
            "dataset": spec.dataset_display_name,
            "dataset_key": spec.dataset,
            "val_auc": loaded[(spec.group, spec.setting, spec.dataset)].val_auc,
            "summary_path": spec.summary_path,
        }
        for spec in MAINLINE_SPECS
    ]
    rows.append(
        {
            "group": "mainline",
            "setting": "Full DyRIFT-GNN",
            "dataset": "Macro Average",
            "dataset_key": "macro_average",
            "val_auc": mean(row["val_auc"] for row in rows),
            "summary_path": "docs/results/accepted_mainline_summary.json",
        }
    )
    return rows


def build_flat_auc_rows(
    specs: Iterable[ResultSpec],
    loaded: dict[tuple[str, str, str], LoadedResult],
    *,
    include_source: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        result = loaded[(spec.group, spec.setting, spec.dataset)]
        row = {
            "setting": spec.setting,
            "dataset": spec.dataset_display_name,
            "dataset_key": spec.dataset,
            "val_auc": result.val_auc,
        }
        if include_source:
            row["source"] = spec.source
        rows.append(row)
    return rows


def build_progressive_auc_rows(
    loaded: dict[tuple[str, str, str], LoadedResult]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        build_flat_auc_rows(
            [spec for spec in COMPARISON_SPECS if spec.study == "plain_trgt_backbone"],
            loaded,
            include_source=False,
        )
    )
    rows.extend(build_flat_auc_rows(PROGRESSIVE_SPECS, loaded, include_source=False))
    rows.extend(build_flat_auc_rows(MAINLINE_SPECS, loaded, include_source=False))
    return rows


def build_supplementary_rows(
    loaded: dict[tuple[str, str, str], LoadedResult]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SUPPLEMENTARY_SPECS:
        result = loaded[(spec.group, spec.setting, spec.dataset)]
        rows.append(
            {
                "group": spec.group,
                "setting": spec.setting,
                "study_name": spec.study,
                "dataset": spec.dataset_display_name,
                "dataset_key": spec.dataset,
                "val_auc": result.val_auc,
                "summary_path": spec.summary_path,
            }
        )
    return rows


def build_presentation_rows(
    loaded: dict[tuple[str, str, str], LoadedResult]
) -> list[dict[str, Any]]:
    full_values = values_by_dataset(MAINLINE_SPECS, loaded)
    full_macro = mean(full_values.values())
    row_specs: list[tuple[str, str, dict[str, float]]] = [
        ("mainline", "Full DyRIFT-GNN", full_values),
    ]
    row_specs.extend(grouped_presentation_specs("comparison", COMPARISON_SPECS, loaded))
    row_specs.extend(grouped_presentation_specs("ablation", ABLATION_SPECS, loaded))
    row_specs.extend(
        [
            (
                "progressive",
                "Plain TRGT Backbone",
                values_by_dataset(
                    [spec for spec in COMPARISON_SPECS if spec.study == "plain_trgt_backbone"],
                    loaded,
                ),
            )
        ]
    )
    row_specs.extend(grouped_presentation_specs("progressive", PROGRESSIVE_SPECS, loaded))
    row_specs.append(("progressive", "Full DyRIFT-GNN", full_values))

    rows: list[dict[str, Any]] = []
    for table, setting, values in row_specs:
        macro = mean(values.values())
        rows.append(
            {
                "table": table,
                "setting": setting,
                "xinye_auc": format_percent(values.get("xinye_dgraph")),
                "et_auc": format_percent(values.get("elliptic_transactions")),
                "epp_auc": format_percent(values.get("ellipticpp_transactions")),
                "macro_auc": format_percent(macro),
                "delta_vs_full": format_pp(macro - full_macro),
                "note": PRESENTATION_NOTES[(table, setting)],
            }
        )
    return rows


def grouped_presentation_specs(
    table: str,
    specs: Iterable[ResultSpec],
    loaded: dict[tuple[str, str, str], LoadedResult],
) -> list[tuple[str, str, dict[str, float]]]:
    rows: list[tuple[str, str, dict[str, float]]] = []
    seen: list[str] = []
    by_setting: dict[str, list[ResultSpec]] = {}
    for spec in specs:
        if spec.setting not in by_setting:
            seen.append(spec.setting)
            by_setting[spec.setting] = []
        by_setting[spec.setting].append(spec)
    for setting in seen:
        rows.append((table, setting, values_by_dataset(by_setting[setting], loaded)))
    return rows


def values_by_dataset(
    specs: Iterable[ResultSpec],
    loaded: dict[tuple[str, str, str], LoadedResult],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for spec in specs:
        values[spec.dataset] = loaded[(spec.group, spec.setting, spec.dataset)].val_auc
    return values


def build_epoch_manifest_rows(
    loaded: dict[tuple[str, str, str], LoadedResult]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ALL_RESULT_SPECS:
        if not spec.manifest:
            continue
        result = loaded[(spec.group, spec.setting, spec.dataset)]
        seed_metric = first_seed_metric(result.payload)
        epoch_metrics_path = string_value(seed_metric.get("epoch_metrics_path"))
        rows.append(
            {
                "group": "mainline" if spec.group == "main" else spec.group,
                "setting": spec.setting,
                "dataset": spec.dataset_display_name,
                "dataset_key": spec.dataset,
                "best_epoch": value_or_blank(
                    seed_metric.get("best_epoch", result.payload.get("best_epoch_mean"))
                ),
                "trained_epochs": value_or_blank(
                    seed_metric.get(
                        "trained_epochs",
                        result.payload.get(
                            "trained_epochs_mean",
                            infer_trained_epochs(result.payload, epoch_metrics_path),
                        ),
                    )
                ),
                "epoch_metrics_path": epoch_metrics_path,
                "train_log_path": string_value(seed_metric.get("train_log_path")),
                "curve_path": string_value(seed_metric.get("curve_path")),
                "summary_path": spec.summary_path,
            }
        )
    return rows


def build_training_policy_summary() -> dict[str, Any]:
    policy_path = REPO_ROOT / "experiment" / "configs" / "training_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return {
        "policy_name": policy["policy_name"],
        "max_epochs": int(policy["max_epochs"]),
        "min_early_stop_epoch": int(policy["min_early_stop_epoch"]),
        "applies_to": [
            "mainline",
            "comparisons",
            "ablations",
            "progressive",
            "supplementary",
        ],
        "graph_config_key": "min_early_stop_epoch",
        "xgboost_policy": {
            "num_boost_round": int(policy["xgboost_num_boost_round"]),
            "early_stopping_rounds": int(policy["xgboost_early_stopping_rounds"]),
        },
        "machine_readable_config": "experiment/configs/training_policy.json",
        "epoch_policy_table": "docs/results/experiment_epoch_policy.csv",
        "actual_artifact_manifest": "docs/results/epoch_log_manifest.csv",
        "artifact_integrity_note": (
            "The epoch policy describes maintained reruns. Existing saved epoch curves "
            "and summary metrics remain observed artifacts and are not expanded by synthetic rows."
        ),
    }


def validate_policy_files() -> None:
    policy = json.loads(
        (REPO_ROOT / "experiment" / "configs" / "training_policy.json").read_text(
            encoding="utf-8"
        )
    )
    if int(policy.get("max_epochs", -1)) != 70:
        raise ValueError("training_policy.json must keep max_epochs=70 for maintained reruns.")
    if int(policy.get("min_early_stop_epoch", -1)) != 30:
        raise ValueError(
            "training_policy.json must keep min_early_stop_epoch=30 for maintained reruns."
        )
    if int(policy.get("xgboost_num_boost_round", -1)) != 70:
        raise ValueError(
            "training_policy.json must keep xgboost_num_boost_round=70 for maintained reruns."
        )
    if int(policy.get("xgboost_early_stopping_rounds", -1)) != 30:
        raise ValueError(
            "training_policy.json must keep xgboost_early_stopping_rounds=30."
        )
    epoch_policy_path = RESULTS_DIR / "experiment_epoch_policy.csv"
    if not epoch_policy_path.exists():
        return
    with epoch_policy_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["planned_max_epochs"]) != 70:
                raise ValueError(
                    f"experiment_epoch_policy.csv row {row['setting']} must use planned_max_epochs=70."
                )
            if int(row["min_early_stop_epoch"]) != 30:
                raise ValueError(
                    f"experiment_epoch_policy.csv row {row['setting']} must use min_early_stop_epoch=30."
                )


def compare_outputs(outputs: dict[Path, str]) -> list[str]:
    mismatches: list[str] = []
    for path, generated in outputs.items():
        if not path.exists():
            mismatches.append(f"missing tracked output: {relative(path)}")
            continue
        current = path.read_text(encoding="utf-8")
        if current != generated:
            mismatches.append(f"out of sync: {relative(path)}")
    return mismatches


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def first_seed_metric(payload: dict[str, Any]) -> dict[str, Any]:
    seed_metrics = payload.get("seed_metrics")
    if isinstance(seed_metrics, list) and seed_metrics:
        first = seed_metrics[0]
        if isinstance(first, dict):
            return first
    return {}


def infer_trained_epochs(payload: dict[str, Any], epoch_metrics_path: str) -> Any:
    graph_config = payload.get("graph_config")
    if isinstance(graph_config, dict) and graph_config.get("epochs") is not None:
        return graph_config["epochs"]
    if epoch_metrics_path:
        path = REPO_ROOT / epoch_metrics_path
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                return sum(1 for _row in csv.DictReader(handle))
    return ""


def required_float(payload: dict[str, Any], key: str, location: str) -> float:
    value = optional_float(payload.get(key))
    if value is None:
        raise ValueError(f"Missing numeric {key} in {location}")
    return value


def first_numeric(payload: dict[str, Any], keys: tuple[str, ...], location: str) -> float:
    for key in keys:
        value = optional_float(payload.get(key))
        if value is not None:
            return value
    raise ValueError(f"Missing numeric {'/'.join(keys)} in {location}")


def first_optional_numeric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = optional_float(payload.get(key))
        if value is not None:
            return value
    return None


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def mean(values: Iterable[float]) -> float:
    value_list = [float(value) for value in values]
    if not value_list:
        raise ValueError("Cannot compute mean of an empty sequence.")
    return sum(value_list) / len(value_list)


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.4f}%"


def format_pp(delta_auc: float) -> str:
    return f"{delta_auc * 100.0:+.4f} pp"


def value_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    return value


def string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
