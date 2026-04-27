from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from dyrift.data_processing.core.registry import get_dataset_spec
from dyrift.models.spec import OFFICIAL_TRAIN_EPOCHS
from dyrift.utils.common import ensure_dir, read_json, write_json
from experiments.common.contracts import REPO_ROOT, load_experiment_config, resolve_dataset_plan
from experiments.common.graph_runner import _resolve_graph_config


FULL_MODEL_DISPLAY_NAME = "DyRIFT-TGAT (Full)"
POLICY_MIN_EARLY_STOP_EPOCH = 0
DATASET_ORDER = ("xinye_dgraph", "elliptic_transactions", "ellipticpp_transactions")

COMPARISON_ORDER = (
    FULL_MODEL_DISPLAY_NAME,
    "Linear Same-Feature Baseline",
    "Temporal GraphSAGE Reference",
    "TGAT Backbone Reference",
)
ABLATION_ORDER = (
    FULL_MODEL_DISPLAY_NAME,
    "DyRIFT-TGAT w/o Target-Context Bridge",
    "DyRIFT-TGAT w/o Drift Expert",
    "DyRIFT-TGAT w/o Prototype Memory",
)


def _display_path(path: str | Path, *, repo_root: Path | None) -> str:
    path_obj = Path(path)
    if repo_root is None:
        return str(path_obj)
    try:
        return path_obj.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path_obj)


@dataclass(frozen=True)
class RecoveredRun:
    experiment_name: str
    display_name: str
    experiment_type: str
    model_name: str
    dataset_name: str
    dataset_display_name: str
    output_dir: Path
    source_config: Path | None
    metric_source: Path
    best_val_auc: float
    best_epoch: int
    trained_epochs: int
    last_epoch: int
    configured_total_epochs: int | None
    configured_stop_floor: int | None
    configured_policy_compliant: bool | None
    observed_policy_compliant: bool
    artifact_paths: dict[str, str]
    notes: list[str]

    def to_dict(self, *, repo_root: Path | None = None) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "method": self.display_name,
            "experiment_type": self.experiment_type,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "dataset_display_name": self.dataset_display_name,
            "best_val_auc": float(self.best_val_auc),
            "best_epoch": int(self.best_epoch),
            "trained_epochs": int(self.trained_epochs),
            "metric_source": _display_path(self.metric_source, repo_root=repo_root),
        }


def recover_results_bundle(
    *,
    repo_root: Path = REPO_ROOT,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    docs_dir = (docs_dir or (repo_root / "docs" / "generated")).resolve()

    recovered_runs: list[RecoveredRun] = []
    missing_runs: list[dict[str, str]] = []

    recovered_runs.extend(_recover_full_training_runs(repo_root=repo_root))

    experiment_root = repo_root / "experiments"
    for config_path in sorted(experiment_root.glob("*/*/config.json")):
        experiment = load_experiment_config(config_path.parent)
        for dataset_name in experiment.datasets:
            dataset_dir = experiment.dataset_output_dir(dataset_name)
            epoch_metrics_path = dataset_dir / "epoch_metrics.csv"
            if not epoch_metrics_path.exists():
                missing_runs.append(
                    {
                        "experiment_name": experiment.experiment_name,
                        "display_name": experiment.display_name,
                        "dataset_name": str(dataset_name),
                        "expected_output_dir": str(dataset_dir),
                    }
                )
                continue
            recovered_runs.append(
                _recover_experiment_run(
                    experiment=experiment,
                    dataset_name=str(dataset_name),
                )
            )

    full_by_dataset = {
        run.dataset_name: run.best_val_auc
        for run in recovered_runs
        if run.experiment_name == "full_dyrift_gnn"
    }
    run_payloads = []
    for run in _sort_runs(recovered_runs):
        payload = run.to_dict(repo_root=repo_root)
        full_value = full_by_dataset.get(run.dataset_name)
        payload["delta_vs_full"] = (
            None if full_value is None else float(run.best_val_auc - float(full_value))
        )
        run_payloads.append(payload)

    comparison_runs = [payload for payload in run_payloads if payload["experiment_type"] in {"full_model", "comparison"}]
    ablation_runs = [payload for payload in run_payloads if payload["experiment_type"] in {"full_model", "ablation"}]

    tables = {
        "comparisons": _build_table_rows(comparison_runs, method_order=COMPARISON_ORDER),
        "ablations": _build_table_rows(ablation_runs, method_order=ABLATION_ORDER),
    }
    ensure_dir(docs_dir)
    summary_csv_path = docs_dir / "results_summary.csv"
    comparison_csv_path = docs_dir / "comparison_table.csv"
    ablation_csv_path = docs_dir / "ablation_table.csv"

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "runs": run_payloads,
        "missing_runs": missing_runs,
        "tables": tables,
        "files": {
            "results_summary_csv": str(summary_csv_path),
            "comparison_table_csv": str(comparison_csv_path),
            "ablation_table_csv": str(ablation_csv_path),
        },
    }

    _write_csv(summary_csv_path, run_payloads)
    _write_csv(comparison_csv_path, tables["comparisons"])
    _write_csv(ablation_csv_path, tables["ablations"])
    return payload


def _recover_full_training_runs(*, repo_root: Path) -> list[RecoveredRun]:
    runs: list[RecoveredRun] = []
    seen_datasets: set[str] = set()
    for config_path in _iter_full_train_config_paths(repo_root=repo_root):
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
        train = dict(payload.get("train") or {})
        dataset_name = str(payload.get("dataset") or config_path.stem)
        if dataset_name in seen_datasets:
            continue
        seen_datasets.add(dataset_name)
        output_dir = (
            repo_root
            / str(train.get("outdir", "outputs/train"))
            / str(train.get("experiment_name", "full_dyrift_gnn"))
            / str(train.get("model", "dyrift_gnn"))
            / dataset_name
        ).resolve()
        epoch_metrics_path = output_dir / "epoch_metrics.csv"
        if not epoch_metrics_path.exists():
            continue
        configured_total_epochs = int(train.get("epochs", OFFICIAL_TRAIN_EPOCHS))
        configured_stop_floor = _extract_override_int(
            train.get("graph_config_overrides"),
            key="min_early_stop_epoch",
        )
        configured_policy_compliant = (
            configured_total_epochs >= OFFICIAL_TRAIN_EPOCHS
            and (configured_stop_floor or 0) >= POLICY_MIN_EARLY_STOP_EPOCH
        )
        run = _recover_run_from_epoch_metrics(
            experiment_name=str(train.get("experiment_name", "full_dyrift_gnn")),
            display_name=FULL_MODEL_DISPLAY_NAME,
            experiment_type="full_model",
            model_name=str(train.get("model", "dyrift_gnn")),
            dataset_name=dataset_name,
            output_dir=output_dir,
            source_config=config_path,
            configured_total_epochs=configured_total_epochs,
            configured_stop_floor=configured_stop_floor,
            configured_policy_compliant=configured_policy_compliant,
        )
        runs.append(run)
    return runs


def _iter_full_train_config_paths(*, repo_root: Path) -> list[Path]:
    private_dir = repo_root / "configs" / "private"
    public_legacy_dir = repo_root / "configs" / "train"
    return [
        *sorted(private_dir.glob("*.json")),
        *sorted(public_legacy_dir.glob("*.json")),
    ]


def _recover_experiment_run(
    *,
    experiment: Any,
    dataset_name: str,
) -> RecoveredRun:
    plan = resolve_dataset_plan(experiment, dataset_name)
    dataset_dir = experiment.dataset_output_dir(dataset_name)

    configured_total_epochs: int | None
    configured_stop_floor: int | None
    configured_policy_compliant: bool | None

    if experiment.runner == "graph":
        graph_config = _resolve_graph_config(graph_spec=experiment.runner_spec, plan=plan)
        configured_total_epochs = int(plan.epochs)
        configured_stop_floor = int(getattr(graph_config, "min_early_stop_epoch", 0))
    elif experiment.runner == "xgboost":
        configured_total_epochs = int(experiment.runner_spec.get("num_boost_round", OFFICIAL_TRAIN_EPOCHS))
        configured_stop_floor = int(experiment.runner_spec.get("early_stopping_rounds", 5))
    elif experiment.runner == "mlp":
        configured_total_epochs = int(experiment.runner_spec.get("epochs", plan.epochs))
        configured_stop_floor = 0
    else:
        configured_total_epochs = None
        configured_stop_floor = None

    configured_policy_compliant = (
        None
        if configured_total_epochs is None or configured_stop_floor is None
        else configured_total_epochs >= OFFICIAL_TRAIN_EPOCHS
        and configured_stop_floor >= POLICY_MIN_EARLY_STOP_EPOCH
    )

    run = _recover_run_from_epoch_metrics(
        experiment_name=experiment.experiment_name,
        display_name=experiment.display_name,
        experiment_type=experiment.experiment_type,
        model_name=experiment.model_name,
        dataset_name=dataset_name,
        output_dir=dataset_dir,
        source_config=experiment.config_path,
        configured_total_epochs=configured_total_epochs,
        configured_stop_floor=configured_stop_floor,
        configured_policy_compliant=configured_policy_compliant,
    )
    return run


def _recover_run_from_epoch_metrics(
    *,
    experiment_name: str,
    display_name: str,
    experiment_type: str,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    source_config: Path | None,
    configured_total_epochs: int | None,
    configured_stop_floor: int | None,
    configured_policy_compliant: bool | None = None,
) -> RecoveredRun:
    dataset_spec = get_dataset_spec(dataset_name)
    epoch_metrics_path = output_dir / "epoch_metrics.csv"
    rows = _read_epoch_rows(epoch_metrics_path)
    if not rows:
        raise ValueError(f"{epoch_metrics_path}: no usable rows found.")
    best_row = max(rows, key=lambda row: float(row["val_auc"]))
    last_epoch = max(int(row["epoch"]) for row in rows)

    fit_metrics_path = _resolve_first_existing(output_dir, pattern="seed_*/fit_metrics.json")
    fit_metrics = read_json(fit_metrics_path) if fit_metrics_path is not None else None
    model_meta_path = _resolve_first_existing(output_dir, pattern="seed_*/model_meta.json")
    model_meta = read_json(model_meta_path) if model_meta_path is not None else None

    best_epoch = int((fit_metrics or {}).get("best_epoch", best_row["epoch"]))
    trained_epochs = int((fit_metrics or {}).get("trained_epochs", last_epoch))

    notes: list[str] = []
    if configured_total_epochs is not None and configured_total_epochs < OFFICIAL_TRAIN_EPOCHS:
        notes.append("configured_total_epochs_below_policy")
    if configured_stop_floor is not None and configured_stop_floor < POLICY_MIN_EARLY_STOP_EPOCH:
        notes.append("configured_stop_floor_below_policy")
    if trained_epochs < POLICY_MIN_EARLY_STOP_EPOCH and (
        configured_total_epochs is None or trained_epochs < configured_total_epochs
    ):
        notes.append("observed_training_stopped_before_policy_floor")
    if best_epoch < POLICY_MIN_EARLY_STOP_EPOCH and trained_epochs >= POLICY_MIN_EARLY_STOP_EPOCH:
        notes.append("best_epoch_before_policy_floor")
    if model_meta is not None and int(model_meta.get("epochs", trained_epochs)) != trained_epochs:
        notes.append("model_meta_epoch_count_differs_from_fit_metrics")

    observed_policy_compliant = trained_epochs >= POLICY_MIN_EARLY_STOP_EPOCH or (
        configured_total_epochs is not None and trained_epochs >= configured_total_epochs
    )

    artifact_paths = {"epoch_metrics": str(epoch_metrics_path)}
    if fit_metrics_path is not None:
        artifact_paths["fit_metrics"] = str(fit_metrics_path)
    if model_meta_path is not None:
        artifact_paths["model_meta"] = str(model_meta_path)

    return RecoveredRun(
        experiment_name=experiment_name,
        display_name=display_name,
        experiment_type=experiment_type,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_display_name=dataset_spec.display_name,
        output_dir=output_dir,
        source_config=source_config,
        metric_source=epoch_metrics_path,
        best_val_auc=float(best_row["val_auc"]),
        best_epoch=best_epoch,
        trained_epochs=trained_epochs,
        last_epoch=last_epoch,
        configured_total_epochs=configured_total_epochs,
        configured_stop_floor=configured_stop_floor,
        configured_policy_compliant=configured_policy_compliant,
        observed_policy_compliant=observed_policy_compliant,
        artifact_paths=artifact_paths,
        notes=notes,
    )


def _read_epoch_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch_raw = row.get("epoch")
            val_raw = row.get("val_auc")
            if epoch_raw in (None, "") or val_raw in (None, ""):
                continue
            rows.append(
                {
                    "epoch": int(float(str(epoch_raw))),
                    "val_auc": float(val_raw),
                }
            )
    return rows


def _resolve_first_existing(base_dir: Path, *, pattern: str) -> Path | None:
    matches = sorted(base_dir.glob(pattern))
    return None if not matches else matches[0]


def _extract_override_int(raw_overrides: Any, *, key: str) -> int | None:
    if raw_overrides is None:
        return None
    if isinstance(raw_overrides, dict):
        value = raw_overrides.get(key)
        return None if value is None else int(value)
    if isinstance(raw_overrides, list):
        for item in raw_overrides:
            text = str(item)
            prefix = f"{key}="
            if text.startswith(prefix):
                return int(text[len(prefix) :])
    return None


def _sort_runs(runs: list[RecoveredRun]) -> list[RecoveredRun]:
    dataset_rank = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    type_rank = {"full_model": 0, "comparison": 1, "ablation": 2, "progressive": 3}
    return sorted(
        runs,
        key=lambda run: (
            dataset_rank.get(run.dataset_name, 99),
            type_rank.get(run.experiment_type, 99),
            run.display_name.lower(),
        ),
    )


def _build_table_rows(
    runs: list[dict[str, Any]],
    *,
    method_order: tuple[str, ...],
) -> list[dict[str, str]]:
    by_display: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_display.setdefault(str(run["method"]), {})[str(run["dataset_name"])] = run

    ordered_methods = list(method_order) + sorted(
        name for name in by_display if name not in method_order
    )

    rows: list[dict[str, str]] = []
    for display_name in ordered_methods:
        dataset_runs = by_display.get(display_name)
        if not dataset_runs:
            continue
        row: dict[str, str] = {"method": display_name}
        for dataset_name in DATASET_ORDER:
            run = dataset_runs.get(dataset_name)
            if run is None:
                row[dataset_name] = "n/a"
                row[f"{dataset_name}_best_epoch"] = "n/a"
                row[f"{dataset_name}_trained_epochs"] = "n/a"
                continue
            row[dataset_name] = _format_metric_with_delta(
                value=float(run["best_val_auc"]),
                delta=run.get("delta_vs_full"),
                is_full=(display_name == FULL_MODEL_DISPLAY_NAME),
            )
            row[f"{dataset_name}_best_epoch"] = str(int(run["best_epoch"]))
            row[f"{dataset_name}_trained_epochs"] = str(int(run["trained_epochs"]))
        rows.append(row)
    return rows


def _format_metric_with_delta(*, value: float, delta: float | None, is_full: bool) -> str:
    metric_text = f"{value * 100.0:.2f}%"
    if is_full or delta is None:
        return metric_text
    return f"{metric_text} ({delta * 100.0:+.2f}%)"


def _stringify_optional(value: Any) -> str:
    return "" if value is None else str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover per-run summaries and percentage-formatted result tables from existing outputs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Defaults to the current project root.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Directory for generated JSON/CSV/Markdown reports. Defaults to docs/generated.",
    )
    args = parser.parse_args(argv)

    bundle = recover_results_bundle(
        repo_root=args.repo_root,
        docs_dir=args.docs_dir,
    )
    print(f"Recovered runs: {len(bundle['runs'])}")
    print(f"Missing runs: {len(bundle['missing_runs'])}")
    print(f"Summary CSV: {bundle['files']['results_summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
