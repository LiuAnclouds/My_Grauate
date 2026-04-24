from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dyrift.config_loader import ATTR_PROJ_ENV_VAR
from dyrift.data_processing.core.registry import get_dataset_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "train"
FEATURE_OUTPUT_ROOT = REPO_ROOT / "outputs" / "features"


@dataclass(frozen=True)
class DatasetPlan:
    dataset_name: str
    dataset_display_name: str
    dataset_short: str
    feature_profile: str
    feature_dir: Path
    attr_proj_dim: int | None
    epochs: int
    batch_size: int
    hidden_dim: int
    rel_dim: int
    fanouts: list[int]
    learning_rate: float | None
    weight_decay: float | None
    dropout: float | None
    target_context_groups: list[str]
    graph_config_overrides: list[str]

    @property
    def feature_env(self) -> dict[str, str]:
        if self.attr_proj_dim is None:
            return {}
        return {ATTR_PROJ_ENV_VAR: str(int(self.attr_proj_dim))}

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_display_name": self.dataset_display_name,
            "dataset_short": self.dataset_short,
            "feature_profile": self.feature_profile,
            "feature_dir": str(self.feature_dir),
            "attr_proj_dim": self.attr_proj_dim,
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "hidden_dim": int(self.hidden_dim),
            "rel_dim": int(self.rel_dim),
            "fanouts": [int(value) for value in self.fanouts],
            "learning_rate": None if self.learning_rate is None else float(self.learning_rate),
            "weight_decay": None if self.weight_decay is None else float(self.weight_decay),
            "dropout": None if self.dropout is None else float(self.dropout),
            "target_context_groups": list(self.target_context_groups),
            "graph_config_overrides": list(self.graph_config_overrides),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_dir: Path
    config_path: Path
    experiment_name: str
    display_name: str
    experiment_type: str
    runner: str
    description: str
    datasets: list[str]
    seeds: list[int]
    dataset_parameters_path: Path
    dataset_parameters: dict[str, Any]
    runner_spec: dict[str, Any]
    shared_dataset_overrides: dict[str, Any]
    dataset_overrides: dict[str, dict[str, Any]]
    output_root: Path
    model_name: str

    def dataset_output_dir(self, dataset_name: str) -> Path:
        return self.output_root / self.model_name / dataset_name


def load_experiment_config(experiment_dir: Path) -> ExperimentConfig:
    config_path = experiment_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: expected a JSON object.")

    runner = str(payload["runner"]).strip()
    runner_spec = payload.get(runner)
    if not isinstance(runner_spec, dict):
        raise ValueError(f"{config_path}: missing `{runner}` configuration block.")

    dataset_parameters_ref = payload.get("dataset_parameters", "../../common/dataset_parameters.json")
    dataset_parameters_path = (experiment_dir / dataset_parameters_ref).resolve()
    dataset_parameters = json.loads(dataset_parameters_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_parameters, dict):
        raise ValueError(f"{dataset_parameters_path}: expected a JSON object.")

    datasets_section = dataset_parameters.get("datasets")
    if not isinstance(datasets_section, dict):
        raise ValueError(f"{dataset_parameters_path}: missing `datasets` mapping.")

    dataset_names = payload.get("datasets")
    datasets = list(datasets_section.keys()) if dataset_names is None else [str(name) for name in dataset_names]

    experiment_name = str(payload.get("experiment_name") or experiment_dir.name).strip()
    group_name = experiment_dir.parent.name
    experiment_type = str(payload.get("experiment_type") or group_name.rstrip("s"))
    model_name = _resolve_model_name(runner=runner, runner_spec=runner_spec)
    return ExperimentConfig(
        experiment_dir=experiment_dir,
        config_path=config_path,
        experiment_name=experiment_name,
        display_name=str(payload.get("display_name") or experiment_name),
        experiment_type=experiment_type,
        runner=runner,
        description=str(payload.get("description") or ""),
        datasets=datasets,
        seeds=[int(seed) for seed in payload.get("seeds", [42])],
        dataset_parameters_path=dataset_parameters_path,
        dataset_parameters=dataset_parameters,
        runner_spec=dict(runner_spec),
        shared_dataset_overrides=_coerce_mapping(payload.get("shared_dataset_overrides"), location=f"{config_path}:shared_dataset_overrides"),
        dataset_overrides=_coerce_nested_mapping(payload.get("dataset_overrides"), location=f"{config_path}:dataset_overrides"),
        output_root=EXPERIMENT_OUTPUT_ROOT / experiment_name,
        model_name=model_name,
    )


def resolve_dataset_plan(experiment: ExperimentConfig, dataset_name: str) -> DatasetPlan:
    datasets_section = experiment.dataset_parameters["datasets"]
    try:
        base_settings = dict(datasets_section[str(dataset_name)])
    except KeyError as exc:
        supported = ", ".join(sorted(datasets_section))
        raise KeyError(
            f"{experiment.dataset_parameters_path}: unsupported dataset `{dataset_name}`. Supported: {supported}"
        ) from exc

    merged = dict(base_settings)
    merged.update(experiment.shared_dataset_overrides)
    merged.update(experiment.dataset_overrides.get(str(dataset_name), {}))

    dataset_spec = get_dataset_spec(str(dataset_name))
    feature_dir = _resolve_feature_dir(
        dataset_name=str(dataset_name),
        feature_dir=merged.get("feature_dir"),
        feature_subdir=merged.get("feature_subdir"),
    )
    return DatasetPlan(
        dataset_name=str(dataset_name),
        dataset_display_name=dataset_spec.display_name,
        dataset_short=str(merged.get("dataset_short") or dataset_name),
        feature_profile=str(merged.get("feature_profile") or "utpm_unified"),
        feature_dir=feature_dir,
        attr_proj_dim=_coerce_optional_int(merged.get("attr_proj_dim")),
        epochs=int(merged.get("epochs", 8)),
        batch_size=int(merged.get("batch_size", 512)),
        hidden_dim=int(merged.get("hidden_dim", 128)),
        rel_dim=int(merged.get("rel_dim", 32)),
        fanouts=[int(value) for value in merged.get("fanouts", [15, 10])],
        learning_rate=_coerce_optional_float(merged.get("learning_rate")),
        weight_decay=_coerce_optional_float(merged.get("weight_decay")),
        dropout=_coerce_optional_float(merged.get("dropout")),
        target_context_groups=[str(value) for value in merged.get("target_context_groups", [])],
        graph_config_overrides=[str(value) for value in merged.get("graph_config_overrides", [])],
    )


def resolve_dataset_output_roots(dataset_name: str) -> tuple[Path, Path]:
    return REPO_ROOT / "outputs" / "analysis" / dataset_name, FEATURE_OUTPUT_ROOT / dataset_name


def _resolve_model_name(*, runner: str, runner_spec: dict[str, Any]) -> str:
    if runner == "graph":
        return str(runner_spec["engine_target"])
    if runner == "xgboost":
        return str(runner_spec.get("model_name") or "xgboost")
    return str(runner)


def _resolve_feature_dir(
    *,
    dataset_name: str,
    feature_dir: Any,
    feature_subdir: Any,
) -> Path:
    if feature_dir is not None:
        candidate = Path(str(feature_dir))
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate
    dataset_feature_root = FEATURE_OUTPUT_ROOT / dataset_name
    if feature_subdir is not None:
        return dataset_feature_root / str(feature_subdir)
    return dataset_feature_root


def _coerce_mapping(raw_value: Any, *, location: str) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError(f"{location}: expected a JSON object.")
    return dict(raw_value)


def _coerce_nested_mapping(raw_value: Any, *, location: str) -> dict[str, dict[str, Any]]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError(f"{location}: expected a JSON object.")
    output: dict[str, dict[str, Any]] = {}
    for key, value in raw_value.items():
        if not isinstance(value, dict):
            raise ValueError(f"{location}.{key}: expected a JSON object.")
        output[str(key)] = dict(value)
    return output


def _coerce_optional_int(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    return int(raw_value)


def _coerce_optional_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    return float(raw_value)
