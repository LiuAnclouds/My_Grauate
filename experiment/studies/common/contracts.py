from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.config_loader import ATTR_PROJ_ENV_VAR
from experiment.datasets.core.registry import get_dataset_spec


REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_OUTPUT_ROOT = REPO_ROOT / "experiment" / "outputs" / "studies"


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

    def to_summary_payload(self) -> dict[str, Any]:
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
class StudyConfig:
    study_dir: Path
    config_path: Path
    study_name: str
    display_name: str
    study_type: str
    runner: str
    description: str
    datasets: list[str]
    seeds: list[int]
    dataset_profile_path: Path
    dataset_profile: dict[str, Any]
    runner_spec: dict[str, Any]
    shared_dataset_overrides: dict[str, Any]
    dataset_overrides: dict[str, dict[str, Any]]
    output_root: Path

    def dataset_output_dir(self, dataset_name: str) -> Path:
        return self.output_root / dataset_name


def load_study_config(study_dir: Path) -> StudyConfig:
    config_path = study_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: expected a JSON object.")

    runner = str(payload["runner"]).strip()
    runner_spec = payload.get(runner)
    if not isinstance(runner_spec, dict):
        raise ValueError(f"{config_path}: missing `{runner}` configuration block.")

    dataset_profile_ref = payload.get("dataset_profile", "../common/tri_dataset_profile.json")
    dataset_profile_path = (study_dir / dataset_profile_ref).resolve()
    dataset_profile = json.loads(dataset_profile_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_profile, dict):
        raise ValueError(f"{dataset_profile_path}: expected a JSON object.")

    datasets_section = dataset_profile.get("datasets")
    if not isinstance(datasets_section, dict):
        raise ValueError(f"{dataset_profile_path}: missing `datasets` mapping.")

    dataset_names = payload.get("datasets")
    if dataset_names is None:
        datasets = list(datasets_section.keys())
    else:
        datasets = [str(name) for name in dataset_names]

    study_name = str(payload.get("study_name") or study_dir.name).strip()
    group_name = study_dir.parent.name
    return StudyConfig(
        study_dir=study_dir,
        config_path=config_path,
        study_name=study_name,
        display_name=str(payload.get("display_name") or study_name),
        study_type=str(payload.get("study_type") or group_name.rstrip("s")),
        runner=runner,
        description=str(payload.get("description") or ""),
        datasets=datasets,
        seeds=[int(seed) for seed in payload.get("seeds", [42])],
        dataset_profile_path=dataset_profile_path,
        dataset_profile=dataset_profile,
        runner_spec=dict(runner_spec),
        shared_dataset_overrides=_coerce_mapping(payload.get("shared_dataset_overrides"), location=f"{config_path}:shared_dataset_overrides"),
        dataset_overrides=_coerce_nested_mapping(payload.get("dataset_overrides"), location=f"{config_path}:dataset_overrides"),
        output_root=STUDY_OUTPUT_ROOT / group_name / study_name,
    )


def resolve_dataset_plan(study: StudyConfig, dataset_name: str) -> DatasetPlan:
    datasets_section = study.dataset_profile["datasets"]
    try:
        base_settings = dict(datasets_section[str(dataset_name)])
    except KeyError as exc:
        supported = ", ".join(sorted(datasets_section))
        raise KeyError(f"{study.dataset_profile_path}: unsupported dataset `{dataset_name}`. Supported: {supported}") from exc

    merged = dict(base_settings)
    merged.update(study.shared_dataset_overrides)
    merged.update(study.dataset_overrides.get(str(dataset_name), {}))

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


def _dataset_training_root(dataset_name: str) -> Path:
    _, training_root = resolve_dataset_output_roots(dataset_name)
    return training_root


def resolve_dataset_output_roots(dataset_name: str) -> tuple[Path, Path]:
    spec = get_dataset_spec(dataset_name)
    outputs_root = REPO_ROOT / "experiment" / "outputs"
    if spec.uses_legacy_output_layout:
        return outputs_root / "eda", outputs_root / "training"
    dataset_root = outputs_root / spec.output_namespace
    return dataset_root / "eda", dataset_root / "training"


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
    if feature_subdir is not None:
        return _dataset_training_root(dataset_name) / str(feature_subdir)
    return _dataset_training_root(dataset_name) / "features"


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
