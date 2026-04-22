from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTR_PROJ_DIM = 32
ATTR_PROJ_ENV_VAR = "GRADPROJ_UTPM_ATTR_PROJ_DIM"

_MAINLINE_ALLOWED_KEYS = {
    "attr_proj_dim",
    "batch_size",
    "dropout",
    "epochs",
    "feature_profile",
    "fanouts",
    "feature_dir",
    "feature_env",
    "feature_subdir",
    "graph_config_overrides",
    "hidden_dim",
    "learning_rate",
    "rel_dim",
    "run_name_template",
    "target_context_groups",
    "weight_decay",
}
_HYBRID_ALLOWED_KEYS = {
    "base_run_name_template",
    "blend_alpha",
    "run_name_template",
    "secondary_run_name_template",
}


@dataclass(frozen=True)
class LoadedThesisHparamProfile:
    path: Path
    payload: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        section_payload = self.payload.get(name)
        if section_payload is None:
            return {}
        if not isinstance(section_payload, dict):
            raise ValueError(f"Section `{name}` in `{self.path}` must be a JSON object.")
        return section_payload


@dataclass(frozen=True)
class MainlineDatasetHparams:
    dataset_name: str
    run_name_template: str | None
    feature_profile: str
    feature_dir: Path | None
    feature_subdir: str | None
    feature_env: dict[str, str]
    epochs: int
    batch_size: int
    hidden_dim: int
    rel_dim: int
    fanouts: list[int]
    learning_rate: float | None
    weight_decay: float | None
    dropout: float | None
    target_context_groups: list[str] | None
    graph_config_overrides: list[str]

    @property
    def attr_proj_dim(self) -> int | None:
        raw_value = self.feature_env.get(ATTR_PROJ_ENV_VAR)
        if raw_value is None:
            return None
        return int(raw_value)

    def to_summary_payload(self) -> dict[str, Any]:
        return {
            "run_name_template": self.run_name_template,
            "feature_profile": self.feature_profile,
            "feature_dir": None if self.feature_dir is None else str(self.feature_dir),
            "feature_subdir": self.feature_subdir,
            "feature_env": dict(self.feature_env),
            "attr_proj_dim": self.attr_proj_dim,
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "hidden_dim": int(self.hidden_dim),
            "rel_dim": int(self.rel_dim),
            "fanouts": [int(value) for value in self.fanouts],
            "learning_rate": None if self.learning_rate is None else float(self.learning_rate),
            "weight_decay": None if self.weight_decay is None else float(self.weight_decay),
            "dropout": None if self.dropout is None else float(self.dropout),
            "target_context_groups": None if self.target_context_groups is None else list(self.target_context_groups),
            "graph_config_overrides": list(self.graph_config_overrides),
        }


@dataclass(frozen=True)
class HybridDatasetHparams:
    dataset_name: str
    blend_alpha: float
    base_run_name_template: str | None
    secondary_run_name_template: str | None
    run_name_template: str | None

    def to_summary_payload(self) -> dict[str, Any]:
        return {
            "blend_alpha": float(self.blend_alpha),
            "base_run_name_template": self.base_run_name_template,
            "secondary_run_name_template": self.secondary_run_name_template,
            "run_name_template": self.run_name_template,
        }


def load_thesis_hparam_profile(path: Path | str | None) -> LoadedThesisHparamProfile | None:
    if path is None:
        return None
    profile_path = Path(path)
    if not profile_path.is_absolute():
        profile_path = REPO_ROOT / profile_path
    profile_path = profile_path.resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Hyperparameter profile `{profile_path}` must contain a JSON object.")
    payload = _expand_dataset_file_refs(profile_path=profile_path, payload=payload)
    return LoadedThesisHparamProfile(path=profile_path, payload=payload)


def _expand_dataset_file_refs(*, profile_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Expand mainline.dataset_files into mainline.datasets.

    The final DyRIFT-GNN profile keeps each dataset's hyperparameters in a separate
    JSON file while preserving the existing single-profile suite entrypoint.
    """
    mainline_section = payload.get("mainline")
    if not isinstance(mainline_section, dict):
        return payload
    dataset_files = mainline_section.get("dataset_files")
    if dataset_files is None:
        return payload
    if not isinstance(dataset_files, dict):
        raise ValueError(f"`{profile_path}:mainline.dataset_files` must be a JSON object.")

    expanded_mainline = dict(mainline_section)
    expanded_mainline.pop("dataset_files", None)
    datasets = _coerce_dataset_mapping(
        expanded_mainline.get("datasets"),
        location=f"{profile_path}:mainline.datasets",
    )
    for dataset_name, raw_ref in dataset_files.items():
        dataset_key = str(dataset_name)
        dataset_path = _resolve_profile_ref(profile_path=profile_path, raw_ref=raw_ref)
        dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(dataset_payload, dict):
            raise ValueError(f"Dataset hyperparameter file `{dataset_path}` must contain a JSON object.")
        dataset_override = _extract_dataset_hparam_payload(
            dataset_name=dataset_key,
            dataset_path=dataset_path,
            dataset_payload=dataset_payload,
        )
        merged_override = dict(datasets.get(dataset_key) or {})
        merged_override.update(dataset_override)
        datasets[dataset_key] = merged_override
    expanded_mainline["datasets"] = datasets
    expanded_payload = dict(payload)
    expanded_payload["mainline"] = expanded_mainline
    return expanded_payload


def _resolve_profile_ref(*, profile_path: Path, raw_ref: Any) -> Path:
    ref_path = Path(str(raw_ref))
    if not ref_path.is_absolute():
        ref_path = (profile_path.parent / ref_path).resolve()
    return ref_path


def _extract_dataset_hparam_payload(
    *,
    dataset_name: str,
    dataset_path: Path,
    dataset_payload: dict[str, Any],
) -> dict[str, Any]:
    if "hparams" in dataset_payload:
        return _coerce_section_mapping(
            dataset_payload.get("hparams"),
            location=f"{dataset_path}:hparams",
        )
    mainline_section = dataset_payload.get("mainline")
    if isinstance(mainline_section, dict):
        dataset_section = _coerce_dataset_mapping(
            mainline_section.get("datasets"),
            location=f"{dataset_path}:mainline.datasets",
        )
        if dataset_name in dataset_section:
            return _coerce_section_mapping(
                dataset_section.get(dataset_name),
                location=f"{dataset_path}:mainline.datasets.{dataset_name}",
            )
    dataset_copy = dict(dataset_payload)
    dataset_copy.pop("schema_version", None)
    dataset_copy.pop("dataset", None)
    dataset_copy.pop("notes", None)
    return _coerce_section_mapping(dataset_copy, location=str(dataset_path))


def resolve_mainline_dataset_hparams(
    *,
    args: Any,
    dataset_name: str,
    profile: LoadedThesisHparamProfile | None,
) -> MainlineDatasetHparams:
    feature_env: dict[str, str] = {}
    env_attr_proj_dim = os.environ.get(ATTR_PROJ_ENV_VAR)
    if env_attr_proj_dim is not None and str(env_attr_proj_dim).strip():
        feature_env[ATTR_PROJ_ENV_VAR] = str(int(env_attr_proj_dim))

    resolved: dict[str, Any] = {
        "run_name_template": str(getattr(args, "run_name_template")),
        "feature_profile": str(getattr(args, "feature_profile")),
        "feature_dir": _coerce_optional_path(getattr(args, "feature_dir", None)),
        "feature_subdir": _coerce_optional_str(getattr(args, "feature_subdir", None)),
        "feature_env": feature_env,
        "epochs": int(getattr(args, "epochs")),
        "batch_size": int(getattr(args, "batch_size")),
        "hidden_dim": int(getattr(args, "hidden_dim")),
        "rel_dim": int(getattr(args, "rel_dim")),
        "fanouts": [int(value) for value in getattr(args, "fanouts")],
        "learning_rate": _coerce_optional_float(getattr(args, "learning_rate", None)),
        "weight_decay": _coerce_optional_float(getattr(args, "weight_decay", None)),
        "dropout": _coerce_optional_float(getattr(args, "dropout", None)),
        "target_context_groups": _normalize_optional_str_list(
            getattr(args, "target_context_groups", None),
            location="cli target_context_groups",
        ),
        "graph_config_overrides": _normalize_graph_config_overrides(
            list(getattr(args, "graph_config_override", [])),
            location="cli graph_config_override",
        ),
    }
    profile_defaults: dict[str, Any] = {}
    dataset_overrides: dict[str, Any] = {}
    if profile is not None:
        mainline_section = profile.section("mainline")
        dataset_section = _coerce_dataset_mapping(
            mainline_section.get("datasets"),
            location=f"{profile.path}:mainline.datasets",
        )
        profile_defaults = _coerce_section_mapping(
            mainline_section.get("defaults"),
            location=f"{profile.path}:mainline.defaults",
        )
        dataset_overrides = _coerce_section_mapping(
            dataset_section.get(dataset_name),
            location=f"{profile.path}:mainline.datasets.{dataset_name}",
        )
        _validate_allowed_keys(
            profile_defaults,
            allowed_keys=_MAINLINE_ALLOWED_KEYS,
            location=f"{profile.path}:mainline.defaults",
        )
        _validate_allowed_keys(
            dataset_overrides,
            allowed_keys=_MAINLINE_ALLOWED_KEYS,
            location=f"{profile.path}:mainline.datasets.{dataset_name}",
        )

    for section_name, section_payload in (
        ("profile defaults", profile_defaults),
        (f"dataset override `{dataset_name}`", dataset_overrides),
    ):
        if not section_payload:
            continue
        resolved = _merge_mainline_section(
            resolved=resolved,
            section=section_payload,
            location=section_name,
        )

    attr_proj_dim = None
    if ATTR_PROJ_ENV_VAR in resolved["feature_env"]:
        attr_proj_dim = int(resolved["feature_env"][ATTR_PROJ_ENV_VAR])
    if resolved["feature_subdir"] is None and resolved["feature_dir"] is None and attr_proj_dim not in {None, DEFAULT_ATTR_PROJ_DIM}:
        resolved["feature_subdir"] = f"features_ap{attr_proj_dim}"

    return MainlineDatasetHparams(
        dataset_name=str(dataset_name),
        run_name_template=_coerce_optional_str(resolved["run_name_template"]),
        feature_profile=str(resolved["feature_profile"]),
        feature_dir=resolved["feature_dir"],
        feature_subdir=resolved["feature_subdir"],
        feature_env=dict(resolved["feature_env"]),
        epochs=int(resolved["epochs"]),
        batch_size=int(resolved["batch_size"]),
        hidden_dim=int(resolved["hidden_dim"]),
        rel_dim=int(resolved["rel_dim"]),
        fanouts=[int(value) for value in resolved["fanouts"]],
        learning_rate=resolved["learning_rate"],
        weight_decay=resolved["weight_decay"],
        dropout=resolved["dropout"],
        target_context_groups=resolved["target_context_groups"],
        graph_config_overrides=_graph_override_map_to_list(resolved["graph_config_overrides"]),
    )


def resolve_hybrid_dataset_hparams(
    *,
    args: Any,
    dataset_name: str,
    profile: LoadedThesisHparamProfile | None,
) -> HybridDatasetHparams:
    resolved_blend_alpha = float(getattr(args, "blend_alpha"))
    resolved_base_run_name_template = str(getattr(args, "base_run_name_template"))
    resolved_secondary_run_name_template = str(getattr(args, "secondary_run_name_template"))
    resolved_run_name_template = str(getattr(args, "run_name_template"))
    if profile is not None:
        hybrid_section = profile.section("hybrid")
        dataset_section = _coerce_dataset_mapping(
            hybrid_section.get("datasets"),
            location=f"{profile.path}:hybrid.datasets",
        )
        defaults = _coerce_section_mapping(
            hybrid_section.get("defaults"),
            location=f"{profile.path}:hybrid.defaults",
        )
        dataset_overrides = _coerce_section_mapping(
            dataset_section.get(dataset_name),
            location=f"{profile.path}:hybrid.datasets.{dataset_name}",
        )
        _validate_allowed_keys(
            defaults,
            allowed_keys=_HYBRID_ALLOWED_KEYS,
            location=f"{profile.path}:hybrid.defaults",
        )
        _validate_allowed_keys(
            dataset_overrides,
            allowed_keys=_HYBRID_ALLOWED_KEYS,
            location=f"{profile.path}:hybrid.datasets.{dataset_name}",
        )
        if "blend_alpha" in defaults:
            resolved_blend_alpha = float(defaults["blend_alpha"])
        if "blend_alpha" in dataset_overrides:
            resolved_blend_alpha = float(dataset_overrides["blend_alpha"])
        if "base_run_name_template" in defaults:
            resolved_base_run_name_template = str(defaults["base_run_name_template"])
        if "base_run_name_template" in dataset_overrides:
            resolved_base_run_name_template = str(dataset_overrides["base_run_name_template"])
        if "secondary_run_name_template" in defaults:
            resolved_secondary_run_name_template = str(defaults["secondary_run_name_template"])
        if "secondary_run_name_template" in dataset_overrides:
            resolved_secondary_run_name_template = str(dataset_overrides["secondary_run_name_template"])
        if "run_name_template" in defaults:
            resolved_run_name_template = str(defaults["run_name_template"])
        if "run_name_template" in dataset_overrides:
            resolved_run_name_template = str(dataset_overrides["run_name_template"])
    return HybridDatasetHparams(
        dataset_name=str(dataset_name),
        blend_alpha=float(resolved_blend_alpha),
        base_run_name_template=resolved_base_run_name_template,
        secondary_run_name_template=resolved_secondary_run_name_template,
        run_name_template=resolved_run_name_template,
    )


def _merge_mainline_section(
    *,
    resolved: dict[str, Any],
    section: dict[str, Any],
    location: str,
) -> dict[str, Any]:
    merged = dict(resolved)
    if "run_name_template" in section:
        merged["run_name_template"] = str(section["run_name_template"])
    if "feature_profile" in section:
        merged["feature_profile"] = _normalize_feature_profile(
            section["feature_profile"],
            location=f"{location}.feature_profile",
        )
    if "feature_dir" in section:
        merged["feature_dir"] = _coerce_optional_path(section.get("feature_dir"))
    if "feature_subdir" in section:
        merged["feature_subdir"] = _coerce_optional_str(section.get("feature_subdir"))
    if "feature_env" in section:
        merged_feature_env = dict(merged["feature_env"])
        merged_feature_env.update(
            _normalize_feature_env(
                section.get("feature_env"),
                location=f"{location}.feature_env",
            )
        )
        merged["feature_env"] = merged_feature_env
    if "attr_proj_dim" in section:
        attr_proj_dim = int(section["attr_proj_dim"])
        if attr_proj_dim <= 0:
            raise ValueError(f"`{location}.attr_proj_dim` must be positive, got {attr_proj_dim}.")
        merged_feature_env = dict(merged["feature_env"])
        merged_feature_env[ATTR_PROJ_ENV_VAR] = str(attr_proj_dim)
        merged["feature_env"] = merged_feature_env
    if "epochs" in section:
        merged["epochs"] = int(section["epochs"])
    if "batch_size" in section:
        merged["batch_size"] = int(section["batch_size"])
    if "hidden_dim" in section:
        merged["hidden_dim"] = int(section["hidden_dim"])
    if "rel_dim" in section:
        merged["rel_dim"] = int(section["rel_dim"])
    if "fanouts" in section:
        merged["fanouts"] = _normalize_int_list(section["fanouts"], location=f"{location}.fanouts")
    if "learning_rate" in section:
        merged["learning_rate"] = _coerce_optional_float(section["learning_rate"])
    if "weight_decay" in section:
        merged["weight_decay"] = _coerce_optional_float(section["weight_decay"])
    if "dropout" in section:
        merged["dropout"] = _coerce_optional_float(section["dropout"])
    if "target_context_groups" in section:
        merged["target_context_groups"] = _normalize_optional_str_list(
            section["target_context_groups"],
            location=f"{location}.target_context_groups",
        )
    if "graph_config_overrides" in section:
        override_map = dict(merged["graph_config_overrides"])
        override_map.update(
            _normalize_graph_config_overrides(
                section["graph_config_overrides"],
                location=f"{location}.graph_config_overrides",
            )
        )
        merged["graph_config_overrides"] = override_map
    return merged


def _normalize_int_list(values: Any, *, location: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"`{location}` must be a JSON array.")
    normalized = [int(value) for value in values]
    if not normalized:
        raise ValueError(f"`{location}` cannot be empty.")
    return normalized


def _normalize_optional_str_list(values: Any, *, location: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        return [text]
    if not isinstance(values, list):
        raise ValueError(f"`{location}` must be null, a string, or a JSON array of strings.")
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return normalized


def _normalize_feature_profile(value: Any, *, location: str) -> str:
    normalized = str(value).strip()
    allowed = {"utpm_unified", "utpm_shift_compact", "utpm_shift_enhanced"}
    if normalized not in allowed:
        raise ValueError(
            f"`{location}` must be one of {', '.join(sorted(allowed))}, got `{value}`."
        )
    return normalized


def _normalize_feature_env(raw: Any, *, location: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"`{location}` must be a JSON object.")
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        normalized[str(key)] = _stringify_override_value(value)
    return normalized


def _normalize_graph_config_overrides(raw: Any, *, location: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, list):
        overrides: dict[str, Any] = {}
        for item in raw:
            key, separator, raw_value = str(item).partition("=")
            if separator != "=" or not key.strip():
                raise ValueError(
                    f"`{location}` items must look like KEY=VALUE strings, got `{item}`."
                )
            overrides[key.strip()] = raw_value.strip()
        return overrides
    raise ValueError(f"`{location}` must be either a JSON object or an array of KEY=VALUE strings.")


def _graph_override_map_to_list(override_map: dict[str, Any]) -> list[str]:
    return [f"{key}={_stringify_override_value(value)}" for key, value in override_map.items()]


def _stringify_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise ValueError("Override values cannot be null.")
    return str(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _coerce_section_mapping(raw: Any, *, location: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"`{location}` must be a JSON object.")
    return dict(raw)


def _coerce_dataset_mapping(raw: Any, *, location: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"`{location}` must be a JSON object keyed by dataset name.")
    return dict(raw)


def _validate_allowed_keys(section: dict[str, Any], *, allowed_keys: set[str], location: str) -> None:
    unknown_keys = sorted(set(section.keys()) - set(allowed_keys))
    if unknown_keys:
        raise ValueError(
            f"Unsupported keys under `{location}`: {', '.join(unknown_keys)}. "
            f"Allowed keys: {', '.join(sorted(allowed_keys))}."
        )
