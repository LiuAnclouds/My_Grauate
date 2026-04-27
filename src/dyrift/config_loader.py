from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTR_PROJ_DIM = 32
ATTR_PROJ_ENV_VAR = "GRADPROJ_UTPM_ATTR_PROJ_DIM"

_TRAIN_PARAMETER_ALLOWED_KEYS = {
    "batch_size",
    "device",
    "dropout",
    "epochs",
    "experiment_name",
    "fanouts",
    "feature_dir",
    "feature_profile",
    "graph_config_overrides",
    "hidden_dim",
    "learning_rate",
    "model",
    "outdir",
    "preset",
    "rel_dim",
    "run_name",
    "seeds",
    "target_context_groups",
    "weight_decay",
}


@dataclass(frozen=True)
class TrainParameters:
    experiment_name: str
    model: str
    preset: str
    run_name: str
    feature_profile: str
    feature_dir: Path
    outdir: Path
    seeds: list[int]
    epochs: int
    batch_size: int
    hidden_dim: int
    rel_dim: int
    fanouts: list[int]
    device: str | None
    target_context_groups: list[str] | None
    learning_rate: float | None
    weight_decay: float | None
    dropout: float | None
    graph_config_overrides: list[str]
    parameter_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_file": None if self.parameter_file is None else str(self.parameter_file),
            "experiment_name": self.experiment_name,
            "model": self.model,
            "preset": self.preset,
            "run_name": self.run_name,
            "feature_profile": self.feature_profile,
            "feature_dir": str(self.feature_dir),
            "outdir": str(self.outdir),
            "seeds": [int(value) for value in self.seeds],
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "hidden_dim": int(self.hidden_dim),
            "rel_dim": int(self.rel_dim),
            "fanouts": [int(value) for value in self.fanouts],
            "device": self.device,
            "target_context_groups": (
                None if self.target_context_groups is None else list(self.target_context_groups)
            ),
            "learning_rate": None if self.learning_rate is None else float(self.learning_rate),
            "weight_decay": None if self.weight_decay is None else float(self.weight_decay),
            "dropout": None if self.dropout is None else float(self.dropout),
            "graph_config_overrides": list(self.graph_config_overrides),
        }


# Public name used by the train interface.
Parameter = TrainParameters


def resolve_train_parameters(
    *,
    args: Any,
    default_epochs: int,
    default_outdir: Path,
) -> TrainParameters:
    parameter_file = _coerce_optional_path(getattr(args, "parameter_file", None))
    file_payload = _load_train_parameter_payload(parameter_file)
    cli_payload = _collect_cli_train_payload(args)
    merged = _merge_train_parameter_payload(file_payload, cli_payload)

    if "epochs" not in merged or merged.get("epochs") is None:
        merged["epochs"] = int(default_epochs)
    if "outdir" not in merged or merged.get("outdir") is None:
        merged["outdir"] = default_outdir
    if "experiment_name" not in merged or _is_missing_parameter_value(merged.get("experiment_name")):
        merged["experiment_name"] = "full_dyrift_gnn"

    _validate_allowed_keys(
        merged,
        allowed_keys=_TRAIN_PARAMETER_ALLOWED_KEYS,
        location="resolved train parameters",
    )

    required_keys = [
        "experiment_name",
        "model",
        "preset",
        "run_name",
        "feature_profile",
        "feature_dir",
        "seeds",
        "batch_size",
        "hidden_dim",
        "rel_dim",
        "fanouts",
    ]
    missing = [key for key in required_keys if _is_missing_parameter_value(merged.get(key))]
    if missing:
        raise ValueError(
            "Missing required train parameters: "
            f"{', '.join(missing)}. Provide them with --parameter-file or explicit CLI flags."
        )

    return TrainParameters(
        experiment_name=str(merged["experiment_name"]),
        model=str(merged["model"]),
        preset=str(merged["preset"]),
        run_name=str(merged["run_name"]),
        feature_profile=_normalize_feature_profile(
            merged["feature_profile"],
            location="train_parameters.feature_profile",
        ),
        feature_dir=_coerce_required_path(merged["feature_dir"]),
        outdir=_coerce_required_path(merged["outdir"]),
        seeds=_normalize_int_list(merged["seeds"], location="train_parameters.seeds"),
        epochs=int(merged["epochs"]),
        batch_size=int(merged["batch_size"]),
        hidden_dim=int(merged["hidden_dim"]),
        rel_dim=int(merged["rel_dim"]),
        fanouts=_normalize_int_list(merged["fanouts"], location="train_parameters.fanouts"),
        device=_coerce_optional_str(merged.get("device")),
        target_context_groups=_normalize_optional_str_list(
            merged.get("target_context_groups"),
            location="train_parameters.target_context_groups",
        ),
        learning_rate=_coerce_optional_float(merged.get("learning_rate")),
        weight_decay=_coerce_optional_float(merged.get("weight_decay")),
        dropout=_coerce_optional_float(merged.get("dropout")),
        graph_config_overrides=_graph_override_map_to_list(
            _normalize_graph_config_overrides(
                merged.get("graph_config_overrides"),
                location="train_parameters.graph_config_overrides",
            )
        ),
        parameter_file=parameter_file,
    )


def _load_train_parameter_payload(parameter_file: Path | None) -> dict[str, Any]:
    if parameter_file is None:
        return {}
    payload = json.loads(parameter_file.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Train parameter file `{parameter_file}` must contain a JSON object.")
    for section_name in ("train", "parameters"):
        section_payload = payload.get(section_name)
        if section_payload is not None:
            return _coerce_section_mapping(
                section_payload,
                location=f"{parameter_file}:{section_name}",
            )
    parameter_payload = dict(payload)
    for metadata_key in ("schema_version", "name", "notes", "dataset"):
        parameter_payload.pop(metadata_key, None)
    return _coerce_section_mapping(parameter_payload, location=str(parameter_file))


def _collect_cli_train_payload(args: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "experiment_name",
        "model",
        "preset",
        "run_name",
        "feature_profile",
        "feature_dir",
        "outdir",
        "seeds",
        "epochs",
        "batch_size",
        "hidden_dim",
        "rel_dim",
        "fanouts",
        "device",
        "target_context_groups",
        "learning_rate",
        "weight_decay",
        "dropout",
    ):
        value = getattr(args, key, None)
        if value is not None:
            payload[key] = value
    graph_config_overrides = list(getattr(args, "graph_config_override", []) or [])
    if graph_config_overrides:
        payload["graph_config_overrides"] = graph_config_overrides
    return payload


def _merge_train_parameter_payload(
    file_payload: dict[str, Any],
    cli_payload: dict[str, Any],
) -> dict[str, Any]:
    _validate_allowed_keys(
        file_payload,
        allowed_keys=_TRAIN_PARAMETER_ALLOWED_KEYS,
        location="train parameter file",
    )
    _validate_allowed_keys(
        cli_payload,
        allowed_keys=_TRAIN_PARAMETER_ALLOWED_KEYS,
        location="train CLI parameters",
    )
    merged = dict(file_payload)
    file_graph_overrides = _normalize_graph_config_overrides(
        merged.get("graph_config_overrides"),
        location="train parameter file.graph_config_overrides",
    )
    cli_graph_overrides = _normalize_graph_config_overrides(
        cli_payload.get("graph_config_overrides"),
        location="train CLI graph_config_overrides",
    )
    merged.update(
        {key: value for key, value in cli_payload.items() if key != "graph_config_overrides"}
    )
    graph_overrides = dict(file_graph_overrides)
    graph_overrides.update(cli_graph_overrides)
    if graph_overrides:
        merged["graph_config_overrides"] = graph_overrides
    elif "graph_config_overrides" in merged:
        merged.pop("graph_config_overrides", None)
    return merged


def _is_missing_parameter_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


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
    return [str(value).strip() for value in values if str(value).strip()]


def _normalize_feature_profile(value: Any, *, location: str) -> str:
    normalized = str(value).strip()
    allowed = {
        "utpm_unified",
        "utpm_shift_compact",
        "utpm_shift_enhanced",
        "utpm_shift_history",
        "utpm_shift_fused",
        "utpm_shift_fused_rawmask",
    }
    if normalized not in allowed:
        raise ValueError(
            f"`{location}` must be one of {', '.join(sorted(allowed))}, got `{value}`."
        )
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


def _coerce_required_path(value: Any) -> Path:
    path = _coerce_optional_path(value)
    if path is None:
        raise ValueError("Expected a path value, got null.")
    return path


def _coerce_section_mapping(raw: Any, *, location: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"`{location}` must be a JSON object.")
    return dict(raw)


def _validate_allowed_keys(section: dict[str, Any], *, allowed_keys: set[str], location: str) -> None:
    unknown_keys = sorted(set(section.keys()) - set(allowed_keys))
    if unknown_keys:
        raise ValueError(
            f"Unsupported keys under `{location}`: {', '.join(unknown_keys)}. "
            f"Allowed keys: {', '.join(sorted(allowed_keys))}."
        )
