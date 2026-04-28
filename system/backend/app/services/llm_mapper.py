from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests


MAPPING_SYSTEM_PROMPT = """
You convert arbitrary CSV schemas into a fraud-detection graph database contract.
Return only JSON with these keys: node_id, source_id, target_id, timestamp,
edge_type, amount, feature_columns, display_columns.
Use null for unavailable scalar fields and [] for unavailable lists.
Labels must not be required because inference data has no labels.
"""


def _extract_toml_string(text: str, key: str) -> str | None:
    match = re.search(rf'^{re.escape(key)}\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def _extract_provider_block(text: str, provider: str) -> str:
    match = re.search(
        rf'(?ms)^\[model_providers\.{re.escape(provider)}\]\s*(.*?)(?=^\[|\Z)',
        text,
    )
    return match.group(1) if match else ""


def load_codex_openai_credentials() -> dict[str, str]:
    codex_dir = Path.home() / ".codex"
    if not (codex_dir / "config.toml").exists():
        users_root = Path("C:/Users")
        if users_root.exists():
            for user_dir in users_root.iterdir():
                candidate = user_dir / ".codex"
                if (candidate / "config.toml").exists():
                    codex_dir = candidate
                    break
    config_path = codex_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Codex config not found: {config_path}")

    config_text = config_path.read_text(encoding="utf-8")
    provider = _extract_toml_string(config_text, "model_provider") or "wmxs"
    model = _extract_toml_string(config_text, "model") or "gpt-5.4"
    provider_block = _extract_provider_block(config_text, provider)
    base_url = _extract_toml_string(provider_block, "base_url")
    if not base_url:
        raise ValueError(f"base_url not found for provider '{provider}' in {config_path}")

    auth_path = codex_dir / f"auth_{provider}.json"
    if not auth_path.exists():
        auth_path = codex_dir / "auth.json"
    if not auth_path.exists():
        raise FileNotFoundError(f"Codex auth file not found for provider '{provider}'")

    auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
    api_key = auth_data.get("OPENAI_API_KEY")
    if not api_key:
        raise KeyError(f"OPENAI_API_KEY missing in {auth_path}")
    return {"provider": provider, "model": model, "base_url": base_url.rstrip("/"), "api_key": api_key}


def infer_mapping_with_llm(headers: list[str], sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    credentials = load_codex_openai_credentials()
    prompt = json.dumps({"headers": headers, "sample_rows": sample_rows[:5]}, ensure_ascii=False)
    response = requests.post(
        f"{credentials['base_url']}/chat/completions",
        headers={
            "Authorization": f"Bearer {credentials['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": credentials["model"],
            "messages": [
                {"role": "system", "content": MAPPING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(_extract_json_object(content))


def infer_mapping(
    *,
    headers: list[str],
    sample_rows: list[dict[str, Any]],
    use_llm: bool = False,
) -> tuple[dict[str, Any], str, str]:
    heuristic = heuristic_mapping(headers)
    if not use_llm:
        return heuristic, "heuristic", "字段映射已按规则识别。"
    try:
        llm_mapping = infer_mapping_with_llm(headers=headers, sample_rows=sample_rows)
        mapping = validate_mapping(llm_mapping, headers=headers, fallback=heuristic)
        return mapping, "llm", "字段映射已由 LLM 识别并通过本地校验。"
    except Exception as exc:
        return heuristic, "heuristic_fallback", f"LLM 映射失败，已回退到规则识别：{exc}"


def heuristic_mapping(headers: list[str]) -> dict[str, Any]:
    lowered = {header.lower(): header for header in headers}

    def first(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        return None

    node_id = first("node_id", "txid", "tx_id", "id", "account_id", "user_id", "uid") or headers[0]
    source_id = first("source_id", "src", "src_id", "from", "from_id")
    target_id = first("target_id", "dst", "dst_id", "to", "to_id")
    timestamp = first("timestamp", "time", "time_step", "time_bucket", "date")
    amount = first("amount", "value", "transaction_amount")
    display_columns = [
        value
        for value in (
            first("display_name", "name", "username", "customer_name", "real_name"),
            first("id_number", "identity_no", "id_card", "certificate_no"),
            first("phone", "mobile", "telephone"),
            first("region", "city", "province", "address"),
            first("occupation", "job", "profession"),
        )
        if value
    ]
    feature_columns = [
        header
        for header in headers
        if header not in {node_id, source_id, target_id, timestamp, amount, *display_columns}
    ]
    return {
        "node_id": node_id,
        "source_id": source_id,
        "target_id": target_id,
        "timestamp": timestamp,
        "edge_type": first("edge_type", "relation", "type"),
        "amount": amount,
        "feature_columns": feature_columns,
        "display_columns": display_columns,
    }


def validate_mapping(
    mapping: dict[str, Any],
    *,
    headers: list[str],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    header_set = set(headers)

    def scalar(key: str) -> str | None:
        value = mapping.get(key)
        if value is None or value == "":
            return None
        text = str(value)
        return text if text in header_set else fallback.get(key)

    def column_list(key: str) -> list[str]:
        values = mapping.get(key)
        if not isinstance(values, list):
            return list(fallback.get(key) or [])
        valid = [str(value) for value in values if str(value) in header_set]
        return valid

    node_id = scalar("node_id") or fallback["node_id"]
    source_id = scalar("source_id")
    target_id = scalar("target_id")
    timestamp = scalar("timestamp")
    amount = scalar("amount")
    edge_type = scalar("edge_type")
    display_columns = column_list("display_columns")
    feature_columns = [
        column
        for column in column_list("feature_columns")
        if column not in {node_id, source_id, target_id, timestamp, amount, edge_type, *display_columns}
    ]
    if not feature_columns:
        feature_columns = list(fallback.get("feature_columns") or [])
    return {
        "node_id": node_id,
        "source_id": source_id,
        "target_id": target_id,
        "timestamp": timestamp,
        "edge_type": edge_type,
        "amount": amount,
        "feature_columns": feature_columns,
        "display_columns": display_columns,
    }


def _extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object.")
    return match.group(0)
