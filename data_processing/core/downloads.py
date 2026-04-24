from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen


USER_AGENT = "GraduationProject-DGraph/1.0"
DEFAULT_HF_MIRROR_BASES = ("https://hf-mirror.com",)
HF_MIRROR_ENV_VARS = ("GRADPROJ_HF_MIRROR", "HF_ENDPOINT", "HF_MIRROR")
HF_HOSTS = {
    "huggingface.co",
    "www.huggingface.co",
    "hf.co",
    "www.hf.co",
}


@dataclass(frozen=True)
class DownloadResult:
    source_url: str
    resolved_url: str
    output_path: Path
    size_bytes: int
    used_mirror: bool


def _normalize_base_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Mirror base URL cannot be empty.")
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid mirror base URL: {value}")
    cleaned = parsed._replace(path=parsed.path.rstrip("/"), params="", query="", fragment="")
    return urlunparse(cleaned)


def is_huggingface_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in HF_HOSTS


def _collect_hf_mirror_bases(extra_bases: list[str] | tuple[str, ...] | None = None) -> list[str]:
    bases: list[str] = []
    for value in extra_bases or ():
        if str(value).strip():
            bases.append(_normalize_base_url(str(value)))
    for env_name in HF_MIRROR_ENV_VARS:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            bases.append(_normalize_base_url(env_value))
    if not bases:
        bases.extend(DEFAULT_HF_MIRROR_BASES)

    unique: list[str] = []
    seen = set()
    for base in bases:
        if base not in seen:
            unique.append(base)
            seen.add(base)
    return unique


def build_download_candidates(
    url: str,
    *,
    hf_mirror_bases: list[str] | tuple[str, ...] | None = None,
    allow_direct_hf: bool = False,
) -> list[str]:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid download URL: {url}")

    if not is_huggingface_url(url):
        return [url]

    candidates = []
    for base in _collect_hf_mirror_bases(hf_mirror_bases):
        mirror = urlparse(base)
        mirrored = parsed._replace(
            scheme=mirror.scheme,
            netloc=mirror.netloc,
            path=f"{mirror.path.rstrip('/')}{parsed.path}",
        )
        candidates.append(urlunparse(mirrored))
    if allow_direct_hf:
        candidates.append(url)
    return candidates


def download_file(
    url: str,
    output_path: Path,
    *,
    force: bool = False,
    hf_mirror_bases: list[str] | tuple[str, ...] | None = None,
    allow_direct_hf: bool = False,
    retries: int = 3,
    timeout_seconds: int = 120,
    chunk_size_bytes: int = 8 * 1024 * 1024,
) -> DownloadResult:
    if output_path.exists() and not force:
        return DownloadResult(
            source_url=url,
            resolved_url=str(output_path),
            output_path=output_path,
            size_bytes=int(output_path.stat().st_size),
            used_mirror=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    errors: list[str] = []
    candidates = build_download_candidates(
        url,
        hf_mirror_bases=hf_mirror_bases,
        allow_direct_hf=allow_direct_hf,
    )

    for candidate in candidates:
        for attempt in range(1, max(int(retries), 1) + 1):
            try:
                tmp_path.unlink(missing_ok=True)
                print(f"Downloading {output_path.name} from {candidate} (attempt {attempt}/{retries})...")
                request = Request(candidate, headers={"User-Agent": USER_AGENT})
                with urlopen(request, timeout=timeout_seconds) as response, tmp_path.open("wb") as handle:
                    while True:
                        chunk = response.read(chunk_size_bytes)
                        if not chunk:
                            break
                        handle.write(chunk)
                size_bytes = int(tmp_path.stat().st_size)
                if size_bytes <= 0:
                    raise RuntimeError("downloaded file is empty")
                tmp_path.replace(output_path)
                return DownloadResult(
                    source_url=url,
                    resolved_url=candidate,
                    output_path=output_path,
                    size_bytes=size_bytes,
                    used_mirror=(candidate != url),
                )
            except Exception as exc:
                tmp_path.unlink(missing_ok=True)
                errors.append(f"{candidate} [attempt {attempt}]: {exc}")

    formatted = "\n".join(f"- {entry}" for entry in errors)
    raise RuntimeError(f"Failed to download {url}. Tried:\n{formatted}")
