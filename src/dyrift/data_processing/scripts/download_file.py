from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dyrift.data_processing.core.downloads import download_file, is_huggingface_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a remote file with mirror-aware Python streaming. "
            "Hugging Face URLs use mirrors first and do not direct-connect by default."
        )
    )
    parser.add_argument("--url", required=True, help="Remote file URL.")
    parser.add_argument("--output", type=Path, required=True, help="Local output path.")
    parser.add_argument(
        "--hf-mirror",
        action="append",
        default=[],
        help=(
            "Optional Hugging Face mirror base URL. Can be passed multiple times. "
            "Examples: https://hf-mirror.com"
        ),
    )
    parser.add_argument(
        "--allow-direct-hf",
        action="store_true",
        help="Allow falling back to direct Hugging Face download after mirror attempts.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if output already exists.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per candidate URL.")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Socket timeout per request.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if is_huggingface_url(args.url) and not args.allow_direct_hf:
        print("Hugging Face URL detected: mirror-only mode is enabled by default.")
    result = download_file(
        args.url,
        args.output,
        force=bool(args.force),
        hf_mirror_bases=list(args.hf_mirror),
        allow_direct_hf=bool(args.allow_direct_hf),
        retries=int(args.retries),
        timeout_seconds=int(args.timeout_seconds),
    )
    print(
        f"Downloaded to {result.output_path} "
        f"(size={result.size_bytes} bytes, resolved_url={result.resolved_url}, used_mirror={result.used_mirror})"
    )


if __name__ == "__main__":
    main()
