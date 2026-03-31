#!/usr/bin/env python3
"""Sync a Markdown note into a Feishu docx document."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BATCH_SIZE = 50
HEADING_MAP = {
    1: (3, "heading1"),
    2: (4, "heading2"),
    3: (5, "heading3"),
    4: (6, "heading4"),
    5: (7, "heading5"),
    6: (8, "heading6"),
    7: (9, "heading7"),
    8: (10, "heading8"),
    9: (11, "heading9"),
}
BULLET_PATTERN = re.compile(r"^\s*[-*]\s+(?P<body>.+?)\s*$")
ORDERED_PATTERN = re.compile(r"^\s*(?P<num>\d+)\.\s+(?P<body>.+?)\s*$")
QUOTE_PATTERN = re.compile(r"^\s*>\s?(?P<body>.*)$")
HEADING_PATTERN = re.compile(r"^(?P<marks>#{1,9})\s+(?P<title>.+?)\s*$")
PIPE_TABLE_PATTERN = re.compile(r"^\|.*\|\s*$")


def load_mentor_module():
    candidates: list[Path] = []
    if os.getenv("ARTICLE_NOTE_MENTOR_FEISHU_SCRIPT"):
        candidates.append(Path(os.environ["ARTICLE_NOTE_MENTOR_FEISHU_SCRIPT"]).expanduser())
    if os.getenv("CODEX_HOME"):
        candidates.append(
            Path(os.environ["CODEX_HOME"])
            / "skills"
            / "article-note-mentor"
            / "scripts"
            / "feishu_docx_finalize.py"
        )
    if os.getenv("USERPROFILE"):
        candidates.append(
            Path(os.environ["USERPROFILE"])
            / ".codex"
            / "skills"
            / "article-note-mentor"
            / "scripts"
            / "feishu_docx_finalize.py"
        )
    candidates.append(
        Path.home()
        / ".codex"
        / "skills"
        / "article-note-mentor"
        / "scripts"
        / "feishu_docx_finalize.py"
    )
    candidates.extend(
        Path("C:/Users").glob("*/.codex/skills/article-note-mentor/scripts/feishu_docx_finalize.py")
    )

    target = None
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            target = resolved
            break
    if target is None:
        looked = "\n".join(str(path) for path in candidates)
        raise SystemExit("Unable to locate article-note-mentor script. Looked in:\n" + looked)

    spec = importlib.util.spec_from_file_location("mentor_feishu_docx_finalize", target)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to import mentor script from: {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mentor = load_mentor_module()
FeishuAPIError = mentor.FeishuAPIError


@dataclass
class MarkdownBlock:
    kind: str
    block_type: int | None = None
    field_name: str | None = None
    elements: list[dict[str, Any]] | None = None
    rows: list[list[str]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace or append Feishu docx content from Markdown.")
    parser.add_argument("--note-file", required=True, help="Local Markdown note file.")
    parser.add_argument("--doc-url", help="Feishu docx URL.")
    parser.add_argument("--doc-token", help="Feishu docx token/document_id.")
    parser.add_argument(
        "--backup-dir",
        help="Backup root directory. Default: <note dir>/feishu-backups",
    )
    parser.add_argument(
        "--replace-root",
        action="store_true",
        help="Delete all root children before writing imported blocks.",
    )
    return parser.parse_args()


def build_text_elements(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return [{"text_run": {"content": ""}}]
    elements = mentor.split_text_run_markdown(stripped, None)
    elements, _ = mentor.render_inline_formulas(elements)
    return elements or [{"text_run": {"content": stripped}}]


def make_equation_block(content: str) -> MarkdownBlock:
    expression = content.strip()
    return MarkdownBlock(
        kind="equation",
        block_type=2,
        field_name="text",
        elements=[{"equation": {"content": expression}}],
    )


def make_text_block(block_type: int, field_name: str, text: str) -> MarkdownBlock:
    return MarkdownBlock(
        kind="text",
        block_type=block_type,
        field_name=field_name,
        elements=build_text_elements(text),
    )


def is_special_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if HEADING_PATTERN.match(stripped):
        return True
    if BULLET_PATTERN.match(line):
        return True
    if ORDERED_PATTERN.match(line):
        return True
    if QUOTE_PATTERN.match(line):
        return True
    if PIPE_TABLE_PATTERN.match(stripped):
        return True
    if stripped.startswith("$$"):
        return True
    return False


def parse_equation_block(lines: list[str], start: int) -> tuple[MarkdownBlock, int]:
    stripped = lines[start].strip()
    if stripped != "$$" and stripped.endswith("$$") and len(stripped) > 4:
        return make_equation_block(stripped[2:-2]), start + 1

    expr_lines: list[str] = []
    index = start + 1
    while index < len(lines):
        current = lines[index].rstrip("\n")
        if current.strip() == "$$":
            break
        expr_lines.append(current)
        index += 1
    if index >= len(lines):
        raise SystemExit(f"Unclosed display equation starting at line {start + 1}.")
    return make_equation_block("\n".join(expr_lines)), index + 1


def parse_table_block(lines: list[str], start: int) -> tuple[MarkdownBlock, int]:
    table_lines: list[str] = []
    index = start
    while index < len(lines) and PIPE_TABLE_PATTERN.match(lines[index].strip()):
        table_lines.append(lines[index].rstrip("\n"))
        index += 1
    rows = mentor.parse_pipe_table_lines(table_lines)
    return MarkdownBlock(kind="table", rows=rows), index


def parse_quote_block(lines: list[str], start: int) -> tuple[MarkdownBlock, int]:
    parts: list[str] = []
    index = start
    while index < len(lines):
        match = QUOTE_PATTERN.match(lines[index])
        if not match:
            break
        body = match.group("body").strip()
        if body:
            parts.append(body)
        index += 1
    return make_text_block(15, "quote", " ".join(parts)), index


def parse_paragraph_block(lines: list[str], start: int) -> tuple[MarkdownBlock, int]:
    parts: list[str] = []
    index = start
    while index < len(lines):
        current = lines[index].rstrip("\n")
        if not current.strip():
            break
        if index != start and is_special_line(current):
            break
        parts.append(current.strip())
        index += 1
    return make_text_block(2, "text", " ".join(parts)), index


def parse_markdown(note_path: Path) -> list[MarkdownBlock]:
    lines = note_path.read_text(encoding="utf-8").splitlines()
    blocks: list[MarkdownBlock] = []
    index = 0

    while index < len(lines):
        line = lines[index].rstrip("\n")
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        heading_match = HEADING_PATTERN.match(stripped)
        if heading_match:
            level = len(heading_match.group("marks"))
            block_type, field_name = HEADING_MAP[level]
            blocks.append(make_text_block(block_type, field_name, heading_match.group("title")))
            index += 1
            continue

        if stripped.startswith("$$"):
            block, index = parse_equation_block(lines, index)
            blocks.append(block)
            continue

        if PIPE_TABLE_PATTERN.match(stripped):
            block, index = parse_table_block(lines, index)
            blocks.append(block)
            continue

        bullet_match = BULLET_PATTERN.match(line)
        if bullet_match:
            blocks.append(make_text_block(12, "bullet", bullet_match.group("body")))
            index += 1
            continue

        ordered_match = ORDERED_PATTERN.match(line)
        if ordered_match:
            blocks.append(make_text_block(13, "ordered", ordered_match.group("body")))
            index += 1
            continue

        quote_match = QUOTE_PATTERN.match(line)
        if quote_match:
            block, index = parse_quote_block(lines, index)
            blocks.append(block)
            continue

        block, index = parse_paragraph_block(lines, index)
        blocks.append(block)

    return blocks


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def block_to_payload(block: MarkdownBlock) -> dict[str, Any]:
    if block.kind != "text" and block.kind != "equation":
        raise ValueError(f"Unsupported non-text payload conversion: {block.kind}")
    if block.block_type is None or block.field_name is None or block.elements is None:
        raise ValueError("Incomplete MarkdownBlock for text payload.")
    return {
        "block_type": block.block_type,
        block.field_name: {"elements": block.elements},
    }


def load_annotations(note_path: Path) -> list[dict[str, str]]:
    annotations_path = note_path.with_suffix("")
    annotations_path = note_path.parent / f"{annotations_path.name}.format-annotations.json"
    if not annotations_path.exists():
        return []
    return json.loads(annotations_path.read_text(encoding="utf-8"))


def get_root_page(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    for block in blocks:
        if block.get("block_type") == 1:
            return block
    raise SystemExit("Unable to locate root page block in Feishu document.")


def get_block_map(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {block["block_id"]: block for block in blocks}


def can_reuse_root_child(root_page: dict[str, Any], block_map: dict[str, dict[str, Any]]) -> str | None:
    children = root_page.get("children", [])
    if len(children) != 1:
        return None
    child = block_map.get(children[0])
    if not child or child.get("block_type") != 2:
        return None
    return child["block_id"]


def create_table(
    client: Any,
    document_id: str,
    access_token: str,
    parent_id: str,
    index: int,
    rows: list[list[str]],
) -> dict[str, Any]:
    children_id, descendants, table_temp_id = mentor.build_descendant_table_payload(rows)
    result = client.create_descendant_blocks(
        document_id,
        parent_id,
        access_token,
        children_id,
        descendants,
        index=index,
    )
    relations = {
        relation.get("temporary_block_id"): relation.get("block_id")
        for relation in result.get("data", result).get("block_id_relations", [])
    }
    return {
        "created_table_block_id": relations.get(table_temp_id),
        "row_count": len(rows),
        "column_count": max((len(row) for row in rows), default=0),
        "inserted_blocks": len(children_id),
    }


def sync_blocks(
    client: Any,
    document_id: str,
    access_token: str,
    root_id: str,
    markdown_blocks: list[MarkdownBlock],
    seed_block_id: str | None = None,
) -> dict[str, Any]:
    inserted = 0
    tables = 0
    current_index = 0
    pending_text: list[MarkdownBlock] = []

    if seed_block_id:
        if len(markdown_blocks) < 2:
            raise SystemExit("Seed-block fallback requires at least two Markdown blocks.")
        first_block = markdown_blocks[0]
        seed_markdown_block = markdown_blocks[1]
        if first_block.kind == "table" or seed_markdown_block.kind == "table":
            raise SystemExit("Seed-block fallback cannot start with table blocks.")

        client.create_child_blocks(
            document_id,
            root_id,
            access_token,
            [block_to_payload(first_block)],
            index=0,
        )
        inserted += 1
        current_index = 2
        client.batch_update_blocks(
            document_id,
            access_token,
            [
                {
                    "block_id": seed_block_id,
                    "update_text_elements": {"elements": seed_markdown_block.elements or []},
                }
            ],
        )
        inserted += 1
        markdown_blocks = markdown_blocks[2:]
        time.sleep(0.2)

    def flush_text_batch() -> None:
        nonlocal inserted, current_index, pending_text
        if not pending_text:
            return
        payloads = [block_to_payload(block) for block in pending_text]
        for batch in chunked(payloads, DEFAULT_BATCH_SIZE):
            client.create_child_blocks(
                document_id,
                root_id,
                access_token,
                batch,
                index=current_index,
            )
            inserted += len(batch)
            current_index += len(batch)
            time.sleep(0.2)
        pending_text = []

    for block in markdown_blocks:
        if block.kind == "table":
            flush_text_batch()
            rows = block.rows or []
            create_table(client, document_id, access_token, root_id, current_index, rows)
            current_index += 1
            inserted += 1
            tables += 1
            time.sleep(0.2)
            continue
        pending_text.append(block)

    flush_text_batch()
    return {"inserted_blocks": inserted, "table_blocks": tables}


def main() -> int:
    mentor.configure_stdio()
    args = parse_args()

    note_path = Path(args.note_file).expanduser().resolve()
    if not note_path.exists():
        raise SystemExit(f"Markdown note not found: {note_path}")

    document_id = mentor.resolve_document_id(args.doc_url, args.doc_token)
    app_id, app_secret = mentor.require_credentials()
    client = mentor.FeishuDocxClient(app_id, app_secret)
    access_token = client.get_tenant_access_token()
    raw_content = client.get_raw_content(document_id, access_token)
    blocks = client.list_blocks(document_id, access_token)
    root_page = get_root_page(blocks)
    block_map = get_block_map(blocks)
    root_id = root_page["block_id"]

    backup_root = (
        Path(args.backup_dir).expanduser().resolve()
        if args.backup_dir
        else note_path.parent / "feishu-backups"
    )
    mentor.ensure_directory(backup_root)
    annotations = load_annotations(note_path)
    backup_dir = mentor.write_backup(
        backup_root,
        document_id,
        raw_content,
        blocks,
        {"mode": "markdown_sync", "replace_root": args.replace_root, "note_file": str(note_path)},
        annotations,
    )

    seed_block_id = None
    if args.replace_root:
        child_count = len(root_page.get("children", []))
        if child_count > 0:
            try:
                client.batch_delete_children(
                    document_id,
                    root_id,
                    access_token,
                    0,
                    child_count,
                )
                time.sleep(0.2)
            except FeishuAPIError as exc:
                reusable = can_reuse_root_child(root_page, block_map)
                if reusable is None:
                    raise
                seed_block_id = reusable
                print(
                    "Root child deletion is forbidden for this document; "
                    "reusing the single existing text child as a seed block."
                )

    markdown_blocks = parse_markdown(note_path)
    result = sync_blocks(
        client,
        document_id,
        access_token,
        root_id,
        markdown_blocks,
        seed_block_id=seed_block_id,
    )

    sync_report = {
        "document_id": document_id,
        "note_file": str(note_path),
        "replace_root": args.replace_root,
        "parsed_blocks": len(markdown_blocks),
        "inserted_blocks": result["inserted_blocks"],
        "table_blocks": result["table_blocks"],
        "backup_dir": str(backup_dir),
    }
    mentor.write_json(backup_dir / "sync-report.json", sync_report)
    print(json.dumps(sync_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FeishuAPIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
