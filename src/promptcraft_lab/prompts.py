import json
import os
from pathlib import Path


def read_text(path: str) -> str:
    # Use utf-8-sig so files with BOM are handled safely on Windows.
    return Path(path).read_text(encoding="utf-8-sig")


def load_prompt_version(path: str) -> dict:
    return json.loads(read_text(path))


def load_jsonl(path: str):
    rows = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_system_prompt(root: Path, version_cfg: dict) -> str:
    prompt_text = os.getenv("PROMPTOPS_SYSTEM_PROMPT_TEXT", "").strip()
    if prompt_text:
        return prompt_text

    prompt_path = os.getenv("PROMPTOPS_SYSTEM_PROMPT_PATH", "").strip()
    if prompt_path:
        return read_text(prompt_path)

    raise RuntimeError(
        "System prompt loading is blocked for public safety. "
        "Set PROMPTOPS_SYSTEM_PROMPT_PATH or PROMPTOPS_SYSTEM_PROMPT_TEXT."
    )
