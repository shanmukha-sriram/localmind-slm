"""Safe file I/O helpers."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temp file, then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:  # newline="" avoids \r\r\n on Windows
        f.write(text)
    tmp.replace(path)


def write_json(obj: Any, path: Path) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, df.to_csv(index=False))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines, skipping a truncated last line left by an interrupted run."""
    rows = []

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("%s line %d is not valid JSON; skipped", path.name, number)
    return rows
