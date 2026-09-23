"""The platform's one atomic write."""

from __future__ import annotations

from pathlib import Path


def atomic_write_text(path: Path, text: str) -> Path:
    """Write via temp + replace, creating the parent directory.

    Cell build records and batch records route through this. Each is read back
    by a different process, and a half-written record does not fail loudly: it
    reads as a record with missing fields, which the parsers then fill from
    defaults.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path
