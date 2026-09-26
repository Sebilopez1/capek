"""Crash-safe file writes: write to a temp file in the same directory, fsync, then ``os.replace``.

A crash (or exception) before the final rename leaves the previous file untouched and removes the temp file.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any


def atomic_write(path: Path, write: Callable[[IO[bytes]], None]) -> None:
    """Call ``write(fh)`` on a temp file next to ``path`` and atomically move it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def dump_json_line(row: dict[str, Any]) -> bytes:
    """Serialize one JSONL row (kept as a separate function so tests can inject a mid-write crash)."""
    return (json.dumps(row, sort_keys=False, allow_nan=False) + "\n").encode("utf-8")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    def _write(fh: IO[bytes]) -> None:
        for row in rows:
            fh.write(dump_json_line(row))

    atomic_write(path, _write)


def atomic_write_json(path: Path, obj: Any) -> None:
    data = (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda fh: fh.write(data))
