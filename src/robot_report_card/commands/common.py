"""Helpers shared by subcommands."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def fail(msg: str) -> int:
    """Print a one-line error to stderr and return exit code 1."""
    print(f"rrc: error: {msg}", file=sys.stderr)
    return 1


def parse_indices(values: list[str] | None) -> list[int] | None:
    """Parse repeated ``--episode`` values: ``3``, ``1,4`` or ranges ``2-5`` (inclusive)."""
    if not values:
        return None
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            try:
                if "-" in part[1:]:
                    lo, hi = part.split("-", 1)
                    a, b = int(lo), int(hi)
                    if b < a:
                        raise ValueError
                    out.extend(range(a, b + 1))
                else:
                    out.append(int(part))
            except ValueError:
                raise argparse.ArgumentTypeError(f"bad episode index {part!r} (use N, N,M or N-M)") from None
    return out


def report_path_problem(path: Path, overwrite: bool, protected: list[Path]) -> str | None:
    """Why a JSON report can't be written to ``path`` (None if it can): never inside ``protected`` dirs
    (datasets, checkpoints), never over an existing file without ``--overwrite``."""
    target = path.resolve()
    for root in protected:
        r = root.resolve()
        if target == r or r in target.parents:
            return f"refusing to write the report inside {root}; pass --json-out somewhere else"
    if path.is_dir():
        return f"--json-out {path} is a directory"
    if path.exists() and not overwrite:
        return f"{path} already exists; pass --overwrite to replace it (or choose another --json-out)"
    return None


def slug(text: str) -> str:
    """File-name-safe version of a policy spec, e.g. 'bc:ckpt/a' -> 'bc_ckpt_a'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "policy"
