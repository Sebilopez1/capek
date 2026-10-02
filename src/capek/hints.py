"""User-facing install hints (PyPI form, not the developer's editable install)."""

from __future__ import annotations

DIST_NAME = "capek-tech"


def install_hint(extra: str) -> str:
    """``pip install "capek-tech[<extra>]"``: what a PyPI user types to get an optional feature."""
    return f'pip install "{DIST_NAME}[{extra}]"'
