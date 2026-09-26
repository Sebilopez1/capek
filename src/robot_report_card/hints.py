"""User-facing install hints (PyPI form, not the developer's editable install)."""

from __future__ import annotations

DIST_NAME = "robot-report-card"


def install_hint(extra: str) -> str:
    """``pip install "robot-report-card[<extra>]"``: what a PyPI user types to get an optional feature."""
    return f'pip install "{DIST_NAME}[{extra}]"'
