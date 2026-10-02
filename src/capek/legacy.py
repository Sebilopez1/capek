"""Backward compatibility with files written by 0.1.0, when the project was called Robot Report Card (``rrc``).

New files always use the Capek names; these helpers only let us *read* the old ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TOOL_NAMES = ("capek", "robot-report-card")  # the "tool" field: current, then 0.1.0
TAGS_FILES = ("capek_tags.json", "rrc_tags.json")
POLICY_FILES = ("capek_policy.json", "rrc_policy.json")


def is_our_tool(value: Any) -> bool:
    return value in TOOL_NAMES


def find_file(directory: Path, names: tuple[str, ...]) -> Path:
    """The first of ``names`` that exists in ``directory``, else the current (first) name."""
    for name in names:
        if (directory / name).is_file():
            return directory / name
    return directory / names[0]


def upgrade_version_key(d: dict[str, Any]) -> dict[str, Any]:
    """Rename 0.1.0's ``rrc_version`` key to ``capek_version`` (returns a copy)."""
    if "rrc_version" in d and "capek_version" not in d:
        d = {**d, "capek_version": d["rrc_version"]}
        d.pop("rrc_version")
    return d
