"""Locate the vendored SO-101 MJCF (TheRobotStudio SO-ARM100, Apache-2.0; see assets/so101/SOURCE.md)."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def so101_xml_path() -> Path:
    """Path to ``so101_new_calib.xml`` inside the installed package (no network, no robot_descriptions)."""
    return Path(str(files("robot_report_card.sim").joinpath("assets/so101/so101_new_calib.xml")))
