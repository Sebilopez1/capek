"""P1-5: end to end through the real CLI in subprocesses: record 3 -> tag -> list -> export -> load offline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

CAPEK = [sys.executable, "-m", "capek.cli"]


def capek(cwd: Path, *argv: str) -> str:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([*CAPEK, *argv], cwd=cwd, capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, f"capek {' '.join(argv)} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_record_tag_list_export_load(tmp_path: Path) -> None:
    capek(tmp_path, "record", "--env", "so101_reach", "--policy", "scripted", "--episodes", "3", "--seed", "0",
        "--out", "runs/demo")  # fmt: skip
    session = tmp_path / "runs" / "demo"
    rows = [json.loads(line) for line in (session / "episodes.jsonl").read_text().splitlines()]
    assert [r["num_frames"] for r in rows] == [90, 90, 90]

    capek(tmp_path, "tag", "runs/demo", "--episode", "2", "--label", "fail", "--note", "overshot")
    listing = capek(tmp_path, "list", "runs/demo")
    for column in ("index", "length", "sim_success", "final_error_m", "label", "notes"):
        assert column in listing.splitlines()[0]
    assert "overshot" in listing
    listed = json.loads(capek(tmp_path, "list", "runs/demo", "--json"))
    assert listed[2]["label"] == "fail" and listed[2]["notes"] == "overshot"

    pytest.importorskip("lerobot")
    capek(tmp_path, "export", "runs/demo", "--out", "datasets/demo", "--repo-id", "local/demo")
    root = tmp_path / "datasets" / "demo"
    assert (root / "meta" / "info.json").is_file()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/demo", root=root)
    assert ds.num_episodes == 3 and ds.num_frames == 270 and ds.fps == 30
    assert ds.meta.info["robot_type"] == "so101_sim" and ds.meta.info["codebase_version"] == "v3.0"
    shapes = {k: tuple(v["shape"]) for k, v in ds.meta.features.items()}
    assert shapes["observation.state"] == (6,) and shapes["observation.environment_state"] == (3,)
    assert shapes["action"] == (6,) and shapes["next.reward"] == (1,) and shapes["next.success"] == (1,)
    assert ds.meta.features["next.success"]["dtype"] == "bool"
    assert ds.meta.features["action"]["names"][0] == "shoulder_pan.pos"

    with np.load(session / "episodes" / "episode_000001.npz") as z:
        item = ds[90 + 45]
        np.testing.assert_allclose(item["observation.state"].numpy(), z["observation.state"][45], atol=1e-7)
        np.testing.assert_allclose(item["action"].numpy(), z["action"][45], atol=1e-7)
        assert item["task"] == "Move the gripper tip to the target point."

    tags = json.loads((root / "meta" / "capek_tags.json").read_text())
    assert tags["dataset"]["total_frames"] == ds.num_frames and tags["dataset"]["total_episodes"] == 3
    for row in listed:
        exported = dict(tags["episodes"][str(row["episode_index"])])
        assert exported.pop("source_episode_index") == row.pop("episode_index")
        assert exported == row
