"""P2-4: pyarrow reader for LeRobot v3.0 / v2.1 / v2.0 (no lerobot, no torch)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from lerobot_files import smooth_reach, v3_to_v2, write_dataset  # noqa: E402

from robot_report_card.score.reader import ReaderError, read_dataset  # noqa: E402


def _episodes(n: int = 5, frames: int = 40, seed: int = 0) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [smooth_reach(rng, frames, noise=0.01) for _ in range(n)]


def test_v3_round_trip_sorted_across_shuffled_files(tmp_path: Path) -> None:
    eps = _episodes()
    ds = read_dataset(write_dataset(tmp_path / "ds", eps, files=3))
    assert ds.codebase_version == "v3.0" and ds.fps == 30 and len(ds.episodes) == 5 and ds.num_frames == 200
    assert ds.state_names and ds.state_names[0] == "shoulder_pan.pos"
    for got, want in zip(ds.episodes, eps, strict=True):
        assert got.state.dtype == np.float64
        np.testing.assert_array_equal(got.state, want["observation.state"].astype(np.float32))
        np.testing.assert_array_equal(got.action, want["action"].astype(np.float32))
        np.testing.assert_array_equal(got.frame_index, np.arange(40))
        np.testing.assert_allclose(got.timestamp, np.arange(40) / 30, atol=1e-6)
        assert got.success is not None and got.success[-1] and not got.success[0]


@pytest.mark.parametrize("version", ["v2.1", "v2.0"])
def test_v2_layouts_read_identically(tmp_path: Path, version: str) -> None:
    eps = _episodes()
    a = read_dataset(write_dataset(tmp_path / "v3", eps))
    b = read_dataset(write_dataset(tmp_path / version, eps, version=version))
    assert b.codebase_version == version
    for x, y in zip(a.episodes, b.episodes, strict=True):
        for f in ("state", "action", "timestamp", "frame_index", "success"):
            np.testing.assert_array_equal(getattr(x, f), getattr(y, f))


def test_optional_success_and_custom_keys(tmp_path: Path) -> None:
    eps = [{k: v for k, v in e.items() if k != "next.success"} for e in _episodes(2)]
    root = write_dataset(tmp_path / "ds", eps)
    assert all(e.success is None for e in read_dataset(root).episodes)
    ds = read_dataset(root, state_key="action", action_key="observation.state")
    np.testing.assert_array_equal(ds.episodes[0].state, eps[0]["action"].astype(np.float32))


def test_nested_names_dict(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "ds", _episodes(1))
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.state"]["names"] = {"motors": ["a", "b", "c", "d", "e", "f"]}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    assert read_dataset(root).state_names == ["a", "b", "c", "d", "e", "f"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: (r / "meta" / "info.json").unlink(), "no meta/info.json"),
        (lambda r: (r / "meta" / "info.json").write_text("{nope"), "not valid JSON"),
        (lambda r: _set_info(r, codebase_version="v1.6"), "unsupported codebase_version 'v1.6'"),
        (lambda r: _set_info(r, fps=0), "no valid fps"),
        (lambda r: [p.unlink() for p in r.glob("data/*/*.parquet")], "no data files"),
        (lambda r: _set_feature_shape(r, "action", [7]), r"6 dims in the data but shape \[7\]"),
    ],
)
def test_clear_errors(tmp_path: Path, mutate, message: str) -> None:
    root = write_dataset(tmp_path / "ds", _episodes(2))
    mutate(root)
    with pytest.raises(ReaderError, match=message):
        read_dataset(root)


def test_missing_key_names_the_flag_and_options(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "ds", _episodes(1))
    with pytest.raises(ReaderError, match=r"no 'observation.joints' column \(pass --state-key; vector features: "):
        read_dataset(root, state_key="observation.joints")
    with pytest.raises(ReaderError, match="--action-key"):
        read_dataset(root, action_key="act")


def test_tags_used_only_when_fingerprint_matches(tmp_path: Path) -> None:
    tags = {"dataset": {"total_episodes": 2, "total_frames": 80}, "episodes": {"0": {"label": "fail"}}}
    ds = read_dataset(write_dataset(tmp_path / "ok", _episodes(2), tags=tags))
    assert ds.tags is not None and ds.tags["episodes"]["0"]["label"] == "fail" and ds.notes == []
    tags["dataset"]["total_frames"] = 81
    stale = read_dataset(write_dataset(tmp_path / "stale", _episodes(2), tags=tags))
    assert stale.tags is None and "stale" in stale.notes[0]


def test_13500_frames_under_one_second(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "big", _episodes(150, 90), files=1)
    t0 = time.perf_counter()
    ds = read_dataset(root)
    elapsed = time.perf_counter() - t0
    assert ds.num_frames == 13_500 and elapsed < 1.0, elapsed


def test_reader_never_imports_lerobot_or_torch(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "ds", _episodes(2))
    code = (
        "import sys; from robot_report_card.score.reader import read_dataset; "
        f"d = read_dataset({str(root)!r}); assert d.num_frames == 80; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('lerobot', 'torch')); print(bad)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def _set_info(root: Path, **kv) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    info.update(kv)
    (root / "meta" / "info.json").write_text(json.dumps(info))


def _set_feature_shape(root: Path, key: str, shape: list[int]) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"][key]["shape"] = shape
    (root / "meta" / "info.json").write_text(json.dumps(info))


# ---- against lerobot's own loader on a real `rrc export` (lerobot venv only) ----------------------------------
def test_arrays_equal_lerobot_dataset_on_real_export(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from robot_report_card import cli

    session = tmp_path / "s"
    assert cli.main(["record", "--episodes", "2", "--noise", "0.1", "--out", str(session), "-q"]) == 0
    assert cli.main(["record", "--episodes", "2", "--policy", "random", "--append", "--out", str(session), "-q"]) == 0
    root = tmp_path / "ds"
    assert cli.main(["export", str(session), "--out", str(root), "--repo-id", "local/ds"]) == 0
    ours = read_dataset(root)
    theirs = lerobot_dataset.LeRobotDataset("local/ds", root=root).hf_dataset.with_format("numpy")
    ep = np.asarray(theirs["episode_index"])
    assert ours.num_frames == len(ep) and len(ours.episodes) == 4
    for e in ours.episodes:
        m = ep == e.index
        np.testing.assert_array_equal(e.state, np.stack(theirs["observation.state"])[m].astype(np.float64))
        np.testing.assert_array_equal(e.action, np.stack(theirs["action"])[m].astype(np.float64))
        np.testing.assert_array_equal(e.timestamp, np.asarray(theirs["timestamp"])[m].astype(np.float64))
        np.testing.assert_array_equal(e.success, np.asarray(theirs["next.success"])[m].reshape(-1))
    assert ours.tags is not None and len(ours.tags["episodes"]) == 4
    for version in ("v2.1", "v2.0"):
        copy = read_dataset(v3_to_v2(root, tmp_path / version, version))
        for x, y in zip(ours.episodes, copy.episodes, strict=True):
            np.testing.assert_array_equal(x.state, y.state)
            np.testing.assert_array_equal(x.action, y.action)
        assert copy.tags == ours.tags
