"""P1-1: package, CLI skeleton, session format, vendored assets."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from helpers import make_synthetic_session, synthetic_episode

from robot_report_card import __version__, cli
from robot_report_card.session import EpisodeMeta, Session, SessionError
from robot_report_card.sim.assets import so101_xml_path


def test_help_lists_four_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("record", "tag", "list", "export"):
        assert cmd in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_console_script_runs() -> None:
    r = subprocess.run([sys.executable, "-m", "robot_report_card.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "export" in r.stdout


def test_episode_round_trips_bit_exactly(tmp_path) -> None:
    s = make_synthetic_session(tmp_path / "sess", n=3)
    for i in range(3):
        original = synthetic_episode(i)
        loaded = Session.open(tmp_path / "sess").load_episode(i)
        assert loaded.meta == original.meta
        assert set(loaded.arrays) == set(original.arrays)
        for k, arr in original.arrays.items():
            assert loaded.arrays[k].dtype == arr.dtype
            assert loaded.arrays[k].shape == arr.shape
            assert loaded.arrays[k].tobytes() == arr.tobytes()
    assert [m.episode_index for m in s.read_metas()] == [0, 1, 2]


def test_session_rejects_bad_arrays_and_reuse(tmp_path) -> None:
    s = make_synthetic_session(tmp_path / "sess", n=1)
    bad = synthetic_episode(1)
    bad.arrays["action"] = bad.arrays["action"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        s.append_episode(bad)
    with pytest.raises(SessionError, match="expected episode_index 1"):
        s.append_episode(synthetic_episode(5))
    with pytest.raises(SessionError, match="already contains a session"):
        make_synthetic_session(tmp_path / "sess", n=1)
    with pytest.raises(SessionError, match="not a session"):
        Session.open(tmp_path / "nope")


def test_episode_meta_validates_label() -> None:
    row = synthetic_episode(0).meta.to_dict()
    row["label"] = "maybe"
    with pytest.raises(ValueError, match="label"):
        EpisodeMeta.from_dict(row)


def test_vendored_mjcf_loads() -> None:
    mujoco = pytest.importorskip("mujoco")
    path = so101_xml_path()
    assert path.is_file() and (path.parent / "LICENSE").is_file()
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.njnt == 6 and m.nu == 6
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper") >= 0
