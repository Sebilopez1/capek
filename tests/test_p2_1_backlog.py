"""P2-1: phase 1 backlog - corrupt .npz on export, and `capek record | head` exiting like SIGPIPE."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from helpers import FakeEnv, make_synthetic_session

from capek import cli
from capek.session import Session, SessionError
from capek.sim import registry


def _corrupt(path: Path) -> None:
    path.write_bytes(b"PK\x03\x04 definitely not a zip archive")


def test_load_arrays_wraps_corrupt_npz(tmp_path: Path) -> None:
    s = make_synthetic_session(tmp_path / "s", n=2)
    _corrupt(s.npz_path(1))
    with pytest.raises(SessionError, match="corrupt episode file"):
        s.load_arrays(1)


def test_record_piped_into_closed_reader_exits_141_in_process(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(registry, "make_env", lambda env_id: FakeEnv())
    printed = {"n": 0}

    def print_then_break(*a, **k):
        printed["n"] += 1
        if printed["n"] == 3:
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr("builtins.print", print_then_break)
    out = tmp_path / "s"
    assert cli.main(["record", "--episodes", "10", "--max-steps", "4", "--out", str(out)]) == cli.EXIT_BROKEN_PIPE
    metas = Session.open(out).read_metas()
    assert [m.episode_index for m in metas] == [0, 1, 2]  # episode 2 was saved before its line was printed
    assert all(Session.open(out).load_arrays(i)["action"].shape == (4, 6) for i in range(3))


def test_record_pipe_to_head_real_cli(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    out = tmp_path / "s"
    cmd = f"{sys.executable} -m capek.cli record --episodes 400 --max-steps 3 --out {out} | head -1"
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == cli.EXIT_BROKEN_PIPE, r.stderr
    assert r.stdout.startswith("episode    0:") and r.stderr == ""
    metas = Session.open(out).read_metas()
    assert 1 <= len(metas) < 400
    assert [m.episode_index for m in metas] == list(range(len(metas)))
    assert r.returncode != 1 and "episodes saved" not in r.stderr


# ---- export of a corrupt / malformed episode file (needs lerobot) ------------------------------------------
@pytest.mark.parametrize("kind", ["not_a_zip", "wrong_shape"])
def test_export_corrupt_npz_fails_cleanly(tmp_path: Path, capsys, kind: str) -> None:
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    s = make_synthetic_session(tmp_path / "s", n=3)
    if kind == "not_a_zip":
        _corrupt(s.npz_path(2))
    else:
        arrays = s.load_arrays(2)
        arrays["action"] = arrays["action"][:, :4]  # a valid .npz whose arrays don't match session.json
        np.savez(s.npz_path(2), **arrays)
    out = tmp_path / "ds"
    assert cli.main(["export", str(tmp_path / "s"), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("capek: error:") and "Traceback" not in err
    assert ("corrupt episode file" in err) if kind == "not_a_zip" else ("action" in err)
    assert not out.exists()
