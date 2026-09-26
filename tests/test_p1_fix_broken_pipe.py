"""PM report: `rrc list runs/demo | head` must not print a BrokenPipeError traceback."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import make_synthetic_session, synthetic_episode

from robot_report_card import cli
from robot_report_card.session import Session

RRC = [sys.executable, "-m", "robot_report_card.cli"]


@pytest.fixture
def big_session(tmp_path: Path) -> Path:
    """A session whose `list` output (~300 KB) is far larger than a pipe buffer (64 KB)."""
    root = tmp_path / "s"
    s = make_synthetic_session(root, n=1)
    metas = []
    for i in range(3000):
        m = synthetic_episode(i).meta
        m.notes = "a fairly long note so every row is wide " * 2
        metas.append(m)
    s.write_metas(metas)  # list only reads episodes.jsonl
    return root


@pytest.mark.parametrize("extra", [[], ["--json"]])
def test_list_piped_into_head_exits_quietly(big_session: Path, extra: list[str]) -> None:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    proc = subprocess.Popen(
        [*RRC, "list", str(big_session), *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )
    assert proc.stdout is not None and proc.stderr is not None
    first = proc.stdout.readline()  # like `head -1`: read a little, then close the pipe
    proc.stdout.close()
    stderr = proc.stderr.read().decode()
    rc = proc.wait(timeout=60)
    assert first
    assert "Traceback" not in stderr and "BrokenPipeError" not in stderr and "Exception ignored" not in stderr
    assert rc == cli.EXIT_BROKEN_PIPE


def test_shell_pipe_to_head(big_session: Path) -> None:
    cmd = " ".join([*RRC, "list", str(big_session)]) + " | head -3"
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    assert len(r.stdout.splitlines()) == 3 and r.stderr == ""
    assert r.returncode == cli.EXIT_BROKEN_PIPE


def test_broken_pipe_in_process_returns_141(monkeypatch, tmp_path: Path, capsys) -> None:
    # capsys: sys.stdout has no real fd here, so the devnull redirect is skipped instead of hijacking pytest's
    make_synthetic_session(tmp_path / "s", n=2)

    def closed_pipe(*a, **k):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr("builtins.print", closed_pipe)
    assert cli.main(["list", str(tmp_path / "s")]) == cli.EXIT_BROKEN_PIPE
    assert len(Session.open(tmp_path / "s").read_metas()) == 2
