"""P4-6 backlog: N3 (`capek record --append` checks that the session's existing episode files are present)."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import FakeEnv

from capek import cli
from capek.session import Session
from capek.sim import registry


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(registry, "make_env", lambda env_id: FakeEnv())


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_append_refuses_a_session_with_missing_episode_files(tmp_path: Path, fake, capsys) -> None:
    out = tmp_path / "s"
    assert cli.main(["record", "--episodes", "3", "--max-steps", "5", "--out", str(out), "-q"]) == 0
    Session.open(out).npz_path(1).unlink()
    before = _snapshot(out)
    assert cli.main(["record", "--append", "--episodes", "1", "--max-steps", "5", "--out", str(out), "-q"]) == 1
    err = capsys.readouterr().err
    assert "1 existing episode file(s) are missing (episodes 1)" in err and "Traceback" not in err
    assert _snapshot(out) == before


def test_append_still_works_on_an_intact_session(tmp_path: Path, fake) -> None:
    out = tmp_path / "s"
    assert cli.main(["record", "--episodes", "2", "--max-steps", "5", "--out", str(out), "-q"]) == 0
    assert cli.main(["record", "--append", "--episodes", "2", "--max-steps", "5", "--out", str(out), "-q"]) == 0
    assert [m.episode_index for m in Session.open(out).read_metas()] == [0, 1, 2, 3]
