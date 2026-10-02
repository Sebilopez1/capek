"""QA R3/R4 regressions: `capek record` fails cleanly (no traceback, valid session) - no mujoco needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import FakeEnv, make_synthetic_session

from capek import cli
from capek.session import Session, SessionError, SessionInfo
from capek.sim import registry


@pytest.fixture
def fake_env(monkeypatch):
    holder: dict[str, FakeEnv] = {}

    def install(fail_at=None) -> None:
        holder["env"] = FakeEnv(fail_at)
        monkeypatch.setattr(registry, "make_env", lambda env_id: holder["env"])

    install()
    return install


@pytest.mark.parametrize("noise", ["nan", "inf", "-inf", "-0.1"])
def test_record_rejects_non_finite_or_negative_noise(tmp_path: Path, capsys, noise: str) -> None:
    out = tmp_path / "s"
    assert cli.main(["record", f"--noise={noise}", "--episodes", "1", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "--noise must be a finite number >= 0" in err and "Traceback" not in err
    assert not out.exists()  # validated before anything is written


def test_record_divergence_exits_cleanly_with_saved_count(tmp_path: Path, capsys, fake_env) -> None:
    fake_env(fail_at=(2, 5))  # third episode diverges at frame 5
    out = tmp_path / "s"
    assert cli.main(["record", "--episodes", "4", "--max-steps", "10", "--out", str(out), "-q"]) == 1
    err = capsys.readouterr().err
    assert "simulation diverged" in err and f"2 episodes saved in {out}" in err
    s = Session.open(out)
    assert [m.episode_index for m in s.read_metas()] == [0, 1]
    assert sorted(p.name for p in (out / "episodes").iterdir()) == ["episode_000000.npz", "episode_000001.npz"]
    assert cli.main(["list", str(out), "--json"]) == 0  # the session is still valid


def test_record_oserror_exits_cleanly(tmp_path: Path, capsys, fake_env, monkeypatch) -> None:
    from capek import session as session_mod

    real = session_mod.atomic_write
    calls = {"n": 0}

    def flaky(path, write):
        calls["n"] += 1
        if str(path).endswith(".npz") and "000001" in str(path):
            raise OSError(28, "No space left on device")
        return real(path, write)

    monkeypatch.setattr(session_mod, "atomic_write", flaky)
    out = tmp_path / "s"
    assert cli.main(["record", "--episodes", "3", "--max-steps", "5", "--out", str(out), "-q"]) == 1
    err = capsys.readouterr().err
    assert "No space left on device" in err and "1 episodes saved" in err and "Traceback" not in err


def test_record_nan_metadata_never_orphans_npz(tmp_path: Path, fake_env) -> None:
    fake_env()
    registry.make_env("x").tip_error = lambda: float("nan")  # type: ignore[method-assign]
    out = tmp_path / "s"
    assert cli.main(["record", "--episodes", "1", "--max-steps", "3", "--out", str(out), "-q"]) == 1
    assert list((out / "episodes").iterdir()) == []
    assert Session.open(out).read_metas() == []


INFO = SessionInfo(env_id="t", fps=30, features={}, seeding="t", created_at="t", capek_version="t")


def test_session_create_refuses_file_path_and_non_empty_dir(tmp_path: Path) -> None:
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    with pytest.raises(SessionError, match="not a directory"):
        Session.create(a_file, INFO)
    with pytest.raises(SessionError, match="cannot create"):
        Session.create(a_file / "sub", INFO)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("x")
    with pytest.raises(SessionError, match="not empty"):
        Session.create(busy, INFO)
    assert sorted(p.name for p in busy.iterdir()) == ["keep.txt"]
    empty = tmp_path / "empty"
    empty.mkdir()
    Session.create(empty, INFO)  # an empty dir is fine


@pytest.mark.parametrize("layout", ["file", "under_file", "dataset", "session"])
def test_record_out_refusals_are_clean(tmp_path: Path, capsys, fake_env, layout: str) -> None:
    if layout == "file":
        out = tmp_path / "f"
        out.write_text("x")
    elif layout == "under_file":
        (tmp_path / "f").write_text("x")
        out = tmp_path / "f" / "s"
    elif layout == "dataset":  # B6: an exported LeRobot root
        out = tmp_path / "datasets" / "demo"
        (out / "meta").mkdir(parents=True)
        (out / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    else:
        out = tmp_path / "s"
        make_synthetic_session(out, n=1)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    assert cli.main(["record", "--episodes", "1", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("capek: error:") and "Traceback" not in err
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
