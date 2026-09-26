"""P1-4: `rrc export` fails cleanly when lerobot isn't importable (runs whether or not it is installed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from helpers import make_synthetic_session

from robot_report_card import cli


def test_missing_lerobot_gives_clear_error(tmp_path: Path, monkeypatch, capsys) -> None:
    make_synthetic_session(tmp_path / "s", n=2)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)  # import -> ImportError
    assert cli.main(["export", str(tmp_path / "s"), "--out", str(tmp_path / "ds")]) == 1
    err = capsys.readouterr().err
    assert 'pip install "robot-report-card[lerobot]"' in err and "Traceback" not in err
    assert not (tmp_path / "ds").exists()


# ---- QA R1: --overwrite must never delete the session or a directory rrc did not write --------------------
# These refusals happen before lerobot is imported, so they run with or without the lerobot extra.
def _fake_rrc_dataset(root: Path, tool: str = "robot-report-card") -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text('{"codebase_version": "v3.0"}')
    (root / "meta" / "rrc_tags.json").write_text(json.dumps({"schema_version": 1, "tool": tool}))


def _snapshot(root: Path) -> list[str]:
    return sorted(str(p) for p in root.rglob("*"))


def test_overwrite_refuses_when_out_is_the_session(tmp_path: Path, capsys) -> None:
    d = tmp_path / "D"
    make_synthetic_session(d, n=2)
    _fake_rrc_dataset(d)  # B1 layout 1: S == D and D looks like an rrc dataset
    before = _snapshot(tmp_path)
    assert cli.main(["export", str(d), "--out", str(d), "--overwrite"]) == 1
    assert "overlaps the session" in capsys.readouterr().err
    assert _snapshot(tmp_path) == before


def test_overwrite_refuses_when_session_is_inside_out(tmp_path: Path, capsys) -> None:
    d = tmp_path / "D"
    _fake_rrc_dataset(d)
    make_synthetic_session(d / "runs" / "s", n=2)  # B1 layout 2: S = D/runs/s
    before = _snapshot(tmp_path)
    assert cli.main(["export", str(d / "runs" / "s"), "--out", str(d), "--overwrite"]) == 1
    assert "overlaps the session" in capsys.readouterr().err
    assert _snapshot(tmp_path) == before
    # the same check through a relative path / '..' spelling
    assert cli.main(["export", str(d / "runs" / "s"), "--out", str(d / "runs" / ".."), "--overwrite"]) == 1


def test_out_inside_session_is_refused(tmp_path: Path, capsys) -> None:
    s = tmp_path / "s"
    make_synthetic_session(s, n=1)
    assert cli.main(["export", str(s), "--out", str(s / "ds")]) == 1
    assert "overlaps the session" in capsys.readouterr().err and not (s / "ds").exists()


@pytest.mark.parametrize("kind", ["foreign_lerobot", "other_tool", "corrupt_tags"])
def test_overwrite_refuses_datasets_rrc_did_not_write(tmp_path: Path, capsys, kind: str) -> None:
    make_synthetic_session(tmp_path / "s", n=1)
    d = tmp_path / "hub_download"
    if kind == "foreign_lerobot":
        (d / "meta").mkdir(parents=True)
        (d / "meta" / "info.json").write_text('{"codebase_version": "v3.0"}')
    elif kind == "other_tool":
        _fake_rrc_dataset(d, tool="someone-else")
    else:
        _fake_rrc_dataset(d)
        (d / "meta" / "rrc_tags.json").write_text("{not json")
    before = _snapshot(d)
    assert cli.main(["export", str(tmp_path / "s"), "--out", str(d), "--overwrite"]) == 1
    err = capsys.readouterr().err
    assert "not a dataset exported by rrc" in err and "delete it yourself" in err
    assert _snapshot(d) == before


def test_overwrite_refuses_symlinked_out(tmp_path: Path, capsys) -> None:
    make_synthetic_session(tmp_path / "s", n=1)
    real = tmp_path / "real"
    _fake_rrc_dataset(real)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert cli.main(["export", str(tmp_path / "s"), "--out", str(link), "--overwrite"]) == 1
    err = capsys.readouterr().err
    assert "is a symlink" in err and "Traceback" not in err
    assert link.is_symlink() and (real / "meta" / "info.json").is_file()


def test_missing_npz_is_reported_before_anything_is_written(tmp_path: Path, capsys) -> None:
    s = make_synthetic_session(tmp_path / "s", n=3)
    s.npz_path(1).unlink()
    assert cli.main(["export", str(tmp_path / "s"), "--out", str(tmp_path / "ds")]) == 1
    assert "missing episode file" in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()
