"""P1-3: `rrc tag` and `rrc list` (synthetic sessions; no mujoco / lerobot needed)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from helpers import make_synthetic_session

from robot_report_card import atomic, cli
from robot_report_card.session import Session


@pytest.fixture
def sess(tmp_path: Path) -> Path:
    make_synthetic_session(tmp_path / "s", n=5)
    return tmp_path / "s"


def metas(path: Path):
    return {m.episode_index: m for m in Session.open(path).read_metas()}


def test_label_note_flag_round_trip(sess: Path) -> None:
    assert cli.main(["tag", str(sess), "--episode", "3", "--label", "fail", "--note", "overshot"]) == 0
    assert cli.main(["tag", str(sess), "-e", "3", "--note", "wrist wobble", "--flag", "jerky", "--flag", "slow"]) == 0
    m = metas(sess)[3]
    assert (m.label, m.notes, m.flags) == ("fail", "overshot; wrist wobble", ["jerky", "slow"])
    assert m.sim_success is False  # never touched by tagging
    assert cli.main(["tag", str(sess), "-e", "3", "--note", "fresh", "--replace-notes", "--unflag", "jerky"]) == 0
    m = metas(sess)[3]
    assert (m.notes, m.flags) == ("fresh", ["slow"])
    assert cli.main(["tag", str(sess), "-e", "3", "--clear-label", "--clear-notes"]) == 0
    m = metas(sess)[3]
    assert (m.label, m.notes) == ("unlabeled", "")
    assert all(x.label == "unlabeled" for i, x in metas(sess).items() if i != 3)


def test_unflag_strips_whitespace_like_flag(sess: Path) -> None:
    # QA R5 / B7: --flag stores the stripped name, so --unflag must strip too
    assert cli.main(["tag", str(sess), "-e", "1", "--flag", " padded "]) == 0
    assert metas(sess)[1].flags == ["padded"]
    assert cli.main(["tag", str(sess), "-e", "1", "--unflag", " padded "]) == 0
    assert metas(sess)[1].flags == []
    assert cli.main(["tag", str(sess), "-e", "1", "--flag", "x ", "--unflag", " x"]) == 1  # same flag both ways


def test_ranges_and_all(sess: Path) -> None:
    assert cli.main(["tag", str(sess), "-e", "0-1,4", "--flag", "x"]) == 0
    assert [i for i, m in metas(sess).items() if m.flags == ["x"]] == [0, 1, 4]
    assert cli.main(["tag", str(sess), "--all", "--label", "success"]) == 0
    assert {m.label for m in metas(sess).values()} == {"success"}


def test_from_sim_keeps_human_labels_unless_overwrite(sess: Path) -> None:
    cli.main(["tag", str(sess), "-e", "0", "--label", "fail"])  # ep 0 has sim_success=True
    assert cli.main(["tag", str(sess), "--from-sim"]) == 0
    got = {i: (m.sim_success, m.label) for i, m in metas(sess).items()}
    assert got[0] == (True, "fail")
    assert all(lab == ("success" if ok else "fail") for i, (ok, lab) in got.items() if i != 0)
    assert cli.main(["tag", str(sess), "--from-sim", "--overwrite-labels", "-e", "0"]) == 0
    assert metas(sess)[0].label == "success"


@pytest.mark.parametrize(
    "argv, message",
    [
        (["-e", "7", "--label", "fail"], "out of range"),
        (["-e", "-1", "--label", "fail"], "out of range"),
        (["-e", "x", "--label", "fail"], "bad episode index"),
        (["--label", "fail"], "choose episodes"),
        (["-e", "1"], "nothing to change"),
        (["-e", "1", "--all", "--label", "fail"], "not both"),
        (["-e", "1", "--replace-notes"], "--replace-notes needs --note"),
        (["-e", "1", "--flag", " "], "non-empty"),
        (["--from-sim", "--note", "x"], "cannot be combined"),
    ],
)
def test_bad_input_exits_nonzero(sess: Path, capsys, argv: list[str], message: str) -> None:
    before = (sess / "episodes.jsonl").read_bytes()
    assert cli.main(["tag", str(sess), *argv]) == 1
    assert message in capsys.readouterr().err
    assert (sess / "episodes.jsonl").read_bytes() == before


def test_bad_label_is_rejected_by_argparse(sess: Path, capsys) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["tag", str(sess), "-e", "1", "--label", "maybe"])
    assert e.value.code != 0 and "invalid choice" in capsys.readouterr().err


def test_missing_session(tmp_path: Path, capsys) -> None:
    assert cli.main(["list", str(tmp_path / "nope")]) == 1
    assert "not a session" in capsys.readouterr().err


def test_crash_mid_write_leaves_old_file(sess: Path, monkeypatch) -> None:
    path = sess / "episodes.jsonl"
    before = path.read_bytes()
    real, calls = atomic.dump_json_line, {"n": 0}

    def flaky(row):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("simulated crash mid-write")
        return real(row)

    monkeypatch.setattr(atomic, "dump_json_line", flaky)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["tag", str(sess), "--all", "--label", "fail"])
    assert calls["n"] == 3  # the temp file really was partially written
    assert path.read_bytes() == before
    assert [p.name for p in sess.iterdir() if p.name.endswith(".tmp")] == []


def test_list_table_and_json(sess: Path, capsys) -> None:
    cli.main(["tag", str(sess), "-e", "3", "--label", "fail", "--note", "overshot"])
    capsys.readouterr()
    assert cli.main(["list", str(sess)]) == 0
    out = capsys.readouterr().out
    header = out.splitlines()[0].split()
    assert header[:5] == ["index", "length", "sim_success", "final_error_m", "label"] and "notes" in header
    row3 = next(line for line in out.splitlines() if line.startswith("3 "))
    assert "fail" in row3 and "overshot" in row3 and "12" in row3
    assert "5 episodes" in out
    assert cli.main(["list", str(sess), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 5 and rows[3]["label"] == "fail" and rows[3]["notes"] == "overshot"
    assert cli.main(["list", str(sess), "--json", "--label", "fail"]) == 0
    assert [r["episode_index"] for r in json.loads(capsys.readouterr().out)] == [3]


BLOCKER = textwrap.dedent(
    """
    import sys
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"mujoco", "lerobot", "torch"}:
                raise ImportError(f"blocked {name}")
    sys.meta_path.insert(0, Block())
    from robot_report_card import cli
    sys.exit(cli.main(sys.argv[1:]))
    """
)


def _rrc_subprocess(*argv: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, "-c", BLOCKER, *argv], capture_output=True, text=True, env=env)


def test_tags_persist_across_processes_without_mujoco_or_lerobot(sess: Path) -> None:
    r = _rrc_subprocess("tag", str(sess), "--episode", "2", "--label", "fail", "--note", "overshot")
    assert r.returncode == 0, r.stderr
    r = _rrc_subprocess("list", str(sess), "--json")
    assert r.returncode == 0, r.stderr
    row = json.loads(r.stdout)[2]
    assert (row["label"], row["notes"]) == ("fail", "overshot")
    r = _rrc_subprocess("export", str(sess), "--out", str(sess.parent / "ds"), "--repo-id", "local/x")
    assert r.returncode != 0 and "Traceback" not in r.stderr
