"""P2-6: `capek score` CLI, JSON report (schema v1), and an end-to-end run through the installed `capek`."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from lerobot_files import smooth_reach, write_dataset  # noqa: E402

from capek import cli  # noqa: E402
from capek.score.engine import HONESTY_STATEMENT, MAJORITY_WARNING, SIGNALS  # noqa: E402

ROW = re.compile(r"^\d+\s+\|")
EPISODE_FIELDS = {
    "episode_index", "length", "length_z", "quality", "combined", "raw", "z", "hard_flags", "reasons",
    "outcome_sim", "outcome_label", "outcome_disagree", "outcome_display",
}  # fmt: skip
TOP_FIELDS = {
    "schema_version", "tool", "capek_version", "dataset", "keys", "thresholds", "config", "signals",
    "skipped_signals", "signal_medians", "notes", "summary", "episodes",
}  # fmt: skip


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    rng = np.random.default_rng(21)
    eps = [smooth_reach(rng, 90, noise=float(rng.uniform(0.005, 0.02))) for _ in range(30)]
    eps += [smooth_reach(rng, 90, noise=0.2) for _ in range(3)]
    tags = {"dataset": {"total_episodes": 33, "total_frames": 33 * 90}, "episodes": {"0": {"label": "fail"}}}
    return write_dataset(tmp_path_factory.mktemp("score") / "demo", eps, tags=tags)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_table_summary_and_json_report(dataset: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    before = _snapshot(dataset)
    assert cli.main(["score", str(dataset)]) == 0
    out = capsys.readouterr().out
    header = [c.strip() for c in out.splitlines()[0].split("|")]
    assert header == ["ep", "frames", "quality", "score", "outcome", "why"]
    assert "fail (label) ≠ sim" in out and "success (sim)" in out
    assert HONESTY_STATEMENT in out and "assume most episodes are good" in out
    assert re.search(r"^flagged: \d+/33 ", out, re.M) and MAJORITY_WARNING not in out
    report_path = tmp_path / "demo.capek_score.json"  # default: ./<dataset dir name>.capek_score.json
    assert f"JSON report: {report_path}" in out
    report = json.loads(report_path.read_text())
    assert set(report) == TOP_FIELDS and report["schema_version"] == 1 and report["tool"] == "capek"
    assert report["dataset"]["codebase_version"] == "v3.0" and report["dataset"]["total_frames"] == 2970
    assert report["keys"] == {"state": "observation.state", "action": "action"}
    assert report["signals"] == list(SIGNALS) and report["skipped_signals"] == {}
    assert report["summary"]["honesty_statement"] == HONESTY_STATEMENT
    ep0 = report["episodes"][0]
    assert set(ep0) == EPISODE_FIELDS and set(ep0["raw"]) == set(SIGNALS)
    assert (ep0["outcome_label"], ep0["outcome_sim"], ep0["outcome_disagree"]) == ("fail", True, True)
    assert [e["quality"] for e in report["episodes"][-3:]] == ["FLAG"] * 3
    assert _snapshot(dataset) == before  # the dataset is never written to


def test_json_out_refused_without_overwrite(dataset: Path, tmp_path: Path, capsys) -> None:
    target = tmp_path / "r.json"
    target.write_text("keep me")
    assert cli.main(["score", str(dataset), "--json-out", str(target)]) == 1
    assert "already exists; pass --overwrite" in capsys.readouterr().err
    assert target.read_text() == "keep me"
    assert cli.main(["score", str(dataset), "--json-out", str(target), "--overwrite"]) == 0
    assert json.loads(target.read_text())["schema_version"] == 1


def test_refuses_to_write_inside_the_dataset(dataset: Path, monkeypatch, capsys) -> None:
    before = _snapshot(dataset)
    assert cli.main(["score", str(dataset), "--json-out", str(dataset / "meta" / "x.json")]) == 1
    monkeypatch.chdir(dataset)
    assert cli.main(["score", "."]) == 1
    assert "inside the dataset" in capsys.readouterr().err
    assert _snapshot(dataset) == before


def test_json_stdout_only_flagged_and_threshold(dataset: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["score", str(dataset), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["episodes"]) == 33 and not list(tmp_path.iterdir())  # --json alone writes no file
    assert cli.main(["score", str(dataset), "--only-flagged", "--json-out", str(tmp_path / "a.json")]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if ROW.match(line)]
    flagged = sum(e["quality"] != "ok" for e in report["episodes"])
    assert len(rows) == flagged >= 3 and all("FLAG" in r for r in rows)
    assert cli.main(["score", str(dataset), "--threshold", "-100", "--json-out", str(tmp_path / "b.json")]) == 0
    out = capsys.readouterr().out
    assert "flagged: 33/33" in out and MAJORITY_WARNING in out
    assert json.loads((tmp_path / "b.json").read_text())["thresholds"]["flag_z"] == -100


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--state-key", "observation.joints"], "no 'observation.joints' column (pass --state-key"),
        (["--threshold", "nan"], "--threshold must be a finite number"),
    ],
)
def test_clean_errors(dataset: Path, tmp_path: Path, capsys, argv: list[str], message: str) -> None:
    assert cli.main(["score", str(dataset), "--json-out", str(tmp_path / "x.json"), *argv]) == 1
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err and not (tmp_path / "x.json").exists()


def test_unsupported_version_and_not_a_dataset(tmp_path: Path, capsys) -> None:
    rng = np.random.default_rng(0)
    old = write_dataset(
        tmp_path / "old", [smooth_reach(rng) for _ in range(3)], info_overrides={"codebase_version": "v1.6"}
    )
    assert cli.main(["score", str(old), "--json-out", str(tmp_path / "o.json")]) == 1
    assert "unsupported codebase_version 'v1.6'" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert cli.main(["score", str(tmp_path / "empty"), "--json-out", str(tmp_path / "o.json")]) == 1
    assert "no meta/info.json" in capsys.readouterr().err


def test_v21_copy_scores_identically(dataset: Path, tmp_path: Path) -> None:
    from lerobot_files import v3_to_v2

    v21 = v3_to_v2(dataset, tmp_path / "v21")
    assert cli.main(["score", str(dataset), "--json-out", str(tmp_path / "a.json")]) == 0
    assert cli.main(["score", str(v21), "--json-out", str(tmp_path / "b.json")]) == 0
    a, b = (json.loads((tmp_path / f).read_text()) for f in ("a.json", "b.json"))
    assert a["episodes"] == b["episodes"] and b["dataset"]["codebase_version"] == "v2.1"


CAPEK_CODE = (
    "import sys; from capek import cli; rc = cli.main(sys.argv[1:]); "
    "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('lerobot', 'torch', 'mujoco')); "
    "print('MODULES', bad, file=sys.stderr); sys.exit(rc)"
)


def test_score_cli_imports_no_lerobot_torch_or_mujoco(dataset: Path, tmp_path: Path) -> None:
    r = subprocess.run(
        [sys.executable, "-c", CAPEK_CODE, "score", str(dataset), "--json-out", str(tmp_path / "r.json")],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert r.returncode == 0, r.stderr
    assert "MODULES []" in r.stderr


def test_piped_into_head_exits_141(dataset: Path, tmp_path: Path) -> None:
    big = write_dataset(
        tmp_path / "big", [smooth_reach(np.random.default_rng(i), 30, noise=0.3 * (i % 2)) for i in range(2000)]
    )
    cmd = f"{sys.executable} -m capek.cli score {big} --json-out {tmp_path / 'r.json'} | head -1"
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    assert r.returncode == cli.EXIT_BROKEN_PIPE and r.stderr == "" and r.stdout.startswith("ep ")


# ---- end to end through the installed console script: record --append -> tag -> export -> score -------------
def _capek_bin() -> str:
    exe = Path(sys.executable).with_name("capek")
    return str(exe) if exe.exists() else (shutil.which("capek") or pytest.skip("installed `capek` not found"))


def _run(cwd: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    return subprocess.run([_capek_bin(), *argv], cwd=cwd, capture_output=True, text=True, env=env, timeout=300)


def test_e2e_record_append_export_score(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    for spec in (
        ["--policy", "scripted", "--noise", "0.02", "--episodes", "12", "--seed", "0"],
        ["--policy", "scripted", "--noise", "0.25", "--episodes", "3", "--seed", "1"],
        ["--policy", "random", "--episodes", "3", "--seed", "2"],
        ["--policy", "stall", "--noise", "0.02", "--episodes", "2", "--seed", "3"],
    ):
        r = _run(tmp_path, "record", "--append", "--out", "runs/mix", "-q", *spec)
        assert r.returncode == 0, r.stderr
    assert _run(tmp_path, "tag", "runs/mix", "-e", "0", "--label", "fail").returncode == 0
    r = _run(tmp_path, "export", "runs/mix", "--out", "datasets/mix")
    assert r.returncode == 0, r.stderr
    r = _run(tmp_path, "score", "datasets/mix")
    assert r.returncode == 0, r.stderr
    rows = {
        int(line.split("|")[0]): [c.strip() for c in line.split("|")]
        for line in r.stdout.splitlines()
        if ROW.match(line)
    }
    assert len(rows) == 20
    assert all(rows[i][2] == "FLAG" for i in range(12, 18))  # noise 0.25 + random
    assert all(rows[i][2] == "ok" for i in range(12))
    assert rows[0][4] == "fail (label) ≠ sim"
    assert all(rows[i][2] == "ok" and rows[i][4] == "fail (sim)" for i in (18, 19))  # stall: motion looks normal
    assert HONESTY_STATEMENT in r.stdout
    report = json.loads((tmp_path / "mix.capek_score.json").read_text())
    assert set(report) == TOP_FIELDS and len(report["episodes"]) == 20
    again = _run(tmp_path, "score", "datasets/mix")
    assert again.returncode == 1 and "--overwrite" in again.stderr
    # junk-majority subset: relative scores hide a junk majority, so force the >50% branch with a low threshold
    r = _run(tmp_path, "score", "datasets/mix", "--threshold", "0.3", "--json-out", "low.json")
    assert r.returncode == 0 and MAJORITY_WARNING in r.stdout
