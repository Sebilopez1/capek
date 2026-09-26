"""P4-4: `rrc report` CLI behaviour on real rrc JSON (the D2 rules themselves are QA's P4-5 test)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("mujoco")

from lerobot_files import smooth_reach, write_dataset  # noqa: E402

from robot_report_card import cli  # noqa: E402


@pytest.fixture(scope="module")
def reports(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("report")
    rng = np.random.default_rng(5)
    eps = [smooth_reach(rng, 90, float(rng.uniform(0.005, 0.02))) for _ in range(28)]
    eps += [smooth_reach(rng, 90, 0.2) for _ in range(4)]
    write_dataset(root / "demo", eps)
    assert cli.main(["score", str(root / "demo"), "--json-out", str(root / "score.json")]) == 0
    assert cli.main(["eval", "scripted", "--episodes", "12", "--json-out", str(root / "eval.json")]) == 0
    other = ["eval", "random", "--episodes", "12", "--eval-seed", "5", "--json-out", str(root / "eval_other.json")]
    assert cli.main(other) == 0
    assert cli.main(["compare", "random", "scripted", "--episodes", "12", "--json-out", str(root / "cmp.json")]) == 0
    return root


def _report(reports: Path, tmp_path: Path, *extra: str) -> tuple[int, dict | None]:
    out = tmp_path / "report.json"
    code = cli.main(["report", *extra, "--json-out", str(out)])
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_full_card_on_real_json(reports: Path, tmp_path: Path, capsys) -> None:
    args = ["--score", str(reports / "score.json"), "--eval", str(reports / "eval.json"),
            "--compare", str(reports / "cmp.json"),
            "--md", str(tmp_path / "c.md"), "--html", str(tmp_path / "c.html")]  # fmt: skip
    code, r = _report(reports, tmp_path, *args)
    out = capsys.readouterr().out
    assert code == 0 and r["kind"] == "report" and r["schema_version"] == 1
    assert "Inputs: score" in out and str(reports / "cmp.json") in out
    assert r["data"]["episodes"] == 32 and r["data"]["motion_flagged"] >= 4
    assert [p["side"] for p in r["policies"]] == [None, "A", "B"]
    assert r["linking"]["A"]["linked"] is False and r["linking"]["B"]["linked"] is False  # scripted/random
    assert r["verdict"][0]["rule"] in ("R3", "INCONCLUSIVE")
    for block in ("Data", "Policy", "Regression", "Verdict", "Can't tell"):
        assert f"\n{block}\n" in out
        assert f"## {block}" in (tmp_path / "c.md").read_text()
    page = (tmp_path / "c.html").read_text()
    assert page.startswith("<!DOCTYPE html>") and "<script" not in page and "http" not in page


def test_mismatched_inputs_warn(reports: Path, tmp_path: Path, capsys) -> None:
    code, r = _report(
        reports, tmp_path, "--eval", str(reports / "eval_other.json"), "--compare", str(reports / "cmp.json")
    )
    assert code == 0
    assert any("other seeds or n than the compare" in w for w in r["warnings"])
    assert "WARNING:" in capsys.readouterr().out
    assert r["verdict"][0]["rule"] == "INCONCLUSIVE" and "missing" in r["verdict"][0]["text"]


def test_eval_only_is_inconclusive_missing(reports: Path, tmp_path: Path) -> None:
    code, r = _report(reports, tmp_path, "--eval", str(reports / "eval.json"))
    assert code == 0 and r["verdict"] == [
        {
            "rule": "INCONCLUSIVE",
            "text": "Inconclusive: no score JSON (missing), so nothing is known about the data; "
            "no compare JSON (missing), so there is no A/B regression to explain.",
        }
    ]


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "give at least one of --score, --eval, --compare"),
        (["--compare", "{score}"], "is not an `rrc compare` JSON report"),
        (["--score", "{cmp}"], "is not an `rrc score` JSON report"),
        (["--score", "{missing}"], "can't read"),
        (["--score", "{bad}"], "is not valid JSON"),
        (["--score", "{score}", "--train-seeds", "0"], "--train-seeds must be >= 1"),
    ],
)
def test_clean_errors(reports: Path, tmp_path: Path, capsys, argv: list[str], message: str) -> None:
    (tmp_path / "bad.json").write_text("{nope")
    subs = {"score": reports / "score.json", "cmp": reports / "cmp.json", "missing": tmp_path / "nope.json",
            "bad": tmp_path / "bad.json"}  # fmt: skip
    argv = [a.format(**subs) for a in argv]
    code, r = _report(reports, tmp_path, *argv)
    err = capsys.readouterr().err
    assert code == 1 and message in err and "Traceback" not in err and r is None


def test_outputs_are_not_overwritten_without_flag(reports: Path, tmp_path: Path, capsys) -> None:
    md = tmp_path / "c.md"
    md.write_text("keep")
    assert cli.main(["report", "--score", str(reports / "score.json"), "--md", str(md), "--json"]) == 1
    assert "--overwrite" in capsys.readouterr().err and md.read_text() == "keep"
    assert cli.main(["report", "--score", str(reports / "score.json"), "--md", str(md), "--json", "--overwrite"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "report" and md.read_text().startswith("# Robot Report Card")
