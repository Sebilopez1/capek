"""P3-5: `rrc eval` / `rrc compare` (policy protocol, runner, paired statistics wording, JSON)."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_report_card import cli  # noqa: E402
from robot_report_card.eval.report import RECIPE_CAVEAT, compare_stats, mde_sentence, verdict  # noqa: E402
from robot_report_card.eval.runner import EpisodeOutcome, RolloutResult  # noqa: E402

DEFINITION = (
    "success = gripper tip within 2 cm of the target on the final frame (90 frames, 3 s); "
    "LeRobot's eval counts success at any step"
)
FORBIDDEN = re.compile(r"(?<!detectable )\bno difference\b|equivalent", re.I)


def _run(tmp_path: Path, *argv: str) -> dict:
    out = tmp_path / "r.json"
    assert cli.main([*argv, "--json-out", str(out), "--overwrite"]) == 0
    return json.loads(out.read_text())


def test_eval_scripted_all_succeed(tmp_path: Path, capsys) -> None:
    r = _run(tmp_path, "eval", "scripted", "--episodes", "50")
    text = capsys.readouterr().out
    assert r["kind"] == "eval" and r["schema_version"] == 1
    assert r["summary"]["successes"] == 50 and r["summary"]["n"] == 50
    assert r["success_definition"] == DEFINITION and DEFINITION in text
    header = [c.strip() for c in text.splitlines()[0].split("|")]
    assert header[:6] == ["policy", "n", "successes", "rate", "Wilson 95% CI", "median final error"]
    assert r["seeds"][:2] == [[900000, 0], [900000, 1]] and len(r["episodes"]) == 50
    ep = r["episodes"][0]
    assert set(ep) == {"episode", "success", "success_any_step", "final_error_m", "final_state_sha256"}
    assert len(ep["final_state_sha256"]) == 64
    assert r["policy"]["kind"] == "scripted" and r["machine"]["mujoco"]


def test_eval_random_rarely_succeeds(tmp_path: Path) -> None:
    r = _run(tmp_path, "eval", "random", "--episodes", "50", "--ci", "clopper-pearson")
    assert r["summary"]["successes"] <= 5 and r["summary"]["ci_method"] == "clopper-pearson"
    assert r["summary"]["successes_any_step"] >= r["summary"]["successes"]


def test_compare_scripted_random_is_significant(tmp_path: Path, capsys) -> None:
    r = _run(tmp_path, "compare", "scripted", "random", "--episodes", "40")
    text = capsys.readouterr().out
    p = r["paired"]
    k_random = r["B"]["summary"]["successes"]
    assert (p["both"], p["a_only"], p["b_only"], p["neither"]) == (k_random, 40 - k_random, 0, 0)
    assert p["significant"] and p["mcnemar_p"] < 1e-6 and p["delta_b_minus_a"] < -0.8
    assert r["verdict"].startswith("B (random) is worse than A (scripted)") and r["verdict"] in text
    assert r["recipe_caveat"] is None and RECIPE_CAVEAT not in text
    assert [e["episode"] for e in r["A"]["episodes"]] == [e["episode"] for e in r["B"]["episodes"]]


def test_compare_scripted_scripted_no_detectable_difference(tmp_path: Path, capsys) -> None:
    r = _run(tmp_path, "compare", "scripted", "scripted", "--episodes", "30")
    text = capsys.readouterr().out
    assert r["paired"]["mcnemar_p"] == 1.0 and not r["paired"]["significant"]
    assert r["verdict"].startswith("No detectable difference (Δ = +0.0 pts")
    assert r["mde_statement"].startswith("With n = 30 and this much disagreement between the policies")
    assert r["mde_statement"] in text
    assert not FORBIDDEN.search(text)


# ---- D2 MDE logic and wording on hand-built paired results ---------------------------------------------------------
def _result(success: list[bool]) -> RolloutResult:
    eps = [EpisodeOutcome(i, s, s, 0.01, "0" * 64) for i, s in enumerate(success)]
    return RolloutResult({"spec": "x"}, "so101_reach", 900000, 90, 30.0, eps)


def test_mde_uses_observed_disagreement_with_floor() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(200) < 0.55
    b = a.copy()
    lose = np.nonzero(a)[0][:24]  # B fails 24 of A's successes
    b[lose] = False
    b[np.nonzero(~a)[0][:20]] = True
    c = compare_stats(_result(list(a)), _result(list(b)))
    assert c.a_only == 24 and c.b_only == 20
    assert c.mde_inputs["lose_observed"] == pytest.approx(24 / a.sum()) and not c.mde_inputs["floored"]
    assert c.mde_inputs["p_A"] == pytest.approx(a.mean())
    assert c.mde == pytest.approx(0.125, abs=0.03)
    s = mde_sentence(c)
    assert s == (
        "With n = 200 and this much disagreement between the checkpoints, this test detects differences of about "
        f"{100 * c.mde:.1f} points or more with 80% power."
    )
    assert verdict(c, "A", "B").startswith("No detectable difference (Δ = -2.0 pts, 95% CI ")


def test_mde_floor_and_no_success_baseline() -> None:
    a = [True] * 50 + [False] * 150
    b = [True] * 50 + [True] * 30 + [False] * 120  # B never loses A's episodes -> floor 0.10
    c = compare_stats(_result(a), _result(b))
    assert c.mde_inputs["floored"] and c.mde_inputs["lose"] == 0.10 and c.mde_inputs["lose_observed"] == 0.0
    assert c.significant and verdict(c, "a", "b").startswith("B (b) is better than A (a) (Δ = +15.0 pts")
    none = compare_stats(_result([False] * 200), _result([False] * 190 + [True] * 10))
    assert none.mde_inputs == {**none.mde_inputs, "p_A": 0.5, "lose": 0.10, "a_had_no_successes": True}
    assert none.mde == pytest.approx(0.09, abs=1e-9)
    top = compare_stats(_result([True] * 20), _result([True] * 20))
    assert math.isnan(top.mde) and "no possible improvement" in mde_sentence(top)


# ---- CLI behaviour --------------------------------------------------------------------------------------------------
def test_existing_json_out_is_refused(tmp_path: Path, capsys) -> None:
    target = tmp_path / "r.json"
    target.write_text("keep")
    assert cli.main(["eval", "scripted", "--episodes", "2", "--json-out", str(target)]) == 1
    assert "--overwrite" in capsys.readouterr().err and target.read_text() == "keep"


def test_default_path_and_json_stdout(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["eval", "random", "--episodes", "2"]) == 0
    assert (tmp_path / "random.rrc_eval.json").is_file()
    assert cli.main(["eval", "random", "--episodes", "2"]) == 1  # second run needs --overwrite
    capsys.readouterr()
    assert cli.main(["compare", "scripted", "random", "--episodes", "2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "compare"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["random.rrc_eval.json"]


@pytest.mark.parametrize(
    "argv, message",
    [
        (["eval", "bc:"], "needs a directory"),
        (["eval", "bc:/definitely/not/here"], "is not a directory"),
        (["eval", "teleop"], "unknown policy 'teleop'"),
        (["eval", "scripted", "--episodes", "0"], "--episodes"),
        (["compare", "scripted", "random", "--threads", "0"], "--threads"),
    ],
)
def test_bad_arguments(tmp_path: Path, capsys, argv: list[str], message: str) -> None:
    assert cli.main([*argv, "--json-out", str(tmp_path / "x.json")]) == 1
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err


RRC = (
    "import sys; from robot_report_card import cli; rc = cli.main(sys.argv[1:]); "
    "print('TORCH', 'torch' in sys.modules, file=sys.stderr); sys.exit(rc)"
)


def test_same_results_across_processes_and_no_torch_for_scripted(tmp_path: Path) -> None:
    runs = []
    for k in range(2):
        out = tmp_path / f"r{k}.json"
        r = subprocess.run(
            [sys.executable, "-c", RRC, "compare", "scripted", "random", "--episodes", "15", "--json-out", str(out)],
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": str(k + 1)},
        )
        assert r.returncode == 0, r.stderr
        assert "TORCH False" in r.stderr
        runs.append(json.loads(out.read_text()))
    for side in ("A", "B"):
        assert runs[0][side]["episodes"] == runs[1][side]["episodes"]
