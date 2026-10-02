"""P3-8 (QA-owned): the phase 3 headline, end to end through the installed `capek` (plan DoD 9, decision D4).

record the pinned mix (bench generators, one session via the --append path) -> capek export -> capek score ->
capek train-bc --keep all  vs  --keep ok-and-success (train seed 0) -> capek compare (200 episodes, eval seed 900000).
Margin measured 2026-09-25 on this recipe: p = 3.7e-9 at train seed 0, worst 6.1e-5 over train seeds 0-4.
The gated full version (CAPEK_E2E_FULL=1) reproduces research brief §4 B vs E (200 clean + 100 junk).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("pyarrow")
pytest.importorskip("lerobot")  # capek export

PINNED_MIX = "clean:60,noise025:10,random:10,hesitation:5,wrong:5"
FULL_MIX = "clean:200,noise025:30,random:30,hesitation:20,wrong:20"


def _capek() -> list[str]:
    exe = Path(sys.executable).parent / "capek"
    found = str(exe) if exe.is_file() else shutil.which("capek")
    if not found:
        pytest.skip("the capek console script is not installed")
    return [found]


def run(cwd: Path, *argv: str, cli: list[str] | None = None) -> str:
    cmd = [*(cli or _capek()), *argv]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, f"{' '.join(cmd)} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}"
    return r.stdout


def headline(tmp: Path, mix: str, seed: int) -> dict:
    bench = [sys.executable, "-m", "capek.bench"]
    run(tmp, "record", "--out", "session", "--groups", mix, "--seed", str(seed), cli=bench)
    run(tmp, "export", "session", "--out", "dataset", "--repo-id", "local/p3_headline")
    run(tmp, "score", "dataset", "--json-out", "score.json")
    run(tmp, "train-bc", "dataset", "--out", "ckpt_all", "--keep", "all", "--seed", "0")
    run(tmp, "train-bc", "dataset", "--out", "ckpt_filtered", "--keep", "ok-and-success", "--score-json",
        "score.json", "--seed", "0")  # fmt: skip
    text = run(tmp, "compare", "bc:ckpt_all", "bc:ckpt_filtered", "--episodes", "200", "--eval-seed", "900000",
               "--json-out", "compare.json")  # fmt: skip
    return {"text": text, "report": json.loads((tmp / "compare.json").read_text()), "tmp": tmp}


def _check_wording(text: str, report: dict) -> None:
    p = report["paired"]
    assert report["kind"] == "compare" and report["schema_version"] == 1
    assert p["significant"] and p["delta_b_minus_a"] > 0 and p["delta_ci"][0] > 0
    assert report["verdict"].startswith("B (bc:ckpt_filtered) is better than A (bc:ckpt_all)")
    assert report["verdict"] in text and "No detectable difference" not in text
    assert report["recipe_caveat"] and report["recipe_caveat"] in text  # two trained checkpoints
    assert "10–15 points" in report["recipe_caveat"] and "≥ 3 training seeds" in report["recipe_caveat"]
    assert report["success_definition"] in text and "final frame" in report["success_definition"]
    assert "any step" in report["success_definition"]
    assert report["mde_statement"] in text
    assert report["seeds"][:2] == [[900000, 0], [900000, 1]] and len(report["seeds"]) == 200
    for side in ("A", "B"):
        assert report[side]["policy"]["train"]["seed"] == 0


def test_headline_filtering_by_score_and_outcome_beats_unfiltered(tmp_path: Path) -> None:
    r = headline(tmp_path, PINNED_MIX, 30000)
    report, p = r["report"], r["report"]["paired"]
    assert p["mcnemar_p"] < 0.01, p  # DoD 9
    _check_wording(r["text"], report)
    # the filter did what it says: quality ok (capek score) AND sim success, recorded in the checkpoint
    meta = json.loads((tmp_path / "ckpt_filtered" / "capek_policy.json").read_text())
    score = json.loads((tmp_path / "score.json").read_text())
    tags = json.loads((tmp_path / "dataset" / "meta" / "capek_tags.json").read_text())["episodes"]
    expected = sorted(
        e["episode_index"]
        for e in score["episodes"]
        if e["quality"] == "ok" and tags[str(e["episode_index"])]["sim_success"]
    )
    assert meta["kept_episodes"] == expected and meta["filter"]["keep"] == "ok-and-success"
    groups = {int(k): v["policy_params"]["bench_group"] for k, v in tags.items()}
    kept_groups = {groups[i] for i in expected}
    assert "random" not in kept_groups and "noise025" not in kept_groups and "wrong" not in kept_groups


@pytest.mark.skipif(os.environ.get("CAPEK_E2E_FULL") != "1", reason="full headline: set CAPEK_E2E_FULL=1 (~90 s)")
def test_full_headline_reproduces_brief_b_vs_e(tmp_path: Path) -> None:
    """Research brief §4: B (unfiltered 300) 1/200 vs E (scorer ok AND success) 121/200 at train seed 0."""
    r = headline(tmp_path, FULL_MIX, 20000)
    a, b = r["report"]["A"]["summary"]["successes"], r["report"]["B"]["summary"]["successes"]
    assert a <= 5 and b >= 100 and r["report"]["paired"]["mcnemar_p"] < 1e-4, (a, b)
    _check_wording(r["text"], r["report"])
