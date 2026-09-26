"""QA phase 4 R1: `rrc report` linking respects the checkpoint's training filter."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from test_p4_5_report_rules import (
    COMPARES,
    MACHINE,
    Data,
    bc_policy,
    paired,
    rollouts,
    run_report,
    score_json,
)

from robot_report_card.eval.report import compare_json, compare_stats


def _write(tmp: Path, data: Data, a_keep: dict, b_keep: dict, compare: str = "significant") -> dict:
    root = tmp / "dataset"
    (tmp / "score.json").write_text(json.dumps(score_json(root, data)))
    policies = []
    for spec, keep in (("bc:ckpt/a", a_keep), ("bc:ckpt/b", b_keep)):
        p = bc_policy(spec, root)
        p["filter"] = {"keep": keep["keep"], "score_json": None}
        if "indices" in keep:
            p["kept_episode_indices"] = keep["indices"]
            p["kept_episodes"] = len(keep["indices"])
        policies.append(p)
    sa, sb = paired(*COMPARES[compare])
    ra, rb = rollouts(policies[0], sa), rollouts(policies[1], sb)
    (tmp / "compare.json").write_text(json.dumps(compare_json(ra, rb, compare_stats(ra, rb), "wilson", True, MACHINE)))
    return {"score": tmp / "score.json", "compare": tmp / "compare.json", "evals": [], "train_seeds": 1}


def _all_text(r: dict) -> str:
    return " ".join(re.sub(r"\s+", " ", r[k]) for k in ("terminal", "md", "html"))


def test_filtered_checkpoint_never_gets_trained_on(tmp_path: Path, capsys) -> None:
    ok = [i for i in range(100) if i < 88]  # the LAST 12 episodes failed; B kept only the good ones
    r = run_report(tmp_path, _write(tmp_path, Data(failed=12), {"keep": "all"},
                                    {"keep": "ok-and-success", "indices": ok}), capsys)  # fmt: skip
    assert r["json"]["verdict"][0]["rule"] == "R2"
    text = _all_text(r)
    assert "trained on" not in text
    assert "A (bc:ckpt/a) learned from all 100 episodes." in r["json"]["verdict"][0]["text"]
    note = (
        "B (bc:ckpt/b) learned from a filtered subset (keep = ok-and-success, 88 of 100 episodes), of which 0% failed."
    )
    assert note in r["json"]["verdict"][0]["text"] and note in text
    link = r["json"]["linking"]["B"]
    assert link["linked"] and not link["full"] and link["subset"]["failed_frac"] == 0.0


def test_both_unfiltered_say_these_checkpoints(tmp_path: Path, capsys) -> None:
    r = run_report(tmp_path, _write(tmp_path, Data(failed=12), {"keep": "all"}, {"keep": "all"}), capsys)
    assert "in the data these checkpoints were trained on" in r["json"]["verdict"][0]["text"]


def test_old_filtered_checkpoint_without_indices(tmp_path: Path, capsys) -> None:
    r = run_report(tmp_path, _write(tmp_path, Data(failed=12), {"keep": "all"}, {"keep": "ok-and-success"}), capsys)
    text = r["json"]["verdict"][0]["text"]
    assert "B (bc:ckpt/b) learned from a filtered subset (keep = ok-and-success; which episodes isn't recorded" in text
    assert "trained on" not in _all_text(r)


@pytest.mark.parametrize("indices", [None, list(range(100))])
def test_filtered_checkpoints_never_satisfy_r5(tmp_path: Path, capsys, indices) -> None:
    keep = {"keep": "quality-ok"} if indices is None else {"keep": "quality-ok", "indices": indices}
    r = run_report(tmp_path, _write(tmp_path, Data(), {"keep": "all"}, keep), capsys)
    first = r["json"]["verdict"][0]
    assert first["rule"] == "INCONCLUSIVE" and "learned from a filtered subset" in first["text"]
    assert "differ in this eval" not in _all_text(r)
