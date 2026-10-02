"""P4-5 (QA-owned): `capek report` rules test, written from docs/dev/phase4-plan.md "Report-card rules (D2)".

Inputs are schema-exact: they come from the real phase 2 / phase 3 JSON builders (`score.report.build_json`,
`eval.report.eval_json` / `compare_json` / `compare_stats`) fed with synthetic results, so every row is one rule's
condition and nothing else. Every rule row, linked vs unlinked, borderline R4, R5's strict conditions, R6's add-on,
the always-present "Can't tell" lines and the forbidden phrases are checked in all three renderings (terminal, --md,
--html) plus report.json. Only `run_report` / `rule_ids` know the CLI and report.json layout (plan leaves them open).
"""

from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from capek import cli
from capek.eval.report import compare_json, compare_stats, eval_json
from capek.eval.runner import EpisodeOutcome, RolloutResult
from capek.score.engine import SIGNALS, EpisodeScore, ScoreConfig, ScoreResult
from capek.score.reader import Dataset, EpisodeData
from capek.score.report import build_json

N_EPISODES, T = 100, 90
MACHINE = {"platform": "test", "machine": "x86_64", "python": "3.11", "numpy": np.__version__, "mujoco": None,
           "torch": None, "torch_threads": None, "requested_threads": None, "cpu_count": 2}  # fmt: skip


# ---- synthetic, schema-exact inputs -----------------------------------------------------------------------------
@dataclass
class Data:
    """How the scored dataset looks. Counts are out of N_EPISODES."""

    hard: int = 0
    hard_type: str = "too short"  # or "non-finite"
    flagged: int = 0
    flag_reason: str = "action_hf_energy"  # or "idle_frac" (hesitation)
    failed: int = 0
    outcome: bool = True  # False -> no outcome evidence at all


def score_json(root: Path, d: Data) -> dict[str, Any]:
    eps, data_eps = [], []
    for i in range(N_EPISODES):
        z = {k: 0.1 for k in SIGNALS}
        quality, hard, reasons, combined = "ok", [], [], 0.5
        if i < d.hard:
            quality = "HARD"
            if d.hard_type == "non-finite":
                hard, combined = ["non-finite: NaN/inf values in state, action or timestamps"], None
                z = {k: None for k in SIGNALS}
            else:
                hard = ["too short: 10 frames (0.33 s); need at least 17 (0.57 s) to score motion"]
        elif i < d.hard + d.flagged:
            quality, combined = "FLAG", 8.0
            z[d.flag_reason] = 8.0
            reasons = [
                "looks jittery: action chatter 40.0x dataset median (z=8.0)"
                if d.flag_reason == "action_hf_energy"
                else "hesitates: idle 30% of frames before its last motion (median 0%) (z=8.0)"
            ]
        failed = N_EPISODES - d.failed <= i  # the LAST `failed` episodes failed (independent of flags)
        eps.append(
            EpisodeScore(
                episode_index=i,
                length=T,
                raw={k: 0.0 for k in SIGNALS},
                z=z,
                combined=combined,
                quality=quality,
                hard_flags=hard,
                reasons=reasons,
                outcome_sim=(not failed) if d.outcome else None,
                outcome_label=None,
                outcome_disagree=False,
                length_z=0.0,
            )  # fmt: skip
        )
        zero = np.zeros((T, 6))
        data_eps.append(EpisodeData(i, zero, zero, np.arange(T) / 30, np.arange(T), None))
    result = ScoreResult(eps, ScoreConfig(), list(SIGNALS), {}, [], {k: 1.0 for k in SIGNALS})
    ds = Dataset(root=root, info={"robot_type": "so101_sim"}, codebase_version="v3.0", fps=30.0,
                 state_key="observation.state", action_key="action", state_names=None, action_names=None,
                 episodes=data_eps)  # fmt: skip
    return build_json(result, ds)


def bc_policy(
    spec: str, trained_on: Path | None, frames: int = N_EPISODES * T, keep: str = "all", kept: list[int] | None = None
) -> dict[str, Any]:
    if trained_on is None:  # a LeRobot policy: never linked
        return {"spec": spec, "kind": "lerobot", "policy_type": "act", "inputs": ["observation.state"], "path": "/x"}
    return {
        "spec": spec, "kind": "bc", "weights_sha256": "0" * 64,
        "train": {"seed": 0, "epochs": 40, "batch_size": 256, "lr": 1e-3, "optimizer": "adam", "threads": 1,
                  "frames": frames, "seconds": 1.0, "torch": "2.10"},
        "dataset": {"path": str(trained_on.resolve()), "codebase_version": "v3.0", "fps": 30,
                    "total_episodes": N_EPISODES, "total_frames": frames},
        "filter": {"keep": keep, "score_json": None},
        "kept_episodes": N_EPISODES if kept is None else len(kept),
        **({} if kept is None else {"kept_episode_indices": kept}),  # None: an old JSON without indices
    }  # fmt: skip


def rollouts(policy: dict[str, Any], success: np.ndarray) -> RolloutResult:
    eps = [EpisodeOutcome(i, bool(s), bool(s), 0.01 if s else 0.1, f"{i:064x}") for i, s in enumerate(success)]
    return RolloutResult(policy, "so101_reach", 900000, T, 30.0, eps)


def paired(both: int, a_only: int, b_only: int, neither: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1] * both + [1] * a_only + [0] * b_only + [0] * neither, bool)
    b = np.array([1] * both + [0] * a_only + [1] * b_only + [0] * neither, bool)
    return a, b


COMPARES = {
    "significant": (60, 2, 40, 98),  # p ~ 1e-9, CI far from 0
    "not_significant": (80, 10, 12, 98),  # p ~ 0.8
    "borderline": (40, 0, 5, 155),  # exact McNemar p = .0625 but Newcombe CI (+0.2, +5.0): disagree
}


@dataclass
class Case:
    name: str
    data: Data = field(default_factory=Data)
    compare: str | None = "significant"
    a_linked: bool = True
    b_linked: bool = True
    b_lerobot: bool = False
    train_seeds: int = 1
    with_score: bool = True
    b_keep: str = "all"  # a filtered B checkpoint (QA phase 4 R1)
    b_kept_recorded: bool = True  # False: an old checkpoint JSON without kept_episode_indices


def write_inputs(tmp: Path, case: Case) -> dict[str, Any]:
    root = tmp / "dataset"
    other = tmp / "some_other_dataset"
    files: dict[str, Any] = {"train_seeds": case.train_seeds, "evals": []}
    if case.with_score:
        (tmp / "score.json").write_text(json.dumps(score_json(root, case.data)))
        files["score"] = tmp / "score.json"
    if case.compare is not None:
        pa = bc_policy("bc:ckpt/a", root if case.a_linked else other)
        # a filtered B keeps the episodes that succeeded (the last `failed` episodes failed)
        b_kept = list(range(N_EPISODES - case.data.failed)) if case.b_keep != "all" else None
        pb = bc_policy(
            "lerobot:act" if case.b_lerobot else "bc:ckpt/b",
            None if case.b_lerobot else (root if case.b_linked else other),
            keep=case.b_keep,
            kept=b_kept if case.b_kept_recorded else None,
        )
        sa, sb = paired(*COMPARES[case.compare])
        ra, rb = rollouts(pa, sa), rollouts(pb, sb)
        c = compare_stats(ra, rb)
        caveat = not case.b_lerobot
        (tmp / "compare.json").write_text(json.dumps(compare_json(ra, rb, c, "wilson", caveat, MACHINE)))
        (tmp / "eval_b.json").write_text(json.dumps(eval_json(rb, "wilson", MACHINE)))
        files["compare"] = tmp / "compare.json"
        files["evals"] = [tmp / "eval_b.json"]
    return files


# ---- the only two functions that know the P4-4 interface --------------------------------------------------------
def run_report(tmp: Path, files: dict[str, Any], capsys) -> dict[str, Any]:
    argv = ["report"]
    if "score" in files:
        argv += ["--score", str(files["score"])]
    for e in files["evals"]:
        argv += ["--eval", str(e)]
    if "compare" in files:
        argv += ["--compare", str(files["compare"])]
    argv += ["--train-seeds", str(files["train_seeds"]), "--md", str(tmp / "card.md"), "--html", str(tmp / "card.html"),
             "--json-out", str(tmp / "report.json")]  # fmt: skip
    capsys.readouterr()
    code = cli.main(argv)
    out = capsys.readouterr()
    assert code == 0, out.err
    html_text = (tmp / "card.html").read_text()
    return {
        "terminal": out.out,
        "md": (tmp / "card.md").read_text(),
        "html": html_lib.unescape(re.sub(r"<[^>]+>", " ", html_text)),
        "html_raw": html_text,
        "json": json.loads((tmp / "report.json").read_text()),
    }


def rule_ids(report: dict[str, Any]) -> list[str]:
    """Rule ids of the verdict lines, in order (first = the winning rule; R6 may follow as an add-on)."""
    return [line["rule"] for line in report["verdict"]]


# ---- expectations ------------------------------------------------------------------------------------------------
RENDERINGS = ("terminal", "md", "html")
CANT_TELL = [  # one fixed line each (plan D2); matched case-insensitively on these key phrases
    r"uniform junk",
    r"wobble",
    r"smooth failed attempts",
    r"10[–-]15",
    r"final[- ]frame",
    r"any[- ]step",
    r"simulation",
    r"across datasets",
    r"same[- ]machine",
    r"sim-tuned",
]
FORBIDDEN = [
    "caused by", "because of", "data looks clean", "data is good", "clean data", "proves", "real robot will",
    "reproducible across", "better recipe", "the recipe is better", "recipe works",
]  # fmt: skip
LEROBOT_ADJECTIVES = ["good policy", "bad policy", "works well", "performs well", "high-quality", "low-quality",
                      "strong policy", "weak policy"]  # fmt: skip


def text_of(r: dict[str, Any], rendering: str) -> str:
    return re.sub(r"\s+", " ", r[rendering])


def assert_everywhere(r: dict[str, Any], pattern: str) -> None:
    for rendering in RENDERINGS:
        assert re.search(pattern, text_of(r, rendering), re.I), (rendering, pattern)


def assert_nowhere(r: dict[str, Any], pattern: str) -> None:
    for rendering in RENDERINGS:
        assert not re.search(pattern, text_of(r, rendering), re.I), (rendering, pattern)


CASES = [
    # (case, expected winning rule, patterns that must appear in every rendering, patterns that must not)
    (
        Case("R0_hard_5pct", Data(hard=5)),
        "R0",
        [r"Fix the data first", r"5 episodes have hard flags", r"too short"],
        [r"corrupt"],
    ),
    (
        Case("R0_any_nonfinite", Data(hard=1, hard_type="non-finite")),
        "R0",
        [r"Fix the data first", r"non-finite"],
        [r"corrupt"],
    ),
    (Case("R1_majority_flagged", Data(flagged=55)), "R1", [r"Most episodes are flagged", r"relative"], []),
    (Case("R0_beats_R1", Data(hard=6, flagged=60)), "R0", [r"Fix the data first"], [r"Most episodes are flagged"]),
    (
        Case("R2_failed_linked", Data(failed=12)),
        "R2",
        [r"12(\.0)?% of demos .*failed", r"Consistent with a data problem", r"trained on", r"24 points"],
        [],
    ),
    (
        Case("R2_failed_unlinked", Data(failed=12), a_linked=False, b_linked=False),
        "R2",
        [r"12(\.0)?% of demos .*failed", r"Consistent with a data problem"],
        [r"trained on", r"training demos"],
    ),
    (
        Case("R3_motion_flagged", Data(flagged=12)),
        "R3",
        [r"12(\.0)?% of demos flagged", r"Consistent with a data problem", r"hesitation didn"],
        [],
    ),
    (Case("R2_beats_R3", Data(flagged=20, failed=20)), "R2", [r"failed"], []),
    (Case("R4_not_significant", compare="not_significant"), "R4", [r"Eval can.t tell", r"detects differences"], []),
    (Case("R4_borderline", compare="borderline"), "R4", [r"Eval can.t tell", r"borderline"], []),
    (
        Case("R5_strict_1_seed"),
        "R5",
        [r"differ in this eval", r"consistent with training variance or the recipe", r"3 training seeds"],
        [],
    ),
    (Case("R5_strict_3_seeds", train_seeds=3), "R5", [r"differ in this eval"], []),
    # R5's strict conditions: break one each -> R5 must NOT win
    (
        Case("R5_needs_A_linked", a_linked=False),
        "INCONCLUSIVE",
        [r"Inconclusive", r"unlinked"],
        [r"differ in this eval"],
    ),
    (Case("R5_needs_B_linked", b_linked=False), "INCONCLUSIVE", [r"Inconclusive"], [r"differ in this eval"]),
    (Case("R5_no_lerobot", b_lerobot=True), "INCONCLUSIVE", [r"Inconclusive"], [r"differ in this eval"]),
    (Case("R5_needs_flagged_lt_5", Data(flagged=6)), "INCONCLUSIVE", [r"Inconclusive"], [r"differ in this eval"]),
    (Case("R5_needs_failed_lt_5", Data(failed=6)), "INCONCLUSIVE", [r"Inconclusive"], [r"differ in this eval"]),
    (
        Case("R5_needs_outcome", Data(outcome=False)),
        "INCONCLUSIVE",
        [r"Inconclusive", r"Outcome unknown"],
        [r"differ in this eval"],
    ),
    (
        Case("R5_small_hard_ok", Data(hard=1)),
        "R5",
        [r"differ in this eval", r"1 episodes? ha(s|ve) hard flags"],
        [r"Fix the data first"],
    ),  # 1% too-short: reported in Data, doesn't trigger R0
    # QA phase 4 R1: a filtered checkpoint is linked but never "trained on" the scored demos, never satisfies R5
    (
        Case("R2_filtered_B_subset_recorded", Data(failed=12), b_keep="ok-and-success"),
        "R2",
        [r"12(\.0)?% of demos .*failed", r"filtered subset", r"ok-and-success", r"of which 0% failed"],
        [r"trained on", r"training demos"],
    ),
    (
        Case("R2_filtered_B_indices_missing", Data(failed=12), b_keep="ok-and-success", b_kept_recorded=False),
        "R2",
        [r"filtered subset"],
        [r"trained on", r"training demos"],
    ),
    (
        Case("R5_never_for_filtered_B", b_keep="quality-ok"),
        "INCONCLUSIVE",
        [r"Inconclusive", r"filter"],
        [r"differ in this eval", r"trained on"],
    ),
    (Case("inconclusive_score_only", compare=None), "INCONCLUSIVE", [r"Inconclusive", r"missing"], []),
    (Case("R3_plus_R6", Data(flagged=12, outcome=False), compare=None), "R3", [r"Outcome unknown"], []),
]


@pytest.mark.parametrize("case, rule, must, must_not", CASES, ids=[c[0].name for c in CASES])
def test_rules(tmp_path: Path, capsys, case: Case, rule: str, must: list[str], must_not: list[str]) -> None:
    r = run_report(tmp_path, write_inputs(tmp_path, case), capsys)
    ids = rule_ids(r["json"])
    assert ids and ids[0] == rule, ids
    if not case.data.outcome:
        assert "R6" in ids[1:] or ids[0] == "R6", ids  # the add-on line
    else:
        assert "R6" not in ids
    for p in must:
        assert_everywhere(r, p)
    for p in must_not:
        assert_nowhere(r, p)
    # never "training demos" unless every bc checkpoint is linked to the scored dataset
    if not (case.a_linked and case.b_linked) or case.b_lerobot or case.b_keep != "all":
        assert_nowhere(r, r"training demos")
    # always-present blocks and fixed Can't-tell lines, forbidden phrases, sim-tuned footer
    for block in ("Data", "Verdict", "Can.t tell"):
        assert_everywhere(r, block)
    for p in CANT_TELL:
        assert_everywhere(r, p)
    for phrase in FORBIDDEN:
        assert_nowhere(r, re.escape(phrase))
    if case.b_lerobot:
        for phrase in LEROBOT_ADJECTIVES:
            assert_nowhere(r, re.escape(phrase))
    if case.train_seeds < 3 and rule == "R5":
        assert_everywhere(r, r"(≥|>=|at least) ?3 training seeds")


def test_html_is_self_contained(tmp_path: Path, capsys) -> None:
    r = run_report(tmp_path, write_inputs(tmp_path, Case("html", Data(flagged=12))), capsys)
    raw = r["html_raw"].lower()
    assert "<script" not in raw and "http://" not in raw and "https://" not in raw.replace("https://github.com", "")
    assert "<style" in raw  # inline CSS


def test_report_json_is_the_source_of_truth(tmp_path: Path, capsys) -> None:
    """Every verdict line in report.json appears verbatim in all three renderings."""
    r = run_report(tmp_path, write_inputs(tmp_path, Case("sot", Data(failed=12))), capsys)
    for line in r["json"]["verdict"]:
        text = re.sub(r"\s+", " ", line["text"]).strip()
        for rendering in RENDERINGS:
            assert text in text_of(r, rendering), (rendering, text)
