"""QA phase 3 review: R1 (borderline McNemar / Newcombe disagreement) and N1-N3 wording fixes."""

from __future__ import annotations

import numpy as np
import pytest

from capek.eval import stats
from capek.eval.report import (
    BORDERLINE_SENTENCE,
    compare_json,
    compare_stats,
    compare_text,
    mde_sentence,
)
from capek.eval.runner import EpisodeOutcome, RolloutResult


def _result(success: np.ndarray, spec: str) -> RolloutResult:
    eps = [EpisodeOutcome(i, bool(s), bool(s), 0.01, "0" * 64) for i, s in enumerate(success)]
    return RolloutResult({"spec": spec}, "so101_reach", 900000, 90, 30.0, eps)


def _paired(both: int, a_only: int, b_only: int, n: int = 200) -> tuple[RolloutResult, RolloutResult]:
    neither = n - both - a_only - b_only
    a = np.array([1] * both + [1] * a_only + [0] * b_only + [0] * neither, bool)
    b = np.array([1] * both + [0] * a_only + [1] * b_only + [0] * neither, bool)
    return _result(a, "bc:a"), _result(b, "bc:b")


@pytest.mark.parametrize("a_only, b_only", [(0, 5), (1, 7), (3, 11), (5, 0)])
def test_borderline_when_exact_test_and_ci_disagree(a_only: int, b_only: int) -> None:
    """QA's tables (n = 200, a = 100): the Newcombe CI excludes 0 but exact McNemar p > .05."""
    ra, rb = _paired(100, a_only, b_only)
    c = compare_stats(ra, rb)
    assert not c.significant and c.ci_excludes_zero and c.borderline
    text = compare_text(ra, rb, c, "wilson", caveat=True)
    lines = text.splitlines()
    verdict_at = next(i for i, line in enumerate(lines) if line.startswith("No detectable difference"))
    assert lines[verdict_at + 1] == BORDERLINE_SENTENCE
    report = compare_json(ra, rb, c, "wilson", True, {})
    assert report["paired"]["borderline"] is True and report["paired"]["ci_excludes_zero"] is True
    assert report["borderline_statement"] == BORDERLINE_SENTENCE


@pytest.mark.parametrize("a_only, b_only", [(0, 0), (2, 3), (0, 20), (20, 0), (10, 12)])
def test_not_borderline_when_they_agree(a_only: int, b_only: int) -> None:
    ra, rb = _paired(100, a_only, b_only)
    c = compare_stats(ra, rb)
    assert c.significant == c.ci_excludes_zero and not c.borderline
    assert BORDERLINE_SENTENCE not in compare_text(ra, rb, c, "wilson", caveat=False)
    report = compare_json(ra, rb, c, "wilson", False, {})
    assert report["paired"]["borderline"] is False and report["borderline_statement"] is None


def test_mde_sentence_names_policies_or_checkpoints() -> None:
    c = compare_stats(*_paired(100, 2, 3))
    assert "between the checkpoints" in mde_sentence(c, checkpoints=True)
    assert "between the policies" in mde_sentence(c, checkpoints=False)
    ra, rb = _paired(100, 2, 3)
    assert "between the policies" in compare_text(ra, rb, c, "wilson", caveat=False)


def test_wilson_touches_exactly_one_at_k_equals_n() -> None:
    for n in (1, 20, 200):
        assert stats.wilson(n, n)[1] == 1.0 and stats.wilson(0, n)[0] == 0.0


def test_hard_rows_mark_reasons_below_the_threshold() -> None:
    pytest.importorskip("pyarrow")
    from lerobot_files import smooth_reach

    from capek.score.engine import ScoreConfig, score_dataset
    from capek.score.reader import Dataset, EpisodeData

    rng = np.random.default_rng(3)
    eps = [smooth_reach(rng, 90, float(rng.uniform(0.005, 0.02))) for _ in range(30)] + [smooth_reach(rng, 90, 0.2)]
    data = []
    for i, e in enumerate(eps):
        ts = np.arange(90) / 30.0
        if i == 30:
            ts = np.r_[ts[:40], ts[45:], ts[-1] + np.arange(1, 6) / 30.0]  # dropped frames -> HARD
        data.append(EpisodeData(i, e["observation.state"], e["action"], ts, np.arange(90), None))
    ds = Dataset(None, {}, "v3.0", 30.0, "observation.state", "action", None, None, data)  # type: ignore[arg-type]
    hard = score_dataset(ds, ScoreConfig(flag_z=1e6)).episodes[30]
    assert hard.quality == "HARD" and hard.reasons
    assert all(r.endswith("(below flag threshold)") for r in hard.reasons)
    flagged = score_dataset(ds).episodes[30]
    assert flagged.quality == "HARD" and not any(r.endswith("(below flag threshold)") for r in flagged.reasons)
