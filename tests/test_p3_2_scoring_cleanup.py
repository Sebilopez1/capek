"""P3-2: R-real-2 (length_z with a floored MAD) and N5 ("below flag threshold" on ok rows)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from lerobot_files import smooth_reach  # noqa: E402

from robot_report_card.score.engine import ScoreConfig, score_dataset  # noqa: E402
from robot_report_card.score.reader import Dataset, EpisodeData  # noqa: E402


def _ds(eps: list[dict[str, np.ndarray]]) -> Dataset:
    data = [
        EpisodeData(
            i,
            e["observation.state"],
            e["action"],
            np.arange(len(e["action"])) / 30.0,
            np.arange(len(e["action"])),
            None,
        )
        for i, e in enumerate(eps)
    ]
    return Dataset(None, {}, "v3.0", 30.0, "observation.state", "action", None, None, data)  # type: ignore[arg-type]


def test_common_time_cap_does_not_explode_length_z() -> None:
    """R-real-2: 90% of episodes stop at the same time cap, so the MAD of log length is 0."""
    rng = np.random.default_rng(0)
    eps = [smooth_reach(rng, 90, 0.01) for _ in range(45)]
    eps += [smooth_reach(rng, int(n), 0.01) for n in (30, 45, 60, 75, 85)]
    res = score_dataset(_ds(eps))
    zs = np.array([e.length_z for e in res.episodes])
    assert np.all(np.abs(zs) < 20), zs
    assert np.all(zs[:45] == 0.0)
    assert zs[45] == pytest.approx(np.log(30 / 90) / 0.1)  # floored MAD: 0.1 in log length
    assert zs[45] < zs[46] < zs[47] < zs[48] < zs[49] < 0


def test_length_is_still_not_part_of_the_combined_score() -> None:
    rng = np.random.default_rng(1)
    eps = [smooth_reach(rng, 90, 0.01) for _ in range(30)] + [smooth_reach(rng, 30, 0.01)]
    e = score_dataset(_ds(eps)).episodes[-1]
    assert e.length_z < -3.5 and e.combined == max(v for v in e.z.values() if v is not None)


def test_ok_rows_mark_reasons_below_the_threshold() -> None:
    rng = np.random.default_rng(2)
    eps = [smooth_reach(rng, 90, float(rng.uniform(0.005, 0.02))) for _ in range(30)]
    eps += [smooth_reach(rng, 90, 0.2) for _ in range(3)]
    flagged = score_dataset(_ds(eps)).episodes
    assert all(e.quality == "FLAG" for e in flagged[-3:])
    assert all(not r.endswith("(below flag threshold)") for e in flagged if e.quality == "FLAG" for r in e.reasons)
    lenient = score_dataset(_ds(eps), ScoreConfig(flag_z=1e6)).episodes  # same z, nothing crosses the threshold
    noisy = lenient[-1]
    assert noisy.quality == "ok" and noisy.reasons
    assert all(r.endswith("(below flag threshold)") for r in noisy.reasons)
    assert [r.removesuffix(" (below flag threshold)") for r in noisy.reasons] == flagged[-1].reasons
