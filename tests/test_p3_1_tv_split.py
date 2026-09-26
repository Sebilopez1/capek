"""P3-1 (code part): R-real-1 split denominator for action_tv_ratio."""

from __future__ import annotations

import numpy as np
import pytest

from robot_report_card.score import signals
from robot_report_card.score.engine import ScoreConfig, score_dataset
from robot_report_card.score.reader import Dataset, EpisodeData


def test_split_is_never_below_net_on_random_traces() -> None:
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = np.cumsum(rng.normal(0, rng.uniform(0.01, 1.0), (int(rng.integers(3, 120)), 6)), axis=0)
        assert signals.tv_denominator(a, "split") >= signals.tv_denominator(a, "net") - 1e-12


def test_split_equals_net_on_one_way_reaches() -> None:
    rng = np.random.default_rng(1)
    for _ in range(100):
        goal = rng.uniform(-2, 2, 6)
        ramp = np.clip(np.arange(90) / rng.uniform(20, 80), 0, 1)[:, None]  # monotone ramp, then hold
        a = rng.uniform(-1, 1, 6) + ramp * goal
        assert signals.tv_denominator(a, "split") == pytest.approx(signals.tv_denominator(a, "net"), rel=1e-12)


def test_out_and_back_counts_both_legs() -> None:
    t = np.arange(90)
    leg = np.clip(np.minimum(t / 30, (80 - t) / 30), 0, 1)[:, None] * np.ones(6)  # out, hold, back to start
    assert signals.tv_denominator(leg, "net") == pytest.approx(0.0)
    assert signals.tv_denominator(leg, "split") == pytest.approx(12.0)
    assert signals.action_tv_ratio(leg, 0.0, "split") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown tv mode"):
        signals.tv_denominator(leg, "max_exc")


def _episode(i: int, a: np.ndarray) -> EpisodeData:
    s = np.vstack([a[:1], a[:-1]])
    return EpisodeData(i, s, a, np.arange(len(a)) / 30.0, np.arange(len(a)), None)


def test_return_home_in_a_smooth_dataset_is_not_flagged_with_split_but_is_with_net() -> None:
    """R-real-1 (DoD 8): in a low-noise dataset of one-way reaches, clean out-and-back episodes looked like
    dithering under the net denominator. The split denominator treats them as ordinary motion."""
    rng = np.random.default_rng(2)
    t = np.arange(90)
    eps = []
    for i in range(40):
        ramp = np.clip(t / rng.uniform(30, 75), 0, 1)[:, None]
        eps.append(_episode(i, ramp * rng.uniform(-1, 1, 6) + rng.normal(0, 0.003, (90, 6))))
    for i in range(40, 45):  # bench-like return_home: out over 0.8-1.2 s, hold 0.1-0.4 s, back over the same time
        r, h = rng.uniform(24, 36), rng.uniform(3, 12)
        out_back = np.clip(np.minimum(t / r, (2 * r + h - t) / r), 0, 1)[:, None]
        eps.append(_episode(i, out_back * rng.uniform(-1, 1, 6) + rng.normal(0, 0.003, (90, 6))))
    ds = Dataset(None, {}, "v3.0", 30.0, "observation.state", "action", None, None, eps)  # type: ignore[arg-type]
    split = score_dataset(ds, ScoreConfig(tv_mode="split")).episodes[40:]
    net = score_dataset(ds, ScoreConfig(tv_mode="net")).episodes[40:]
    assert sum(e.quality != "ok" for e in split) == 0
    assert sum(e.z["action_tv_ratio"] > 3.5 for e in net) >= 2  # the phase 2 formula false-flags them
    assert max(e.z["action_tv_ratio"] for e in split) < min(e.z["action_tv_ratio"] for e in net)


def test_default_config_is_split() -> None:
    assert ScoreConfig().tv_mode == "split"
