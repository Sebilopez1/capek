"""P3-3: eval/stats.py (port of docs/spikes/phase3_stats.py; QA validates against scipy in P3-4)."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from capek.eval import stats

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spike():
    spec = importlib.util.spec_from_file_location("spike_stats", REPO / "docs" / "spikes" / "phase3_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_identical_to_the_spike(spike) -> None:
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 20, 50, 200):
        for k in range(1, n):
            assert stats.wilson(k, n) == spike.wilson(k, n)
        # k = 0 / k = n: clamped to exactly 0.0 / 1.0 (the spike can give 5e-17 / 1 - 1e-16), QA N2
        assert stats.wilson(0, n) == (0.0, spike.wilson(0, n)[1]) and spike.wilson(0, n)[0] < 1e-12
        assert stats.wilson(n, n) == (spike.wilson(n, n)[0], 1.0) and spike.wilson(n, n)[1] > 1 - 1e-12
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            assert stats.clopper_pearson(k, n) == spike.clopper_pearson(k, n)
    for b in range(40):
        for c in range(0, 40, 3):
            assert stats.mcnemar_exact(b, c) == spike.mcnemar_exact(b, c)
    for _ in range(200):
        n = int(rng.integers(1, 250))
        x, y = rng.random(n) < rng.random(), rng.random(n) < rng.random()
        d, (lo, hi) = stats.newcombe_paired_diff(x, y)
        sd, (slo, shi) = spike.newcombe_paired_diff(x, y)
        assert d == sd and lo == pytest.approx(slo, abs=1e-12) and hi == pytest.approx(shi, abs=1e-12)
    assert stats.mcnemar_power(100, 0.05, 0.15) == spike.mcnemar_power(100, 0.05, 0.15)


def test_known_values() -> None:
    lo, hi = stats.wilson(25, 50)
    assert (round(lo, 3), round(hi, 3)) == (0.366, 0.634)
    assert stats.wilson(100, 200)[1] - 0.5 == pytest.approx(0.0686, abs=5e-4)  # +-6.9 pts at n = 200
    assert stats.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert stats.mcnemar_exact(3, 3) == 1.0
    lo, hi = stats.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-12)


@pytest.mark.parametrize(
    "args, expected",
    [((200, 0.5, 0.0), 0.04), ((200, 0.5, 0.1), 0.09), ((200, 0.555, 0.22), 0.125), ((100, 0.5, 0.0), 0.08)],
)
def test_mde_table_from_the_brief(args, expected: float) -> None:
    assert stats.mde_paired(*args) == pytest.approx(expected, abs=1e-9)


def test_mde_not_reachable_is_nan() -> None:
    assert math.isnan(stats.mde_paired(50, 0.8, 0.1))


def test_edge_cases_are_finite_and_documented() -> None:
    assert stats.wilson(0, 0) == (0.0, 1.0) and stats.clopper_pearson(0, 0) == (0.0, 1.0)
    lo, hi = stats.wilson(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = stats.wilson(20, 20)
    assert hi == 1.0 and 0.8 < lo < 1
    assert stats.clopper_pearson(20, 20)[1] == 1.0
    for k in (0, 1):  # n = 1
        for iv in (stats.wilson(k, 1), stats.clopper_pearson(k, 1)):
            assert 0.0 <= iv[0] < iv[1] <= 1.0
    assert stats.mcnemar_exact(0, 0) == 1.0  # b + c = 0
    same = np.array([True, False, True])
    d, (lo, hi) = stats.newcombe_paired_diff(same, same)
    assert d == 0.0 and -1 <= lo <= 0 <= hi <= 1
    assert stats.newcombe_paired_diff([], []) == (0.0, (-1.0, 1.0))
    d, (lo, hi) = stats.newcombe_paired_diff(np.ones(10, bool), np.zeros(10, bool))
    assert d == 1.0 and hi == 1.0 and 0 < lo < 1
    assert stats.newcombe_paired_diff([True], [False])[0] == 1.0  # n = 1


def test_large_n_does_not_overflow() -> None:
    lo, hi = stats.clopper_pearson(3, 5000)
    assert 0 < lo < 3 / 5000 < hi < 0.002
    assert 0 < stats.mcnemar_exact(700, 800) < 0.05


def test_bad_input_raises() -> None:
    for k, n in ((5, 4), (-1, 3), (0, -1)):
        with pytest.raises(ValueError):
            stats.wilson(k, n)
    with pytest.raises(ValueError):
        stats.mcnemar_exact(-1, 2)
    with pytest.raises(ValueError, match="same length"):
        stats.newcombe_paired_diff([True, False], [True])


def test_no_scipy_at_runtime() -> None:
    code = "import sys, capek.eval.stats as s; s.mde_paired(50, .5); print('scipy' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "False"
