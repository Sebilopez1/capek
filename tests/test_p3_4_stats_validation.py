"""P3-4 (QA-owned): validate capek.eval.stats against independent references.

scipy is used only here (dev venv), never by capek at runtime. Bars from docs/dev/phase3-plan.md P3-4:
(a) Wilson / Clopper-Pearson == scipy binomtest(...).proportion_ci("wilson" / "exact") to 1e-9, n <= 200
(b) mcnemar_exact(b, c) == scipy binomtest(b, b + c, 0.5).pvalue
(c) Newcombe method 10 == QA's own implementation from the paper; exact enumeration of the paired multinomial on a
    (p_A, p_B, phi) grid at n = 50 / 200: mean coverage 0.94-0.96, worst >= 0.90
(d) mde_paired: simulated power at the reported MDE >= 0.78 (2,000 draws, 3 grid points)
(e) Wilson coverage on a 1e-4 grid over p in [0.05, 0.95], n = 20-200: mean 0.945-0.955, worst >= 0.91
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from capek.eval import stats

scipy_stats = pytest.importorskip("scipy.stats")

Z = 1.959963984540054
N_GRID = (1, 2, 3, 5, 10, 20, 37, 50, 100, 137, 200)


# ---- (a) intervals ------------------------------------------------------------------------------------------------
def test_wilson_matches_scipy_all_k() -> None:
    """Every k for n = 1-60 plus larger n up to 200 (all n <= 200 takes ~8 s of scipy; run gated grids by hand)."""
    worst = 0.0
    for n in [*range(1, 61), 75, 99, 100, 137, 150, 199, 200]:
        for k in range(n + 1):
            ref = scipy_stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
            lo, hi = stats.wilson(k, n)
            worst = max(worst, abs(lo - ref.low), abs(hi - ref.high))
    assert worst < 1e-9, worst


def test_clopper_pearson_matches_scipy_exact() -> None:
    worst = 0.0
    for n in N_GRID:
        for k in range(n + 1):
            ref = scipy_stats.binomtest(k, n).proportion_ci(0.95, method="exact")
            lo, hi = stats.clopper_pearson(k, n)
            worst = max(worst, abs(lo - ref.low), abs(hi - ref.high))
    assert worst < 1e-9, worst


def test_interval_edge_cases_are_finite_and_documented() -> None:
    assert stats.wilson(0, 0) == (0.0, 1.0) and stats.clopper_pearson(0, 0) == (0.0, 1.0)
    for f in (stats.wilson, stats.clopper_pearson):
        lo0, hi0 = f(0, 30)
        lon, hin = f(30, 30)
        # k = 0 / k = n touch 0 / 1 (Wilson's k = n upper bound is 1 - 1e-16 in floating point: see review N-item)
        assert lo0 == 0.0 and 0 < hi0 < 1 and hin == pytest.approx(1.0, abs=1e-12) and 0 < lon < 1
        lo1, hi1 = f(1, 1)
        assert math.isfinite(lo1) and hi1 == 1.0
    with pytest.raises(ValueError):
        stats.wilson(5, 3)


# ---- (b) McNemar -------------------------------------------------------------------------------------------------
def test_mcnemar_exact_matches_scipy_binomtest() -> None:
    pairs = [(b, c) for b in range(61) for c in range(61) if b + c] + [(150, 180), (0, 200), (97, 103), (1, 199)]
    worst = max(abs(stats.mcnemar_exact(b, c) - scipy_stats.binomtest(b, b + c, 0.5).pvalue) for b, c in pairs)
    assert worst < 1e-9, worst
    assert stats.mcnemar_exact(0, 0) == 1.0


# ---- (c) Newcombe method 10 --------------------------------------------------------------------------------------
def _wilson_vec(x: np.ndarray, n: float) -> tuple[np.ndarray, np.ndarray]:
    p = x / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return np.maximum(0.0, c - h), np.minimum(1.0, c + h)


def newcombe10(a, b, c, d):
    """QA's own implementation, written from Newcombe (1998) method 10: Wilson limits l1,u1 / l2,u2 for the two
    marginal proportions, combined as theta - sqrt(dl1^2 - 2 phi dl1 du2 + du2^2) and
    theta + sqrt(du1^2 - 2 phi du1 dl2 + dl2^2), phi = (AD - BC) / sqrt((A+B)(C+D)(A+C)(B+D)) (0 if undefined)."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson_vec(a + b, n)
    l2, u2 = _wilson_vec(a + c, n)
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = np.divide(a * d - b * c, den, out=np.zeros_like(den), where=den > 0)
    lo = p1 - p2 - np.sqrt(np.maximum(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = p1 - p2 + np.sqrt(np.maximum(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return np.maximum(-1.0, lo), np.minimum(1.0, hi)


def _vectors(a: int, b: int, c: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([1] * a + [1] * b + [0] * c + [0] * d, bool)
    y = np.array([1] * a + [0] * b + [1] * c + [0] * d, bool)
    return x, y


def test_newcombe_matches_independent_implementation() -> None:
    rng = np.random.default_rng(0)
    tables = [
        tuple(int(v) for v in rng.multinomial(int(rng.integers(1, 300)), rng.dirichlet([1, 1, 1, 1])))
        for _ in range(2000)
    ]
    tables += [(0, 0, 0, 5), (5, 0, 0, 0), (0, 5, 0, 0), (0, 0, 5, 0), (3, 0, 0, 3), (0, 4, 4, 0), (1, 0, 0, 0)]
    worst = 0.0
    for a, b, c, d in tables:
        t, (lo, hi) = stats.newcombe_paired_diff(*_vectors(a, b, c, d))
        rlo, rhi = newcombe10(a, b, c, d)
        assert t == pytest.approx((b - c) / (a + b + c + d), abs=1e-12)
        worst = max(worst, abs(lo - float(rlo)), abs(hi - float(rhi)))
        assert -1.0 <= lo <= t <= hi <= 1.0
    assert worst < 1e-12, worst


@pytest.mark.parametrize("n", [50, 200])
def test_newcombe_exact_coverage(n: int) -> None:
    A, B, C = np.meshgrid(np.arange(n + 1), np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = A + B + C <= n
    A, B, C = A[keep], B[keep], C[keep]
    D = n - A - B - C
    lo, hi = newcombe10(A, B, C, D)  # == stats.newcombe_paired_diff (previous test), vectorized for enumeration
    lf = np.array([math.lgamma(k + 1) for k in range(n + 1)])
    logc = lf[n] - lf[A] - lf[B] - lf[C] - lf[D]
    covs = []
    for pa in (0.1, 0.3, 0.5, 0.7, 0.9):
        for pb in (0.1, 0.3, 0.5, 0.7, 0.9):
            for phi in (0.0, 0.3, 0.6):
                p11 = pa * pb + phi * math.sqrt(pa * (1 - pa) * pb * (1 - pb))
                p10, p01 = pa - p11, pb - p11
                p00 = 1 - p11 - p10 - p01
                if min(p11, p10, p01, p00) <= 0:
                    continue
                lp = logc + A * math.log(p11) + B * math.log(p10) + C * math.log(p01) + D * math.log(p00)
                inside = (lo <= pa - pb + 1e-12) & (pa - pb - 1e-12 <= hi)
                covs.append(float(np.exp(lp)[inside].sum()))
    covs = np.array(covs)
    assert len(covs) >= 40
    assert 0.94 <= covs.mean() <= 0.96, covs.mean()
    assert covs.min() >= 0.90, covs.min()


# ---- (d) MDE -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n, p_a, lose", [(200, 0.5, 0.0), (200, 0.5, 0.1), (100, 0.5, 0.1)])
def test_simulated_power_at_reported_mde(n: int, p_a: float, lose: float) -> None:
    mde = stats.mde_paired(n, p_a, lose)
    assert math.isfinite(mde)
    p10 = lose * p_a
    p01 = mde + p10
    draws = np.random.default_rng(1).multinomial(n, [p10, p01, 1 - p10 - p01], size=2000)
    power = np.mean([stats.mcnemar_exact(int(s[0]), int(s[1])) < 0.05 for s in draws])
    assert power >= 0.78, (n, p_a, lose, mde, power)
    # and one step smaller really is underpowered (the MDE is the smallest, not just a sufficient, effect)
    assert stats.mcnemar_power(n, p10, p01 - 0.005) < 0.8


def test_mde_not_reachable_is_nan() -> None:
    assert math.isnan(stats.mde_paired(50, 0.8, 0.1))


# ---- (e) Wilson coverage ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [20, 50, 100, 200])
def test_wilson_exact_coverage_fine_grid(n: int) -> None:
    grid = np.arange(0.05, 0.95 + 1e-12, 1e-4)
    ks = np.arange(n + 1)
    iv = np.array([stats.wilson(int(k), n) for k in ks])
    pmf = scipy_stats.binom.pmf(ks[None, :], n, grid[:, None])
    inside = (iv[None, :, 0] <= grid[:, None]) & (grid[:, None] <= iv[None, :, 1])
    cov = (pmf * inside).sum(1)
    assert 0.945 <= cov.mean() <= 0.955, cov.mean()
    assert cov.min() >= 0.91, cov.min()
