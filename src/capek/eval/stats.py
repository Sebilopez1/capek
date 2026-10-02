"""Statistics for policy evaluation: success-rate CIs, paired tests and power. Pure Python + numpy, no scipy.

Ported from ``docs/spikes/phase3_stats.py`` (plan decision D1; QA-validated against scipy in P3-4).

- ``wilson(k, n)``: Wilson score interval (Wilson 1927), the default 95% CI for a success rate.
- ``clopper_pearson(k, n)``: exact (conservative) interval by bisection on the binomial CDF.
- ``mcnemar_exact(b, c)``: exact two-sided McNemar test = binomial test of the discordant pairs at 1/2.
- ``newcombe_paired_diff(x, y)``: paired difference p_x − p_y with Newcombe (1998) method 10 (Wilson-based,
  φ-corrected) 95% CI.
- ``mcnemar_power`` / ``mde_paired``: exact power of the McNemar test and the minimum detectable effect.

Edge cases (all finite): ``n = 0`` gives the uninformative interval (0, 1) (and Δ = 0 with CI (−1, 1)); ``k = 0``
and ``k = n`` give one-sided Wilson/Clopper-Pearson intervals touching 0 or 1; ``b + c = 0`` gives p = 1.0;
``n = 1`` works like any other n. ``mde_paired`` returns NaN when no effect within [0, 1 − p_base] reaches the
target power (documented "not reachable"), e.g. small n with a high baseline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

Z95 = 1.959963984540054
EXACT_MAX_N = 1000  # above this, binomial terms use log-gamma (math.comb * p**k would overflow a float)


def _pmf(i: int, n: int, p: float) -> float:
    if n <= EXACT_MAX_N:
        return math.comb(n, i) * p**i * (1 - p) ** (n - i)
    if p <= 0.0:
        return 1.0 if i == 0 else 0.0
    if p >= 1.0:
        return 1.0 if i == n else 0.0
    log = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)
    return math.exp(log)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (Wilson 1927). ``n = 0`` → (0, 1)."""
    _check_counts(k, n)
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    return (0.0 if k == 0 else lo), (1.0 if k == n else hi)  # touch exactly 0 / 1 at the edges (symmetric)


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(sum(_pmf(i, n, p) for i in range(k + 1)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson interval (bisection, 60 steps). ``n = 0`` → (0, 1)."""
    _check_counts(k, n)
    if n == 0:
        return 0.0, 1.0

    def solve(f: Callable[[float], bool], lo: float = 0.0, hi: float = 1.0) -> float:
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return lower, upper


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of min(b, c) under Binomial(b + c, 1/2). b + c = 0 → 1.0."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2 * _binom_cdf(min(b, c), m, 0.5))


def discordant(x: Sequence[bool] | np.ndarray, y: Sequence[bool] | np.ndarray) -> tuple[int, int, int, int]:
    """Paired 2x2 table for success vectors on the same seeds: (both, x only, y only, neither)."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("paired success vectors must be 1-D and the same length")
    return int(np.sum(x & y)), int(np.sum(x & ~y)), int(np.sum(~x & y)), int(np.sum(~x & ~y))


def newcombe_paired_diff(
    x: Sequence[bool] | np.ndarray, y: Sequence[bool] | np.ndarray, z: float = Z95
) -> tuple[float, tuple[float, float]]:
    """Paired difference p_x − p_y with the Newcombe (1998) method 10 95% CI (Wilson-based, φ-corrected).

    ``x`` and ``y`` are success vectors on the same seeds. Empty input → (0.0, (−1.0, 1.0)).
    """
    a, b, c, d = discordant(x, y)
    n = a + b + c + d
    if n == 0:
        return 0.0, (-1.0, 1.0)
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = 0.0 if den == 0 else (a * d - b * c) / den
    t = p1 - p2
    lo = t - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = t + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return t, (max(-1.0, lo), min(1.0, hi))


def mcnemar_power(n: int, p10: float, p01: float, alpha: float = 0.05) -> float:
    """Exact power of the two-sided McNemar test for a paired design with n pairs.

    D ~ Bin(n, p10 + p01) discordant pairs; given D, the "y only" count ~ Bin(D, p01 / (p10 + p01)).
    """
    pd = p10 + p01
    if pd <= 0:
        return 0.0
    q = p01 / pd
    power = 0.0
    for dd in range(1, n + 1):
        w = _pmf(dd, n, pd)
        if w < 1e-12:
            continue
        rej = sum(_pmf(j, dd, q) for j in range(dd + 1) if mcnemar_exact(dd - j, j) < alpha)
        power += w * rej
    return power


def mde_paired(n: int, p_base: float, disc_frac: float = 0.0, target: float = 0.8) -> float:
    """Smallest improvement Δ of B over A (as a proportion) detectable with ``target`` power at α = .05.

    ``p_base`` = A's success rate; ``disc_frac`` = share of A's successes that B loses (0 = B never loses an
    episode A won). Scans Δ in steps of 0.005 from 0.01. Returns NaN if no Δ ≤ 1 − p_base reaches the target.
    """
    for delta in np.arange(0.01, 1 - p_base, 0.005):
        p10 = disc_frac * p_base  # A success, B fail
        p01 = delta + p10  # A fail, B success
        if p01 > 1 - p_base:
            break
        if mcnemar_power(n, p10, p01) >= target:
            return float(delta)
    return float("nan")


def _check_counts(k: int, n: int) -> None:
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
