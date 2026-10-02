"""Research spike: success-rate CIs and paired regression tests for phase 3 (pure Python + numpy at use time;
scipy is used ONLY by the coverage study in __main__).

    wilson(k, n)                       Wilson score interval
    clopper_pearson(k, n)              exact interval (bisection on the binomial CDF, no scipy)
    mcnemar_exact(b, c)                two-sided exact McNemar p-value on discordant counts
    newcombe_paired_diff(x, y)         paired difference p_x - p_y with Newcombe (1998) method-10 95% CI
    mcnemar_power(n, p10, p01)         exact power of McNemar at alpha for a paired design (for MDE statements)
    mde_paired(n, p_base, rho_disc)    smallest improvement detectable with 80% power

Run `python phase3_stats.py` for the coverage study and the MDE table quoted in docs/dev/phase3-research-brief.md.
"""
from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return lower, upper


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: binomial test of min(b, c) under Binomial(b + c, 1/2)."""
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2 * _binom_cdf(min(b, c), m, 0.5))


def newcombe_paired_diff(x: np.ndarray, y: np.ndarray, z: float = Z95) -> tuple[float, tuple[float, float]]:
    """Difference of paired proportions p_x - p_y, Newcombe (1998) method 10 (Wilson-based, phi-corrected)."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    n = len(x)
    a, b = int(np.sum(x & y)), int(np.sum(x & ~y))
    c, d = int(np.sum(~x & y)), int(np.sum(~x & ~y))
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
    """Exact power: D ~ Bin(n, p10 + p01) discordant pairs, then B-only ~ Bin(D, p01 / (p10 + p01))."""
    pd = p10 + p01
    if pd <= 0:
        return 0.0
    q = p01 / pd
    power = 0.0
    for dd in range(1, n + 1):
        w = math.comb(n, dd) * pd**dd * (1 - pd) ** (n - dd)
        if w < 1e-12:
            continue
        rej = sum(math.comb(dd, j) * q**j * (1 - q) ** (dd - j) for j in range(dd + 1) if mcnemar_exact(dd - j, j) < alpha)
        power += w * rej
    return power


def mde_paired(n: int, p_base: float, disc_frac: float = 0.0, target: float = 0.8) -> float:
    """Smallest improvement delta (B over A) with >= target power. disc_frac = fraction of A's successes that
    B loses (0 = B dominates A: only B-only discordant pairs), a knob for how 'noisy' the two checkpoints are."""
    for delta in np.arange(0.01, 1 - p_base, 0.005):
        p10 = disc_frac * p_base            # A success, B fail
        p01 = delta + p10                   # A fail, B success
        if p01 > 1 - p_base:
            break
        if mcnemar_power(n, p10, p01) >= target:
            return float(delta)
    return float("nan")


if __name__ == "__main__":
    from scipy.stats import beta, binom

    def jeffreys(k, n):
        return (0.0 if k == 0 else beta.ppf(0.025, k + .5, n - k + .5), 1.0 if k == n else beta.ppf(0.975, k + .5, n - k + .5))

    print("Exact coverage of nominal 95% intervals (mean / min over p in [0.05, 0.95], step 0.01):")
    grid = np.arange(0.05, 0.951, 0.01)
    for n in (20, 50, 100, 200):
        ks = np.arange(n + 1)
        ivs = {"wilson": [wilson(k, n) for k in ks], "clopper": [clopper_pearson(k, n) for k in ks],
               "jeffreys": [jeffreys(k, n) for k in ks]}
        line = []
        for name, iv in ivs.items():
            cov = [sum(binom.pmf(k, n, p) for k in ks if iv[k][0] <= p <= iv[k][1]) for p in grid]
            width = np.mean([iv[k][1] - iv[k][0] for k in ks])
            line.append(f"{name} {np.mean(cov):.3f}/{np.min(cov):.3f} w={width:.3f}")
        print(f"  n={n:3d}: " + " | ".join(line))
    print("Wilson at 50%: n=50 ->", [round(v, 3) for v in wilson(25, 50)], " n=100 ->", [round(v, 3) for v in wilson(50, 100)],
          " n=200 ->", [round(v, 3) for v in wilson(100, 200)])
    print("Paired MDE (80% power, alpha .05, McNemar exact); disc_frac = share of A's successes that B loses:")
    for n in (50, 100, 200, 400):
        row = []
        for pb in (0.5, 0.8):
            for df in (0.0, 0.1):
                row.append(f"p_A={pb} lose={df}: {mde_paired(n, pb, df):.3f}")
        print(f"  n={n:3d}: " + " | ".join(row))
