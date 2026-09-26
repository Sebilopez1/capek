"""Per-episode motion-quality signals (plan D5; research brief §1). Pure numpy, higher = worse for every signal.

Inputs are already normalized per joint (see ``engine.normalize``): ``s`` is state (T, D), ``a`` is action (T, A).
Formulas are ported from ``docs/spikes/phase2_score_proto.py`` and must stay numerically identical to it.
"""

from __future__ import annotations

import numpy as np

NUMERIC_EPS = 1e-12


def sparc(speed: np.ndarray, fps: float, fc: float = 10.0, amp_th: float = 0.05, pad: int = 4) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2012/2015), returned positive.

    Normalized FFT magnitude, adaptive cutoff (last frequency with amplitude >= ``amp_th``, at most ``fc`` Hz),
    zero padding to 2^(ceil(log2 N) + pad). Longer arc = less smooth. Computed on joint-space speed.
    """
    n = int(2 ** (np.ceil(np.log2(len(speed))) + pad))
    f = np.arange(n) * fps / n
    mag = np.abs(np.fft.fft(speed, n))
    mag = mag / (mag.max() + NUMERIC_EPS)
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    idx = np.nonzero(mag >= amp_th)[0]
    if len(idx) < 2:
        return 0.0
    f, mag = f[: idx[-1] + 1], mag[: idx[-1] + 1]
    df = np.diff(f) / (f[-1] - f[0] + NUMERIC_EPS)
    dm = np.diff(mag)
    return float(np.sum(np.sqrt(df**2 + dm**2)))


def ldlj(s: np.ndarray, fps: float) -> float:
    """Log dimensionless jerk (velocity form, Hogan & Sternad 2009), sign flipped so higher = jerkier."""
    dt = 1.0 / fps
    speed = joint_speed(s, fps)
    jerk = np.diff(s, n=3, axis=0) / dt**3
    vpk = speed.max() + 1e-9
    t_total = len(s) * dt
    return float(np.log(t_total**3 / vpk**2 * np.sum(jerk**2) * dt + NUMERIC_EPS))


def joint_speed(s: np.ndarray, fps: float) -> np.ndarray:
    """Joint-space speed ‖ṡ‖ per frame (divides by dt, like the prototype, for identical floats)."""
    return np.linalg.norm(np.diff(s, axis=0) / (1.0 / fps), axis=1)


def net_motion(s: np.ndarray, fps: float, win_s: float) -> np.ndarray:
    """``‖s[t+w] − s[t]‖`` with ``w = win_s`` seconds: net motion, so jitter around a fixed point cancels."""
    w = max(1, int(round(win_s * fps)))
    return np.linalg.norm(s[w:] - s[:-w], axis=1) if len(s) > w else np.zeros(1)


def idle_before_last_motion(
    s: np.ndarray, fps: float, win_s: float, eps_jitter: float, r_idle: float, r_move: float
) -> float:
    """Fraction of frames idle *before* the episode's last substantial motion (terminal hold excluded).

    Idle = net motion < max(eps_jitter, r_idle · p95); last motion = last frame with net motion >=
    max(eps_jitter, r_move · p95). Returns 1.0 if the episode never really moves.
    """
    net = net_motion(s, fps, win_s)
    pk = np.percentile(net, 95)
    if pk < eps_jitter:
        return 1.0
    moving = np.nonzero(net >= max(eps_jitter, r_move * pk))[0]
    last = moving[-1]
    return float(np.sum(net[:last] < max(eps_jitter, r_idle * pk)) / len(net))


TV_MODES: tuple[str, ...] = ("split", "net")


def tv_denominator(a: np.ndarray, mode: str = "split") -> float:
    """Displacement used by ``action_tv_ratio`` (L1).

    - ``"split"`` (phase 3, R-real-1): ``|a_k − a_0| + |a_T − a_k|`` with k the frame farthest from the start. An
      out-and-back (reach, then return home) counts both legs; for a one-way reach it equals ``net``, and by the
      triangle inequality it is never smaller than ``net``.
    - ``"net"`` (phase 2): ``|a_T − a_0|``. Kept for comparisons; it under-counts return-to-home motions.
    """
    if mode == "net":
        return float(np.abs(a[-1] - a[0]).sum())
    if mode != "split":
        raise ValueError(f"unknown tv mode {mode!r}; choose from {TV_MODES}")
    exc = np.abs(a - a[0]).sum(1)
    k = int(np.argmax(exc))
    return float(exc[k] + np.abs(a[-1] - a[k]).sum())


def action_tv_ratio(a: np.ndarray, den_floor: float, mode: str = "split") -> float:
    """Commanded path length / displacement (L1, see ``tv_denominator``), denominator floored at ``den_floor``."""
    return float(np.abs(np.diff(a, axis=0)).sum() / max(tv_denominator(a, mode), den_floor, 1e-6))


def action_hf_energy(a: np.ndarray) -> float:
    """Mean squared second difference of the action: frame-to-frame chatter."""
    return float(np.mean(np.diff(a, n=2, axis=0) ** 2))


def saturation_frac(raw_a: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of action entries within 1% of the dataset's per-joint min/max."""
    rng = np.maximum(hi - lo, 1e-9)
    return float(np.mean((raw_a <= lo + 0.01 * rng) | (raw_a >= hi - 0.01 * rng)))


def track_err(s: np.ndarray, a: np.ndarray) -> float:
    """Mean ``|a_t − s_{t+1}|`` (position control, same joint order for action and state)."""
    return float(np.mean(np.abs(a[:-1] - s[1:])))
