"""Benchmark-v2 episode generators (plan D1), built on the phase 1 policy library.

Every scripted group shares the clean group's heterogeneity: ramp ~U(1.0, 2.5) s and, unless the group fixes it,
action noise sigma ~U(0.01, 0.04) drawn per episode. That shared spread is what keeps the benchmark from being
separable by generator artifacts (fixed timing / fixed noise floor) instead of by quality.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capek.policies import NoisyPolicy, RandomPolicy, ScriptedReach

RAMP_S = (1.0, 2.5)
SIGMA = (0.01, 0.04)
FIXED_SIGMA = {"noise005": 0.05, "noise010": 0.10, "noise025": 0.25}
HESITATION_S = 0.8
HESITATION_AT = (0.25, 0.60)  # pause starts at this fraction of the ramp (random per episode)
WOBBLE_SIGMA, WOBBLE_PHI = 0.10, 0.9
TARGET_FRACTION = 0.6  # so101_reach samples targets from U(0.6 * joint range); wrong goals use the same distribution
STALL_AT = (0.3, 0.7)  # stall: the command freezes at this fraction of the ramp (random per episode)
RETURN_OUT_S = (0.8, 1.2)  # return_home: out to the target over this long, hold, then back over the same time
RETURN_HOLD_S = (0.1, 0.4)


class ReturnHomeReach:
    """Out-and-back reach: linear ramp from the start pose to the target, hold, linear ramp back to the start pose.

    Commands include the same gravity offset as ``ScriptedReach`` so both legs actually get there. Clean motion:
    the phase 2 ``net`` tv denominator (|a_T - a_0|, about zero here) is what made it look like dithering.
    """

    name = "scripted"

    def __init__(self, out_s: float, hold_s: float) -> None:
        self.out_s, self.hold_s = out_s, hold_s

    def params(self) -> dict[str, Any]:
        return {"out_s": round(self.out_s, 6), "hold_s": round(self.hold_s, 6)}

    def reset(self, env: Any, obs: Any, info: dict[str, Any], rng: np.random.Generator) -> None:
        def with_offset(q: np.ndarray) -> np.ndarray:
            offset = getattr(env, "gravity_offset", None)
            return q + offset(q) if offset is not None else q

        self._home = with_offset(np.asarray(obs.state, dtype=np.float64))
        self._start = np.asarray(obs.state, dtype=np.float64).copy()
        self._goal = with_offset(np.asarray(info["target_qpos"], dtype=np.float64))
        self._out = max(1, round(self.out_s * env.fps))
        self._hold = round(self.hold_s * env.fps)

    def act(self, obs: Any, t: int) -> np.ndarray:
        if t < self._out:
            return self._start + (t + 1) / self._out * (self._goal - self._start)
        if t < self._out + self._hold:
            return self._goal.copy()
        f = min(1.0, (t - self._out - self._hold + 1) / self._out)
        return self._goal + f * (self._home - self._goal)


class GroupPolicy:
    """One benchmark group as a phase 1 ``Policy``: per-episode parameters are drawn in ``reset`` from its rng."""

    def __init__(self, group: str, sigma_range: tuple[float, float] = SIGMA) -> None:
        self.group = group
        self.sigma_range = sigma_range
        self.name = "random" if group == "random" else "scripted"
        self._params: dict[str, Any] = {}

    def params(self) -> dict[str, Any]:
        return {"bench_group": self.group, **self._params}

    def reset(self, env: Any, obs: Any, info: dict[str, Any], rng: np.random.Generator) -> None:
        # spawn(7): children 0-4 are identical to the original spawn(5), so the other groups' draws are unchanged
        r_inner, r_ramp, r_sigma, r_pause, r_wobble, r_stall, r_goal = rng.spawn(7)
        g = self.group
        self._pause: tuple[int, int] | None = None
        self._wobble: np.random.Generator | None = None
        if g == "random":
            self._inner: Any = NoisyPolicy(RandomPolicy(), 0.0)
            self._inner.reset(env, obs, info, r_inner)
            self._params = {}
            return
        sigma = FIXED_SIGMA.get(g, float(r_sigma.uniform(*self.sigma_range)))
        if g == "return_home":
            out_s, hold_s = float(r_ramp.uniform(*RETURN_OUT_S)), float(r_pause.uniform(*RETURN_HOLD_S))
            self._inner = NoisyPolicy(ReturnHomeReach(out_s, hold_s), sigma)
            self._inner.reset(env, obs, info, r_inner)
            self._params = {"out_s": round(out_s, 6), "hold_s": round(hold_s, 6), "noise": round(sigma, 6)}
            return
        ramp = float(r_ramp.uniform(*RAMP_S))
        stall_after = float(r_stall.uniform(*STALL_AT)) * ramp if g == "stall" else None
        base = ScriptedReach(ramp_s=ramp, gravity_comp=g != "nearmiss", stall_after_s=stall_after)
        self._params = {"ramp_s": round(ramp, 6), "noise": round(sigma, 6), "gravity_comp": g != "nearmiss"}
        if g == "wrong":
            # A wrong goal drawn independently from the SAME distribution as real targets, so the motion looks like
            # any clean reach and only the outcome differs (no joint-limit artifact from mirroring the target).
            low, high = np.asarray(env.action_low), np.asarray(env.action_high)
            goal = r_goal.uniform(TARGET_FRACTION * low, TARGET_FRACTION * high)
            info = {**info, "target_qpos": goal}
            self._params["wrong_goal_qpos"] = [round(float(x), 6) for x in goal]
        if stall_after is not None:
            self._params["stall_after_s"] = round(stall_after, 6)
        self._inner = NoisyPolicy(base, sigma)
        self._inner.reset(env, obs, info, r_inner)
        if g == "hesitation":
            start = int(float(r_pause.uniform(*HESITATION_AT)) * ramp * env.fps)
            self._pause = (start, round(HESITATION_S * env.fps))
            self._params["pause"] = {"start_frame": start, "frames": self._pause[1]}
        if g == "wobble":
            self._wobble, self._e = np.random.default_rng(r_wobble), np.zeros(len(env.action_low))
            self._params["wobble"] = {"sigma": WOBBLE_SIGMA, "phi": WOBBLE_PHI}

    def act(self, obs: Any, t: int) -> np.ndarray:
        if self._pause is not None:  # the command schedule stops for the pause; noise keeps going
            start, n = self._pause
            t = t if t < start else (start if t < start + n else t - n)
        a = self._inner.act(obs, t)
        if self._wobble is not None:
            self._e = WOBBLE_PHI * self._e + np.sqrt(1 - WOBBLE_PHI**2) * self._wobble.normal(
                0.0, WOBBLE_SIGMA, self._e.shape
            )
            a = a + self._e
        return a
