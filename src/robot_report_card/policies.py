"""Data-generating policies for sim recording.

Each policy gets its own ``np.random.Generator`` per episode (see ``record.episode_seeds``) and may read
privileged reset info (``target_qpos``) from the env, because they exist to produce labelled good/bad demos.

- ``scripted``: linear ramp from the start pose to the target joint config over ``ramp_s`` seconds, then hold.
  The command includes a gravity offset (bias torque / kp at the target) so the position servos don't sag.
- ``random``: uniform random joint targets within the action bounds every frame (a junk baseline).
- ``stall``: ``scripted`` but the command freezes after ``stall_after_s`` seconds (nice-to-have failure mode).
- ``wrong``: ``scripted`` towards a wrong goal: by default an independent reachable goal drawn like the env's targets
  (from 0.1.0), or the mirrored config ``-q*`` with ``wrong_goal="mirrored"`` (phase 1 behaviour).

``--noise σ`` wraps any policy and adds ``N(0, σ²)`` rad per joint per frame (0 = off).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from robot_report_card.sim.base import EnvAdapter, Observation

POLICY_NAMES: tuple[str, ...] = ("scripted", "random", "stall", "wrong")
WRONG_GOALS: tuple[str, ...] = ("independent", "mirrored")
WRONG_GOAL_FRACTION = 0.6  # so101_reach draws targets from U(0.6 * joint range); independent wrong goals too


class Policy(Protocol):
    name: str

    def params(self) -> dict[str, Any]: ...

    def reset(self, env: EnvAdapter, obs: Observation, info: dict[str, Any], rng: np.random.Generator) -> None: ...

    def act(self, obs: Observation, t: int) -> np.ndarray: ...


@dataclass
class ScriptedReach:
    ramp_s: float = 2.0
    gravity_comp: bool = True
    target_sign: float = 1.0  # -1 => head to the mirrored pose -q* ("wrong" with --wrong-goal mirrored)
    stall_after_s: float | None = None  # set => "stall" policy
    wrong_goal: str | None = None  # "independent" | "mirrored" for the wrong policy (recorded as params["goal"])
    name: str = "scripted"
    _start: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _goal: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _ramp_frames: int = 1
    _stall_frame: int | None = None
    _last: np.ndarray | None = None
    _goal_qpos: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        p: dict[str, Any] = {"ramp_s": self.ramp_s, "gravity_comp": self.gravity_comp}
        if self.target_sign != 1.0:
            p["target_sign"] = self.target_sign
        if self.stall_after_s is not None:
            p["stall_after_s"] = self.stall_after_s
        if self.wrong_goal is not None:
            p["goal"] = self.wrong_goal
            if self.wrong_goal == "independent" and self._goal_qpos is not None:
                p["goal_qpos"] = [round(float(x), 6) for x in self._goal_qpos]
        return p

    def reset(self, env: EnvAdapter, obs: Observation, info: dict[str, Any], rng: np.random.Generator) -> None:
        if self.wrong_goal == "independent":
            # a different reachable goal drawn from the same distribution as the env's targets: the motion looks
            # like any clean reach and only the outcome differs (no joint-limit artifact, unlike mirroring)
            low, high = np.asarray(env.action_low), np.asarray(env.action_high)
            self._goal_qpos = rng.uniform(WRONG_GOAL_FRACTION * low, WRONG_GOAL_FRACTION * high)
            goal = self._goal_qpos.copy()
        else:
            goal = self.target_sign * np.asarray(info["target_qpos"], dtype=np.float64)
        offset = getattr(env, "gravity_offset", None)
        if self.gravity_comp and offset is not None:
            goal = goal + offset(goal)
        self._start = obs.state.astype(np.float64).copy()
        self._goal = goal
        self._ramp_frames = max(1, round(self.ramp_s * env.fps))
        self._stall_frame = None if self.stall_after_s is None else round(self.stall_after_s * env.fps)
        self._last = None

    def act(self, obs: Observation, t: int) -> np.ndarray:
        if self._stall_frame is not None and t >= self._stall_frame and self._last is not None:
            return self._last.copy()
        frac = min(1.0, (t + 1) / self._ramp_frames)
        self._last = self._start + frac * (self._goal - self._start)
        return self._last.copy()


@dataclass
class RandomPolicy:
    name: str = "random"
    _rng: np.random.Generator | None = None
    _low: np.ndarray | None = None
    _high: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {"distribution": "uniform(action bounds)"}

    def reset(self, env: EnvAdapter, obs: Observation, info: dict[str, Any], rng: np.random.Generator) -> None:
        self._rng, self._low, self._high = rng, env.action_low, env.action_high

    def act(self, obs: Observation, t: int) -> np.ndarray:
        assert self._rng is not None, "reset() first"
        return self._rng.uniform(self._low, self._high)


@dataclass
class NoisyPolicy:
    """Adds i.i.d. Gaussian action noise (rad) on top of ``inner``."""

    inner: Policy
    sigma: float
    _rng: np.random.Generator | None = None

    @property
    def name(self) -> str:
        return self.inner.name

    def params(self) -> dict[str, Any]:
        return {**self.inner.params(), "noise": self.sigma}

    def reset(self, env: EnvAdapter, obs: Observation, info: dict[str, Any], rng: np.random.Generator) -> None:
        inner_rng, self._rng = rng.spawn(2)
        self.inner.reset(env, obs, info, inner_rng)

    def act(self, obs: Observation, t: int) -> np.ndarray:
        assert self._rng is not None, "reset() first"
        a = self.inner.act(obs, t)
        return a + self._rng.normal(0.0, self.sigma, size=a.shape) if self.sigma > 0 else a


def make_policy(name: str, noise: float = 0.0, gravity_comp: bool = True, wrong_goal: str = "independent") -> Policy:
    """Build a policy by CLI name; ``noise`` (rad, σ) is always recorded in ``policy_params``.

    ``gravity_comp=False`` gives the uncompensated ramp from the research spike (sags under gravity, ~75% success).
    ``wrong_goal`` (wrong only): "independent" (default from 0.1.0) or "mirrored" (the phase 1 behaviour).
    """
    if wrong_goal not in WRONG_GOALS:
        raise ValueError(f"--wrong-goal must be one of {', '.join(WRONG_GOALS)}")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError(f"--noise must be a finite number >= 0, got {noise!r}")
    base: Policy
    if name == "scripted":
        base = ScriptedReach(gravity_comp=gravity_comp)
    elif name == "stall":
        base = ScriptedReach(gravity_comp=gravity_comp, stall_after_s=1.0, name="stall")
    elif name == "wrong":
        sign = -1.0 if wrong_goal == "mirrored" else 1.0
        base = ScriptedReach(gravity_comp=gravity_comp, target_sign=sign, wrong_goal=wrong_goal, name="wrong")
    elif name == "random":
        base = RandomPolicy()
    else:
        raise ValueError(f"unknown policy {name!r}; available: {', '.join(POLICY_NAMES)}")
    return NoisyPolicy(base, noise)
