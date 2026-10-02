"""Policies that `capek eval` / `capek compare` can roll out, and the spec parser (`scripted`, `random`, `bc:<dir>`,
`lerobot:<dir>`).

Protocol (plan "Policy protocol"): ``reset(seed, ctx)`` once per episode, then ``act(obs) -> action`` per frame, where
``obs`` is ``{"observation.state": (6,), "observation.environment_state": (3,)}`` (float64). ``ctx`` carries the env
and its reset info; learned policies ignore it, the scripted baseline reads its privileged target from it. Torch is
imported only by the ``bc:`` / ``lerobot:`` loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from capek.policies import RandomPolicy, ScriptedReach
from capek.sim.base import EnvAdapter, Observation

STATE = "observation.state"
ENV_STATE = "observation.environment_state"
TRAINED_KINDS = ("bc", "lerobot")


class PolicySpecError(ValueError):
    """A POLICY argument that can't be parsed or loaded."""


@dataclass(frozen=True)
class ResetContext:
    env: EnvAdapter
    observation: Observation
    info: dict[str, Any]


class EvalPolicy(Protocol):
    spec: str  # what the user typed, e.g. "bc:ckpt/a"
    kind: str  # scripted | random | bc | lerobot

    def metadata(self) -> dict[str, Any]: ...

    def reset(self, seed: np.random.SeedSequence, ctx: ResetContext) -> None: ...

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...


def to_observation(obs: dict[str, np.ndarray]) -> Observation:
    return Observation(state=obs[STATE], env_state=obs[ENV_STATE])


class _LegacyAdapter:
    """Wraps a phase 1 data-generating policy (reset(env, obs, info, rng) / act(obs, t))."""

    def __init__(self, spec: str, kind: str, inner: Any) -> None:
        self.spec, self.kind, self._inner, self._t = spec, kind, inner, 0

    def metadata(self) -> dict[str, Any]:
        return {"spec": self.spec, "kind": self.kind, "params": self._inner.params()}

    def reset(self, seed: np.random.SeedSequence, ctx: ResetContext) -> None:
        self._t = 0
        self._inner.reset(ctx.env, ctx.observation, ctx.info, np.random.default_rng(seed))

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        a = self._inner.act(to_observation(obs), self._t)
        self._t += 1
        return np.asarray(a, dtype=np.float64)


def is_trained(spec: str) -> bool:
    return spec.split(":", 1)[0] in TRAINED_KINDS


def load_policy(spec: str, threads: int | None = None) -> EvalPolicy:
    """Parse and load a POLICY spec. ``threads`` sets torch's CPU thread count for learned policies."""
    kind, _, arg = spec.partition(":")
    if kind == "scripted" and not arg:
        return _LegacyAdapter(spec, "scripted", ScriptedReach())
    if kind == "random" and not arg:
        return _LegacyAdapter(spec, "random", RandomPolicy())
    if kind in TRAINED_KINDS:
        if not arg:
            raise PolicySpecError(f"{kind}: needs a directory, e.g. {kind}:path/to/checkpoint")
        path = Path(arg).expanduser()
        if not path.is_dir():
            raise PolicySpecError(f"{spec}: {path} is not a directory")
        if kind == "bc":
            from capek.eval.bc import load_bc_policy

            return load_bc_policy(spec, path, threads)
        from capek.eval.lerobot_adapter import load_lerobot_policy

        return load_lerobot_policy(spec, path, threads)
    raise PolicySpecError(f"unknown policy {spec!r}: use scripted, random, bc:<dir> or lerobot:<dir>")


def policy_dir(spec: str) -> Path | None:
    kind, _, arg = spec.partition(":")
    return Path(arg).expanduser() if kind in TRAINED_KINDS and arg else None
