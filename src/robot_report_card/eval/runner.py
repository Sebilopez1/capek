"""Seeded closed-loop rollouts in ``so101_reach``.

Episode i: the env resets from ``SeedSequence([eval_seed, i])`` (this draws the target), and the policy gets its
own child ``SeedSequence([eval_seed, i]).spawn(1)[0]``. Actions are clipped to the action range by the env. Success
is the env's final-frame success (tip within 2 cm on the last frame); ``success_any_step`` is LeRobot's definition.
Same seeds + same machine => identical success vectors and final-state hashes (plan DoD 7).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from robot_report_card.eval.policies import ENV_STATE, STATE, EvalPolicy, ResetContext
from robot_report_card.sim.registry import make_env

ENV_ID = "so101_reach"
DEFAULT_EPISODES = 200
DEFAULT_EVAL_SEED = 900_000
DEFAULT_MAX_STEPS = 90


def success_definition(max_steps: int, fps: float) -> str:
    return (
        f"success = gripper tip within 2 cm of the target on the final frame ({max_steps} frames, "
        f"{max_steps / fps:g} s); LeRobot's eval counts success at any step"
    )


@dataclass
class EpisodeOutcome:
    episode: int  # i in SeedSequence([eval_seed, i])
    success: bool
    success_any_step: bool
    final_error_m: float
    final_state_sha256: str


@dataclass
class RolloutResult:
    policy: dict[str, Any]
    env_id: str
    eval_seed: int
    max_steps: int
    fps: float
    episodes: list[EpisodeOutcome]

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def success(self) -> np.ndarray:
        return np.array([e.success for e in self.episodes], dtype=bool)

    @property
    def success_any_step(self) -> np.ndarray:
        return np.array([e.success_any_step for e in self.episodes], dtype=bool)

    @property
    def median_final_error_m(self) -> float:
        return float(np.median([e.final_error_m for e in self.episodes])) if self.episodes else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "env_id": self.env_id,
            "eval_seed": self.eval_seed,
            "max_steps": self.max_steps,
            "fps": self.fps,
            "seeds": [[self.eval_seed, e.episode] for e in self.episodes],
            "episodes": [asdict(e) for e in self.episodes],
        }


def _obs_dict(obs: Any) -> dict[str, np.ndarray]:
    return {STATE: obs.state.copy(), ENV_STATE: obs.env_state.copy()}


def run_rollouts(
    policy: EvalPolicy,
    episodes: int = DEFAULT_EPISODES,
    eval_seed: int = DEFAULT_EVAL_SEED,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> RolloutResult:
    if episodes < 1 or max_steps < 1 or eval_seed < 0:
        raise ValueError("--episodes and --max-steps must be >= 1 and --eval-seed >= 0")
    env = make_env(ENV_ID)
    out = []
    for i in range(episodes):
        obs, info = env.reset(np.random.SeedSequence([eval_seed, i]))
        policy.reset(np.random.SeedSequence([eval_seed, i]).spawn(1)[0], ResetContext(env, obs, info))
        any_step = False
        res = None
        for _ in range(max_steps):
            action = np.asarray(policy.act(_obs_dict(obs)), dtype=np.float64)
            if action.shape != env.action_low.shape or not np.all(np.isfinite(action)):
                raise ValueError(f"policy {policy.spec} returned an invalid action {action!r} in episode {i}")
            res = env.step(action)
            any_step = any_step or res.success
            obs = res.observation
        assert res is not None
        digest = hashlib.sha256(np.ascontiguousarray(obs.state, dtype=np.float64).tobytes()).hexdigest()
        out.append(EpisodeOutcome(i, bool(res.success), bool(any_step), float(env.tip_error()), digest))
    return RolloutResult(policy.metadata(), ENV_ID, eval_seed, max_steps, float(env.fps), out)


def machine_info(threads: int | None) -> dict[str, Any]:
    def version(pkg: str) -> str | None:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    torch_threads = None
    if "torch" in sys.modules:
        torch_threads = sys.modules["torch"].get_num_threads()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": version("mujoco"),
        "torch": version("torch"),
        "torch_threads": torch_threads,
        "requested_threads": threads,
        "cpu_count": os.cpu_count(),
    }
