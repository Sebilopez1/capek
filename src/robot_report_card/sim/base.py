"""Environment adapter protocol: the only surface the recorder and policies use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from robot_report_card.features import FeatureSpec

SeedLike = int | np.random.SeedSequence


@dataclass(frozen=True)
class Observation:
    state: np.ndarray  # observation.state (e.g. joint positions)
    env_state: np.ndarray  # observation.environment_state (e.g. target xyz)


@dataclass(frozen=True)
class StepResult:
    observation: Observation  # observation after the step
    action: np.ndarray  # the action actually applied (after clipping to the action bounds)
    reward: float  # next.reward
    success: bool  # next.success
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EnvAdapter(Protocol):
    env_id: str
    fps: int
    features: FeatureSpec
    action_low: np.ndarray
    action_high: np.ndarray
    success_threshold_m: float

    def reset(self, seed: SeedLike) -> tuple[Observation, dict[str, Any]]:
        """Start an episode. ``info`` may carry privileged data for scripted policies (e.g. ``target_qpos``)."""
        ...

    def step(self, action: np.ndarray) -> StepResult: ...

    def tip_error(self) -> float:
        """Current distance (m) from the end effector to the target."""
        ...
