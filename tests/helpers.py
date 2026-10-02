"""Synthetic sessions for tests that must not need mujoco or lerobot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from capek import __version__
from capek.features import make_features
from capek.session import Episode, EpisodeMeta, Session, SessionInfo

JOINTS = [f"{j}.pos" for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]
FEATURES = make_features(JOINTS, ["target_x", "target_y", "target_z"], JOINTS)


def synthetic_episode(index: int, num_frames: int = 12, seed: int = 0, fps: int = 30) -> Episode:
    rng = np.random.default_rng([seed, index])
    arrays = {
        "observation.state": rng.normal(size=(num_frames, 6)).astype(np.float32),
        "observation.environment_state": rng.normal(size=(num_frames, 3)).astype(np.float32),
        "action": rng.normal(size=(num_frames, 6)).astype(np.float32),
        "next.reward": -rng.random((num_frames, 1)).astype(np.float32),
        "next.success": rng.random((num_frames, 1)) < 0.3,
    }
    success = bool(index % 2 == 0)
    meta = EpisodeMeta(
        episode_index=index,
        env_id="synthetic",
        policy_name="synthetic",
        policy_params={"noise": 0.0},
        seed=seed,
        fps=fps,
        num_frames=num_frames,
        duration_s=num_frames / fps,
        sim_success=success,
        final_error_m=0.01 if success else 0.05,
        termination_reason="max_steps",
        recorded_at="2026-09-24T00:00:00+00:00",
        capek_version=__version__,
    )
    return Episode(meta=meta, arrays=arrays)


def make_synthetic_session(root: Path, n: int = 4, num_frames: int = 12) -> Session:
    info = SessionInfo(
        env_id="synthetic",
        fps=30,
        features=FEATURES,
        seeding="test",
        created_at="2026-09-24T00:00:00+00:00",
        capek_version=__version__,
    )
    s = Session.create(root, info)
    for i in range(n):
        s.append_episode(synthetic_episode(i, num_frames=num_frames))
    return s


class FakeEnv:
    """Pure-python EnvAdapter stand-in (no mujoco). ``fail_at=(episode, frame)`` raises ``FloatingPointError``."""

    env_id = "fake_reach"
    fps = 30
    success_threshold_m = 0.02
    features = FEATURES

    def __init__(self, fail_at: tuple[int, int] | None = None) -> None:
        self.action_low = -np.ones(6)
        self.action_high = np.ones(6)
        self.fail_at = fail_at
        self.episode = -1
        self.t = 0
        self.q = np.zeros(6)

    def reset(self, seed):
        from capek.sim.base import Observation

        self.episode += 1
        self.t = 0
        self.q = np.zeros(6)
        target = np.random.default_rng(seed).uniform(-0.5, 0.5, 6)
        return Observation(self.q.copy(), np.zeros(3)), {"target_qpos": target}

    def step(self, action):
        from capek.sim.base import Observation, StepResult

        if self.fail_at == (self.episode, self.t):
            raise FloatingPointError("simulation diverged (non-finite qpos)")
        self.t += 1
        applied = np.clip(np.asarray(action, dtype=np.float64), self.action_low, self.action_high)
        self.q = applied
        err = self.tip_error()
        return StepResult(Observation(self.q.copy(), np.zeros(3)), applied, -err, err < 0.02)

    def tip_error(self) -> float:
        return 0.05
