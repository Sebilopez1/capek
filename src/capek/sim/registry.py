"""Env lookup by id (imports mujoco lazily so the rest of the CLI works without the sim extra)."""

from __future__ import annotations

from capek.hints import install_hint
from capek.sim.base import EnvAdapter

ENV_IDS: tuple[str, ...] = ("so101_reach",)


def make_env(env_id: str) -> EnvAdapter:
    if env_id not in ENV_IDS:
        raise ValueError(f"unknown env {env_id!r}; available: {', '.join(ENV_IDS)}")
    try:
        from capek.sim.so101_reach import SO101Reach
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(f"the sim extra is not installed: {install_hint('sim')}") from e
    return SO101Reach()
