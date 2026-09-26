"""Record episodes from an env + policy into a session.

Seeding: episode ``i`` of a run with base ``--seed s`` uses ``np.random.SeedSequence([s, i])``, spawned into two
independent children (env reset stream, policy stream). Consequences:

- any episode is reproducible on its own from ``(seed, episode_index)``;
- runs with neighbouring seeds do not share episodes (unlike ``default_rng(seed + i)``);
- changing ``--noise`` or the policy does not change the sampled targets for a given ``(seed, i)``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from robot_report_card import __version__
from robot_report_card.features import ACTION, ENV_STATE, REWARD, STATE, SUCCESS
from robot_report_card.policies import Policy
from robot_report_card.session import Episode, EpisodeMeta, Session, SessionError, SessionInfo
from robot_report_card.sim.base import EnvAdapter

SEEDING_SCHEME = "numpy SeedSequence([seed, episode_index]).spawn(2) -> (env reset, policy)"


def episode_seeds(seed: int, episode_index: int) -> tuple[np.random.SeedSequence, np.random.SeedSequence]:
    """Independent (env, policy) seed sequences for one episode."""
    if seed < 0 or episode_index < 0:
        raise ValueError("seed and episode_index must be >= 0")
    env_ss, policy_ss = np.random.SeedSequence([seed, episode_index]).spawn(2)
    return env_ss, policy_ss


def record_episode(env: EnvAdapter, policy: Policy, seed: int, episode_index: int, max_steps: int) -> Episode:
    """Run exactly ``max_steps`` frames (no early termination) and return the episode."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    env_ss, policy_ss = episode_seeds(seed, episode_index)
    obs, info = env.reset(env_ss)
    policy.reset(env, obs, info, np.random.default_rng(policy_ss))
    states, env_states, actions, rewards, successes = [], [], [], [], []
    for t in range(max_steps):
        res = env.step(policy.act(obs, t))
        states.append(obs.state)
        env_states.append(obs.env_state)
        actions.append(res.action)
        rewards.append(res.reward)
        successes.append(res.success)
        obs = res.observation
    final_error = env.tip_error()
    arrays = {
        STATE: np.asarray(states, dtype=np.float32),
        ENV_STATE: np.asarray(env_states, dtype=np.float32),
        ACTION: np.asarray(actions, dtype=np.float32),
        REWARD: np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        SUCCESS: np.asarray(successes, dtype=bool).reshape(-1, 1),
    }
    meta = EpisodeMeta(
        episode_index=episode_index,
        env_id=env.env_id,
        policy_name=policy.name,
        policy_params=policy.params(),
        seed=seed,
        fps=env.fps,
        num_frames=max_steps,
        duration_s=max_steps / env.fps,
        sim_success=bool(final_error < env.success_threshold_m),
        final_error_m=round(float(final_error), 6),
        termination_reason="max_steps",
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rrc_version=__version__,
    )
    return Episode(meta=meta, arrays=arrays)


def new_session(env: EnvAdapter, out: str | Path, max_steps: int | None = None) -> Session:
    """Create an empty session for ``env`` at ``out`` (refuses existing files and non-empty dirs)."""
    info = SessionInfo(
        env_id=env.env_id,
        fps=env.fps,
        features=env.features,
        seeding=SEEDING_SCHEME,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rrc_version=__version__,
        max_steps=max_steps,
    )
    return Session.create(out, info)


def open_for_append(env: EnvAdapter, out: str | Path, max_steps: int) -> Session:
    """Open an existing session to add episodes, refusing any env / fps / max_steps / feature-spec mismatch.

    Policy, noise and seed may differ per append: they are stored per episode. Episode ``i`` still uses
    ``SeedSequence([seed, i])`` with ``i`` the session-wide index, so appended episodes never repeat earlier ones.
    """
    session = Session.open(out)
    info = session.info()
    metas = session.read_metas()
    stored_steps = info.max_steps
    if stored_steps is None and metas:  # sessions from before max_steps was stored
        lengths = {m.num_frames for m in metas}
        stored_steps = lengths.pop() if len(lengths) == 1 else None
    problems = []
    if info.env_id != env.env_id:
        problems.append(f"env {env.env_id!r} != session's {info.env_id!r}")
    if info.fps != env.fps:
        problems.append(f"fps {env.fps} != session's {info.fps}")
    if stored_steps is not None and stored_steps != max_steps:
        problems.append(f"--max-steps {max_steps} != session's {stored_steps}")
    if json.loads(json.dumps(env.features)) != info.features:
        problems.append("feature spec differs from session.json")
    missing = [m.episode_index for m in metas if not session.npz_path(m.episode_index).is_file()]
    if missing:  # N3: don't extend a session whose existing episodes are already broken
        shown = ", ".join(str(i) for i in missing[:10]) + (" ..." if len(missing) > 10 else "")
        problems.append(f"{len(missing)} existing episode file(s) are missing (episodes {shown})")
    if problems:
        raise SessionError(f"cannot append to {session.root}: " + "; ".join(problems))
    return session


def record_into(
    session: Session,
    env: EnvAdapter,
    policy: Policy,
    episodes: int,
    seed: int,
    max_steps: int,
    on_episode: Callable[[EpisodeMeta], None] | None = None,
) -> None:
    """Record ``episodes`` episodes; each row is written only after its episode completes, so an exception
    mid-run leaves a valid session holding the episodes finished so far."""
    if episodes < 1:
        raise ValueError("--episodes must be >= 1")
    start = len(session.read_metas())
    for i in range(start, start + episodes):
        ep = record_episode(env, policy, seed, i, max_steps)
        session.append_episode(ep)
        if on_episode is not None:
            on_episode(ep.meta)


def record_session(
    env: EnvAdapter,
    policy: Policy,
    out: str | Path,
    episodes: int,
    seed: int,
    max_steps: int,
    on_episode: Callable[[EpisodeMeta], None] | None = None,
) -> Session:
    """Create a new session at ``out`` and record ``episodes`` episodes into it."""
    session = new_session(env, out, max_steps)
    record_into(session, env, policy, episodes, seed, max_steps, on_episode)
    return session
