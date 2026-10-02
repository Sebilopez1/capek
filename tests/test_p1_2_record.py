"""P1-2: so101_reach env, policies, `capek record`."""

from __future__ import annotations

import json
import time
from dataclasses import fields

import numpy as np
import pytest

pytest.importorskip("mujoco")

from capek import cli  # noqa: E402
from capek.policies import make_policy  # noqa: E402
from capek.record import episode_seeds, record_episode  # noqa: E402
from capek.session import EpisodeMeta, Session  # noqa: E402
from capek.sim.base import EnvAdapter  # noqa: E402
from capek.sim.registry import make_env  # noqa: E402

N_RATE = 50


@pytest.fixture(scope="module")
def env():
    return make_env("so101_reach")


def _rate(env, policy_name: str, noise: float = 0.0, seed: int = 0, gravity_comp: bool = True) -> float:
    policy = make_policy(policy_name, noise=noise, gravity_comp=gravity_comp)
    return float(np.mean([record_episode(env, policy, seed, i, 90).meta.sim_success for i in range(N_RATE)]))


def test_env_matches_brief(env) -> None:
    assert isinstance(env, EnvAdapter)
    assert env.fps == 30 and env.substeps == 17
    assert abs(env.substeps * env.model.opt.timestep - 1 / env.fps) < 1e-12
    assert env.success_threshold_m == 0.02
    assert env.features["observation.state"]["names"][0] == "shoulder_pan.pos"
    obs, info = env.reset(0)
    assert np.allclose(obs.state, 0) and obs.env_state.shape == (3,)
    assert np.all(np.abs(info["target_qpos"]) <= 0.6 * np.maximum(-env.action_low, env.action_high) + 1e-12)


def test_record_five_episodes_under_10s(tmp_path) -> None:
    t0 = time.perf_counter()
    rc = cli.main(["record", "--episodes", "5", "--seed", "0", "--out", str(tmp_path / "s"), "-q"])
    assert rc == 0 and time.perf_counter() - t0 < 10
    metas = Session.open(tmp_path / "s").read_metas()
    assert [m.episode_index for m in metas] == list(range(5))


def test_same_seed_identical_arrays(tmp_path) -> None:
    for name in ("a", "b"):
        assert (
            cli.main(
                ["record", "--episodes", "3", "--seed", "4", "--noise", "0.1", "--out", str(tmp_path / name), "-q"]
            )
            == 0
        )
    a, b = Session.open(tmp_path / "a"), Session.open(tmp_path / "b")
    for i in range(3):
        xa, xb = a.load_arrays(i), b.load_arrays(i)
        assert all(xa[k].tobytes() == xb[k].tobytes() for k in xa)


def test_episode_reproducible_alone_and_neighbour_seeds_differ(env, tmp_path) -> None:
    cli.main(["record", "--episodes", "4", "--seed", "0", "--out", str(tmp_path / "s"), "-q"])
    alone = record_episode(env, make_policy("scripted"), seed=0, episode_index=3, max_steps=90)
    stored = Session.open(tmp_path / "s").load_arrays(3)
    assert all(np.array_equal(alone.arrays[k], stored[k]) for k in stored)
    # the old seed + index scheme made (seed 0, ep 1) == (seed 1, ep 0); SeedSequence([seed, i]) must not
    t01 = record_episode(env, make_policy("scripted"), 0, 1, 5).arrays["observation.environment_state"][0]
    t10 = record_episode(env, make_policy("scripted"), 1, 0, 5).arrays["observation.environment_state"][0]
    assert not np.allclose(t01, t10)
    assert episode_seeds(0, 1)[0].entropy != episode_seeds(1, 0)[0].entropy


def test_noise_does_not_change_targets(env) -> None:
    clean = record_episode(env, make_policy("scripted"), 3, 2, 10)
    noisy = record_episode(env, make_policy("scripted", noise=0.2), 3, 2, 10)
    key = "observation.environment_state"
    assert np.array_equal(clean.arrays[key], noisy.arrays[key])
    assert not np.array_equal(clean.arrays["action"], noisy.arrays["action"])


@pytest.mark.parametrize("max_steps", [1, 17, 90])
def test_exact_frame_count_and_metadata(env, max_steps: int) -> None:
    ep = record_episode(env, make_policy("random"), 0, 0, max_steps)
    assert ep.num_frames == max_steps == ep.meta.num_frames
    for key, spec in env.features.items():
        assert ep.arrays[key].shape == (max_steps, *spec["shape"])
        assert ep.arrays[key].dtype == np.dtype(spec["dtype"])
    m = ep.meta
    for f in fields(EpisodeMeta):
        assert getattr(m, f.name) is not None, f.name
    assert m.env_id == "so101_reach" and m.policy_name == "random" and m.termination_reason == "max_steps"
    assert m.duration_s == pytest.approx(max_steps / 30) and m.fps == 30 and m.recorded_at and m.capek_version
    assert m.policy_params["noise"] == 0.0 and m.label == "unlabeled"
    assert m.sim_success == bool(ep.arrays["next.success"][-1, 0])
    assert m.final_error_m == pytest.approx(-float(ep.arrays["next.reward"][-1, 0]), abs=1e-6)
    # actions are clipped to the joint limits
    assert np.all(ep.arrays["action"] >= env.action_low.astype(np.float32) - 1e-6)
    assert np.all(ep.arrays["action"] <= env.action_high.astype(np.float32) + 1e-6)


def test_success_rate_bars(env) -> None:
    scripted = _rate(env, "scripted")
    sagging = _rate(env, "scripted", gravity_comp=False)
    random = _rate(env, "random")
    print(f"\nscripted: {scripted:.2f}, scripted --no-gravity-comp: {sagging:.2f}, random: {random:.2f} ({N_RATE} eps)")
    assert scripted >= 0.60
    assert sagging >= 0.60
    assert random <= 0.10


def test_failure_policies_fail(env) -> None:
    for name in ("stall", "wrong"):
        assert not any(record_episode(env, make_policy(name), 0, i, 90).meta.sim_success for i in range(10))


def test_record_cli_errors(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    out = str(tmp_path / "s")
    assert cli.main(["record", "--episodes", "1", "--out", out, "-q"]) == 0
    assert cli.main(["record", "--episodes", "1", "--out", out]) == 1
    assert "already contains a session" in capsys.readouterr().err
    assert cli.main(["record", "--episodes", "0", "--out", str(tmp_path / "z")]) == 1
    assert cli.main(["record", "--noise", "-1", "--out", str(tmp_path / "z")]) == 1
    info = json.loads((tmp_path / "s" / "session.json").read_text())
    assert info["env_id"] == "so101_reach" and "SeedSequence" in info["seeding"]
