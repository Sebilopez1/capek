"""P2-2: `capek record --append` (one session mixing policies / noise)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import FakeEnv

from capek import cli
from capek.session import Session
from capek.sim import registry


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# ---- no mujoco needed (FakeEnv) -------------------------------------------------------------------------------
@pytest.fixture
def fake(monkeypatch):
    envs: dict[str, FakeEnv] = {}

    def use(env: FakeEnv) -> FakeEnv:
        envs["env"] = env
        monkeypatch.setattr(registry, "make_env", lambda env_id: envs["env"])
        return env

    use(FakeEnv())
    return use


def rec(out: Path, *extra: str, steps: int = 6) -> int:
    return cli.main(["record", "--out", str(out), "--max-steps", str(steps), "-q", *extra])


def test_append_mismatches_are_refused_and_session_untouched(tmp_path: Path, fake, capsys) -> None:
    out = tmp_path / "s"
    assert rec(out, "--episodes", "2") == 0
    before = _snapshot(out)
    assert rec(out, "--episodes", "1", "--append", steps=7) == 1
    assert "--max-steps 7 != session's 6" in capsys.readouterr().err
    other_fps = FakeEnv()
    other_fps.fps = 15
    fake(other_fps)
    assert rec(out, "--episodes", "1", "--append") == 1
    assert "fps 15 != session's 30" in capsys.readouterr().err
    other_env = FakeEnv()
    other_env.env_id = "other"
    other_env.features = {k: dict(v) for k, v in FakeEnv.features.items()}
    other_env.features["action"] = {**other_env.features["action"], "names": ["a"] * 6}
    fake(other_env)
    assert rec(out, "--episodes", "1", "--append") == 1
    err = capsys.readouterr().err
    assert "env 'other' != session's 'fake_reach'" in err and "feature spec differs" in err
    assert _snapshot(out) == before


def test_without_append_the_refusal_is_unchanged(tmp_path: Path, fake, capsys) -> None:
    out = tmp_path / "s"
    assert rec(out, "--episodes", "1") == 0
    assert rec(out, "--episodes", "1") == 1
    assert "already contains a session" in capsys.readouterr().err
    assert len(Session.open(out).read_metas()) == 1


def test_append_to_new_dir_creates_session(tmp_path: Path, fake) -> None:
    assert rec(tmp_path / "new", "--episodes", "2", "--append") == 0
    assert Session.open(tmp_path / "new").info().max_steps == 6


def test_interrupted_append_leaves_session_valid(tmp_path: Path, fake, capsys) -> None:
    out = tmp_path / "s"
    assert rec(out, "--episodes", "2") == 0
    fake(FakeEnv(fail_at=(1, 3)))  # second appended episode (global index 3) diverges
    assert rec(out, "--episodes", "3", "--append") == 1
    assert "3 episodes saved" in capsys.readouterr().err
    s = Session.open(out)
    assert [m.episode_index for m in s.read_metas()] == [0, 1, 2]
    assert sorted(p.name for p in (out / "episodes").iterdir()) == [f"episode_00000{i}.npz" for i in range(3)]
    fake(FakeEnv())
    assert rec(out, "--episodes", "1", "--append") == 0  # and it can be extended again
    assert [m.episode_index for m in Session.open(out).read_metas()] == [0, 1, 2, 3]


def test_legacy_session_without_max_steps_uses_episode_length(tmp_path: Path, fake, capsys) -> None:
    out = tmp_path / "s"
    assert rec(out, "--episodes", "1") == 0
    info = json.loads((out / "session.json").read_text())
    del info["max_steps"]
    (out / "session.json").write_text(json.dumps(info))
    assert rec(out, "--episodes", "1", "--append", steps=5) == 1
    assert "--max-steps 5 != session's 6" in capsys.readouterr().err
    assert rec(out, "--episodes", "1", "--append") == 0


# ---- real sim ----------------------------------------------------------------------------------------------------
APPENDS = [
    ["--policy", "scripted", "--episodes", "3", "--seed", "0"],
    ["--policy", "scripted", "--noise", "0.25", "--episodes", "2", "--seed", "1", "--append"],
    ["--policy", "random", "--episodes", "2", "--seed", "2", "--append"],
]


def _record_mixed(out: Path) -> None:
    for argv in APPENDS:
        assert cli.main(["record", "--out", str(out), "--max-steps", "20", "-q", *argv]) == 0


@pytest.fixture(scope="module")
def mixed(tmp_path_factory) -> Path:
    pytest.importorskip("mujoco")
    out = tmp_path_factory.mktemp("append") / "mixed"
    _record_mixed(out)
    return out


def test_three_appends_give_one_contiguous_session(mixed: Path) -> None:
    metas = Session.open(mixed).read_metas()
    assert [m.episode_index for m in metas] == list(range(7))
    assert [m.policy_name for m in metas] == ["scripted"] * 5 + ["random"] * 2
    assert [m.policy_params["noise"] for m in metas] == [0.0] * 3 + [0.25] * 2 + [0.0] * 2
    assert [m.seed for m in metas] == [0, 0, 0, 1, 1, 2, 2]
    assert {m.num_frames for m in metas} == {20}


def test_appended_episode_uses_session_wide_index(mixed: Path) -> None:
    from capek.policies import make_policy
    from capek.record import record_episode

    env = registry.make_env("so101_reach")
    alone = record_episode(env, make_policy("scripted", noise=0.25), seed=1, episode_index=4, max_steps=20)
    stored = Session.open(mixed).load_arrays(4)
    assert all(np.array_equal(alone.arrays[k], stored[k]) for k in stored)


def test_rerunning_the_same_commands_gives_identical_arrays(mixed: Path, tmp_path: Path) -> None:
    again = tmp_path / "again"
    _record_mixed(again)
    a, b = Session.open(mixed), Session.open(again)
    for i in range(7):
        xa, xb = a.load_arrays(i), b.load_arrays(i)
        assert all(xa[k].tobytes() == xb[k].tobytes() for k in xa)


def test_export_carries_per_episode_policy_into_tags(mixed: Path, tmp_path: Path) -> None:
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from capek.export.lerobot_writer import read_tags

    out = tmp_path / "ds"
    assert cli.main(["export", str(mixed), "--out", str(out)]) == 0
    tags = read_tags(out)["episodes"]
    for m in Session.open(mixed).read_metas():
        row = tags[str(m.episode_index)]
        assert (row["policy_name"], row["policy_params"], row["seed"]) == (m.policy_name, m.policy_params, m.seed)
