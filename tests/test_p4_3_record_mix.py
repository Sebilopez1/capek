"""P4-3: `capek record --mix` (byte-identical to `bench record`), --list-groups, and the independent `wrong` goal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from capek import cli  # noqa: E402
from capek.bench.__main__ import main as bench_main  # noqa: E402
from capek.bench.spec import MIXED  # noqa: E402
from capek.session import Session  # noqa: E402

SMALL_MIX = "clean:3,noise025:1,random:1,hesitation:1,wrong:1,return_home:1"


def _rows(root: Path) -> list[dict]:
    rows = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("recorded_at")
    return rows


def _info(root: Path) -> dict:
    info = json.loads((root / "session.json").read_text())
    info.pop("created_at")
    return info


@pytest.mark.parametrize("profile", ["standard", "low"])
def test_mix_is_byte_identical_to_bench_record(tmp_path: Path, profile: str) -> None:
    a, b = tmp_path / "bench", tmp_path / "capek"
    assert bench_main(["record", "--out", str(a), "--groups", SMALL_MIX, "--seed", "30000", "--noise", profile]) == 0
    argv = ["record", "--mix", SMALL_MIX, "--seed", "30000", "--noise-profile", profile, "--out", str(b), "-q"]
    assert cli.main(argv) == 0
    assert _rows(a) == _rows(b) and _info(a) == _info(b)
    files = sorted(p.name for p in (a / "episodes").iterdir())
    assert files == sorted(p.name for p in (b / "episodes").iterdir()) and len(files) == 8
    for name in files:
        assert (a / "episodes" / name).read_bytes() == (b / "episodes" / name).read_bytes(), name
    groups = [r["policy_params"]["bench_group"] for r in _rows(b)]
    assert groups == ["clean"] * 3 + ["noise025", "random", "hesitation", "wrong", "return_home"]
    assert [r["seed"] for r in _rows(b)] == [30000] * 3 + [30001, 30002, 30003, 30004, 30005]


def test_mix_appends_to_an_existing_session(tmp_path: Path) -> None:
    out = tmp_path / "s"
    assert cli.main(["record", "--episodes", "2", "--out", str(out), "-q"]) == 0
    assert cli.main(["record", "--mix", "clean:1,random:1", "--seed", "7", "--append", "--out", str(out), "-q"]) == 0
    metas = Session.open(out).read_metas()
    assert [m.episode_index for m in metas] == [0, 1, 2, 3]
    assert [m.policy_params.get("bench_group") for m in metas] == [None, None, "clean", "random"]


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--mix", "clean:2", "--policy", "random"], "--mix can't be combined with --policy"),
        (["--mix", "clean:2", "--episodes", "3"], "--mix can't be combined with --episodes"),
        (["--mix", "clean:2", "--noise", "0.1"], "--mix can't be combined with --noise"),
        (["--mix", "clean:2", "--max-steps", "60"], "--mix needs --max-steps 90"),
        (["--mix", "cleen:2"], "bad group 'cleen:2'"),
        (["--mix", "clean:0"], "bad group 'clean:0'"),
        (["--noise-profile", "low"], "--noise-profile only applies to --mix"),
        (["--wrong-goal", "mirrored"], "--wrong-goal only applies to --policy wrong"),
    ],
)
def test_mix_refusals(tmp_path: Path, capsys, argv: list[str], message: str) -> None:
    out = tmp_path / "s"
    assert cli.main(["record", *argv, "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err
    assert not out.exists()


def test_list_groups(capsys) -> None:
    assert cli.main(["record", "--list-groups"]) == 0
    out = capsys.readouterr().out
    lines = {
        line.split()[0]: line.split()[1] for line in out.splitlines()[1:] if line and not line.startswith("classes")
    }
    assert lines == {g: cls for g, cls, *_ in MIXED}
    assert lines["return_home"] == "clean_variant" and lines["wrong"] == "outcome_only"


def test_wrong_policy_independent_by_default_and_mirrored_on_request(tmp_path: Path) -> None:
    assert cli.main(["record", "--policy", "wrong", "--episodes", "3", "--out", str(tmp_path / "ind"), "-q"]) == 0
    argv = [
        "record",
        "--policy",
        "wrong",
        "--wrong-goal",
        "mirrored",
        "--episodes",
        "3",
        "--out",
        str(tmp_path / "mir"),
    ]
    assert cli.main([*argv, "-q"]) == 0
    ind, mir = Session.open(tmp_path / "ind"), Session.open(tmp_path / "mir")
    from capek.sim.registry import make_env

    env = make_env("so101_reach")
    for m in ind.read_metas():
        p = m.policy_params
        assert p["goal"] == "independent" and "target_sign" not in p and not m.sim_success
        goal = np.array(p["goal_qpos"])
        assert np.all(np.abs(goal) <= 0.6 * np.maximum(-env.action_low, env.action_high) + 1e-6)
    for m in mir.read_metas():
        assert m.policy_params["goal"] == "mirrored" and m.policy_params["target_sign"] == -1.0
        assert not m.sim_success
    # same seeds, same targets: the two variants command different goals
    a, b = ind.load_arrays(0)["action"][-1], mir.load_arrays(0)["action"][-1]
    assert not np.allclose(a, b)


def test_bad_wrong_goal_value_and_old_sessions_still_load(tmp_path: Path) -> None:
    from capek.policies import make_policy

    with pytest.raises(ValueError, match="--wrong-goal"):
        make_policy("wrong", wrong_goal="sideways")
    out = tmp_path / "old"
    assert (
        cli.main(
            ["record", "--policy", "wrong", "--wrong-goal", "mirrored", "--episodes", "1", "--out", str(out), "-q"]
        )
        == 0
    )
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    rows[0]["policy_params"] = {"ramp_s": 2.0, "gravity_comp": True, "target_sign": -1.0, "noise": 0.0}  # phase 1 form
    (out / "episodes.jsonl").write_text(json.dumps(rows[0]) + "\n")
    assert Session.open(out).read_metas()[0].policy_params["target_sign"] == -1.0
    assert cli.main(["list", str(out)]) == 0
