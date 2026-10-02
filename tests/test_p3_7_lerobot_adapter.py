"""P3-7: LeRobot adapter (`lerobot:<pretrained_model dir>`), smoke-tested on a tiny ACT trained in-test.

No success level is asserted for ACT (brief §5: it doesn't reach the target within minutes of CPU training).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from capek import cli
from capek.eval.lerobot_adapter import check_inputs
from capek.eval.policies import PolicySpecError


def test_eval_lerobot_writes_a_valid_report(tiny_act: Path, tmp_path: Path) -> None:
    out = tmp_path / "act.json"
    assert cli.main(["eval", f"lerobot:{tiny_act}", "--episodes", "5", "--json-out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["kind"] == "eval" and r["schema_version"] == 1 and r["summary"]["n"] == 5
    assert r["policy"]["kind"] == "lerobot" and r["policy"]["policy_type"] == "act"
    assert r["policy"]["inputs"] == ["observation.environment_state", "observation.state"]
    assert len(r["episodes"]) == 5 and all(len(e["final_state_sha256"]) == 64 for e in r["episodes"])
    assert 0 <= r["summary"]["successes"] <= 5  # smoke test only: no success bar for ACT


def test_json_stdout_stays_valid(tiny_act: Path, capsys) -> None:
    assert cli.main(["eval", f"lerobot:{tiny_act}", "--episodes", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["policy"]["kind"] == "lerobot"


def test_reset_clears_the_action_queue(tiny_act: Path) -> None:
    from capek.eval.policies import ResetContext, load_policy
    from capek.sim.registry import make_env

    policy = load_policy(f"lerobot:{tiny_act}")
    env = make_env("so101_reach")
    obs, info = env.reset(np.random.SeedSequence([1, 0]))
    ctx = ResetContext(env, obs, info)
    policy.reset(np.random.SeedSequence(0), ctx)
    o = {"observation.state": obs.state, "observation.environment_state": obs.env_state}
    a = policy.act(o)
    assert a.shape == (6,) and np.all(np.isfinite(a))
    queue = policy.policy._action_queue
    assert len(queue) == policy.policy.config.n_action_steps - 1  # one chunk predicted, one action popped
    policy.reset(np.random.SeedSequence(1), ctx)
    assert len(policy.policy._action_queue) == 0


def _copy_with_inputs(src: Path, dst: Path, inputs: dict) -> Path:
    shutil.copytree(src, dst)
    cfg = json.loads((dst / "config.json").read_text())
    cfg["input_features"] = inputs
    (dst / "config.json").write_text(json.dumps(cfg))
    return dst


def test_image_policy_is_refused_naming_the_key(tiny_act: Path, tmp_path: Path, capsys) -> None:
    inputs = {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 96, 96]},
    }
    img = _copy_with_inputs(tiny_act, tmp_path / "img", inputs)
    assert cli.main(["eval", f"lerobot:{img}", "--episodes", "1", "--json-out", str(tmp_path / "x.json")]) == 1
    err = capsys.readouterr().err
    assert "observation.images.front" in err and "doesn't provide" in err and "Traceback" not in err


def test_check_inputs_without_lerobot() -> None:
    ok = {"input_features": {"observation.state": {"shape": [6]}, "observation.environment_state": {"shape": [3]}}}
    assert check_inputs(ok, "x") == ["observation.environment_state", "observation.state"]
    with pytest.raises(PolicySpecError, match="observation.images.wrist"):
        check_inputs({"input_features": {"observation.images.wrist": {"shape": [3, 64, 64]}}}, "x")
    with pytest.raises(PolicySpecError, match=r"shape \[14\]"):
        check_inputs({"input_features": {"observation.state": {"shape": [14]}}}, "x")
    with pytest.raises(PolicySpecError, match="no input_features"):
        check_inputs({}, "x")


def test_missing_lerobot_or_config_is_a_clean_error(tmp_path: Path, monkeypatch, capsys) -> None:
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    out = ["--episodes", "1", "--json-out", str(tmp_path / "x.json")]
    assert cli.main(["eval", f"lerobot:{ckpt}", *out]) == 1
    assert "no config.json" in capsys.readouterr().err
    cfg = {"type": "act", "input_features": {"observation.state": {"type": "STATE", "shape": [6]}}}
    (ckpt / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", None)
    assert cli.main(["eval", f"lerobot:{ckpt}", *out]) == 1
    err = capsys.readouterr().err
    assert 'pip install "capek-tech[lerobot]"' in err and "Traceback" not in err
