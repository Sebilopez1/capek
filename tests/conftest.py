"""Test configuration: force offline Hugging Face access (research brief G7) before anything imports lerobot."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")


# ---- shared fixtures ------------------------------------------------------------------------------------------------
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

ACT_STEPS_FULL = 300  # RRC_ACT_FULL=1 repeats the research brief's 300-step run (about 1 min)


@pytest.fixture(scope="session")
def tiny_act(tmp_path_factory) -> Path:
    """A state-only ACT trained by `lerobot_train` on a small rrc export (P3-7 smoke test; not a quality claim).

    Default: 20 steps with a tiny model (~15 s). ``RRC_ACT_FULL=1``: the brief's 300-step configuration.
    Returns the ``.../checkpoints/<step>/pretrained_model`` directory. Skips when mujoco/lerobot are missing.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from robot_report_card.export.lerobot_writer import export_session, quiet_lerobot
    from robot_report_card.policies import make_policy
    from robot_report_card.record import new_session, record_into
    from robot_report_card.sim.registry import make_env

    root = tmp_path_factory.mktemp("act")
    env = make_env("so101_reach")
    session = new_session(env, root / "session", max_steps=90)
    record_into(session, env, make_policy("scripted", noise=0.02), 8, 42000, 90)
    with quiet_lerobot():
        export_session(root / "session", root / "ds", "local/act_smoke")
    full = os.environ.get("RRC_ACT_FULL") == "1"
    steps = ACT_STEPS_FULL if full else 20
    model = (
        ["--policy.chunk_size=20", "--policy.n_action_steps=20", "--policy.dim_model=128",
         "--policy.n_encoder_layers=2", "--batch_size=32"]
        if full
        else ["--policy.chunk_size=10", "--policy.n_action_steps=10", "--policy.dim_model=64",
              "--policy.n_encoder_layers=1", "--policy.n_decoder_layers=1", "--policy.dim_feedforward=128",
              "--batch_size=8"]
    )  # fmt: skip
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train", "--dataset.repo_id=local/act_smoke",
        f"--dataset.root={root / 'ds'}", "--policy.type=act", "--policy.device=cpu", "--policy.push_to_hub=false",
        *model, f"--steps={steps}", f"--save_freq={steps}", "--num_workers=0", "--eval_freq=0", "--log_freq=1000",
        "--wandb.enable=false", f"--output_dir={root / 'train'}",
    ]  # fmt: skip
    env_vars = {**os.environ, "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env_vars, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    path = root / "train" / "checkpoints" / f"{steps:06d}" / "pretrained_model"
    assert (path / "config.json").is_file()
    return path
