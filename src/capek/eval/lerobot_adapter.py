"""Run a LeRobot (0.4.4) pretrained policy directory in so101_reach: ``capek eval lerobot:<pretrained_model dir>``.

Ported from ``docs/spikes/phase3_lerobot_adapter.py``. Loading follows ``lerobot_eval``: the config from
``config.json`` (forced to CPU), weights via ``get_policy_class(cfg.type).from_pretrained``, normalization via the saved
pre/post-processor pipelines. ``reset()`` is called every episode because chunked policies (ACT) keep an action queue.
Policies that need inputs the sim doesn't provide (cameras, other state keys) are refused, naming the keys.
Needs the lerobot extra. No success level is claimed for any LeRobot policy.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from capek.eval.policies import ENV_STATE, STATE, PolicySpecError, ResetContext
from capek.hints import install_hint

AVAILABLE_INPUTS = (STATE, ENV_STATE)


def check_inputs(config: dict[str, Any], spec: str) -> list[str]:
    """Return the policy's input keys, or raise if it needs anything so101_reach can't provide."""
    features = config.get("input_features") or {}
    need = sorted(features)
    missing = [k for k in need if k not in AVAILABLE_INPUTS]
    if missing:
        raise PolicySpecError(
            f"{spec}: policy needs input(s) {', '.join(missing)} that so101_reach doesn't provide "
            f"(it has {', '.join(AVAILABLE_INPUTS)}); image and other-robot policies can't run here"
        )
    for key in need:
        shape = features[key].get("shape") if isinstance(features[key], dict) else None
        expected = 6 if key == STATE else 3
        if shape and list(shape) != [expected]:
            raise PolicySpecError(f"{spec}: input {key} has shape {shape}, so101_reach provides [{expected}]")
    if not need:
        raise PolicySpecError(f"{spec}: config.json lists no input_features")
    return need


class LeRobotPolicy:
    kind = "lerobot"

    def __init__(self, spec: str, path: Path, threads: int | None = None) -> None:
        cfg_path = path / "config.json"
        if not cfg_path.is_file():
            raise PolicySpecError(
                f"{spec}: {path} has no config.json; pass a LeRobot `.../checkpoints/<step>/pretrained_model` directory"
            )
        try:
            raw_cfg = json.loads(cfg_path.read_text())
        except ValueError as e:
            raise PolicySpecError(f"{spec}: {cfg_path} is not valid JSON: {e}") from e
        self.inputs = check_inputs(raw_cfg, spec)
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        except ImportError as e:
            raise PolicySpecError(f"{spec}: lerobot: policies need the lerobot extra: {install_hint('lerobot')}") from e
        if threads is not None:
            torch.set_num_threads(threads)
        try:
            # lerobot print()s while loading; keep stdout clean for `capek eval --json`
            with contextlib.redirect_stdout(sys.stderr):
                cfg, policy, pre, post = _load(PreTrainedConfig, get_policy_class, make_pre_post_processors, path)
        except Exception as e:  # lerobot raises many types for a bad checkpoint; report it cleanly
            raise PolicySpecError(f"{spec}: can't load LeRobot policy from {path}: {e}") from e
        self.policy, self.pre, self.post = policy, pre, post
        self.spec, self.path, self.type, self._torch = spec, path, cfg.type, torch
        self._resets = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "kind": self.kind,
            "policy_type": self.type,
            "inputs": self.inputs,
            "path": str(self.path.resolve()),
        }

    def reset(self, seed: np.random.SeedSequence, ctx: ResetContext) -> None:
        self.policy.reset()  # clears ACT's action-chunk queue / temporal ensemble
        self._resets += 1

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = {k: torch.as_tensor(obs[k], dtype=torch.float32)[None] for k in self.inputs}
        with torch.no_grad():
            action = self.policy.select_action(self.pre(batch))
            action = self.post(action)
        return action[0].cpu().numpy().astype(np.float64)


def _load(config_cls: Any, get_policy_class: Any, make_processors: Any, path: Path) -> tuple[Any, Any, Any, Any]:
    cfg = config_cls.from_pretrained(str(path))
    cfg.device = "cpu"
    policy = get_policy_class(cfg.type).from_pretrained(str(path), config=cfg)
    policy.eval()
    pre, post = make_processors(cfg, pretrained_path=str(path))
    return cfg, policy, pre, post


def load_lerobot_policy(spec: str, path: Path, threads: int | None = None) -> LeRobotPolicy:
    return LeRobotPolicy(spec, path, threads)
