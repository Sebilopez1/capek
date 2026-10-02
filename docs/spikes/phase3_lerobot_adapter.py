"""Research spike: plug-in adapter that runs a LeRobot (0.4.4) pretrained policy directory in so101_reach.

    adapter = LeRobotAdapter(path)          # a `.../checkpoints/NNNNNN/pretrained_model` dir (or a Hub id, if online)
    adapter.reset(); a = adapter.act(obs)   # obs: capek Observation -> np.ndarray joint targets (rad)

The capek evaluator only needs this 2-method protocol (reset per episode, act per frame); our BC MLP implements the
same protocol. Loading follows lerobot_eval: config from `config.json`, weights via `Policy.from_pretrained`,
normalization via the saved pre/post-processor pipelines (`make_pre_post_processors(cfg, pretrained_path=...)`).

Usage: python phase3_lerobot_adapter.py PRETRAINED_DIR [n_episodes]
Verified 2026-09-24 with lerobot 0.4.4, torch 2.10 CPU, a state-only ACT trained by lerobot_train for 300 steps.
"""
from __future__ import annotations

import sys, time

import numpy as np
import torch


class LeRobotAdapter:
    STATE, ENV_STATE = "observation.state", "observation.environment_state"

    def __init__(self, pretrained: str, device: str = "cpu"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(pretrained)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(pretrained, config=cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=pretrained)
        need = set(cfg.input_features)
        have = {self.STATE, self.ENV_STATE}
        if not need <= have:  # e.g. an image policy: can't run in a state-only sim
            raise ValueError(f"policy needs inputs {sorted(need - have)} that so101_reach does not provide")
        self.inputs = need

    def reset(self) -> None:
        self.policy.reset()  # clears ACT's action-chunk queue / temporal ensemble

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        batch = {}
        if self.STATE in self.inputs:
            batch[self.STATE] = torch.as_tensor(obs.state, dtype=torch.float32)[None]
        if self.ENV_STATE in self.inputs:
            batch[self.ENV_STATE] = torch.as_tensor(obs.env_state, dtype=torch.float32)[None]
        action = self.policy.select_action(self.pre(batch))
        return self.post(action)[0].cpu().numpy().astype(np.float64)


def rollout_success(adapter, n: int, base: int = 900_000, T: int = 90) -> np.ndarray:
    from capek.sim.registry import make_env
    env = make_env("so101_reach")
    out = np.zeros(n, bool)
    for i in range(n):
        obs, _ = env.reset(np.random.SeedSequence([base, i]))
        adapter.reset()
        for _ in range(T):
            res = env.step(np.clip(adapter.act(obs), env.action_low, env.action_high))
            obs = res.observation
        out[i] = res.success
    return out


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phase3_stats import wilson
    torch.set_num_threads(2)
    t0 = time.time(); ad = LeRobotAdapter(sys.argv[1]); print(f"loaded in {time.time() - t0:.1f}s; inputs={sorted(ad.inputs)}")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    t0 = time.time(); s = rollout_success(ad, n); dt = time.time() - t0
    lo, hi = wilson(int(s.sum()), n)
    print(f"{int(s.sum())}/{n} success, Wilson95=[{lo:.3f},{hi:.3f}], {dt:.1f}s ({dt / n:.2f}s/episode)")
