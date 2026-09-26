"""Research spike (not production code): phase 3 tiny behavior-cloning policy + paired evaluation.

Pipeline (all CPU, no Hub):
  1. record sessions with the phase 1 library + bench v2 generators (clean / junk groups),
  2. `export_session` -> LeRobot v3.0, read back with pyarrow (state, environment_state, action),
  3. score the mixed dataset with the phase 2 engine and drop flagged episodes ("filtered"),
  4. train an MLP  [state(6), target xyz(3)] -> action - state (6)  with torch on CPU,
  5. evaluate every checkpoint on the SAME eval seeds in so101_reach (paired), success = final tip error < 2 cm,
  6. report success with Wilson 95% CIs and paired McNemar exact tests (phase3_stats.py).

Usage:  python phase3_bc_eval.py WORKDIR            (WORKDIR is created; reruns reuse the exports)
Verified 2026-09-24: Python 3.11, torch 2.10 (CPU, 2 threads), mujoco 3.14, lerobot 0.4.4 (export only).
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_stats import mcnemar_exact, newcombe_paired_diff, wilson  # noqa: E402

from robot_report_card.bench.generators import GroupPolicy
from robot_report_card.record import new_session, record_into
from robot_report_card.sim.registry import make_env

FPS, T = 30, 90
CLEAN_N = 200
JUNK = {"noise025": 30, "random": 30, "hesitation": 20, "wrong": 20}  # 100 junk episodes
EVAL_SEEDS = 200
TRAIN_SEEDS = int(os.environ.get("TRAIN_SEEDS", 3))


def build(work: Path) -> dict[str, Path]:
    """Two exports: clean-only (200) and mixed (same 200 clean + 100 junk)."""
    from robot_report_card.export.lerobot_writer import export_session, quiet_lerobot
    env = make_env("so101_reach")
    out = {}
    for name, groups in [("clean", {"clean": CLEAN_N}), ("mixed", {"clean": CLEAN_N, **JUNK})]:
        root = work / name / "dataset"
        if not root.exists():
            s = new_session(env, work / name / "session", max_steps=T)
            for k, (g, n) in enumerate(groups.items()):
                record_into(s, env, GroupPolicy(g), n, 20000 + k, T)  # same clean seeds in both sets
            with quiet_lerobot():
                export_session(work / name / "session", root, f"rrc_p3/{name}")
        out[name] = root
    return out


def load(root: Path, keep: set[int] | None = None):
    import glob, pyarrow.parquet as pq
    t = pq.ParquetDataset(sorted(glob.glob(f"{root}/data/*/*.parquet"))).read(
        columns=["observation.state", "observation.environment_state", "action", "episode_index", "frame_index"])
    col = lambda c: np.asarray(t.column(c).to_pylist(), dtype=np.float32)
    s, e, a, ep = col("observation.state"), col("observation.environment_state"), col("action"), col("episode_index").astype(int)
    m = np.ones(len(ep), bool) if keep is None else np.isin(ep, sorted(keep))
    return np.concatenate([s, e], 1)[m], (a - s)[m], len(np.unique(ep[m]))


def scorer_keep(root: Path) -> set[int]:
    from robot_report_card.score.engine import score_dataset
    from robot_report_card.score.reader import read_dataset
    res = score_dataset(read_dataset(root))
    return {e.episode_index for e in res.episodes if e.quality == "ok"}


def outcome_keep(root: Path) -> set[int]:
    tags = json.loads((root / "meta" / "rrc_tags.json").read_text())
    return {int(k) for k, v in tags["episodes"].items() if v["sim_success"]}


class MLP(torch.nn.Module):
    def __init__(self, x_mu, x_sd, y_mu, y_sd, hidden=256):
        super().__init__()
        for n, v in dict(x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd).items():
            self.register_buffer(n, torch.as_tensor(v, dtype=torch.float32))
        self.net = torch.nn.Sequential(torch.nn.Linear(9, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.ReLU(), torch.nn.Linear(hidden, 6))

    def forward(self, x):  # raw obs -> raw delta action
        return self.net((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu


def train(X, Y, epochs=40, seed=0, bs=256, lr=1e-3):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = MLP(X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    ysd = model.y_sd
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = (((model(Xt[idx]) - Yt[idx]) / ysd) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, seeds=EVAL_SEEDS, base=900_000) -> np.ndarray:
    """Closed-loop rollouts on fixed seeds; returns a bool success vector (paired across checkpoints)."""
    env = make_env("so101_reach")
    out = np.zeros(seeds, bool)
    for i in range(seeds):
        obs, _ = env.reset(np.random.SeedSequence([base, i]))
        for _ in range(T):
            x = torch.as_tensor(np.concatenate([obs.state, obs.env_state])[None], dtype=torch.float32)
            a = obs.state + model(x)[0].numpy().astype(np.float64)
            res = env.step(np.clip(a, env.action_low, env.action_high))
            obs = res.observation
        out[i] = res.success
    return out


def main(work: Path):
    torch.set_num_threads(2)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); roots = build(work); print(f"build/export: {time.time() - t0:.0f} s")
    keep_score = scorer_keep(roots["mixed"])
    keep_outc = outcome_keep(roots["mixed"])
    n_mixed = CLEAN_N + sum(JUNK.values())
    junk_idx = set(range(CLEAN_N, n_mixed))
    print(f"scorer kept {len(keep_score)}/{n_mixed}: dropped clean {CLEAN_N - len(keep_score - junk_idx)}, "
          f"kept junk {len(keep_score & junk_idx)}/{len(junk_idx)}")
    variants = {
        "A clean (200)": (roots["clean"], None),
        "B mixed (300)": (roots["mixed"], None),
        "C mixed, scorer-filtered": (roots["mixed"], keep_score),
        "D mixed, sim-success-filtered": (roots["mixed"], keep_outc),
        "E mixed, scorer-ok AND success": (roots["mixed"], keep_score & keep_outc),
    }
    tags = json.loads((roots["mixed"] / "meta" / "rrc_tags.json").read_text())["episodes"]
    grp = {int(k): v["policy_params"].get("bench_group", "?") for k, v in tags.items()}
    for label, keep in [("scorer", keep_score), ("sim-success", keep_outc), ("both", keep_score & keep_outc)]:
        comp = {}
        for i in sorted(keep):
            comp[grp[i]] = comp.get(grp[i], 0) + 1
        print(f"  kept by {label}: {comp}")
    results, per_seed = {}, {}
    for name, (root, keep) in variants.items():
        X, Y, n_ep = load(root, keep)
        runs = []
        for seed in range(TRAIN_SEEDS):
            t1 = time.time(); model = train(X, Y, seed=seed); tt = time.time() - t1
            t2 = time.time(); succ = evaluate(model); te = time.time() - t2
            runs.append(succ)
            torch.save(model.state_dict(), work / f"ckpt_{name.split()[0]}_s{seed}.pt")
        results[name] = runs[0]
        per_seed[name] = [int(r.sum()) for r in runs]
        pooled = np.concatenate(runs); k, n = int(pooled.sum()), len(pooled); lo, hi = wilson(k, n)
        print(f"{name:32s} eps={n_ep:3d} frames={len(X):6d} train={tt:5.1f}s eval={te:5.1f}s "
              f"success per train seed={[f'{v}/{EVAL_SEEDS}' for v in per_seed[name]]} "
              f"pooled={k / n:.3f} Wilson95(rollouts only)=[{lo:.3f},{hi:.3f}]")
    names = list(results)
    for a, b in [(names[0], names[1]), (names[1], names[2]), (names[1], names[3]), (names[0], names[2]), (names[0], names[4]), (names[2], names[4])]:
        x, y = results[a], results[b]
        b01, b10 = int(np.sum(x & ~y)), int(np.sum(~x & y))
        d, (dlo, dhi) = newcombe_paired_diff(x, y)
        print(f"  [train seed 0] {a.split()[0]} vs {b.split()[0]}: diff={d:+.3f} Newcombe95=[{dlo:+.3f},{dhi:+.3f}] "
              f"discordant A-only={b01} B-only={b10} McNemar exact p={mcnemar_exact(b01, b10):.4f}")
    np.savez(work / "eval_results.npz", **{n.split()[0]: v for n, v in results.items()})
    json.dump(per_seed, open(work / "per_seed.json", "w"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
