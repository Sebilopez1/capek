"""Research spike: R-real-1 (action_tv_ratio false-flags return-to-home motions) — compare denominators.

Builds the production benchmark v2 (robot_report_card.bench generators, QA's fixed wrong/stall) for a seed set,
plus a new `return_home` group (reach out, short hold, come back to the start pose; clean motion, same noise
spread as clean), in memory, and scores it with docs/spikes/phase2_score_proto.py (numerics identical to the
P2-5 engine) under tv_mode = net (current) / max_exc / split. Reports the DoD 4 bars and the return_home flag rate.

Usage:  python phase3_tv_fix.py dev|heldout [lownoise]
  lownoise: white action noise sigma ~ U(0.002, 0.008) instead of U(0.01, 0.04) for clean/return_home/etc. —
  closer to a smooth leader-arm teleop, which is where the real R-real-1 false positive appeared (DoD 8).
"""
from __future__ import annotations

import copy, importlib.util, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("P", os.path.join(HERE, "phase2_score_proto.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

import robot_report_card.bench.generators as GEN
from robot_report_card.bench.generators import GroupPolicy
from robot_report_card.bench.spec import CLEAN_ONLY, MIXED, SEED_SETS
from robot_report_card.policies import ScriptedReach
from robot_report_card.record import record_episode
from robot_report_card.sim.registry import make_env

JUNK = ("noise010", "noise025", "random", "hesitation")


class ReturnHome:
    """Out-and-back: ramp to the target over r ~ U(0.8, 1.2) s, hold h ~ U(0.1, 0.4) s, ramp back to the start
    pose over r; white action noise sigma ~ U(0.01, 0.04) like clean. Total <= 2.8 s of the 3 s episode."""
    name = "scripted"

    def params(self):
        return {"bench_group": "return_home"}

    def reset(self, env, obs, info, rng):
        r1, r2, r3, self.noise_rng = rng.spawn(4)
        self.r = max(1, round(float(r1.uniform(0.8, 1.2)) * env.fps))
        self.h = round(float(r2.uniform(0.1, 0.4)) * env.fps)
        self.sigma = float(r3.uniform(*GEN.SIGMA))
        self.out = ScriptedReach(ramp_s=self.r / env.fps)
        self.out.reset(env, obs, info, rng)
        self.home = obs.state + env.gravity_offset(obs.state)

    def act(self, obs, t):
        if t < self.r + self.h:
            a = self.out.act(obs, min(t, self.r - 1))
        else:
            f = min(1.0, (t - self.r - self.h + 1) / self.r)
            a = self.out._goal + f * (self.home - self.out._goal)
        return a + self.noise_rng.normal(0.0, self.sigma, a.shape)


def build(base_mixed: int, base_clean: int):
    env = make_env("so101_reach")
    eps, groups = [], []
    table = [(g, n) for g, _, n, _ in MIXED] + [("return_home", 10)]
    for k, (g, n) in enumerate(table):
        pol = ReturnHome() if g == "return_home" else GroupPolicy(g)
        for i in range(n):
            a = record_episode(env, pol, base_mixed + k, i, 90).arrays
            eps.append({"observation.state": a["observation.state"].astype(np.float64), "action": a["action"].astype(np.float64)})
            groups.append(g)
    clean = []
    for i in range(CLEAN_ONLY[0][2]):
        a = record_episode(env, GroupPolicy("clean"), base_clean, i, 90).arrays
        clean.append({"observation.state": a["observation.state"].astype(np.float64), "action": a["action"].astype(np.float64)})
    return eps, np.array(groups), clean


def bars(eps, G, clean, cfg, with_rth: bool):
    keep_rows = np.ones(len(G), bool) if with_rth else (G != "return_home")
    e2, G2 = [e for e, k in zip(eps, keep_rows) if k], G[keep_rows]
    _, Z, comb = P.score_episodes(e2, 30, cfg)
    flag = comb > cfg["flag_z"]
    cl, jk = G2 == "clean", np.isin(G2, JUNK)
    tp, fp, fn = np.sum(flag & jk), np.sum(flag & cl), np.sum(~flag & jk)
    per = {g: (P.auroc(comb[cl | (G2 == g)], (G2 == g)[cl | (G2 == g)]), int(np.sum(flag & (G2 == g)))) for g in dict.fromkeys(G2) if g != "clean"}
    _, _, cc = P.score_episodes(clean, 30, cfg)
    r = dict(auroc=P.auroc(comb[cl | jk], jk[cl | jk]), prec=tp / max(tp + fp, 1), rec=tp / max(tp + fn, 1),
             clean_flag=int(fp), clean_only=float(np.mean(cc > cfg["flag_z"])))
    ok = (r["auroc"] >= .95 and r["prec"] >= .9 and r["rec"] >= .85 and r["clean_flag"] <= 3 and r["clean_only"] <= .05
          and all(per[g][1] >= 9 for g in ("noise010", "noise025", "random")) and per["hesitation"][0] >= .9 and per["nearmiss"][1] <= 2)
    rth = per.get("return_home")
    tv_rth = float(np.median(Z["action_tv_ratio"][G2 == "return_home"])) if with_rth else float("nan")
    return r, per, ok, rth, tv_rth


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if "lownoise" in sys.argv:
        GEN.SIGMA = (0.002, 0.008)
        name_tag = name + "/lownoise"
    else:
        name_tag = name
    eps, G, clean = build(SEED_SETS[name]["mixed"], SEED_SETS[name]["clean_only"])
    for mode in ("net", "max_exc", "split"):
        cfg = copy.deepcopy(P.CFG); cfg["tv_mode"] = mode
        for with_rth in (False, True):
            r, per, ok, rth, tvz = bars(eps, G, clean, cfg, with_rth)
            extra = f" | return_home AUROC={rth[0]:.2f} flagged={rth[1]}/10 median tv z={tvz:+.1f}" if with_rth else ""
            print(f"[{name_tag}] tv_mode={mode:7s} {'+rth' if with_rth else 'std '} AUROC={r['auroc']:.3f} prec={r['prec']:.2f} "
                  f"rec={r['rec']:.2f} clean_flag={r['clean_flag']}/60 clean_only={r['clean_only']:.2f} "
                  f"hes={per['hesitation'][0]:.2f}/{per['hesitation'][1]} nearmiss={per['nearmiss'][1]} "
                  f"BARS={'PASS' if ok else 'FAIL'}{extra}")
