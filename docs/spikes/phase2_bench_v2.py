"""Benchmark v2 (plan D1) built in memory via the phase-1 library; cached to npz per seed set.
dev: base 1000 (mixed) / 1500 (clean-only 100);  held-out: base 5000 / 5500.  Tuning uses dev only."""
import sys, numpy as np
from capek.sim.registry import make_env
from capek.policies import ScriptedReach, NoisyPolicy, RandomPolicy
from capek.record import record_episode

env = make_env("so101_reach")

class Hesitate:  # 0.8 s pause at ~40% of the ramp, then resume
    def __init__(s, inner, pause_s=0.8): s.inner, s.pause_s = inner, pause_s
    @property
    def name(s): return s.inner.name
    def params(s): return {**s.inner.params(), "hesitate": s.pause_s}
    def reset(s, env, obs, info, rng):
        s.inner.reset(env, obs, info, rng); s.p = int(0.4 * s.inner._ramp_frames); s.n = round(s.pause_s * env.fps)
    def act(s, obs, t):
        te = t if t < s.p else (s.p if t < s.p + s.n else t - s.n)
        return s.inner.act(obs, te)

class ARNoise:  # band-limited AR(1) noise, stationary std sigma
    def __init__(s, inner, sigma, phi=0.9): s.inner, s.sigma, s.phi = inner, sigma, phi
    @property
    def name(s): return s.inner.name
    def params(s): return {**s.inner.params(), "arnoise": s.sigma}
    def reset(s, env, obs, info, rng):
        r1, s.rng = rng.spawn(2); s.inner.reset(env, obs, info, r1); s.e = np.zeros(6)
    def act(s, obs, t):
        s.e = s.phi * s.e + np.sqrt(1 - s.phi ** 2) * s.rng.normal(0, s.sigma, 6)
        return s.inner.act(obs, t) + s.e

class Group:  # every scripted-based group gets ramp ~U(1.0, 2.5) s; sigma fixed or ~U(0.01, 0.04)
    def __init__(s, kind, sigma=None): s.kind, s.sigma = kind, sigma; s.name = kind
    def params(s): return {"group": s.kind}
    def reset(s, env, obs, info, rng):
        r1, r2, r3 = rng.spawn(3)
        k = s.kind
        if k == "random":
            s.p = NoisyPolicy(RandomPolicy(), 0.0); s.p.reset(env, obs, info, r2); return
        base = ScriptedReach(ramp_s=float(r1.uniform(1.0, 2.5)), gravity_comp=(k != "nearmiss"),
                             stall_after_s=1.0 if k == "stall" else None, target_sign=-1.0 if k == "wrong" else 1.0)
        sig = s.sigma if s.sigma is not None else float(r3.uniform(0.01, 0.04))
        if k == "hesitation": base = Hesitate(base)   # schedule pauses; noise continues during the pause
        p = NoisyPolicy(base, sig)
        if k == "wobble": p = ARNoise(p, 0.10)
        s.p = p; s.p.reset(env, obs, info, r2)
    def act(s, obs, t): return s.p.act(obs, t)

MIXED = [("clean", None, 60), ("nearmiss", None, 10), ("noise005", 0.05, 10), ("noise010", 0.10, 10), ("noise025", 0.25, 10),
         ("hesitation", None, 10), ("wobble", None, 10), ("stall", None, 10), ("wrong", None, 10), ("random", None, 10)]

def build(spec, seed0):
    S, A, G, OK = [], [], [], []
    for k, (g, sig, n) in enumerate(spec):
        kind = "scripted" if g.startswith("noise") else g
        pol = Group(kind, sig)
        for i in range(n):
            e = record_episode(env, pol, seed0 + k, i, 90).arrays
            S.append(e["observation.state"]); A.append(e["action"]); G.append(g); OK.append(bool(e["next.success"][-1]))
    return np.array(S, np.float32), np.array(A, np.float32), np.array(G), np.array(OK)

if __name__ == "__main__":
    for name, s0, s1 in [("dev", 1000, 1500), ("heldout", 5000, 5500)]:
        S, A, G, OK = build(MIXED, s0); np.savez(f"bench_{name}_mixed.npz", S=S, A=A, G=G, OK=OK)
        S, A, G, OK = build([("clean", None, 100)], s1); np.savez(f"bench_{name}_cleanonly.npz", S=S, A=A, G=G, OK=OK)
        print(name, "built")
