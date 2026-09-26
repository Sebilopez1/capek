"""Evaluate the prototype on benchmark v2 npz files (run phase2_bench_v2.py first, same cwd).
   python phase2_eval_v2.py dev heldout   |   python phase2_eval_v2.py grid   (dev-only constant sweep)"""
import importlib.util, itertools, sys, copy, numpy as np
spec = importlib.util.spec_from_file_location("P", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "phase2_score_proto.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
JUNK = ("noise010", "noise025", "random", "hesitation")

def eps_of(f):
    d = np.load(f); return [{"observation.state": s.astype(np.float64), "action": a.astype(np.float64)} for s, a in zip(d["S"], d["A"])], d["G"], d["OK"]

def evaluate(name, cfg, verbose=True):
    eps, G, OK = eps_of(f"bench_{name}_mixed.npz")
    raw, Z, comb = P.score_episodes(eps, 30, cfg)
    flag = comb > cfg["flag_z"]
    cl = G == "clean"; jk = np.isin(G, JUNK); keep = cl | jk
    tp, fp, fn = np.sum(flag & jk), np.sum(flag & cl), np.sum(~flag & jk)
    res = {"auroc": P.auroc(comb[keep], jk[keep]), "prec": tp / max(tp + fp, 1), "rec": tp / max(tp + fn, 1), "clean_flag": fp / cl.sum()}
    per = {g: (P.auroc(comb[cl | (G == g)], (G == g)[cl | (G == g)]), int(np.sum(flag & (G == g)))) for g in dict.fromkeys(G) if g != "clean"}
    ce, _, _ = eps_of(f"bench_{name}_cleanonly.npz")
    _, _, cc = P.score_episodes(ce, 30, cfg); res["cleanonly_flag"] = float(np.mean(cc > cfg["flag_z"]))
    sig_auc = {k: P.auroc(Z[k][keep], jk[keep]) for k in P.SIG7}
    bars = (res["auroc"] >= .95 and res["prec"] >= .9 and res["rec"] >= .85 and res["clean_flag"] <= .05 and res["cleanonly_flag"] <= .05
            and all(per[g][1] >= 9 for g in ("noise010", "noise025", "random")) and per["hesitation"][0] >= .9 and per["nearmiss"][1] <= 2)
    if verbose:
        print(f"[{name}] AUROC={res['auroc']:.3f} prec={res['prec']:.2f} rec={res['rec']:.2f} clean_flag(mixed)={res['clean_flag']:.3f} "
              f"clean_only_flag={res['cleanonly_flag']:.2f}  ALL BARS {'PASS' if bars else 'FAIL'}")
        print("   per group vs clean (AUROC, flagged/10):", {g: (round(a, 2), f) for g, (a, f) in per.items()})
        print("   per signal AUROC clean vs motion-junk:", {k: round(v, 2) for k, v in sig_auc.items()})
    return res, per, bars

if __name__ == "__main__":
    if sys.argv[1] == "grid":
        best = []
        for ldlj, tv, win, ri, ff, mult in itertools.product([0.25, 0.5, 1.0], [0.25, 0.5, 0.75], [0.33, 0.5], [0.1, 0.2, 0.3], [0.02, 0.03, 0.05], [1.5, 2.0, 3.0]):
            cfg = copy.deepcopy(P.CFG); cfg["abs_floor"] = {"ldlj_state": ldlj, "idle_frac": ff, "saturation_frac": ff}
            cfg.update(tv_den_frac=tv, idle_win_s=win, idle_r_idle=ri, idle_eps_mult=mult)
            r, per, ok = evaluate("dev", cfg, verbose=False)
            best.append((ok, r["rec"], per["hesitation"][1], -r["clean_flag"] - r["cleanonly_flag"], (ldlj, tv, win, ri, ff, mult), r, per["hesitation"]))
        best.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        print("passing combos on dev:", sum(b[0] for b in best), "/", len(best))
        for b in best[:12]: print(b[0], b[4], {k: round(v, 3) for k, v in b[5].items()}, "hes", b[6])
        # sensitivity: how many pass per value of each constant
        names = ["ldlj", "tv", "win", "r_idle", "ff", "mult"]
        for j, n in enumerate(names):
            vals = sorted({b[4][j] for b in best}); print(n, {v: sum(b[0] for b in best if b[4][j] == v) for v in vals})
    else:
        for name in sys.argv[1:]: evaluate(name, P.CFG)
