"""Research spike (not production code): phase 2 per-episode motion-quality signals (plan D5, 7 signals).
Reads a LeRobot v3.0 dataset with pyarrow + numpy only (no lerobot/torch); normalisation is data-derived
(per-joint q01-q99 of observation.state, per-joint min/max of action), so it also works without stats.json.
Constants in CFG were chosen on benchmark-v2 DEV seeds only (docs/dev/phase2-research-brief.md §4).
Run:  python phase2_score_proto.py <dataset_root>        -> prints flagged episodes with reasons
Verified 2026-09-24, Python 3.11, numpy 2.3.5, pyarrow 25.0.1.
"""
import glob, json, sys
import numpy as np

CFG = {
    "rel_floor": 0.10,        # MAD floor for ratio-scale signals: >= 10% of |median|
    "abs_floor": {"ldlj_state": 0.5, "idle_frac": 0.03, "saturation_frac": 0.03},  # absolute MAD floors
    "tv_den_frac": 0.5,       # action_tv_ratio denominator floor = 0.5 x dataset median net displacement
    "idle_win_s": 0.5,        # net-motion window
    "idle_eps_mult": 2.0,     # jitter eps = max(0.02, mult x dataset jitter floor)
    "idle_r_idle": 0.2,       # a frame is idle if net motion < r_idle x episode p95 net motion
    "idle_r_move": 0.5,       # "last motion" = last frame with net motion >= r_move x episode p95
    "flag_z": 3.5,
    "tv_mode": "net",         # action_tv_ratio denominator: "net" (phase 2) | "max_exc" | "split" (phase 3 R-real-1)
}
SIG7 = ["sparc_state", "ldlj_state", "action_tv_ratio", "action_hf_energy", "idle_frac", "saturation_frac", "track_err"]
REL = {"sparc_state", "action_tv_ratio", "action_hf_energy", "track_err"}


def load(root):
    """-> fps, list of episodes {'observation.state','action'} (float64, frame order)."""
    import pyarrow.parquet as pq
    info = json.load(open(f"{root}/meta/info.json"))
    if not info["codebase_version"].startswith("v3"):
        raise SystemExit(f"unsupported codebase_version {info['codebase_version']}")
    cols = ["observation.state", "action", "frame_index", "episode_index"]
    t = pq.ParquetDataset(sorted(glob.glob(f"{root}/data/*/*.parquet"))).read(columns=cols)
    d = {c: np.asarray(t.column(c).to_pylist(), dtype=np.float64) for c in cols}
    o = np.lexsort((d["frame_index"], d["episode_index"]))
    eps = []
    for e in np.unique(d["episode_index"]):
        m = d["episode_index"][o] == e
        eps.append({"observation.state": d["observation.state"][o][m], "action": d["action"][o][m]})
    return info["fps"], eps


def sparc(speed, fps, fc=10.0, amp_th=0.05, pad=4):
    """Spectral arc length (Balasubramanian et al. 2012/2015); returns the (positive) arc length."""
    n = int(2 ** (np.ceil(np.log2(len(speed))) + pad))
    f = np.arange(n) * fps / n
    M = np.abs(np.fft.fft(speed, n)); M = M / (M.max() + 1e-12)
    sel = f <= fc; f, M = f[sel], M[sel]
    idx = np.nonzero(M >= amp_th)[0]
    if len(idx) < 2: return 0.0
    f, M = f[: idx[-1] + 1], M[: idx[-1] + 1]
    df, dM = np.diff(f) / (f[-1] - f[0] + 1e-12), np.diff(M)
    return float(np.sum(np.sqrt(df ** 2 + dM ** 2)))


def net_motion(s, fps, win_s):
    w = max(1, int(round(win_s * fps)))
    return np.linalg.norm(s[w:] - s[:-w], axis=1) if len(s) > w else np.zeros(1)


def idle_before_last_motion(s, fps, win_s, eps_jitter, r_idle, r_move):
    """Fraction of frames that are idle *before* the episode's last substantial motion (terminal hold excluded).
    Thresholds are relative to the episode's own p95 net motion, so short/slow reaches are not called idle,
    and floored by the dataset jitter level so noise around a fixed point is not called motion."""
    net = net_motion(s, fps, win_s)
    pk = np.percentile(net, 95)
    if pk < eps_jitter: return 1.0                                        # never really moves
    moving = np.nonzero(net >= max(eps_jitter, r_move * pk))[0]
    last = moving[-1]
    return float(np.sum(net[:last] < max(eps_jitter, r_idle * pk)) / len(net))


def score_episodes(eps, fps, cfg=CFG):
    S = np.concatenate([e["observation.state"] for e in eps]); A = np.concatenate([e["action"] for e in eps])
    scale = np.maximum(np.percentile(S, 99, axis=0) - np.percentile(S, 1, axis=0), 1e-6)
    lo, hi = A.min(0), A.max(0); rng_a = np.maximum(hi - lo, 1e-9)
    ns = [e["observation.state"] / scale for e in eps]; na = [e["action"] / scale for e in eps]
    # jitter floor from the quietest episodes (p25 across episodes of each episode's p10 net motion): robust to junk-heavy mixes
    floor = np.percentile([np.percentile(net_motion(s, fps, cfg["idle_win_s"]), 10) for s in ns], 25)
    eps_idle = max(0.02, cfg["idle_eps_mult"] * floor)
    def tv_den(a):
        mode = cfg.get("tv_mode", "net")
        if mode == "net":
            return np.abs(a[-1] - a[0]).sum()
        exc = np.abs(a - a[0]).sum(1)
        if mode == "max_exc":
            return exc.max()
        k = int(np.argmax(exc))                 # "split": out-and-back = |a_k - a_0| + |a_T - a_k| >= |a_T - a_0|
        return exc[k] + np.abs(a[-1] - a[k]).sum()
    disp_med = np.median([tv_den(a) for a in na])
    dt = 1.0 / fps; rows = []
    for e, s, a in zip(eps, ns, na):
        T = len(s); v = np.diff(s, axis=0) / dt; speed = np.linalg.norm(v, axis=1)
        jerk = np.diff(s, n=3, axis=0) / dt ** 3; vpk = speed.max() + 1e-9
        ldlj = -np.log((T * dt) ** 3 / vpk ** 2 * np.sum(jerk ** 2) * dt + 1e-12)
        raw_a = e["action"]
        rows.append({
            "sparc_state": sparc(speed, fps),
            "ldlj_state": -ldlj,
            "action_tv_ratio": np.abs(np.diff(a, axis=0)).sum() / max(tv_den(a), cfg["tv_den_frac"] * disp_med, 1e-6),
            "action_hf_energy": float(np.mean(np.diff(a, n=2, axis=0) ** 2)),
            "idle_frac": idle_before_last_motion(s, fps, cfg["idle_win_s"], eps_idle, cfg["idle_r_idle"], cfg["idle_r_move"]),
            "saturation_frac": float(np.mean((raw_a <= lo + 0.01 * rng_a) | (raw_a >= hi - 0.01 * rng_a))),
            "track_err": float(np.mean(np.abs(a[:-1] - s[1:]))),
        })
    raw = {k: np.array([r[k] for r in rows]) for k in SIG7}
    Z = {}
    for k, x in raw.items():
        med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med))
        fl = cfg["rel_floor"] * abs(med) if k in REL else 0.0
        Z[k] = (x - med) / max(mad, fl, cfg["abs_floor"].get(k, 0.0), 1e-9)
    comb = np.max([Z[k] for k in SIG7], axis=0)
    return raw, Z, comb


def auroc(scores, pos):
    s, y = np.asarray(scores), np.asarray(pos, bool)
    p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float(np.mean(p[:, None] > n[None]) + 0.5 * np.mean(p[:, None] == n[None]))


if __name__ == "__main__":
    fps, eps = load(sys.argv[1])
    raw, Z, comb = score_episodes(eps, fps)
    for i in np.argsort(-comb):
        if comb[i] <= CFG["flag_z"]: break
        def say(k):
            med = np.median(raw[k])
            rel = f"{100 * raw[k][i]:.0f}% of frames (median {100 * med:.0f}%)" if k in ("idle_frac", "saturation_frac") \
                else f"{raw[k][i] / med:.1f}x median" if med > 1e-12 else f"{raw[k][i]:.3g}"
            return f"{k} z={min(Z[k][i], 10):.1f} ({rel})"   # z capped at 10 for display
        why = [say(k) for k in SIG7 if Z[k][i] > 3]
        print(f"ep {i}: FLAG  " + "; ".join(why))
    print(f"{int(np.sum(comb > CFG['flag_z']))}/{len(eps)} flagged")
