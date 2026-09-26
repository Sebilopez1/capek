"""P2-5: motion-quality signals and scoring engine."""

from __future__ import annotations

import importlib.util
import inspect
import re
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from lerobot_files import JOINTS, smooth_reach, write_dataset  # noqa: E402

from robot_report_card.score import engine, signals  # noqa: E402
from robot_report_card.score.engine import SIGNALS, ScoreConfig, score_dataset  # noqa: E402
from robot_report_card.score.reader import Dataset, EpisodeData, read_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0


def _mixed(seed: int = 0, clean: int = 40, junk: int = 5) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    eps = [smooth_reach(rng, 90, noise=float(rng.uniform(0.005, 0.02))) for _ in range(clean)]
    return eps + [smooth_reach(rng, 90, noise=0.2) for _ in range(junk)]


def _in_memory(eps: list[dict[str, np.ndarray]], names=None, tags=None) -> Dataset:
    episodes = [
        EpisodeData(
            index=i,
            state=np.asarray(e["observation.state"], np.float64),
            action=np.asarray(e["action"], np.float64),
            timestamp=np.asarray(e.get("timestamp", np.arange(len(e["action"])) / FPS), np.float64),
            frame_index=np.arange(len(e["action"])),
            success=e.get("next.success"),
        )
        for i, e in enumerate(eps)
    ]
    return Dataset(Path("mem"), {}, "v3.0", FPS, "observation.state", "action", names, names, episodes, tags)


# ---- parity with the Researcher's prototype -----------------------------------------------------------------
@pytest.mark.parametrize("tv_mode", ["split", "net"])
def test_matches_prototype_bit_for_bit(tmp_path: Path, tv_mode: str) -> None:
    spec = importlib.util.spec_from_file_location("proto", REPO / "docs" / "spikes" / "phase2_score_proto.py")
    proto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proto)  # type: ignore[union-attr]
    root = write_dataset(tmp_path / "ds", _mixed())
    fps, proto_eps = proto.load(str(root))
    raw, z, comb = proto.score_episodes(proto_eps, fps, {**proto.CFG, "tv_mode": tv_mode})
    res = score_dataset(read_dataset(root), ScoreConfig(tv_mode=tv_mode))
    for k in SIGNALS:
        np.testing.assert_array_equal([e.raw[k] for e in res.episodes], raw[k], err_msg=k)
        np.testing.assert_array_equal([e.z[k] for e in res.episodes], z[k], err_msg=k)
    np.testing.assert_array_equal([e.combined for e in res.episodes], comb)
    cfg = ScoreConfig()
    assert (cfg.rel_floor, cfg.tv_den_frac, cfg.idle_win_s, cfg.idle_eps_mult) == (
        proto.CFG["rel_floor"],
        proto.CFG["tv_den_frac"],
        proto.CFG["idle_win_s"],
        proto.CFG["idle_eps_mult"],
    )
    assert (cfg.idle_r_idle, cfg.idle_r_move, cfg.flag_z) == (
        proto.CFG["idle_r_idle"],
        proto.CFG["idle_r_move"],
        proto.CFG["flag_z"],
    )
    assert {k: cfg.abs_floor(k) for k in proto.CFG["abs_floor"]} == proto.CFG["abs_floor"]


# ---- signals on hand-built traces ---------------------------------------------------------------------------
T = np.arange(90) / FPS


def _sine(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    base = np.stack([np.sin(2 * np.pi * 0.5 * T + p) for p in np.linspace(0, 1, 6)], axis=1)
    return base + np.random.default_rng(seed).normal(0, noise, base.shape)


def test_sine_vs_noisy_sine() -> None:
    clean, noisy = _sine(), _sine(0.05)
    assert signals.sparc(signals.joint_speed(noisy, FPS), FPS) > signals.sparc(signals.joint_speed(clean, FPS), FPS)
    assert signals.ldlj(noisy, FPS) > signals.ldlj(clean, FPS) + 3
    assert signals.action_hf_energy(noisy) > 100 * signals.action_hf_energy(clean)
    assert signals.action_tv_ratio(noisy, 0.1) > signals.action_tv_ratio(clean, 0.1)


def _ramp_hold(ramp_frames: int, pause: tuple[int, int] | None = None) -> np.ndarray:
    t = np.arange(90, dtype=float)
    if pause:  # hold between pause[0] and pause[1], then resume the ramp
        start, stop = pause
        t = np.where(t < start, t, np.where(t < stop, start, t - (stop - start)))
    return np.clip(t / ramp_frames, 0, 1)[:, None] * np.linspace(0.5, 1.0, 6)[None]


def test_terminal_hold_is_not_idle_but_a_pause_is() -> None:
    kw = {"fps": FPS, "win_s": 0.5, "eps_jitter": 0.02, "r_idle": 0.2, "r_move": 0.5}
    assert signals.idle_before_last_motion(_ramp_hold(30), **kw) == 0.0  # ramp then 2 s hold
    paused = signals.idle_before_last_motion(_ramp_hold(40, pause=(16, 40)), **kw)
    assert paused > 0.1


def test_stall_plateau_and_never_moving() -> None:
    kw = {"fps": FPS, "win_s": 0.5, "eps_jitter": 0.02, "r_idle": 0.2, "r_move": 0.5}
    stall = _ramp_hold(60)
    stall[30:] = stall[30]  # stops at half way and holds: a terminal hold, so motion quality looks fine
    assert signals.idle_before_last_motion(stall, **kw) == 0.0
    assert signals.idle_before_last_motion(np.zeros((90, 6)), **kw) == 1.0


def test_saturation_and_tracking() -> None:
    a = np.linspace(-1, 1, 90)[:, None].repeat(6, 1)
    lo, hi = np.full(6, -1.0), np.full(6, 1.0)
    base = signals.saturation_frac(a, lo, hi)
    pinned = a.copy()
    pinned[:45] = -1.0
    assert signals.saturation_frac(pinned, lo, hi) > 0.45 > base
    s = np.vstack([np.zeros((1, 6)), a[:-1]])
    assert signals.track_err(s, a) == pytest.approx(0.0, abs=1e-12)
    assert signals.track_err(s, a + 0.3) == pytest.approx(0.3)


# ---- hard flags ------------------------------------------------------------------------------------------------
def test_hard_flags_each_fixture_gets_the_right_flag() -> None:
    eps = _mixed(1, clean=20, junk=0)
    eps[1] = {**eps[1], "observation.state": eps[1]["observation.state"].copy()}
    eps[1]["observation.state"][10, 2] = np.nan
    eps[2] = {**eps[2], "observation.state": eps[2]["observation.state"].copy()}
    eps[2]["observation.state"][:, 3] = 0.25  # frozen joint
    eps[3] = {**eps[3], "timestamp": np.r_[np.arange(40), np.arange(41, 91)] / FPS}  # dropped frame
    sat = eps[4]["action"].copy()
    sat[:40] = np.max([e["action"].max(0) for e in eps], axis=0)  # pinned at the dataset-wide limits
    eps[4] = {**eps[4], "action": sat}
    eps[5] = {k: v[:3] for k, v in eps[5].items()}  # too short
    res = score_dataset(_in_memory(eps, names=JOINTS))
    by = {e.episode_index: e for e in res.episodes}
    assert by[1].quality == "HARD" and by[1].hard_flags[0].startswith("non-finite") and by[1].combined is None
    assert by[2].quality == "HARD" and any("frozen joint: wrist_flex.pos" in h for h in by[2].hard_flags)
    assert by[3].quality == "HARD" and any(h.startswith("timestamp gap") for h in by[3].hard_flags)
    assert by[4].quality == "HARD" and any(h.startswith("saturation") for h in by[4].hard_flags)
    assert by[5].quality == "HARD" and by[5].hard_flags[0].startswith("too short")
    assert all(by[i].hard_flags == [] for i in range(6, 20))


def test_non_finite_episode_does_not_poison_the_others() -> None:
    eps = _mixed(2, clean=20, junk=2)
    base = score_dataset(_in_memory(eps))
    bad = {**eps[0], "action": np.full_like(eps[0]["action"], np.nan)}
    with_nan = score_dataset(_in_memory([bad, *eps]))
    assert with_nan.episodes[0].quality == "HARD"
    for a, b in zip(base.episodes, with_nan.episodes[1:], strict=True):
        assert a.combined == b.combined and a.quality == b.quality


# ---- outcome axis (D3 / D4) --------------------------------------------------------------------------------
def test_stall_without_outcome_evidence_is_ok_and_unknown() -> None:
    rng = np.random.default_rng(4)
    eps = [smooth_reach(rng, 90, noise=0.01) for _ in range(30)]
    eps = [{k: v for k, v in e.items() if k != "next.success"} for e in eps]
    stall = smooth_reach(rng, 90, noise=0.01)
    stall["action"][30:] = stall["action"][30]  # stops early and holds
    stall["observation.state"][31:] = stall["observation.state"][31]
    del stall["next.success"]
    res = score_dataset(_in_memory([*eps, stall]))
    e = res.episodes[-1]
    assert e.quality == "ok" and e.outcome == ("unknown", "none") and e.outcome_sim is None
    assert engine.HONESTY_STATEMENT in res.summary()["honesty_statement"]


def test_label_first_then_sim_and_disagreement() -> None:
    eps = _mixed(5, clean=12, junk=0)
    tags = {"episodes": {"0": {"label": "fail"}, "1": {"label": "unlabeled"}, "2": {"label": "success"}}}
    eps[2] = {**eps[2], "next.success": np.zeros(90, bool)}
    res = score_dataset(_in_memory(eps, tags=tags))
    e0, e1, e2 = res.episodes[:3]
    assert (e0.outcome_label, e0.outcome_sim, e0.outcome, e0.outcome_disagree) == (
        "fail",
        True,
        ("fail", "label"),
        True,
    )
    assert (e1.outcome_label, e1.outcome, e1.outcome_disagree) == (None, ("success", "sim"), False)
    assert (e2.outcome_label, e2.outcome_sim, e2.outcome_disagree) == ("success", False, True)
    assert res.summary()["outcome_disagreements"] == 2


def test_motion_reasons_never_claim_outcome() -> None:
    cfg = ScoreConfig()
    texts = [engine.reason_text(k, 5.0, 1.0, 12.0, cfg) for k in SIGNALS]
    res = score_dataset(_in_memory(_mixed(6, clean=30, junk=8)))
    texts += [r for e in res.episodes for r in e.reasons]
    assert any(e.quality == "FLAG" for e in res.episodes)
    for t in texts:
        assert not re.search(r"fail|success", t, re.I), t
    assert not re.search(r"fail|success", inspect.getsource(engine.reason_text), re.I)


# ---- unit invariance (DoD 6) and variable lengths --------------------------------------------------------------
def _realify(x: np.ndarray) -> np.ndarray:
    y = np.degrees(x)
    y[:, 5] = (x[:, 5] + 1.0) * 50.0  # gripper to 0-100
    return y.astype(np.float32)


def test_units_names_float32_give_identical_flags_and_reasons(tmp_path: Path) -> None:
    eps = _mixed(7, clean=40, junk=6)
    src = score_dataset(read_dataset(write_dataset(tmp_path / "src", eps)))
    real = [{**e, "observation.state": _realify(e["observation.state"]), "action": _realify(e["action"])} for e in eps]
    names = [
        "main_shoulder_pan",
        "main_shoulder_lift",
        "main_elbow_flex",
        "main_wrist_flex",
        "main_wrist_roll",
        "main_gripper",
    ]
    dst = score_dataset(read_dataset(write_dataset(tmp_path / "real", real, names=names)))
    assert [e.quality for e in dst.episodes] == [e.quality for e in src.episodes]
    assert [e.reasons for e in dst.episodes] == [e.reasons for e in src.episodes]
    assert sum(e.quality != "ok" for e in src.episodes) >= 5
    for a, b in zip(src.episodes, dst.episodes, strict=True):
        for k in SIGNALS:
            np.testing.assert_allclose(b.raw[k], a.raw[k], rtol=1e-4, atol=1e-9, err_msg=k)
            np.testing.assert_allclose(b.z[k], a.z[k], rtol=1e-4, atol=1e-4, err_msg=k)


def test_variable_lengths_score_without_error(tmp_path: Path) -> None:
    eps = _mixed(8, clean=30, junk=4)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(eps), size=len(eps) // 5, replace=False):
        n = int(rng.integers(30, 80))
        eps[i] = {k: v[:n] for k, v in eps[i].items()}
    res = score_dataset(read_dataset(write_dataset(tmp_path / "var", eps)))
    assert len({e.length for e in res.episodes}) > 1
    assert all(e.combined is not None and np.isfinite(e.combined) for e in res.episodes)
    assert all(e.length_z is not None for e in res.episodes)


# ---- dataset-level behaviour ---------------------------------------------------------------------------------------
def test_track_err_skipped_when_dims_or_names_differ() -> None:
    eps = _mixed(9, clean=12, junk=0)
    short = [{**e, "action": e["action"][:, :5]} for e in eps]
    res = score_dataset(_in_memory(short))
    assert "track_err" in res.skipped and "track_err" not in res.signals_used
    assert all(e.raw["track_err"] is None and e.z["track_err"] is None for e in res.episodes)
    ds = _in_memory(eps, names=JOINTS)
    ds.action_names = [f"cmd_{j}" for j in JOINTS]
    assert "names differ" in score_dataset(ds).skipped["track_err"]


def test_majority_and_few_episode_warnings() -> None:
    summary = score_dataset(_in_memory(_mixed(10, clean=40, junk=3))).summary()
    assert summary["warnings"] == [] and 0 < summary["flagged"] < 10
    few = score_dataset(_in_memory(_mixed(11, clean=4, junk=0))).summary()
    assert any("Only 4 episodes" in w for w in few["warnings"])
    everything = score_dataset(_in_memory(_mixed(12, clean=20, junk=0)), ScoreConfig(flag_z=-1e9)).summary()
    assert everything["flagged_frac"] == 1.0 and engine.MAJORITY_WARNING in everything["warnings"]


def test_150_episode_benchmark_sized_dataset_scores_fast(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    eps = [smooth_reach(rng, 90, noise=float(rng.uniform(0.005, 0.03))) for _ in range(150)]
    root = write_dataset(tmp_path / "big", eps, files=1)
    t0 = time.perf_counter()
    res = score_dataset(read_dataset(root))
    assert len(res.episodes) == 150 and time.perf_counter() - t0 < 5.0
