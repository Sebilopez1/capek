"""Scoring engine: data-derived normalization, per-dataset robust z, max-z combined score, hard flags, reasons,
and the two outcome fields (plan D4/D5; research brief §1, constants in ``ScoreConfig`` = brief ``CFG``).

The score is *relative to the dataset*: it assumes most episodes are good. Motion quality and outcome are kept
separate: motion reasons describe how an episode moves and never claim it failed or succeeded.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from capek.score import signals as sg
from capek.score.reader import Dataset

SIGNALS: tuple[str, ...] = (
    "sparc_state",
    "ldlj_state",
    "action_tv_ratio",
    "action_hf_energy",
    "idle_frac",
    "saturation_frac",
    "track_err",
)
RATIO_SIGNALS = frozenset({"sparc_state", "action_tv_ratio", "action_hf_energy", "track_err"})
FRACTION_SIGNALS = frozenset({"idle_frac", "saturation_frac"})
FROZEN_CMD_FRAC = 0.01  # a flat reading is "frozen" only if the command moves > 1% of the joint's range
FEW_EPISODES = 10

HONESTY_STATEMENT = (
    "Motion quality can't detect failed attempts that move normally (early stop, wrong goal). "
    "Only outcome evidence (success column or labels) can."
)
RELATIVE_ASSUMPTION = (
    "Scores are relative to this dataset and assume most episodes are good. "
    "If every episode shares the same problem (for example a jittery leader arm), few or none will be flagged."
)
MAJORITY_WARNING = (
    "More than 50% of episodes are flagged: scores assume most episodes are good, so treat them with care."
)


@dataclass(frozen=True)
class ScoreConfig:
    """Constants chosen on benchmark-v2 dev seeds only (research brief §1 ``CFG``)."""

    flag_z: float = 3.5  # combined score above this -> FLAG (``--threshold``)
    reason_z: float = 3.0  # signals above this are listed in "why"
    display_cap: float = 10.0  # z shown in text is capped here
    rel_floor: float = 0.10  # MAD floor for ratio-scale signals: 10% of |median|
    abs_floor_ldlj: float = 0.5
    abs_floor_fraction: float = 0.03  # idle_frac, saturation_frac
    abs_floor_log_length: float = 0.1  # length_z: a 10% length change is at most 1 robust sd (R-real-2)
    tv_den_frac: float = 0.5  # action_tv_ratio denominator floor = 0.5 x dataset median displacement
    tv_mode: str = "split"  # action_tv_ratio displacement: "split" (phase 3, R-real-1) or "net" (phase 2)
    idle_win_s: float = 0.5
    idle_eps_mult: float = 2.0
    idle_r_idle: float = 0.2
    idle_r_move: float = 0.5
    hard_saturation: float = 0.20
    timestamp_tol_s: float = 1e-4

    def min_frames(self, fps: float) -> int:
        """Shortest scorable episode: the idle window plus 2 frames (jerk, SPARC and idle need that much)."""
        return max(1, int(round(self.idle_win_s * fps))) + 2

    def abs_floor(self, signal: str) -> float:
        if signal == "ldlj_state":
            return self.abs_floor_ldlj
        if signal == "length":
            return self.abs_floor_log_length
        return self.abs_floor_fraction if signal in FRACTION_SIGNALS else 0.0


@dataclass
class EpisodeScore:
    episode_index: int
    length: int
    raw: dict[str, float | None]
    z: dict[str, float | None]
    combined: float | None  # max z over scored signals (None when the episode can't be scored)
    quality: str  # "ok" | "FLAG" | "HARD"
    hard_flags: list[str]
    reasons: list[str]
    outcome_sim: bool | None  # last-frame next.success
    outcome_label: str | None  # "success" | "fail" from capek_tags.json ("unlabeled" -> None)
    outcome_disagree: bool
    length_z: float | None = None  # information only, not in the combined score

    @property
    def outcome(self) -> tuple[str, str]:
        """(value, source) for display: label first, then sim, else unknown."""
        if self.outcome_label is not None:
            return self.outcome_label, "label"
        if self.outcome_sim is not None:
            return ("success" if self.outcome_sim else "fail"), "sim"
        return "unknown", "none"


@dataclass
class ScoreResult:
    episodes: list[EpisodeScore]
    config: ScoreConfig
    signals_used: list[str]
    skipped: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        n = len(self.episodes)
        flagged = [e for e in self.episodes if e.quality != "ok"]
        reasons: Counter[str] = Counter()
        for e in flagged:
            reasons.update("hard: " + h.split(":")[0] for h in e.hard_flags)
            reasons.update(k for k in self.signals_used if (e.z.get(k) or 0) > self.config.reason_z)
        frac = len(flagged) / n if n else 0.0
        outcomes = Counter(f"{v} ({src})" for v, src in (e.outcome for e in self.episodes))
        warnings = [MAJORITY_WARNING] if frac > 0.5 else []
        if n < FEW_EPISODES:
            warnings.append(f"Only {n} episodes: robust z-scores need a larger dataset to be meaningful.")
        return {
            "episodes": n,
            "flagged": len(flagged),
            "flagged_frac": frac,
            "hard": sum(e.quality == "HARD" for e in self.episodes),
            "top_reasons": reasons.most_common(5),
            "outcomes": dict(sorted(outcomes.items())),
            "outcome_disagreements": sum(e.outcome_disagree for e in self.episodes),
            "warnings": warnings,
            "assumption": RELATIVE_ASSUMPTION,
            "honesty_statement": HONESTY_STATEMENT,
        }


# ---- numeric core (identical to docs/spikes/phase2_score_proto.py score_episodes) ---------------------------
def robust_z(x: np.ndarray, signal: str, cfg: ScoreConfig) -> tuple[np.ndarray, float]:
    """(x - median) / max(1.4826 MAD, floors). Returns z and the median."""
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    rel = cfg.rel_floor * abs(med) if signal in RATIO_SIGNALS else 0.0
    return (x - med) / max(mad, rel, cfg.abs_floor(signal), 1e-9), med


def compute_signals(
    states: list[np.ndarray], actions: list[np.ndarray], fps: float, cfg: ScoreConfig, use_track_err: bool = True
) -> dict[str, np.ndarray]:
    """Raw signals per episode (float64 arrays). Actions share the state's per-joint scale when dims match."""
    s_all, a_all = np.concatenate(states), np.concatenate(actions)
    scale = np.maximum(np.percentile(s_all, 99, axis=0) - np.percentile(s_all, 1, axis=0), 1e-6)
    if a_all.shape[1] == s_all.shape[1]:
        a_scale = scale
    else:  # different action space: normalize it by its own q01-q99 range
        a_scale = np.maximum(np.percentile(a_all, 99, axis=0) - np.percentile(a_all, 1, axis=0), 1e-6)
    lo, hi = a_all.min(0), a_all.max(0)
    ns = [s / scale for s in states]
    na = [a / a_scale for a in actions]
    jitter = np.percentile([np.percentile(sg.net_motion(s, fps, cfg.idle_win_s), 10) for s in ns], 25)
    eps_idle = max(0.02, cfg.idle_eps_mult * jitter)
    disp_med = np.median([sg.tv_denominator(a, cfg.tv_mode) for a in na])
    rows: dict[str, list[float]] = {k: [] for k in SIGNALS}
    for raw_a, s, a in zip(actions, ns, na, strict=True):
        rows["sparc_state"].append(sg.sparc(sg.joint_speed(s, fps), fps))
        rows["ldlj_state"].append(sg.ldlj(s, fps))
        rows["action_tv_ratio"].append(sg.action_tv_ratio(a, cfg.tv_den_frac * disp_med, cfg.tv_mode))
        rows["action_hf_energy"].append(sg.action_hf_energy(a))
        rows["idle_frac"].append(
            sg.idle_before_last_motion(s, fps, cfg.idle_win_s, eps_idle, cfg.idle_r_idle, cfg.idle_r_move)
        )
        rows["saturation_frac"].append(sg.saturation_frac(raw_a, lo, hi))
        rows["track_err"].append(sg.track_err(s, a) if use_track_err else math.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}


# ---- hard flags ----------------------------------------------------------------------------------------------
def hard_flags(
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    fps: float,
    varying_joints: np.ndarray,
    joint_names: list[str],
    cfg: ScoreConfig,
    action_range: np.ndarray | None = None,
) -> list[str]:
    """Absolute checks for corrupt recordings. ``varying_joints``: joints whose reading moves somewhere in the
    dataset; ``action_range``: dataset per-joint action range (None when action dims don't match state)."""
    flags = []
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(action)) and np.all(np.isfinite(timestamp))):
        flags.append("non-finite: NaN/inf values in state, action or timestamps")
        return flags
    need = cfg.min_frames(fps)
    if len(state) < need:
        flags.append(
            f"too short: {len(state)} frames ({len(state) / fps:.2f} s); "
            f"need at least {need} ({need / fps:.2f} s) to score motion"
        )
    if action_range is not None and len(state) > 1:
        # a dead motor / stuck encoder: the reading is flat while the command moves. A joint that is simply
        # unused (flat reading AND flat command, e.g. a gripper held still) is not flagged.
        commanded = np.ptp(action, axis=0) > FROZEN_CMD_FRAC * action_range
        stuck = (np.ptp(state, axis=0) == 0) & varying_joints & commanded & (action_range > 0)
        frozen = [joint_names[j] for j in np.nonzero(stuck)[0]]
        if frozen:
            flags.append(f"frozen joint: {', '.join(frozen)} reading never changes although its command moves")
    if len(timestamp) > 1:
        gap = float(np.max(np.abs(np.diff(timestamp) - 1.0 / fps)))
        if gap > cfg.timestamp_tol_s:
            flags.append(f"timestamp gap: frame spacing off by up to {gap:.3g} s (dropped or duplicated frames?)")
    return flags


# ---- plain-words reasons (never "fail"/"success": motion is not outcome) ------------------------------------
def _ratio(x: float, med: float) -> str:
    return f"{x / med:.1f}x dataset median" if abs(med) > 1e-12 else f"{x:.3g} (dataset median ~0)"


def reason_text(signal: str, x: float, med: float, z: float, cfg: ScoreConfig) -> str:
    zs = f"z={min(z, cfg.display_cap):.1f}"
    if signal == "sparc_state":
        return f"irregular speed profile: spectral arc length {_ratio(x, med)} ({zs})"
    if signal == "ldlj_state":
        return f"jerky motion: log dimensionless jerk {x - med:+.1f} above dataset median ({zs})"
    if signal == "action_tv_ratio":
        return f"dithering commands: action path / net displacement {_ratio(x, med)} ({zs})"
    if signal == "action_hf_energy":
        return f"looks jittery: action chatter {_ratio(x, med)} ({zs})"
    if signal == "idle_frac":
        return f"hesitates: idle {100 * x:.0f}% of frames before its last motion (median {100 * med:.0f}%) ({zs})"
    if signal == "saturation_frac":
        return f"actions pinned at their limits {100 * x:.0f}% of frames (median {100 * med:.0f}%) ({zs})"
    if signal == "track_err":
        return f"arm doesn't follow the commands: tracking error {_ratio(x, med)} ({zs})"
    raise KeyError(signal)


# ---- outcome ------------------------------------------------------------------------------------------------
def outcome_fields(ds: Dataset, episode_index: int, success: np.ndarray | None) -> tuple[bool | None, str | None]:
    sim = bool(success[-1]) if success is not None and len(success) else None
    label = None
    if ds.tags is not None:
        row = ds.tags.get("episodes", {}).get(str(episode_index))
        if isinstance(row, dict) and row.get("label") in ("success", "fail"):
            label = row["label"]
    return sim, label


# ---- top level ------------------------------------------------------------------------------------------------
def score_dataset(ds: Dataset, cfg: ScoreConfig | None = None) -> ScoreResult:
    cfg = cfg or ScoreConfig()
    eps = ds.episodes
    if not eps:
        raise ValueError("dataset has no episodes")
    n_joints = eps[0].state.shape[1]
    names = (
        ds.state_names
        if ds.state_names and len(ds.state_names) == n_joints
        else [f"joint {j}" for j in range(n_joints)]
    )
    skipped: dict[str, str] = {}
    notes = list(ds.notes)
    same_dims = eps[0].action.shape[1] == n_joints
    names_differ = bool(ds.state_names and ds.action_names and ds.state_names != ds.action_names)
    if not same_dims:
        skipped["track_err"] = f"action has {eps[0].action.shape[1]} dims but state has {n_joints}"
    elif names_differ:
        skipped["track_err"] = "action and state joint names differ, so they can't be compared joint by joint"
    signals_used = [k for k in SIGNALS if k not in skipped]

    finite = [bool(np.all(np.isfinite(e.state)) and np.all(np.isfinite(e.action))) for e in eps]
    finite_states = [e.state for e, ok in zip(eps, finite, strict=True) if ok]
    all_state = np.concatenate(finite_states) if finite_states else np.zeros((0, n_joints))
    varying = np.ptp(all_state, axis=0) > 0 if len(all_state) else np.zeros(n_joints, bool)
    action_range = None
    if same_dims and finite_states:
        all_action = np.concatenate([e.action for e, ok in zip(eps, finite, strict=True) if ok])
        action_range = np.ptp(all_action, axis=0)
    flags = [hard_flags(e.state, e.action, e.timestamp, ds.fps, varying, names, cfg, action_range) for e in eps]
    need = cfg.min_frames(ds.fps)
    scorable = [ok and len(e.state) >= need for e, ok in zip(eps, finite, strict=True)]
    idx = [i for i, ok in enumerate(scorable) if ok]

    raw_arrays: dict[str, np.ndarray] = {}
    z_arrays: dict[str, np.ndarray] = {}
    medians: dict[str, float] = {}
    if idx:
        raw_arrays = compute_signals(
            [eps[i].state for i in idx], [eps[i].action for i in idx], ds.fps, cfg, "track_err" in signals_used
        )
        for k in signals_used:
            z_arrays[k], medians[k] = robust_z(raw_arrays[k], k, cfg)
        log_len = np.log([len(eps[i].state) for i in idx]).astype(np.float64)
        len_z, _ = robust_z(log_len, "length", cfg)
    else:
        notes.append("no episode could be scored (all non-finite or too short)")

    pos = {i: j for j, i in enumerate(idx)}
    out = []
    for i, e in enumerate(eps):
        sim, label = outcome_fields(ds, e.index, e.success)
        raw: dict[str, float | None] = {k: None for k in SIGNALS}
        z: dict[str, float | None] = {k: None for k in SIGNALS}
        combined = None
        length_z = None
        reasons: list[str] = []
        hard = list(flags[i])
        if i in pos:
            j = pos[i]
            for k in SIGNALS:
                raw[k] = None if k in skipped else float(raw_arrays[k][j])
            for k in signals_used:
                z[k] = float(z_arrays[k][j])
            combined = max(z[k] for k in signals_used)  # type: ignore[type-var]
            length_z = float(len_z[j])
            ranked = sorted(signals_used, key=lambda k: -z[k])  # type: ignore[operator]
            reasons = [
                reason_text(k, raw[k], medians[k], z[k], cfg)  # type: ignore[arg-type]
                for k in ranked
                if z[k] > cfg.reason_z  # type: ignore[operator]
            ]
            sat = raw["saturation_frac"]
            if sat is not None and sat >= cfg.hard_saturation:
                hard.append(f"saturation: actions at their limits in {100 * sat:.0f}% of entries (>= 20%)")
        quality = "HARD" if hard else ("FLAG" if combined is not None and combined > cfg.flag_z else "ok")
        if combined is not None and combined <= cfg.flag_z:  # N5: listed reasons didn't flag the episode
            reasons = [f"{r} (below flag threshold)" for r in reasons]
        out.append(
            EpisodeScore(
                episode_index=e.index,
                length=e.length,
                raw=raw,
                z=z,
                combined=combined,
                quality=quality,
                hard_flags=hard,
                reasons=reasons,
                outcome_sim=sim,
                outcome_label=label,
                outcome_disagree=sim is not None and label is not None and (label == "success") != sim,
                length_z=length_z,
            )
        )
    return ScoreResult(
        episodes=out, config=cfg, signals_used=signals_used, skipped=skipped, notes=notes, medians=medians
    )


def config_dict(cfg: ScoreConfig) -> dict[str, Any]:
    return asdict(cfg)
