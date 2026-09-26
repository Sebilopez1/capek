"""Render a ScoreResult as the terminal table + summary, and as the JSON report (schema_version 1).

JSON field names are frozen once QA approves them (phase 3 and the hosted tier will read them).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from robot_report_card import __version__
from robot_report_card.score.engine import EpisodeScore, ScoreResult
from robot_report_card.score.reader import Dataset

REPORT_SCHEMA_VERSION = 1


def outcome_text(e: EpisodeScore) -> str:
    value, source = e.outcome
    if source == "none":
        return "unknown"
    text = f"{value} ({source})"
    return text + " ≠ sim" if e.outcome_disagree else text


def score_text(e: EpisodeScore, cap: float) -> str:
    if e.combined is None:
        return "-"
    return f"{min(e.combined, cap):.1f}" + ("+" if e.combined > cap else "")


def format_table(result: ScoreResult, only_flagged: bool = False) -> str:
    cap = result.config.display_cap
    header = ["ep", "frames", "quality", "score", "outcome", "why"]
    rows = [
        [
            str(e.episode_index),
            str(e.length),
            e.quality,
            score_text(e, cap),
            outcome_text(e),
            "; ".join([*e.hard_flags, *e.reasons]),
        ]
        for e in result.episodes
        if not only_flagged or e.quality != "ok"
    ]
    widths = [max(len(r[c]) for r in [header, *rows]) for c in range(len(header) - 1)]
    lines = []
    for r in [header, *rows]:
        cells = [cell.ljust(w) for cell, w in zip(r[:-1], widths, strict=True)]
        lines.append(" | ".join([*cells, r[-1]]).rstrip())
    if only_flagged and not rows:
        lines.append("(no flagged episodes)")
    return "\n".join(lines)


def format_summary(result: ScoreResult, ds: Dataset, json_path: str | None) -> str:
    s = result.summary()
    n = s["episodes"]
    pct = 100 * s["flagged_frac"]
    lines = [
        "",
        f"{n} episodes, {ds.num_frames} frames, {ds.fps:g} fps, LeRobot {ds.codebase_version}, "
        f"keys {ds.state_key} / {ds.action_key}",
        f"flagged: {s['flagged']}/{n} ({pct:.1f}%): {s['flagged'] - s['hard']} by motion score > "
        f"{result.config.flag_z:g}, {s['hard']} with hard flags",
    ]
    if s["top_reasons"]:
        lines.append("top reasons: " + ", ".join(f"{k} {c}" for k, c in s["top_reasons"]))
    lines.append("outcome: " + ", ".join(f"{k} {c}" for k, c in s["outcomes"].items()))
    if s["outcome_disagreements"]:
        lines.append(f"label and sim disagree on {s['outcome_disagreements']} episode(s)")
    for name, why in result.skipped.items():
        lines.append(f"skipped signal {name}: {why}")
    lines += [f"note: {note}" for note in result.notes]
    lines += [f"WARNING: {w}" for w in s["warnings"]]
    lines += [s["assumption"], s["honesty_statement"]]
    if json_path:
        lines.append(f"JSON report: {json_path}")
    return "\n".join(lines)


def _episode_json(e: EpisodeScore) -> dict[str, Any]:
    d = asdict(e)
    value, source = e.outcome
    d["outcome_display"] = {"value": value, "source": source}
    return d


def build_json(result: ScoreResult, ds: Dataset) -> dict[str, Any]:
    s = result.summary()
    s["top_reasons"] = [{"reason": k, "count": c} for k, c in s["top_reasons"]]
    cfg = asdict(result.config)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool": "robot-report-card",
        "rrc_version": __version__,
        "dataset": {
            "path": str(ds.root.resolve()),
            "codebase_version": ds.codebase_version,
            "robot_type": ds.info.get("robot_type"),
            "fps": ds.fps,
            "total_episodes": len(ds.episodes),
            "total_frames": ds.num_frames,
        },
        "keys": {"state": ds.state_key, "action": ds.action_key},
        "thresholds": {
            "flag_z": cfg["flag_z"],
            "reason_z": cfg["reason_z"],
            "display_cap": cfg["display_cap"],
            "hard_saturation": cfg["hard_saturation"],
            "timestamp_tol_s": cfg["timestamp_tol_s"],
        },
        "config": cfg,
        "signals": list(result.signals_used),
        "skipped_signals": dict(result.skipped),
        "signal_medians": dict(result.medians),
        "notes": list(result.notes),
        "summary": s,
        "episodes": [_episode_json(e) for e in result.episodes],
    }
