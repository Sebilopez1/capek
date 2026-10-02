"""Judge the plan's DoD 4 bars from ``capek score`` JSON reports (schema_version 1) and QA's ``gt.json``.

This is the only code that decides pass/fail. It uses nothing from the scorer except its JSON output:
per episode ``episode_index``, ``combined`` (max z; null = unscorable) and ``quality`` (ok | FLAG | HARD).
"Flagged" means quality != "ok" (what a user sees). For ranking, a HARD / unscorable episode counts as +inf.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from capek.bench.spec import CLASSES

JUNK_ALWAYS = ("noise010", "noise025", "random")  # each flagged >= 90%
BARS = (
    # (key, description, comparator, threshold)
    ("auroc", "clean vs motion junk AUROC", ">=", 0.95),
    ("precision", "precision at the default threshold", ">=", 0.90),
    ("recall", "recall at the default threshold", ">=", 0.85),
    ("clean_flagged", "clean flagged in the mixed set", "<=", 0.05),
    ("clean_only_flagged", "clean-only set flagged", "<=", 0.05),
    ("noise010_flagged", "noise 0.1 flagged", ">=", 0.90),
    ("noise025_flagged", "noise 0.25 flagged", ">=", 0.90),
    ("random_flagged", "random flagged", ">=", 0.90),
    ("hesitation_auroc", "hesitation vs clean AUROC", ">=", 0.90),
    ("nearmiss_flagged", "nearmiss flagged", "<=", 0.20),
    ("return_home_flagged", "return_home flagged (<= 1/10)", "<=", 0.10),
)
# Which bars decide pass/fail per noise profile (plan D3). On the whole-set low-noise variant only the R-real-1 bar
# binds; everything else is reported (low-noise clean-only was 2-5% with either formula;
# real noise levels decide later).
BINDING: dict[str, frozenset[str] | None] = {"standard": None, "low": frozenset({"return_home_flagged"})}
DEFAULT_FLAG_Z = 3.5


class EvalError(Exception):
    """Inputs that can't be evaluated (wrong files, mismatched datasets...)."""


def auroc(scores: Any, positive: Any) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie); NaN if a class is empty. Works with +inf scores."""
    s, y = np.asarray(scores, dtype=float), np.asarray(positive, dtype=bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def _load(path: Path | str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise EvalError(f"cannot read {path}: {e}") from e


def _check_report(score: Any, what: str) -> None:
    """Structure of a ``capek score`` JSON report (schema_version 1), as far as the evaluator reads it."""
    if not isinstance(score, dict) or score.get("schema_version") != 1 or not isinstance(score.get("episodes"), list):
        raise EvalError(f"{what}: not a capek score JSON report (schema_version 1, episodes list); arguments swapped?")
    ds = score.get("dataset")
    if not isinstance(ds, dict) or not all(isinstance(ds.get(k), int) for k in ("total_episodes", "total_frames")):
        raise EvalError(f"{what}: report has no dataset.total_episodes / dataset.total_frames")
    for i, e in enumerate(score["episodes"]):
        if not isinstance(e, dict) or not isinstance(e.get("episode_index"), int) or "quality" not in e:
            raise EvalError(f"{what}: report episode #{i} lacks episode_index / quality")
        c = e.get("combined")
        if c is not None and (isinstance(c, bool) or not isinstance(c, (int, float))):
            raise EvalError(f"{what}: report episode {e['episode_index']} has a non-numeric combined score")
    thresholds = score.get("thresholds")
    fz = thresholds.get("flag_z") if isinstance(thresholds, dict) else None
    if fz != DEFAULT_FLAG_Z:
        raise EvalError(f"{what}: report was scored at flag_z={fz}; the DoD 4 bars are defined at {DEFAULT_FLAG_Z}")


def _check_gt(gt: Any, what: str) -> None:
    """Structure of a bench ``gt.json``."""
    if not isinstance(gt, dict) or not isinstance(gt.get("episodes"), dict):
        raise EvalError(f"{what}: not a bench gt.json (episodes object keyed by index); arguments swapped?")
    ds = gt.get("dataset")
    if not isinstance(ds, dict) or not all(isinstance(ds.get(k), int) for k in ("total_episodes", "total_frames")):
        raise EvalError(f"{what}: gt.json has no dataset.total_episodes / dataset.total_frames")
    for k, row in gt["episodes"].items():
        if not str(k).isdigit() or not isinstance(row, dict) or row.get("class") not in CLASSES or not row.get("group"):
            raise EvalError(f"{what}: gt.json episode {k!r} needs a group and a class in {CLASSES}")


def _aligned(
    score: dict[str, Any], gt: dict[str, Any], what: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (ranking score, flagged, group, class) in gt order, after checking both describe the same dataset."""
    _check_report(score, what)
    _check_gt(gt, what)
    ds, fp = score["dataset"], gt["dataset"]
    if (ds.get("total_episodes"), ds.get("total_frames")) != (fp["total_episodes"], fp["total_frames"]):
        raise EvalError(
            f"{what}: report is for {ds.get('total_episodes')} episodes / {ds.get('total_frames')} frames, "
            f"gt.json expects {fp['total_episodes']} / {fp['total_frames']} (wrong dataset?)"
        )
    by_index = {int(e["episode_index"]): e for e in score["episodes"]}
    if sorted(by_index) != sorted(int(k) for k in gt["episodes"]):
        raise EvalError(f"{what}: episode indices in the report don't match gt.json")
    keys = sorted(gt["episodes"], key=int)
    rank, flagged = [], []
    for k in keys:
        e = by_index[int(k)]
        if e.get("quality") not in ("ok", "FLAG", "HARD"):
            raise EvalError(f"{what}: episode {k} has unknown quality {e.get('quality')!r}")
        c = e.get("combined")
        hard = e["quality"] == "HARD" or c is None or (isinstance(c, float) and math.isnan(c))
        rank.append(math.inf if hard else float(c))
        flagged.append(e["quality"] != "ok")
    groups = np.array([gt["episodes"][k]["group"] for k in keys])
    classes = np.array([gt["episodes"][k]["class"] for k in keys])
    return np.array(rank), np.array(flagged), groups, classes


def evaluate(mixed_score: dict, mixed_gt: dict, clean_score: dict | None, clean_gt: dict | None) -> dict[str, Any]:
    rank, flag, groups, classes = _aligned(mixed_score, mixed_gt, "mixed")
    clean, junk = classes == "clean", classes == "motion_junk"
    keep = clean | junk
    tp, fp, fn = int(np.sum(flag & junk)), int(np.sum(flag & clean)), int(np.sum(~flag & junk))
    m: dict[str, Any] = {
        "auroc": auroc(rank[keep], junk[keep]),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "clean_flagged": float(np.mean(flag[clean])),
    }
    per_group = {}
    for g in dict.fromkeys(groups):
        sel = groups == g
        per_group[g] = {
            "class": str(classes[sel][0]),
            "n": int(sel.sum()),
            "flagged": int(flag[sel].sum()),
            "auroc_vs_clean": None if g == "clean" else auroc(rank[clean | sel], sel[clean | sel]),
        }
    for g in (*JUNK_ALWAYS, "nearmiss", "return_home"):
        m[f"{g}_flagged"] = per_group[g]["flagged"] / per_group[g]["n"] if g in per_group else float("nan")
    m["hesitation_auroc"] = per_group.get("hesitation", {}).get("auroc_vs_clean", float("nan"))
    if (clean_score is None) != (clean_gt is None):
        raise EvalError("the clean-only set needs both its report and its gt.json")
    if clean_score is not None and clean_gt is not None:
        _, cflag, _, cclasses = _aligned(clean_score, clean_gt, "clean-only")
        if not np.all(cclasses == "clean"):
            raise EvalError("clean-only gt.json contains non-clean episodes")
        m["clean_only_flagged"] = float(np.mean(cflag))
    else:
        m["clean_only_flagged"] = float("nan")  # not evaluated -> the bar fails
    profile = str(mixed_gt.get("noise_profile", "standard"))
    if profile not in BINDING:
        raise EvalError(f"mixed: unknown noise_profile {profile!r} in gt.json")
    if clean_gt is not None and str(clean_gt.get("noise_profile", "standard")) != profile:
        raise EvalError("mixed and clean-only gt.json have different noise profiles")
    binding = BINDING[profile]
    bars = []
    for key, desc, op, thr in BARS:
        v = m[key]
        ok = (v >= thr if op == ">=" else v <= thr) if v is not None and not math.isnan(v) else False
        bars.append(
            {
                "bar": key,
                "description": desc,
                "value": v,
                "op": op,
                "threshold": thr,
                "pass": bool(ok),
                "binding": binding is None or key in binding,
            }
        )
    warnings: list[str] = []
    if mixed_gt.get("small"):
        warnings.append("small build: bars are computed but not meaningful; acceptance needs a full build")
    return {
        "seed_set": mixed_gt.get("seed_set"),
        "noise_profile": profile,
        "small": bool(mixed_gt.get("small")),
        "all_pass": all(b["pass"] for b in bars if b["binding"]),
        "bars": bars,
        "per_group": per_group,
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "warnings": warnings,
    }


def evaluate_files(mixed: str, mixed_gt: str, clean: str | None = None, clean_gt: str | None = None) -> dict[str, Any]:
    return evaluate(
        _load(mixed), _load(mixed_gt), _load(clean) if clean else None, _load(clean_gt) if clean_gt else None
    )


def format_result(r: dict[str, Any]) -> str:
    fmt = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"  # noqa: E731
    small = " (small: bars not meaningful)" if r["small"] else ""
    lines = [f"benchmark v2, seed set {r['seed_set']}, noise profile {r.get('noise_profile', 'standard')}{small}", ""]
    lines.append(f"{'bar':40s} {'value':>7s}  {'target':>8s}  result")
    for b in r["bars"]:
        lines.append(
            f"{b['description']:40s} {fmt(b['value']):>7s}  {b['op']} {b['threshold']:<5g}  "
            + ("PASS" if b["pass"] else "FAIL")
            + ("" if b.get("binding", True) else "  (reported, not binding)")
        )
    lines += ["", "per group (reported; only the bars above are judged):"]
    lines.append(f"  {'group':11s} {'class':13s} {'flagged':>8s}  AUROC vs clean")
    for g, p in r["per_group"].items():
        lines.append(f"  {g:11s} {p['class']:13s} {p['flagged']:>3d}/{p['n']:<4d}  {fmt(p['auroc_vs_clean'])}")
    lines += [f"WARNING: {w}" for w in r["warnings"]]
    lines += [
        "",
        "ALL BINDING BARS PASS"
        if r["all_pass"]
        else "BARS FAILED: " + ", ".join(b["bar"] for b in r["bars"] if not b["pass"] and b.get("binding", True)),
    ]
    return "\n".join(lines)
