"""The report card's rules (docs/phase4-plan.md "Report-card rules (D2)"): dataset linking, typed HARD counts,
R0–R6 (first match wins, R6 is an add-on), the fixed "Can't tell" lines. Builds ``report.json`` (the source of
truth every rendering prints verbatim).

Wording is deliberate: verdicts say "consistent with", never assign a cause; they call scored episodes the data a
policy was trained on only when that policy's checkpoint is *linked* to the scored dataset. Thresholds are named
constants and are sim-tuned proposals.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from robot_report_card import __version__

REPORT_SCHEMA_VERSION = 1

# ---- thresholds (sim-tuned proposals) ------------------------------------------------------------------------------
HARD_FRAC_R0 = 0.05  # R0: hard-flagged share that means "fix the data first" (any non-finite also triggers it)
FLAGGED_FRAC_R1 = 0.50  # R1: flagged share above which per-episode flags aren't reliable
FAILED_FRAC_R2 = 0.10  # R2: failed share (among episodes with outcome evidence)
MOTION_FRAC_R3 = 0.10  # R3: motion-flagged (FLAG) share
FLAGGED_FRAC_R5 = 0.05  # R5 needs fewer flagged episodes than this ...
FAILED_FRAC_R5 = 0.05  # ... and fewer failed ones
RECIPE_SEEDS = 3
WRONG_GOAL_COST_PTS = 24  # phase 3: clean + 20 wrong-goal demos cost ~24 points (52% -> 28%)

HARD_TYPES = {  # hard-flag prefix in the score JSON -> wording (never "corrupt")
    "non-finite": "non-finite values",
    "timestamp gap": "dropped frames",
    "too short": "too short",
    "frozen joint": "frozen joint",
    "saturation": "saturated actions",
}
SIGNAL_WORDS = {
    "sparc_state": "irregular speed profile",
    "ldlj_state": "jerky motion",
    "action_tv_ratio": "dithering commands",
    "action_hf_energy": "action chatter",
    "idle_frac": "hesitation",
    "saturation_frac": "actions pinned at limits",
    "track_err": "commands not followed",
}

CANT_TELL = (
    "Uniform junk: if every episode shares the same problem (for example a jittery leader arm), few or no episodes "
    "are flagged.",
    "Wobble (band-limited jitter) isn't flagged by the motion signals.",
    "Smooth failed attempts (early stop, wrong goal) move normally; only outcome evidence (success column or labels) "
    "can show them.",
    "Checkpoint vs recipe: a compare covers two fixed checkpoints. Retraining the same recipe with another seed moved "
    "success by 10–15 points in our sim study, so a claim about data or training settings needs ≥ 3 training seeds "
    "per side.",
    "Success is final-frame (tip within 2 cm on the last frame); LeRobot's any-step rate is shown alongside and "
    "differs.",
    "Policy eval runs in simulation (so101_reach), not on a real robot.",
    "Scores are relative to one dataset and aren't comparable across datasets.",
    "Reproducibility is same-machine only: a rerun on another machine is a new sample, not a replay.",
    "Thresholds and rules are sim-tuned proposals, not calibrated on real teleoperation data.",
)
FOOTER = (
    f"Rules and thresholds are sim-tuned proposals: R0 hard ≥ {HARD_FRAC_R0:.0%} or any non-finite; "
    f"R1 flagged ≥ {FLAGGED_FRAC_R1:.0%}; R2 failed ≥ {FAILED_FRAC_R2:.0%}; R3 motion-flagged ≥ {MOTION_FRAC_R3:.0%}; "
    f"R5 flagged < {FLAGGED_FRAC_R5:.0%} and failed < {FAILED_FRAC_R5:.0%}. First matching rule wins; R6 is an add-on."
)


class ReportInputError(ValueError):
    """An input JSON that isn't what the flag says it is."""


# ---- helpers -----------------------------------------------------------------------------------------------
def pct(x: float) -> str:
    v = 100 * x
    return f"{v:.0f}%" if abs(v - round(v)) < 0.05 else f"{v:.1f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f} pts"


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def check_kind(obj: dict[str, Any], expected: str, path: str) -> None:
    if expected == "score":
        ok = obj.get("tool") == "robot-report-card" and "episodes" in obj and "summary" in obj and "kind" not in obj
    else:
        ok = obj.get("tool") == "robot-report-card" and obj.get("kind") == expected
    if not ok:
        raise ReportInputError(f"{path} is not an `rrc {expected}` JSON report")


# ---- data block --------------------------------------------------------------------------------------------
def _outcome(e: dict[str, Any]) -> tuple[bool | None, str | None]:
    """(succeeded?, source) with the plan's precedence: label if present, else sim, else unknown."""
    if e.get("outcome_label") in ("success", "fail"):
        return e["outcome_label"] == "success", "label"
    if e.get("outcome_sim") is not None:
        return bool(e["outcome_sim"]), "sim"
    return None, None


def data_block(score: dict[str, Any] | None) -> dict[str, Any]:
    if score is None:
        return {"available": False}
    eps = score["episodes"]
    n = len(eps)
    reason_z = float(score.get("thresholds", {}).get("reason_z", 3.0))
    hard_types: Counter[str] = Counter()
    hard, nonfinite, motion = 0, 0, 0
    motion_reasons: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    known = failed = 0
    for e in eps:
        if e["quality"] == "HARD":
            hard += 1
            types = {t for h in e.get("hard_flags", []) for p, t in HARD_TYPES.items() if h.startswith(p)}
            hard_types.update(types or {"other"})
            nonfinite += any(h.startswith("non-finite") for h in e.get("hard_flags", []))
        elif e["quality"] == "FLAG":
            motion += 1
            motion_reasons.update(k for k, z in (e.get("z") or {}).items() if z is not None and z > reason_z)
        ok, src = _outcome(e)
        if src is not None:
            sources[src] += 1
            known += 1
            failed += not ok
    flagged = hard + motion
    ds = score.get("dataset", {})
    return {
        "available": True,
        "dataset_path": ds.get("path"),
        "dataset_name": Path(str(ds.get("path") or "dataset")).name,
        "total_frames": ds.get("total_frames"),
        "codebase_version": ds.get("codebase_version"),
        "episodes": n,
        "flagged": flagged,
        "flagged_frac": flagged / n if n else 0.0,
        "motion_flagged": motion,
        "motion_flagged_frac": motion / n if n else 0.0,
        "motion_reasons": dict(motion_reasons.most_common()),
        "hard": hard,
        "hard_frac": hard / n if n else 0.0,
        "hard_types": dict(sorted(hard_types.items())),
        "nonfinite": nonfinite,
        "outcome_known": known,
        "outcome_sources": dict(sources),
        "failed": failed,
        "failed_frac": failed / known if known else None,
    }


def hard_line(d: dict[str, Any]) -> str:
    n = d["hard"]
    types = " / ".join(d["hard_types"]) or "none"
    return f"{n} episode{' has' if n == 1 else 's have'} hard flags: {types}."


def reasons_text(d: dict[str, Any]) -> str:
    return ", ".join(f"{SIGNAL_WORDS.get(k, k)} {c}" for k, c in d["motion_reasons"].items()) or "no single signal"


# ---- linking -----------------------------------------------------------------------------------------------
def _subset_stats(score: dict[str, Any], kept: set[int]) -> dict[str, Any]:
    eps = [e for e in score["episodes"] if int(e["episode_index"]) in kept]
    known = [ok for ok, src in (_outcome(e) for e in eps) if src is not None]
    return {
        "episodes": len(eps),
        "failed_frac": (sum(not ok for ok in known) / len(known)) if known else None,
        "outcome_known": len(known),
        "motion_flagged_frac": (sum(e["quality"] == "FLAG" for e in eps) / len(eps)) if eps else 0.0,
        "flagged_frac": (sum(e["quality"] != "ok" for e in eps) / len(eps)) if eps else 0.0,
    }


def link(policy: dict[str, Any], score: dict[str, Any] | None) -> dict[str, Any]:
    """Does this policy's training data relate to the scored dataset? (plan D2 linking, QA phase 4 R1)

    ``linked``: a `bc:` checkpoint whose fingerprint (path + total_frames) is the scored dataset. ``full``: it also
    trained on every episode (keep = all). A linked checkpoint trained with a filter is *not* full; if its kept
    episode indices are recorded, ``subset`` holds the rule statistics over just those episodes.
    """
    spec, kind = policy.get("spec", "?"), policy.get("kind")
    out: dict[str, Any] = {"spec": spec, "linked": False, "full": False, "keep": None, "subset": None}
    if kind != "bc":
        return {**out, "why": f"{kind or 'this'} policies carry no dataset fingerprint"}
    if score is None:
        return {**out, "why": "no score JSON to compare against"}
    fp, ds = policy.get("dataset") or {}, score.get("dataset") or {}
    same_path = bool(fp.get("path")) and Path(str(fp["path"])).resolve() == Path(str(ds.get("path"))).resolve()
    if not (same_path and fp.get("total_frames") == ds.get("total_frames")):
        return {**out, "why": "its dataset fingerprint doesn't match the scored dataset"}
    keep = (policy.get("filter") or {}).get("keep", "all")
    n_all = len(score["episodes"])
    indices = policy.get("kept_episode_indices")
    if keep == "all":
        return {**out, "linked": True, "full": True, "keep": keep, "why": "fingerprint matches the scored dataset"}
    if isinstance(indices, list):
        kept = {int(i) for i in indices}
        stats = _subset_stats(score, kept)
        why = f"filtered subset: keep = {keep}, {stats['episodes']} of {n_all} episodes"
        return {**out, "linked": True, "keep": keep, "subset": stats, "why": why}
    why = f"filtered subset: keep = {keep}; which episodes isn't recorded in this JSON"
    return {**out, "linked": True, "keep": keep, "why": why}


def subset_note(lk: dict[str, Any], side: str | None, measure: str) -> str:
    """One neutral sentence about what a linked checkpoint learned from (never says 'trained on')."""
    who = f"{side} ({lk['spec']})" if side else lk["spec"]
    if lk["full"]:
        return f"{who} learned from all {lk['_all']} episodes."
    if lk["subset"] is None:
        return f"{who} learned from a filtered subset ({lk['why'].split(': ', 1)[1]})."
    st, total = lk["subset"], lk["why"]
    detail = total.split(": ", 1)[1]
    if measure == "failed":
        rate = "unknown" if st["failed_frac"] is None else pct(st["failed_frac"])
        return f"{who} learned from a filtered subset ({detail}), of which {rate} failed."
    return f"{who} learned from a filtered subset ({detail}), of which {pct(st['motion_flagged_frac'])} are flagged."


# ---- policy / regression blocks ----------------------------------------------------------------------------
def _policy_row(side: str | None, run: dict[str, Any], source: str) -> dict[str, Any]:
    s = run["summary"]
    return {
        "side": side,
        "spec": run["policy"].get("spec"),
        "kind": run["policy"].get("kind"),
        "n": s["n"],
        "successes": s["successes"],
        "rate": s["rate"],
        "ci": s["ci"],
        "ci_method": s.get("ci_method", "wilson"),
        "rate_any_step": s.get("rate_any_step"),
        "median_final_error_m": s.get("median_final_error_m"),
        "eval_seed": run.get("eval_seed"),
        "max_steps": run.get("max_steps"),
        "source": source,
    }


def regression_block(compare: dict[str, Any] | None) -> dict[str, Any] | None:
    if compare is None:
        return None
    p = compare["paired"]
    return {
        "a": compare["A"]["policy"].get("spec"),
        "b": compare["B"]["policy"].get("spec"),
        "n": p["n"],
        "delta": p["delta_b_minus_a"],
        "ci": p["delta_ci"],
        "p": p["mcnemar_p"],
        "significant": bool(p["significant"]),
        "borderline": bool(p.get("borderline", False)),
        "discordant": {"a_only": p["a_only"], "b_only": p["b_only"]},
        "mde": p.get("mde"),
        "mde_inputs": p.get("mde_inputs"),
        "mde_statement": compare.get("mde_statement"),
        "borderline_statement": compare.get("borderline_statement"),
        "recipe_caveat": compare.get("recipe_caveat"),
    }


def _effect(r: dict[str, Any]) -> str:
    return f"Δ = {pts(r['delta'])}, 95% CI {pts(r['ci'][0])} to {pts(r['ci'][1])}, McNemar p = {fmt_p(r['p'])}"


# ---- verdict -----------------------------------------------------------------------------------------------
def verdict(d: dict[str, Any], reg: dict[str, Any] | None, links: dict[str, dict[str, Any]], train_seeds: int,
            missing: list[str]) -> list[dict[str, str]]:  # fmt: skip
    lines: list[dict[str, str]] = []
    trained = [lk for lk in links.values() if lk is not None]
    # "trained on the scored demos" only when every checkpoint trained on ALL of them (QA phase 4 R1)
    all_full = bool(trained) and all(lk["full"] for lk in trained)
    filtered = [(side, lk) for side, lk in links.items() if lk is not None and lk["linked"] and not lk["full"]]
    if filtered:  # when any checkpoint is filtered, say what each linked checkpoint learned from (no blanket clause)
        n_all = d["episodes"] if d["available"] else 0
        full_notes = [(side, lk) for side, lk in links.items() if lk is not None and lk["full"]]
        filtered = [*[(side, {**lk, "_all": n_all}) for side, lk in full_notes], *filtered]
    if d["available"]:
        n = d["episodes"]
        if d["hard_frac"] >= HARD_FRAC_R0 or d["nonfinite"] > 0:
            lines.append({"rule": "R0", "text": f"Fix the data first: {hard_line(d)}"})
        elif d["flagged_frac"] >= FLAGGED_FRAC_R1:
            lines.append({"rule": "R1", "text": (
                "Most episodes are flagged. The score is relative and assumes most episodes are good, so per-episode "
                "flags aren't reliable here.")})  # fmt: skip
        elif d["failed_frac"] is not None and d["failed_frac"] >= FAILED_FRAC_R2:
            src = ", ".join(f"{k} {v}" for k, v in sorted(d["outcome_sources"].items()))
            where = ""
            if all_full:
                where = " in the data this policy was trained on" if len(trained) == 1 else (
                    " in the data these checkpoints were trained on")  # fmt: skip
            notes = "".join(f" {subset_note(lk, side, 'failed')}" for side, lk in filtered)
            lines.append({"rule": "R2", "text": (
                f"{pct(d['failed_frac'])} of demos in {d['dataset_name']} failed (outcome source: {src} of {n} "
                f"episodes). Consistent with a data problem{where}.{notes} In our sim study, training on failed "
                f"wrong-goal demos cost ~{WRONG_GOAL_COST_PTS} points.")})  # fmt: skip
        elif d["motion_flagged_frac"] >= MOTION_FRAC_R3:
            lines.append({"rule": "R3", "text": (
                f"{pct(d['motion_flagged_frac'])} of demos flagged ({reasons_text(d)}). "
                "Consistent with a data problem."
                + "".join(f" {subset_note(lk, side, 'flagged')}" for side, lk in filtered)
                + " In our sim study, random and high-noise demos hurt the policy; hesitation didn't.")})  # fmt: skip
    if not lines and reg is not None:
        if not reg["significant"] or reg["borderline"]:
            if reg["mde"] is not None and not (isinstance(reg["mde"], float) and math.isnan(reg["mde"])):
                power = f"With n = {reg['n']} it detects differences ≥ {100 * reg['mde']:.1f} points (80% power)."
            else:
                power = f"With n = {reg['n']} no possible improvement over A reaches 80% power."
            text = f"Eval can't tell A from B ({_effect(reg)}). {power}"
            if reg["borderline"] and reg.get("borderline_statement"):
                text += f" {reg['borderline_statement']}"
            lines.append({"rule": "R4", "text": text})
        else:
            unmet = _r5_unmet(d, links)
            if not unmet:
                text = (
                    f"A and B differ in this eval ({_effect(reg)}). The scorer found no problems in the data both were "
                    "trained on; it can't see uniform junk, wobble or smooth failed attempts. So the difference is "
                    "consistent with training variance or the recipe."
                )
                if train_seeds < RECIPE_SEEDS:
                    text += (
                        f" A claim about the recipe needs ≥ {RECIPE_SEEDS} training seeds per side with a consistent "
                        f"direction; this report has {train_seeds}."
                    )
                lines.append({"rule": "R5", "text": text})
            else:
                missing = [*missing, *unmet]
    if not lines:
        lines.append(
            {"rule": "INCONCLUSIVE", "text": "Inconclusive: " + ("; ".join(missing) or "no rule applies") + "."}
        )
    if d["available"] and d["outcome_known"] == 0:
        lines.append({"rule": "R6", "text": "Outcome unknown: wrong-goal and early-stop demos can't be ruled out."})
    return lines


def _r5_unmet(d: dict[str, Any], links: dict[str, dict[str, Any] | None]) -> list[str]:
    unmet = []
    for side in ("A", "B"):
        lk = links.get(side)
        if lk is None or not lk["linked"]:
            spec = lk["spec"] if lk else "?"
            why = lk["why"] if lk else "not a checkpoint"
            unmet.append(f"{side} ({spec}) is unlinked ({why}), so the scored dataset says nothing about its data")
        elif not lk["full"]:
            unmet.append(
                f"{side} ({lk['spec']}) learned from a {lk['why']}, so the whole-dataset reading doesn't describe "
                "its data"
            )
    if not d["available"]:
        unmet.append("no score JSON (missing)")
        return unmet
    if d["flagged_frac"] >= FLAGGED_FRAC_R5:
        unmet.append(f"{pct(d['flagged_frac'])} of episodes are flagged (the no-data-problem reading needs < "
                     f"{FLAGGED_FRAC_R5:.0%})")  # fmt: skip
    if d["outcome_known"] < d["episodes"]:
        unmet.append(f"outcome evidence covers {d['outcome_known']} of {d['episodes']} episodes")
    elif d["failed_frac"] is not None and d["failed_frac"] >= FAILED_FRAC_R5:
        unmet.append(f"{pct(d['failed_frac'])} of demos failed (needs < {FAILED_FRAC_R5:.0%})")
    return unmet


# ---- whole report ------------------------------------------------------------------------------------------
def build_report(
    score: dict[str, Any] | None,
    evals: list[dict[str, Any]],
    compare: dict[str, Any] | None,
    train_seeds: int,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    d = data_block(score)
    policies = [_policy_row(None, e, str(p)) for e, p in zip(evals, inputs.get("evals", []), strict=True)]
    links: dict[str, Any] = {}
    warnings: list[str] = []
    if compare is not None:
        for side in ("A", "B"):
            policies.append(_policy_row(side, compare[side], str(inputs.get("compare"))))
            links[side] = link(compare[side]["policy"], score)
    for e in evals:
        spec = e["policy"].get("spec")
        if compare is not None:
            specs = {compare["A"]["policy"].get("spec"), compare["B"]["policy"].get("spec")}
            if spec not in specs:
                warnings.append(f"eval of {spec} isn't one of the compared policies")
            if e.get("eval_seed") != compare.get("eval_seed") or len(e.get("episodes", [])) != compare["paired"]["n"]:
                warnings.append(f"eval of {spec} used other seeds or n than the compare")
    seeds = {e.get("eval_seed") for e in evals}
    if len(seeds) > 1:
        warnings.append("the eval JSONs use different eval seeds; their rates aren't paired")
    missing = []
    if score is None:
        missing.append("no score JSON (missing), so nothing is known about the data")
    if compare is None:
        missing.append("no compare JSON (missing), so there is no A/B regression to explain")
    reg = regression_block(compare)
    lines = verdict(d, reg, links, train_seeds, missing)
    definitions = sorted({x.get("success_definition") for x in [*evals, *([compare] if compare else [])] if x})
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "report",
        "tool": "robot-report-card",
        "rrc_version": __version__,
        "inputs": {k: (str(v) if not isinstance(v, list) else [str(x) for x in v]) for k, v in inputs.items()},
        "train_seeds": train_seeds,
        "data": d,
        "policies": policies,
        "regression": reg,
        "linking": links,
        "verdict": lines,
        "cant_tell": list(CANT_TELL),
        "warnings": warnings,
        "success_definitions": definitions,
        "thresholds": {
            "hard_frac_r0": HARD_FRAC_R0,
            "flagged_frac_r1": FLAGGED_FRAC_R1,
            "failed_frac_r2": FAILED_FRAC_R2,
            "motion_flagged_frac_r3": MOTION_FRAC_R3,
            "flagged_frac_r5": FLAGGED_FRAC_R5,
            "failed_frac_r5": FAILED_FRAC_R5,
            "recipe_seeds": RECIPE_SEEDS,
        },
        "footer": FOOTER,
    }
