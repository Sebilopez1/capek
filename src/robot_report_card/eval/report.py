"""Tables, verdict wording and JSON (schema_version 1, kind eval|compare) for `rrc eval` / `rrc compare`.

Wording rules (plan DoD 2/3): a non-significant result starts "No detectable difference" and always carries the
MDE at the observed disagreement; a significant one names the direction. The text never says "no difference" or
"equivalent".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from robot_report_card import __version__
from robot_report_card.eval import stats
from robot_report_card.eval.runner import RolloutResult, success_definition

REPORT_SCHEMA_VERSION = 1
ALPHA = 0.05
LOSE_FLOOR = 0.10
RECIPE_CAVEAT = (
    "This compares two fixed checkpoints; the CI covers rollout noise only. Retraining the same recipe with another "
    "seed moved success by 10–15 points on this task. A claim about data or training settings needs ≥ 3 training "
    "seeds per side with a consistent direction."
)
CI_METHODS = ("wilson", "clopper-pearson")
BORDERLINE_SENTENCE = (
    "The CI and the exact test disagree at the margin (the exact test is more conservative); "
    "treat this result as borderline."
)


def rate_ci(k: int, n: int, method: str = "wilson") -> tuple[float, float]:
    return stats.clopper_pearson(k, n) if method == "clopper-pearson" else stats.wilson(k, n)


def _ci_label(method: str) -> str:
    return "Clopper-Pearson 95% CI" if method == "clopper-pearson" else "Wilson 95% CI"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---- eval --------------------------------------------------------------------------------------------------------
def summarize(r: RolloutResult, method: str = "wilson") -> dict[str, Any]:
    k, n = int(r.success.sum()), r.n
    lo, hi = rate_ci(k, n, method)
    k_any = int(r.success_any_step.sum())
    lo_any, hi_any = rate_ci(k_any, n, method)
    return {
        "n": n,
        "successes": k,
        "rate": k / n,
        "ci": [lo, hi],
        "ci_method": method,
        "successes_any_step": k_any,
        "rate_any_step": k_any / n,
        "ci_any_step": [lo_any, hi_any],
        "median_final_error_m": r.median_final_error_m,
    }


def eval_table(rows: list[tuple[str, dict[str, Any]]], method: str = "wilson") -> str:
    header = ["policy", "n", "successes", "rate", _ci_label(method), "median final error", "any-step rate"]
    body = [
        [
            name,
            str(s["n"]),
            str(s["successes"]),
            _pct(s["rate"]),
            f"{_pct(s['ci'][0])} – {_pct(s['ci'][1])}",
            f"{100 * s['median_final_error_m']:.1f} cm",
            _pct(s["rate_any_step"]),
        ]
        for name, s in rows
    ]
    widths = [max(len(r[c]) for r in [header, *body]) for c in range(len(header))]
    return "\n".join(" | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in [header, *body])


def eval_text(r: RolloutResult, method: str = "wilson") -> str:
    s = summarize(r, method)
    return "\n".join(
        [
            eval_table([(r.policy["spec"], s)], method),
            "",
            success_definition(r.max_steps, r.fps) + ".",
            f"Seeds: env reset from SeedSequence([{r.eval_seed}, i]) for i = 0..{r.n - 1}.",
        ]
    )


def eval_json(r: RolloutResult, method: str, machine: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "eval",
        "tool": "robot-report-card",
        "rrc_version": __version__,
        "success_definition": success_definition(r.max_steps, r.fps),
        "machine": machine,
        "summary": summarize(r, method),
        **r.to_dict(),
    }


# ---- compare -----------------------------------------------------------------------------------------------------
@dataclass
class Comparison:
    n: int
    both: int  # a: A and B succeed
    a_only: int  # b: A succeeds, B fails
    b_only: int  # c: B succeeds, A fails
    neither: int  # d
    p_value: float
    delta: float  # p_B - p_A
    delta_ci: tuple[float, float]
    mde: float  # NaN if not reachable
    mde_inputs: dict[str, Any]

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA

    @property
    def ci_excludes_zero(self) -> bool:
        return self.delta_ci[0] > 0 or self.delta_ci[1] < 0

    @property
    def borderline(self) -> bool:
        """Exact McNemar (the verdict) and the Newcombe CI disagree about significance, as happens near α."""
        return self.significant != self.ci_excludes_zero


def compare_stats(a: RolloutResult, b: RolloutResult) -> Comparison:
    if a.n != b.n or a.eval_seed != b.eval_seed or a.max_steps != b.max_steps:
        raise ValueError("paired comparison needs both policies on the same seeds")
    sa, sb = a.success, b.success
    both, a_only, b_only, neither = stats.discordant(sa, sb)
    p = stats.mcnemar_exact(a_only, b_only)
    delta, ci = stats.newcombe_paired_diff(sb, sa)
    n = a.n
    a_successes = both + a_only
    if a_successes == 0:
        p_a, lose_obs, lose, floored = 0.5, None, LOSE_FLOOR, True
    else:
        p_a = a_successes / n
        lose_obs = a_only / a_successes
        lose = max(lose_obs, LOSE_FLOOR)
        floored = lose_obs < LOSE_FLOOR
    mde = stats.mde_paired(n, p_a, lose)
    inputs = {
        "n": n,
        "p_A": p_a,
        "lose": lose,
        "lose_observed": lose_obs,
        "floored": floored,
        "a_had_no_successes": a_successes == 0,
        "power": 0.8,
        "alpha": ALPHA,
    }
    return Comparison(n, both, a_only, b_only, neither, p, delta, ci, mde, inputs)


def _fmt_p(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def verdict(c: Comparison, name_a: str, name_b: str) -> str:
    effect = (
        f"Δ = {100 * c.delta:+.1f} pts, 95% CI {100 * c.delta_ci[0]:+.1f} to {100 * c.delta_ci[1]:+.1f}; "
        f"McNemar p = {_fmt_p(c.p_value)}"
    )
    if not c.significant:
        return f"No detectable difference ({effect})."
    direction = "better" if c.delta > 0 else "worse"
    return f"B ({name_b}) is {direction} than A ({name_a}) ({effect})."


def mde_sentence(c: Comparison, checkpoints: bool = True) -> str:
    """The MDE at the observed disagreement (plan D2). ``checkpoints`` = both sides are trained checkpoints."""
    what = "checkpoints" if checkpoints else "policies"
    if math.isnan(c.mde):
        return (
            f"With n = {c.n} and this much disagreement between the {what}, no possible improvement over "
            f"A's {100 * c.mde_inputs['p_A']:.1f}% reaches 80% power."
        )
    return (
        f"With n = {c.n} and this much disagreement between the {what}, this test detects differences of about "
        f"{100 * c.mde:.1f} points or more with 80% power."
    )


def compare_text(a: RolloutResult, b: RolloutResult, c: Comparison, method: str, caveat: bool) -> str:
    rows = [(f"A: {a.policy['spec']}", summarize(a, method)), (f"B: {b.policy['spec']}", summarize(b, method))]
    lines = [
        eval_table(rows, method),
        "",
        f"paired on the same {c.n} seeds: both succeed {c.both}, A only {c.a_only}, B only {c.b_only}, "
        f"neither {c.neither}",
        verdict(c, a.policy["spec"], b.policy["spec"]),
        *([BORDERLINE_SENTENCE] if c.borderline else []),
        mde_sentence(c, caveat),
    ]
    if caveat:
        lines.append(RECIPE_CAVEAT)
    lines.append(success_definition(a.max_steps, a.fps) + ".")
    return "\n".join(lines)


def compare_json(
    a: RolloutResult, b: RolloutResult, c: Comparison, method: str, caveat: bool, machine: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "compare",
        "tool": "robot-report-card",
        "rrc_version": __version__,
        "success_definition": success_definition(a.max_steps, a.fps),
        "machine": machine,
        "eval_seed": a.eval_seed,
        "seeds": [[a.eval_seed, e.episode] for e in a.episodes],
        "A": {"summary": summarize(a, method), **a.to_dict()},
        "B": {"summary": summarize(b, method), **b.to_dict()},
        "paired": {
            "n": c.n,
            "both": c.both,
            "a_only": c.a_only,
            "b_only": c.b_only,
            "neither": c.neither,
            "mcnemar_p": c.p_value,
            "alpha": ALPHA,
            "significant": c.significant,
            "ci_excludes_zero": c.ci_excludes_zero,
            "borderline": c.borderline,
            "delta_b_minus_a": c.delta,
            "delta_ci": list(c.delta_ci),
            "delta_ci_method": "newcombe-1998-method-10",
            "mde": None if math.isnan(c.mde) else c.mde,
            "mde_inputs": c.mde_inputs,
        },
        "verdict": verdict(c, a.policy["spec"], b.policy["spec"]),
        "borderline_statement": BORDERLINE_SENTENCE if c.borderline else None,
        "mde_statement": mde_sentence(c, caveat),
        "recipe_caveat": RECIPE_CAVEAT if caveat else None,
    }
