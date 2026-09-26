"""Render report.json as terminal text, Markdown and a self-contained HTML page (stdlib template, inline CSS, no JS).

All three print the same lines from report.json; verdict lines appear verbatim.
"""

from __future__ import annotations

import html
from string import Template
from typing import Any

from robot_report_card.report.rules import hard_line, pct, pts, reasons_text

TITLE = "Robot Report Card"


def data_lines(r: dict[str, Any]) -> list[str]:
    d = r["data"]
    if not d["available"]:
        return ["No score JSON given (missing): nothing is known about the data."]
    lines = [
        f"Dataset: {d['dataset_name']} (LeRobot {d['codebase_version']}, {d['episodes']} episodes, "
        f"{d['total_frames']} frames)",
        f"Flagged: {d['flagged']} of {d['episodes']} ({pct(d['flagged_frac'])}): "
        f"{d['motion_flagged']} by motion score, {d['hard']} with hard flags",
        hard_line(d),
    ]
    if d["motion_flagged"]:
        lines.append(f"Motion flags by signal: {reasons_text(d)}")
    if d["outcome_known"]:
        src = ", ".join(f"{k} {v}" for k, v in sorted(d["outcome_sources"].items()))
        lines.append(
            f"Outcome evidence: {d['outcome_known']} of {d['episodes']} episodes ({src}); "
            f"failed {d['failed']} ({pct(d['failed_frac'])})"
        )
    else:
        lines.append("Outcome evidence: none (no success column, no labels)")
    return lines


def policy_lines(r: dict[str, Any]) -> list[str]:
    if not r["policies"]:
        return ["No eval or compare JSON given."]
    lines = []
    for p in r["policies"]:
        who = f"{p['side']} ({p['spec']})" if p["side"] else p["spec"]
        ci_name = "Clopper-Pearson" if p["ci_method"] == "clopper-pearson" else "Wilson"
        any_step = f", any-step rate {pct(p['rate_any_step'])}" if p.get("rate_any_step") is not None else ""
        err = p.get("median_final_error_m")
        err_txt = f", median final error {100 * err:.1f} cm" if err is not None else ""
        line = (
            f"{who}: {p['successes']}/{p['n']} ({pct(p['rate'])}), {ci_name} 95% CI {pct(p['ci'][0])} to "
            f"{pct(p['ci'][1])}{any_step}{err_txt} (eval seed {p['eval_seed']})"
        )
        lk = r["linking"].get(p["side"]) if p["side"] else None
        if lk is not None:
            if lk["linked"] and lk["full"]:
                line += "; linked to the scored dataset (all episodes)"
            elif lk["linked"]:
                line += f"; linked to the scored dataset ({lk['why']})"
            else:
                line += f"; unlinked ({lk['why']})"
        lines.append(line)
    return lines


def regression_lines(r: dict[str, Any]) -> list[str]:
    g = r["regression"]
    if g is None:
        return ["No compare JSON given."]
    lines = [
        f"B − A = {pts(g['delta'])}, 95% CI {pts(g['ci'][0])} to {pts(g['ci'][1])} (Newcombe), exact McNemar "
        f"p = {g['p']:.3g}; paired on {g['n']} seeds: A only {g['discordant']['a_only']}, "
        f"B only {g['discordant']['b_only']}",
    ]
    for key in ("mde_statement", "borderline_statement", "recipe_caveat"):
        if g.get(key):
            lines.append(g[key])
    return lines


def footer_lines(r: dict[str, Any]) -> list[str]:
    lines = [r["footer"]]
    lines += [f"Definition: {d}." for d in r["success_definitions"]]
    lines.append(
        f"rrc {r['rrc_version']}, report schema {r['schema_version']}; training seeds per side: {r['train_seeds']}"
    )
    return lines


def sections(r: dict[str, Any]) -> list[tuple[str, list[str]]]:
    return [
        ("Data", data_lines(r)),
        ("Policy", policy_lines(r)),
        ("Regression", regression_lines(r)),
        ("Verdict", [f"[{v['rule']}] {v['text']}" for v in r["verdict"]]),
        ("Can't tell", list(r["cant_tell"])),
    ]


def inputs_line(r: dict[str, Any]) -> str:
    i = r["inputs"]
    used = [f"score {i['score']}"] if i.get("score") else []
    used += [f"eval {e}" for e in i.get("evals", [])]
    used += [f"compare {i['compare']}"] if i.get("compare") else []
    return "Inputs: " + ("; ".join(used) or "none")


# ---- terminal ----------------------------------------------------------------------------------------------
def to_terminal(r: dict[str, Any]) -> str:
    out = [TITLE, "=" * len(TITLE), inputs_line(r)]
    out += [f"WARNING: {w}" for w in r["warnings"]]
    for name, lines in sections(r):
        out += ["", name, "-" * len(name), *[f"  {line}" for line in lines]]
    out += ["", *footer_lines(r)]
    return "\n".join(out)


# ---- markdown ----------------------------------------------------------------------------------------------
def to_markdown(r: dict[str, Any]) -> str:
    out = [f"# {TITLE}", "", inputs_line(r), ""]
    out += [f"> **Warning:** {w}" for w in r["warnings"]]
    for name, lines in sections(r):
        out += [f"## {name}", ""]
        if name == "Verdict":
            out += [f"- **{v['rule']}** {v['text']}" for v in r["verdict"]]
        else:
            out += [f"- {line}" for line in lines]
        out.append("")
    out += ["---", "", *[f"<sub>{line}</sub>  " for line in footer_lines(r)], ""]
    return "\n".join(out)


# ---- html --------------------------------------------------------------------------------------------------
PAGE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root { --fg: #1d2330; --muted: #5b6475; --bg: #ffffff; --card: #f5f7fa; --line: #d9dee7; --accent: #2f5fb3; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e9ef; --muted: #a3acbb; --bg: #14171d; --card: #1c2028; --line: #2d3340; --accent: #7aa2f7; }
}
body { font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg);
       max-width: 860px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 1.6em; margin: 0 0 4px; }
h2 { font-size: 1.1em; margin: 0 0 8px; color: var(--accent); }
section { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px;
          margin: 12px 0; }
ul { margin: 0; padding-left: 20px; }
li { margin: 4px 0; }
.rule { font-weight: 600; font-family: ui-monospace, monospace; margin-right: 6px; }
.muted, footer { color: var(--muted); font-size: 0.9em; }
.warn { border-color: #c98a1a; }
</style>
</head>
<body>
<h1>$title</h1>
<p class="muted">$inputs</p>
$warnings
$sections
<footer>$footer</footer>
</body>
</html>
"""
)


def to_html(r: dict[str, Any]) -> str:
    esc = html.escape
    parts = []
    for name, lines in sections(r):
        if name == "Verdict":
            items = "".join(
                f'<li><span class="rule">{esc(v["rule"])}</span>{esc(v["text"])}</li>' for v in r["verdict"]
            )
        else:
            items = "".join(f"<li>{esc(line)}</li>" for line in lines)
        parts.append(f"<section><h2>{esc(name)}</h2><ul>{items}</ul></section>")
    warnings = "".join(f'<section class="warn"><p>Warning: {esc(w)}</p></section>' for w in r["warnings"])
    return PAGE.substitute(
        title=esc(TITLE),
        inputs=esc(inputs_line(r)),
        warnings=warnings,
        sections="\n".join(parts),
        footer="<br>".join(esc(line) for line in footer_lines(r)),
    )
