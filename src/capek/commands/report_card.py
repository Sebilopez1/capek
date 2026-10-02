"""`capek report`: combine `capek score`, `capek eval` and `capek compare` JSON into one report card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from capek.commands.common import fail, report_path_problem

DESCRIPTION = """\
Build a one-page report card from the JSON reports of `capek score` (the dataset), `capek eval` (policies) and
`capek compare` (an A/B regression test). Blocks: Data, Policy, Regression, Verdict and Can't tell. The verdict
applies fixed, ordered rules (sim-tuned thresholds) and says what the evidence is consistent with; it never
assigns a cause. A bc: checkpoint counts as trained on the scored dataset only when its fingerprint matches.

The card is always printed; --md and --html also write it as Markdown or a self-contained HTML page, and
report.json (--json-out) holds everything the renderings show.

example:
  capek report --score mix.capek_score.json --compare all_vs_filtered.capek_compare.json --md card.md --html card.html
"""


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "report",
        help="one-page report card from score / eval / compare JSON",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--score", help="`capek score` JSON for the dataset")
    p.add_argument("--eval", action="append", default=[], metavar="EVAL.json", help="`capek eval` JSON; repeatable")
    p.add_argument("--compare", help="`capek compare` JSON")
    p.add_argument(
        "--train-seeds", type=int, default=1, help="training seeds per side behind the compared checkpoints (default 1)"
    )
    p.add_argument("--md", help="also write the card as Markdown")
    p.add_argument("--html", help="also write the card as a self-contained HTML page")
    p.add_argument("--json-out", help="report.json path (default ./capek_report.json)")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    p.add_argument("--json", action="store_true", help="print report.json instead of the terminal card")
    p.set_defaults(func=run)


def _load(path: str, kind: str) -> dict[str, Any]:
    from capek.report.rules import ReportInputError, check_kind

    try:
        obj = json.loads(Path(path).read_text())
    except OSError as e:
        raise ReportInputError(f"can't read {path}: {e.strerror or e}") from e
    except ValueError as e:
        raise ReportInputError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ReportInputError(f"{path} is not a `capek {kind}` JSON report")
    check_kind(obj, kind, path)
    return obj


def run(args: argparse.Namespace) -> int:
    from capek.atomic import atomic_write
    from capek.report.render import to_html, to_markdown, to_terminal
    from capek.report.rules import ReportInputError, build_report

    if args.train_seeds < 1:
        return fail("--train-seeds must be >= 1")
    if not (args.score or args.eval or args.compare):
        return fail("give at least one of --score, --eval, --compare")
    outputs = {"json": Path(args.json_out) if args.json_out else (None if args.json else Path("capek_report.json"))}
    outputs["md"] = Path(args.md) if args.md else None
    outputs["html"] = Path(args.html) if args.html else None
    for path in outputs.values():
        if path is not None:
            problem = report_path_problem(path, args.overwrite, [])
            if problem:
                return fail(problem)
    try:
        score = _load(args.score, "score") if args.score else None
        evals = [_load(p, "eval") for p in args.eval]
        compare = _load(args.compare, "compare") if args.compare else None
        inputs = {"score": args.score, "evals": list(args.eval), "compare": args.compare}
        report = build_report(score, evals, compare, args.train_seeds, {k: v for k, v in inputs.items() if v})
    except (ReportInputError, KeyError, TypeError) as e:
        msg = str(e) if isinstance(e, ReportInputError) else f"an input JSON is missing a field ({e})"
        return fail(msg)
    rendered = {
        "json": json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        "md": to_markdown(report),
        "html": to_html(report),
    }
    for key, path in outputs.items():
        if path is not None:
            data = rendered[key].encode("utf-8")
            try:
                atomic_write(path, lambda fh, data=data: fh.write(data))
            except OSError as e:
                return fail(f"could not write {path}: {e}")
    print(rendered["json"] if args.json else to_terminal(report))
    return 0
