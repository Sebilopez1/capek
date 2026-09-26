"""`rrc score`: motion-quality report for a LeRobot dataset on disk (needs the score extra; never lerobot/torch)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from robot_report_card.commands.common import fail

DESCRIPTION = """\
Score every episode of a LeRobot dataset (v3.0, v2.1 or v2.0 on disk) for motion quality, and show outcome
evidence separately. Quality is relative to the dataset (robust z per signal, flag when the max z > --threshold),
so it assumes most episodes are good. Motion quality can't detect failed attempts that move normally (early stop,
wrong goal). Only outcome evidence (success column or labels) can.

Outcome shows the human label from meta/rrc_tags.json if present, else the last-frame next.success, else
"unknown", with the source; "≠ sim" marks a label that contradicts the sim. The dataset is never modified.

examples:
  rrc score datasets/demo
  rrc score datasets/demo --only-flagged --json-out reports/demo.json --overwrite
  rrc score ~/rrc-real/so101 --state-key observation.state --action-key action
  rrc score datasets/demo --json | jq '.summary'
"""


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "score",
        help="score a LeRobot dataset's episodes for motion quality (needs the score extra)",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset", help="LeRobot dataset root (the folder containing meta/info.json)")
    p.add_argument(
        "--json-out",
        help="where to write the JSON report (default: ./<dataset dir name>.rrc_score.json; with --json, "
        "only written when given)",
    )
    p.add_argument("--overwrite", action="store_true", help="replace an existing --json-out file")
    p.add_argument("--threshold", type=float, default=3.5, help="flag when the combined z exceeds this (default 3.5)")
    p.add_argument("--state-key", default="observation.state", help="state column (default observation.state)")
    p.add_argument("--action-key", default="action", help="action column (default action)")
    p.add_argument("--only-flagged", action="store_true", help="table: only show FLAG / HARD episodes")
    p.add_argument("--json", action="store_true", help="print the JSON report to stdout instead of the table")
    p.set_defaults(func=run)


def _json_path(args: argparse.Namespace, dataset: Path) -> Path | None:
    if args.json_out:
        return Path(args.json_out)
    if args.json:
        return None
    return Path.cwd() / f"{dataset.resolve().name}.rrc_score.json"


def _check_json_path(path: Path, dataset: Path, overwrite: bool) -> str | None:
    """Return an error message if the report can't be written to ``path``."""
    target, root = path.resolve(), dataset.resolve()
    if target == root or root in target.parents:
        return f"refusing to write the report inside the dataset ({path}); pass --json-out somewhere else"
    if path.is_dir():
        return f"--json-out {path} is a directory"
    if path.exists() and not overwrite:
        return f"{path} already exists; pass --overwrite to replace it (or choose another --json-out)"
    return None


def run(args: argparse.Namespace) -> int:

    from robot_report_card.atomic import atomic_write_json
    from robot_report_card.score.engine import ScoreConfig, score_dataset
    from robot_report_card.score.reader import ReaderError, read_dataset
    from robot_report_card.score.report import build_json, format_summary, format_table

    if not math.isfinite(args.threshold):
        return fail("--threshold must be a finite number")
    dataset = Path(args.dataset)
    json_path = _json_path(args, dataset)
    if json_path is not None:
        problem = _check_json_path(json_path, dataset, args.overwrite)
        if problem:
            return fail(problem)
    try:
        ds = read_dataset(dataset, state_key=args.state_key, action_key=args.action_key)
        result = score_dataset(ds, ScoreConfig(flag_z=args.threshold))
    except (ReaderError, ValueError) as e:
        return fail(str(e))
    report = build_json(result, ds)
    if json_path is not None:
        try:
            atomic_write_json(json_path, report)
        except OSError as e:
            return fail(f"could not write {json_path}: {e}")
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(format_table(result, only_flagged=args.only_flagged))
        print(format_summary(result, ds, str(json_path) if json_path else None))
    return 0
