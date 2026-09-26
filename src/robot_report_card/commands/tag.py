"""`rrc tag`: edit labels / notes / flags in a session's episodes.jsonl (atomic rewrite)."""

from __future__ import annotations

import argparse

from robot_report_card.commands.common import fail, parse_indices
from robot_report_card.session import LABELS, Session, SessionError
from robot_report_card.tagging import TagChange, apply_tags, labels_from_sim

DESCRIPTION = """\
Tag episodes of a session. Changes go to <session>/episodes.jsonl (written atomically) and are carried
into meta/rrc_tags.json on the next `rrc export`. sim_success is never edited; label is the human verdict.

examples:
  rrc tag runs/demo --episode 3 --label fail --note "overshot"
  rrc tag runs/demo --episode 0-2,5 --flag jerky
  rrc tag runs/demo --episode 3 --clear-label --clear-notes --unflag jerky
  rrc tag runs/demo --from-sim                 # label = success/fail from sim_success, unlabeled episodes only
"""


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tag",
        help="set labels / notes / flags on episodes of a session",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("session", help="session directory created by `rrc record`")
    sel = p.add_argument_group("which episodes")
    sel.add_argument(
        "-e", "--episode", action="append", metavar="N", help="episode index; repeatable, accepts N,M and N-M"
    )
    sel.add_argument("--all", action="store_true", help="apply to every episode")
    what = p.add_argument_group("what to change")
    lab = what.add_mutually_exclusive_group()
    lab.add_argument("--label", choices=LABELS, help="set the human label")
    lab.add_argument("--clear-label", action="store_true", help="reset the label to 'unlabeled'")
    lab.add_argument(
        "--from-sim", action="store_true", help="set label = success/fail from sim_success (default: all episodes)"
    )
    what.add_argument("--overwrite-labels", action="store_true", help="with --from-sim: also relabel labelled episodes")
    what.add_argument("--note", help="append a note (joined with '; ')")
    what.add_argument("--replace-notes", action="store_true", help="with --note: replace the notes instead")
    what.add_argument("--clear-notes", action="store_true", help="remove all notes")
    what.add_argument("--flag", action="append", default=[], metavar="NAME", help="add a flag; repeatable")
    what.add_argument("--unflag", action="append", default=[], metavar="NAME", help="remove a flag; repeatable")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        indices = parse_indices(args.episode)
    except argparse.ArgumentTypeError as e:
        return fail(str(e))
    if indices is not None and args.all:
        return fail("use either --episode or --all, not both")
    if args.overwrite_labels and not args.from_sim:
        return fail("--overwrite-labels only applies to --from-sim")
    try:
        session = Session.open(args.session)
        if args.from_sim:
            if args.note is not None or args.clear_notes or args.flag or args.unflag:
                return fail("--from-sim cannot be combined with note/flag changes; run two commands")
            changed, skipped = labels_from_sim(session, indices, overwrite=args.overwrite_labels)
            extra = f", {skipped} already-labelled skipped (use --overwrite-labels)" if skipped else ""
            print(f"labels from sim_success: {changed} changed{extra}")
            return 0
        if indices is None and not args.all:
            return fail("choose episodes with --episode N (or --all)")
        change = TagChange(
            label="unlabeled" if args.clear_label else args.label,
            note=args.note,
            replace_notes=args.replace_notes,
            clear_notes=args.clear_notes,
            add_flags=args.flag,
            remove_flags=args.unflag,
        )
        updated = apply_tags(session, indices, change)
    except SessionError as e:
        return fail(str(e))
    for m in updated:
        print(f"episode {m.episode_index}: label={m.label} flags={m.flags} notes={m.notes!r}")
    return 0
