"""`capek list`: show a session's episodes, sim results and tags."""

from __future__ import annotations

import argparse

from capek.commands.common import fail
from capek.session import LABELS, Session, SessionError
from capek.tagging import format_json, format_table


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "list",
        help="show a session's episodes and their tags",
        description="Show index, length, sim_success, final_error_m, label, policy, flags and notes per episode.",
    )
    p.add_argument("session", help="session directory created by `capek record`")
    p.add_argument("--json", action="store_true", help="print all fields of every episode as a JSON array")
    p.add_argument("--label", choices=LABELS, help="only show episodes with this label")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        metas = Session.open(args.session).read_metas()
    except SessionError as e:
        return fail(str(e))
    if args.label:
        metas = [m for m in metas if m.label == args.label]
    print(format_json(metas) if args.json else format_table(metas))
    return 0
