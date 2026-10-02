"""`capek export`: session -> LeRobot v3.0 dataset + meta/capek_tags.json (needs the lerobot extra)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from capek.commands.common import fail
from capek.session import LABELS, SessionError

DESCRIPTION = """\
Export a session to a local LeRobot v3.0 dataset (lerobot==0.4.4 writer, state-only, no videos) and
write the tag snapshot <out>/meta/capek_tags.json. Episodes are renumbered 0..N-1 when some are excluded;
capek_tags.json keeps each one's source_episode_index. To retag: `capek tag` the session, re-export with
--overwrite. Nothing is uploaded (HF_HUB_OFFLINE defaults to 1 for this command).

example:
  capek export runs/demo --out datasets/demo --repo-id local/demo --exclude-label fail
"""


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "export",
        help="export a session to a LeRobot v3.0 dataset (needs the lerobot extra)",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("session", help="session directory created by `capek record`")
    p.add_argument("--out", required=True, help="dataset root to create (must not exist unless --overwrite)")
    p.add_argument("--repo-id", help="dataset repo id stored in the dataset (default: local/<out dir name>)")
    p.add_argument(
        "--exclude-label", action="append", default=[], choices=LABELS, help="skip episodes with this label; repeatable"
    )
    p.add_argument("--overwrite", action="store_true", help="delete an existing dataset at --out first")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # phase 1 is local-only; never fall through to the Hub (G7)
    from capek.export.lerobot_writer import ExportError, export_session, quiet_lerobot

    out = Path(args.out)
    repo_id = args.repo_id or f"local/{out.resolve().name}"
    try:
        with quiet_lerobot():
            result = export_session(args.session, out, repo_id, args.exclude_label, args.overwrite)
    except (ExportError, SessionError, ValueError) as e:  # ValueError: e.g. arrays that don't match the spec
        return fail(str(e))
    n = len(result.source_indices)
    print(f"exported {n} episodes / {result.total_frames} frames to {result.root} (repo_id {result.repo_id})")
    print(f"tags snapshot: {result.root / 'meta' / 'capek_tags.json'}")
    return 0
