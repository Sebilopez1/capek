"""`capek` command-line entry point. Heavy dependencies (mujoco, lerobot) are imported lazily per subcommand."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from capek import __version__
from capek.commands import evaluate, export, listing, record, report_card, score, tag, train_bc

EPILOG = """\
typical flow:
  capek record --env so101_reach --policy scripted --episodes 10 --seed 0 --out runs/demo
  capek tag runs/demo --episode 3 --label fail --note "overshot"
  capek list runs/demo
  capek export runs/demo --out datasets/demo --repo-id local/demo
  capek score datasets/demo
  capek eval scripted --episodes 50
  capek compare scripted random --episodes 50
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="capek",
        description="Capek: record sim episodes, tag them, export to LeRobot v3.0, score datasets.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"capek {__version__}")
    sub = p.add_subparsers(
        dest="command", metavar="{record,tag,list,export,score,train-bc,eval,compare,report}", required=True
    )
    for command in (record, tag, listing, export, score, train_bc, evaluate, report_card):
        command.add_parser(sub)
    return p


EXIT_BROKEN_PIPE = 141  # 128 + SIGPIPE, what shells report for e.g. `yes | head`


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = int(args.func(args) or 0)
        sys.stdout.flush()  # surface a closed pipe here, not as "Exception ignored" at interpreter exit
    except BrokenPipeError:
        # The reader went away (`capek list runs/demo | head`): stop quietly. Point stdout at devnull so the
        # interpreter's final flush doesn't raise again (pattern from the Python `signal` docs).
        _silence_stdout()
        return EXIT_BROKEN_PIPE
    return code


def _silence_stdout() -> None:
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):  # stdout without a real fd (e.g. captured in tests)
        pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
