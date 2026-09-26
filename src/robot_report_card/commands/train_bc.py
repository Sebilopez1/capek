"""`rrc train-bc`: train the tiny BC MLP on an rrc export (CPU, torch; needs the eval extra)."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_report_card.commands.common import fail

DESCRIPTION = """\
Train a small behavior-cloning MLP ([observation.state, observation.environment_state] -> action - state,
9 -> 256 -> 256 -> 6) on a LeRobot dataset exported by `rrc export`, on the CPU. The checkpoint directory holds
model.pt + rrc_policy.json and can be evaluated as `bc:<dir>` with `rrc eval` / `rrc compare`.

--keep filters the training episodes:
  all             every episode
  quality-ok      `rrc score` quality == ok (needs --score-json)
  success         outcome evidence says success (label in rrc_tags.json, else last-frame next.success)
  ok-and-success  both

The same --seed on the same machine gives identical weights (weights_sha256 in rrc_policy.json).

examples:
  rrc score datasets/mix --json-out mix.score.json
  rrc train-bc datasets/mix --out ckpt/all
  rrc train-bc datasets/mix --out ckpt/filtered --keep ok-and-success --score-json mix.score.json
"""


def add_parser(sub: argparse._SubParsersAction) -> None:
    from robot_report_card.eval.bc import KEEP_CHOICES

    p = sub.add_parser(
        "train-bc",
        help="train a tiny BC policy on an exported dataset (needs the eval extra)",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset", help="LeRobot dataset root from `rrc export`")
    p.add_argument("--out", required=True, help="checkpoint directory to create")
    p.add_argument("--keep", choices=KEEP_CHOICES, default="all", help="which episodes to train on (default all)")
    p.add_argument("--score-json", help="`rrc score` JSON report for this dataset (quality filters)")
    p.add_argument("--seed", type=int, default=0, help="training seed (default 0)")
    p.add_argument("--epochs", type=int, default=40, help="default 40")
    p.add_argument("--batch-size", type=int, default=256, help="default 256")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default 1e-3)")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--overwrite", action="store_true", help="replace an existing rrc checkpoint at --out")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    import math

    from robot_report_card.eval.bc import TrainConfig, TrainError, train_bc
    from robot_report_card.score.reader import ReaderError

    if args.epochs < 1 or args.batch_size < 1 or not (math.isfinite(args.lr) and args.lr > 0) or args.seed < 0:
        return fail("--epochs and --batch-size must be >= 1, --lr > 0 and --seed >= 0")
    if args.threads is not None and args.threads < 1:
        return fail("--threads must be >= 1")
    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, threads=args.threads)
    try:
        meta = train_bc(
            Path(args.dataset),
            Path(args.out),
            keep=args.keep,
            score_json=Path(args.score_json) if args.score_json else None,
            cfg=cfg,
            overwrite=args.overwrite,
        )
    except (TrainError, ReaderError, OSError) as e:
        return fail(str(e))
    t = meta["train"]
    print(
        f"trained bc:{args.out} on {len(meta['kept_episodes'])}/{meta['dataset']['total_episodes']} episodes "
        f"({t['frames']} frames, --keep {args.keep}) in {t['seconds']:.1f} s; seed {t['seed']}, "
        f"weights {meta['weights_sha256'][:12]}"
    )
    return 0
