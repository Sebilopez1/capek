"""`rrc eval` and `rrc compare`: seeded rollouts of a policy in so101_reach, with honest statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_report_card.commands.common import fail, report_path_problem, slug

POLICY_HELP = "scripted | random | bc:<checkpoint dir> (rrc train-bc) | lerobot:<pretrained_model dir>"
EVAL_DESCRIPTION = """\
Roll a policy out on N seeded episodes of so101_reach and report its success rate with a 95% CI.
success = gripper tip within 2 cm of the target on the final frame (90 frames, 3 s); LeRobot's eval counts
success at any step (reported too). Episode i resets from SeedSequence([eval_seed, i]); on the same machine the
same command gives identical results. Across machines, a rerun is a new sample, not a replay.

examples:
  rrc eval scripted --episodes 50
  rrc eval bc:ckpt/clean --episodes 200 --json-out reports/clean.json
"""
COMPARE_DESCRIPTION = """\
Run policies A and B on the SAME seeds and test whether B differs from A: discordant pairs, exact McNemar
p-value, Δ = p_B − p_A with a Newcombe paired 95% CI, and the smallest difference this n could detect at the
disagreement observed (80% power). A non-significant result is "No detectable difference", never "no difference".
For two trained checkpoints the CI covers rollout noise only, not retraining noise.

examples:
  rrc compare scripted random --episodes 50
  rrc compare bc:ckpt/all bc:ckpt/filtered --episodes 200 --eval-seed 900000
"""


def _common(p: argparse.ArgumentParser) -> None:
    from robot_report_card.eval.report import CI_METHODS
    from robot_report_card.eval.runner import DEFAULT_EPISODES, DEFAULT_EVAL_SEED, DEFAULT_MAX_STEPS

    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help=f"rollouts (default {DEFAULT_EPISODES})")
    p.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED, help=f"base seed (default {DEFAULT_EVAL_SEED})")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="frames per episode (default 90)")
    p.add_argument("--ci", choices=CI_METHODS, default="wilson", help="CI for success rates (default wilson)")
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "torch CPU threads for learned policies "
            "(keep it fixed when comparing runs; a different thread count can change results)"
        ),
    )
    p.add_argument("--json-out", help="JSON report path (default ./<policy>.rrc_eval.json / ...rrc_compare.json)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --json-out file")
    p.add_argument("--json", action="store_true", help="print the JSON report instead of the table")


def add_parser(sub: argparse._SubParsersAction) -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    e = sub.add_parser(
        "eval", help="success rate of a policy with a 95%% CI", description=EVAL_DESCRIPTION, formatter_class=fmt
    )
    e.add_argument("policy", help=POLICY_HELP)
    _common(e)
    e.set_defaults(func=run_eval)
    c = sub.add_parser(
        "compare",
        help="paired A/B test of two policies on the same seeds",
        description=COMPARE_DESCRIPTION,
        formatter_class=fmt,
    )
    c.add_argument("a", metavar="A", help="baseline policy: " + POLICY_HELP)
    c.add_argument("b", metavar="B", help="candidate policy")
    _common(c)
    c.set_defaults(func=run_compare)


def _json_path(args: argparse.Namespace, default_name: str) -> Path | None:
    if args.json_out:
        return Path(args.json_out)
    return None if args.json else Path.cwd() / default_name


def _emit(args: argparse.Namespace, path: Path | None, report: dict[str, Any], text: str) -> int:
    from robot_report_card.atomic import atomic_write_json

    if path is not None:
        try:
            atomic_write_json(path, report)
        except OSError as e:
            return fail(f"could not write {path}: {e}")
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(text if path is None else f"{text}\nJSON report: {path}")
    return 0


def _prepare(args: argparse.Namespace, specs: list[str], default_name: str) -> tuple[Path | None, str | None]:
    from robot_report_card.eval.policies import policy_dir

    if args.episodes < 1 or args.max_steps < 1 or args.eval_seed < 0:
        return None, "--episodes and --max-steps must be >= 1 and --eval-seed >= 0"
    if args.threads is not None and args.threads < 1:
        return None, "--threads must be >= 1"
    path = _json_path(args, default_name)
    if path is not None:
        protected = [d for d in (policy_dir(s) for s in specs) if d is not None]
        problem = report_path_problem(path, args.overwrite, protected)
        if problem:
            return None, problem
    return path, None


def _first_line(e: BaseException) -> str:
    """torch errors can carry several paragraphs of advice; the first line says what went wrong."""
    return (str(e).strip().splitlines() or [type(e).__name__])[0]


def _load(spec: str, threads: int | None) -> Any:
    from robot_report_card.eval.policies import load_policy

    return load_policy(spec, threads)


def run_eval(args: argparse.Namespace) -> int:
    from robot_report_card.eval.policies import PolicySpecError
    from robot_report_card.eval.report import eval_json, eval_text
    from robot_report_card.eval.runner import machine_info, run_rollouts

    path, problem = _prepare(args, [args.policy], f"{slug(args.policy)}.rrc_eval.json")
    if problem:
        return fail(problem)
    try:
        policy = _load(args.policy, args.threads)
        result = run_rollouts(policy, args.episodes, args.eval_seed, args.max_steps)
    except (PolicySpecError, ImportError, ValueError, OSError, RuntimeError) as e:
        return fail(_first_line(e))
    report = eval_json(result, args.ci, machine_info(args.threads))
    return _emit(args, path, report, eval_text(result, args.ci))


def run_compare(args: argparse.Namespace) -> int:
    from robot_report_card.eval.policies import PolicySpecError, is_trained
    from robot_report_card.eval.report import compare_json, compare_stats, compare_text
    from robot_report_card.eval.runner import machine_info, run_rollouts

    name = f"{slug(args.a)}_vs_{slug(args.b)}.rrc_compare.json"
    path, problem = _prepare(args, [args.a, args.b], name)
    if problem:
        return fail(problem)
    try:
        pa, pb = _load(args.a, args.threads), _load(args.b, args.threads)
        ra = run_rollouts(pa, args.episodes, args.eval_seed, args.max_steps)
        rb = run_rollouts(pb, args.episodes, args.eval_seed, args.max_steps)
    except (PolicySpecError, ImportError, ValueError, OSError, RuntimeError) as e:
        return fail(_first_line(e))
    comp = compare_stats(ra, rb)
    caveat = is_trained(args.a) and is_trained(args.b)
    report = compare_json(ra, rb, comp, args.ci, caveat, machine_info(args.threads))
    return _emit(args, path, report, compare_text(ra, rb, comp, args.ci, caveat))
