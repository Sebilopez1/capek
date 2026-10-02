"""`capek record`: record sim episodes into a session directory (one policy, or a --mix of benchmark groups)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from capek.commands.common import fail
from capek.session import Session, SessionError

MIX_MAX_STEPS = 90  # the benchmark generators assume 3 s episodes (ramps <= 2.5 s, return_home <= 2.8 s)
DEFAULT_POLICY, DEFAULT_EPISODES, DEFAULT_NOISE = "scripted", 10, 0.0
GROUP_HELP = {  # one line per benchmark group (names are a public API from 0.1.0)
    "clean": "scripted reach, ramp 1.0-2.5 s, small action noise (sigma 0.01-0.04; low profile 0.002-0.008)",
    "nearmiss": "like clean without gravity compensation: sags, often ends just outside 2 cm",
    "noise005": "clean timing with action noise sigma 0.05",
    "noise010": "clean timing with action noise sigma 0.10",
    "noise025": "clean timing with action noise sigma 0.25",
    "hesitation": "clean reach that pauses 0.8 s partway, then resumes",
    "wobble": "clean reach plus band-limited AR(1) jitter (phi 0.9, sigma 0.10)",
    "stall": "clean reach whose command freezes partway (early stop)",
    "wrong": "clean-looking reach to an independent wrong goal (only the outcome differs)",
    "random": "uniform random joint targets every frame",
    "return_home": "out to the target, short hold, back to the start pose (clean motion, ends away from the target)",
}
EPILOG = """\
examples:
  capek record --policy scripted --episodes 10 --seed 0 --out runs/demo
  capek record --mix clean:60,noise025:10,random:10,hesitation:5,wrong:5 --seed 30000 --out runs/mix
  capek record --list-groups
"""


def add_parser(sub: argparse._SubParsersAction) -> None:
    from capek.policies import POLICY_NAMES, WRONG_GOALS
    from capek.sim.registry import ENV_IDS

    rec = sub.add_parser(
        "record",
        help="record episodes from a simulated env into a session directory",
        description=(
            "Record fixed-length episodes (no early termination) into a new session directory (or, with --append, "
            "an existing one): session.json + episodes.jsonl + episodes/*.npz. Needs the sim extra (mujoco). "
            "Episode i uses numpy SeedSequence([seed, i]), so every episode is reproducible on its own "
            "and runs with different --seed values never share episodes. "
            "--mix records several benchmark groups into one session (group k uses seed + k)."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rec.add_argument("--env", default="so101_reach", choices=ENV_IDS, help="simulated environment")
    rec.add_argument(
        "--policy",
        choices=POLICY_NAMES,
        help="scripted = ramp to target (good demos, the default); random = uniform joint targets (junk); "
        "stall = freezes after 1 s; wrong = reaches an independent wrong goal (see --wrong-goal)",
    )
    rec.add_argument(
        "--wrong-goal",
        choices=WRONG_GOALS,
        help="--policy wrong only: independent (default) = a different reachable goal drawn like real targets; "
        "mirrored = the mirrored pose -q* (phase 1 behaviour; hits joint limits)",
    )
    rec.add_argument("--episodes", type=int, help=f"number of episodes to record (default {DEFAULT_EPISODES})")
    rec.add_argument("--seed", type=int, default=0, help="base seed (>= 0; default 0)")
    rec.add_argument("--noise", type=float, help="std-dev (rad) of Gaussian noise added to actions (default 0)")
    rec.add_argument("--max-steps", type=int, default=90, help="frames per episode (fps=30, so 90 = 3 s; default 90)")
    rec.add_argument(
        "--no-gravity-comp",
        action="store_true",
        help="scripted/stall/wrong: drop the gravity offset (arm sags; about 75%% success instead of ~100%%)",
    )
    mix = rec.add_argument_group("benchmark mixes")
    mix.add_argument(
        "--mix",
        metavar="GROUP:N,...",
        help="record benchmark groups into one session, e.g. clean:60,noise025:10,random:10,hesitation:5,wrong:5; "
        "group k uses seed + k; needs --max-steps 90; not combinable with --policy/--episodes/--noise",
    )
    mix.add_argument(
        "--noise-profile", choices=("standard", "low"), help="--mix only: per-episode noise level (default standard)"
    )
    mix.add_argument("--list-groups", action="store_true", help="list the --mix groups with their class and exit")
    rec.add_argument("--out", help="session directory to create (or to extend with --append)")
    rec.add_argument(
        "--append",
        action="store_true",
        help="add episodes to the existing session at --out (env, fps, --max-steps and features must match; "
        "indices continue, so you can mix policies/noise in one session). Creates the session if --out is new",
    )
    rec.add_argument("-q", "--quiet", action="store_true", help="only print the summary line")
    rec.set_defaults(func=run)


def list_groups() -> str:
    from capek.bench.spec import MIXED

    width = max(len(g) for g, *_ in MIXED)
    lines = [f"{'group':{width}s}  {'class':13s}  description"]
    lines += [f"{g:{width}s}  {cls:13s}  {GROUP_HELP.get(g, '')}" for g, cls, *_ in MIXED]
    lines.append("classes: clean_variant = clean motion that ends away from the target (not junk)")
    return "\n".join(lines)


def _mix_problem(args: argparse.Namespace) -> str | None:
    clash = [f for f, v in (("--policy", args.policy), ("--episodes", args.episodes), ("--noise", args.noise),
                             ("--wrong-goal", args.wrong_goal)) if v is not None]  # fmt: skip
    if args.no_gravity_comp:
        clash.append("--no-gravity-comp")
    if clash:
        return f"--mix can't be combined with {', '.join(clash)} (each group sets its own policy and noise)"
    if args.max_steps != MIX_MAX_STEPS:
        return f"--mix needs --max-steps {MIX_MAX_STEPS} (the benchmark generators assume 3 s episodes)"
    return None


def _mix_plan(args: argparse.Namespace) -> list[tuple[str, int, Any]]:
    """[(group, episodes, policy)] for --mix; raises ValueError on a bad spec."""
    from capek.bench.build import BenchError, parse_groups
    from capek.bench.generators import GroupPolicy
    from capek.bench.spec import NOISE_PROFILES

    try:
        groups = parse_groups(args.mix)
    except BenchError as e:
        raise ValueError(str(e)) from e
    sigma = NOISE_PROFILES[args.noise_profile or "standard"]
    return [(g, n, GroupPolicy(g, sigma)) for g, n in groups]


def run(args: argparse.Namespace) -> int:
    from capek.policies import make_policy
    from capek.record import new_session, open_for_append, record_into
    from capek.session import EpisodeMeta
    from capek.sim.registry import make_env

    if args.list_groups:
        print(list_groups())
        return 0
    if not args.out:
        return fail("--out is required (the session directory to record into)")
    if args.seed < 0 or args.max_steps < 1:
        return fail("--max-steps must be >= 1 and --seed >= 0")
    if args.mix is not None:
        problem = _mix_problem(args)
        if problem:
            return fail(problem)
    elif args.noise_profile is not None:
        return fail("--noise-profile only applies to --mix")
    elif args.wrong_goal is not None and args.policy != "wrong":
        return fail("--wrong-goal only applies to --policy wrong")
    episodes = DEFAULT_EPISODES if args.episodes is None else args.episodes
    if args.mix is None and episodes < 1:
        return fail("--episodes must be >= 1")
    try:
        if args.mix is not None:
            plan = _mix_plan(args)
        else:
            policy = make_policy(
                args.policy or DEFAULT_POLICY,
                noise=DEFAULT_NOISE if args.noise is None else args.noise,
                gravity_comp=not args.no_gravity_comp,
                wrong_goal=args.wrong_goal or "independent",
            )
            plan = [(None, episodes, policy)]
        env = make_env(args.env)
        if args.append and Path(args.out, "episodes.jsonl").is_file():
            session = open_for_append(env, args.out, args.max_steps)
        else:
            session = new_session(env, args.out, args.max_steps)
    except (ImportError, ValueError, SessionError) as e:
        return fail(str(e))

    def report(m: EpisodeMeta) -> None:
        if not args.quiet:
            ok = "success" if m.sim_success else "miss"
            err_cm = m.final_error_m * 100
            group = m.policy_params.get("bench_group")
            tag = f" [{group}]" if group else ""
            print(f"episode {m.episode_index:4d}{tag}: {m.num_frames} frames, final error {err_cm:6.2f} cm ({ok})")

    before = len(session.read_metas())
    try:
        for k, (_, n, policy) in enumerate(plan):
            seed = args.seed + k if args.mix is not None else args.seed
            record_into(session, env, policy, n, seed, args.max_steps, on_episode=report)
    except BrokenPipeError:
        raise  # `capek record | head`: cli.main exits 141; every finished episode is already saved
    except (ValueError, FloatingPointError, OSError, SessionError) as e:
        saved = _saved_count(session)
        return fail(f"recording stopped: {e}. {saved} episodes saved in {session.root}")
    metas = session.read_metas()
    new = metas[before:]
    n_ok = sum(m.sim_success for m in new)
    total = f"; session now has {len(metas)}" if before else ""
    what = f" ({args.mix})" if args.mix is not None else ""
    print(f"recorded {len(new)} episodes{what} to {session.root} (sim success {n_ok}/{len(new)}{total})")
    return 0


def _saved_count(session: Session) -> str:
    try:
        return str(len(session.read_metas()))
    except (OSError, SessionError):
        return "an unknown number of"
