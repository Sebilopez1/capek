"""``python -m capek.bench {build,check,evaluate,realify}`` (QA-owned, plan P2-3)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _score(dataset: Path, out: Path) -> None:
    """Score through the plan's public interface: the ``capek score`` CLI and its JSON report."""
    from capek.bench.build import BenchError

    if not (dataset / "meta" / "info.json").is_file():
        raise BenchError(f"{dataset} is not a built benchmark dataset (run `bench build` first)")
    cmd = [sys.executable, "-m", "capek.cli", "score", str(dataset), "--json-out", str(out), "--overwrite"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise BenchError(f"capek score {dataset} failed (exit {r.returncode}): {r.stderr.strip()}")


def net_formula_power_check(mixed_dir: Path) -> dict:
    """Score a (low-noise) mixed set with the phase 2 ``net`` tv denominator and count return_home flags.

    This checks the *benchmark*, not the scorer under test: the R-real-1 bar only means something if the old
    formula fails it somewhere (plan D3: >= 2/10 on at least one low-noise seed set).
    """
    from capek.score.engine import ScoreConfig, score_dataset
    from capek.score.reader import read_dataset

    gt = json.loads((mixed_dir / "gt.json").read_text())["episodes"]
    res = score_dataset(read_dataset(mixed_dir / "dataset"), ScoreConfig(tv_mode="net"))
    rth = [e for e in res.episodes if gt[str(e.episode_index)]["group"] == "return_home"]
    return {"formula": "net", "return_home_flagged": sum(e.quality != "ok" for e in rth), "return_home_n": len(rth)}


def _power_text(r: dict) -> str:
    pc = r.get("power_check")
    if not pc:
        return ""
    return (
        f"\npower check (phase 2 net formula on this low-noise set): return_home flagged "
        f"{pc['return_home_flagged']}/{pc['return_home_n']} (the old formula must flag >= 2/10 on some low-noise set)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m capek.bench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser(
        "build", help="build benchmark v2 (mixed + clean-only sets, gt.json); needs sim + lerobot extras"
    )
    b.add_argument("--out", required=True)
    b.add_argument("--seeds", required=True, help='dev | heldout | file:PATH with {"mixed": int, "clean_only": int}')
    b.add_argument("--small", action="store_true", help="47 + 20 episodes for fast tests (bars not meaningful)")
    rm = sub.add_parser("record", help="record a custom mix of bench groups into one session (e.g. the P3-8 recipe)")
    rm.add_argument("--out", required=True, help="new session directory")
    rm.add_argument("--groups", required=True, help="e.g. clean:60,noise025:10,random:10,hesitation:5,wrong:5")
    rm.add_argument("--seed", type=int, required=True, help="group k records with seed + k")
    rm.add_argument("--noise", default="standard", help="noise profile: standard | low")
    c = sub.add_parser("check", help="run `capek score` on a built benchmark dir, then evaluate the DoD 4 bars")
    c.add_argument("dir")
    c.add_argument("--json", action="store_true")
    e = sub.add_parser("evaluate", help="judge DoD 4 bars from capek score JSON reports + gt.json")
    e.add_argument("mixed_score")
    e.add_argument("mixed_gt")
    e.add_argument(
        "--clean-only", nargs=2, metavar=("SCORE_JSON", "GT_JSON"), help="clean-only set (needed for its bar)"
    )
    e.add_argument("--json", action="store_true")
    r = sub.add_parser("realify", help="real-like + corrupt fixtures from a capek-exported v3.0 dataset")
    r.add_argument("dataset")
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from capek.bench.build import BenchError, build
    from capek.bench.evaluate import EvalError, evaluate_files, format_result
    from capek.session import SessionError

    try:
        if args.cmd == "build":
            t0 = time.perf_counter()
            gt = build(args.out, args.seeds, args.small)
            for name, g in gt.items():
                d = g["dataset"]
                print(f"{name}: {d['total_episodes']} episodes / {d['total_frames']} frames -> {Path(args.out, name)}")
            print(f"built in {time.perf_counter() - t0:.1f} s (seed set {gt['mixed']['seed_set']})")
            return 0
        if args.cmd == "record":
            from capek.bench.build import record_mix

            rows = record_mix(args.out, args.groups, args.seed, args.noise)
            print(f"recorded {len(rows)} episodes ({args.groups}) to {args.out}")
            return 0
        if args.cmd == "realify":
            from capek.bench.realify import realify

            m = realify(args.dataset, args.out, args.seed)
            print(
                "wrote "
                + ", ".join(k for k in m if k not in ("source", "source_codebase_version"))
                + f" under {args.out}"
            )
            return 0
        if args.cmd == "check":
            d = Path(args.dir)
            profiles = [("standard", "")] + ([("low", "_low")] if (d / "mixed_low").is_dir() else [])
            results = {}
            for profile, suffix in profiles:
                for s in (f"mixed{suffix}", f"clean_only{suffix}"):
                    _score(d / s / "dataset", d / s / "score.json")
                res = evaluate_files(
                    d / f"mixed{suffix}" / "score.json",
                    d / f"mixed{suffix}" / "gt.json",
                    d / f"clean_only{suffix}" / "score.json",
                    d / f"clean_only{suffix}" / "gt.json",
                )
                if profile == "low":
                    res["power_check"] = net_formula_power_check(d / "mixed_low")
                results[profile] = res
            ok = all(r["all_pass"] for r in results.values())
            if args.json:
                print(json.dumps({"all_pass": ok, **results}, indent=2, allow_nan=True))
            else:
                print("\n\n".join(format_result(r) + _power_text(r) for r in results.values()))
                print("\nCHECK PASSED" if ok else "\nCHECK FAILED")
            return 0 if ok else 1
        co = args.clean_only or (None, None)
        res = evaluate_files(args.mixed_score, args.mixed_gt, co[0], co[1])
    except (BenchError, EvalError, SessionError, ValueError, OSError) as err:
        print(f"bench: error: {err}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2, allow_nan=True) if args.json else format_result(res))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
