"""Build benchmark v2: one appended session per set, exported once, plus ``gt.json`` kept OUTSIDE the dataset.

Layout of ``--out DIR``::

    DIR/mixed/{session/, dataset/, gt.json}            160 episodes (50 with --small), standard noise
    DIR/clean_only/{session/, dataset/, gt.json}       100 clean episodes (20 with --small), standard noise
    DIR/mixed_low/..., DIR/clean_only_low/...           the same tables and seeds, whole-set low noise (plan D3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from capek.bench.generators import GroupPolicy
from capek.bench.spec import CLEAN_ONLY, MAX_STEPS, MIXED, NOISE_PROFILES, SEED_SETS

GT_SCHEMA_VERSION = 1


class BenchError(Exception):
    """User-facing benchmark problem."""


def resolve_seeds(spec: str) -> tuple[str, dict[str, int], bool]:
    """``dev`` / ``heldout`` / ``file:PATH`` -> (name, {"mixed", "clean_only"}, private)."""
    if spec in SEED_SETS:
        return spec, dict(SEED_SETS[spec]), False
    if spec.startswith("file:"):
        path = Path(spec[5:]).expanduser()
        try:
            d = json.loads(path.read_text())
            seeds = {"mixed": int(d["mixed"]), "clean_only": int(d["clean_only"])}
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise BenchError(f"cannot read seed file {path}: {e}") from e
        published = {b + k for s in SEED_SETS.values() for b in s.values() for k in range(len(MIXED))}
        mine = {b + k for b in seeds.values() for k in range(len(MIXED))}
        if published & mine:
            raise BenchError(f"seed file {path} overlaps the published dev/heldout seeds; pick other bases")
        return str(d.get("name", "private")), seeds, True
    raise BenchError(f"--seeds must be dev, heldout or file:PATH, got {spec!r}")


def _record(
    session_dir: Path, table: tuple[tuple[str, str, int, int], ...], base: int, small: bool, profile: str = "standard"
) -> list[dict[str, Any]]:
    from capek.record import new_session, record_into
    from capek.sim.registry import make_env

    env = make_env("so101_reach")
    session = new_session(env, session_dir, max_steps=MAX_STEPS)
    rows: list[dict[str, Any]] = []
    for k, (group, cls, n_full, n_small) in enumerate(table):
        n = n_small if small else n_full
        policy = GroupPolicy(group, NOISE_PROFILES[profile])
        record_into(session, env, policy, n, base + k, MAX_STEPS)  # the phase 1 --append path
        rows += [{"group": group, "class": cls, "seed": base + k}] * n
    return rows


def build_set(
    out: Path, name: str, table: tuple, base: int, small: bool, private: bool, profile: str = "standard"
) -> dict[str, Any]:
    from capek.export.lerobot_writer import export_session, quiet_lerobot, read_tags

    out.mkdir(parents=True, exist_ok=False)
    rows = _record(out / "session", table, base, small, profile)
    with quiet_lerobot():
        result = export_session(out / "session", out / "dataset", f"capek_bench/{out.name}")
    tags = read_tags(out / "dataset")
    episodes = {}
    for idx, row in enumerate(rows):
        t = tags["episodes"][str(idx)]
        if t["source_episode_index"] != idx or t["policy_params"].get("bench_group") != row["group"]:
            raise BenchError(f"gt/capek_tags mismatch at episode {idx}")  # cross-check (P2-3 acceptance)
        episodes[str(idx)] = {
            "group": row["group"],
            "class": row["class"],
            "policy": t["policy_name"],
            "params": {k: v for k, v in t["policy_params"].items() if k != "bench_group"},
            "sim_success": t["sim_success"],
        }
    gt = {
        "schema_version": GT_SCHEMA_VERSION,
        "tool": "capek bench v2",
        "seed_set": name,
        "seeds": "private" if private else {"base": base, "group_k_uses": "base + k"},
        "small": small,
        "noise_profile": profile,
        "sigma_range": list(NOISE_PROFILES[profile]),
        "dataset": {
            "path": "dataset",
            "total_episodes": tags["dataset"]["total_episodes"],
            "total_frames": result.total_frames,
        },
        "episodes": episodes,
    }
    (out / "gt.json").write_text(json.dumps(gt, indent=1) + "\n")
    return gt


def build(out: Path | str, seeds: str, small: bool = False) -> dict[str, Any]:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise BenchError(f"{out} is not empty; choose a new --out")
    name, bases, private = resolve_seeds(seeds)
    sets = {}
    for profile, suffix in (("standard", ""), ("low", "_low")):
        sets[f"mixed{suffix}"] = build_set(out / f"mixed{suffix}", name, MIXED, bases["mixed"], small, private, profile)
        sets[f"clean_only{suffix}"] = build_set(
            out / f"clean_only{suffix}", name, CLEAN_ONLY, bases["clean_only"], small, private, profile
        )
    return sets


def parse_groups(spec: str) -> list[tuple[str, int]]:
    """``clean:60,noise025:10`` -> [("clean", 60), ("noise025", 10)]; order kept (group k uses seed base + k)."""
    from capek.bench.spec import MIXED

    known = {g for g, *_ in MIXED}
    out = []
    for part in spec.split(","):
        name, _, count = part.strip().partition(":")
        if name not in known or not count.isdigit() or int(count) < 1:
            raise BenchError(f"bad group {part!r}: use NAME:COUNT with NAME in {sorted(known)}")
        out.append((name, int(count)))
    return out


def record_mix(out: Path | str, groups: str, seed: int, profile: str = "standard") -> list[dict[str, Any]]:
    """Record a custom mix of benchmark groups into ONE session (the phase 1 --append path), e.g. the phase 3
    headline recipe (plan D4). Returns the per-episode rows (group, seed); ground truth lives in capek_tags.json."""
    table = tuple((g, "", n, n) for g, n in parse_groups(groups))
    if profile not in NOISE_PROFILES:
        raise BenchError(f"--noise must be one of {sorted(NOISE_PROFILES)}")
    return _record(Path(out), table, seed, small=False, profile=profile)
