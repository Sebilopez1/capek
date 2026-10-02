"""Benchmark v2 declaration (plan D1): groups, classes, seed sets. One table, no logic."""

from __future__ import annotations

# (group, class, episodes in the full set, episodes in --small)
MIXED: tuple[tuple[str, str, int, int], ...] = (
    ("clean", "clean", 60, 20),
    ("nearmiss", "outcome_only", 10, 3),
    ("noise005", "borderline", 10, 3),
    ("noise010", "motion_junk", 10, 3),
    ("noise025", "motion_junk", 10, 3),
    ("hesitation", "motion_junk", 10, 3),
    ("wobble", "tracked_gap", 10, 3),
    ("stall", "outcome_only", 10, 3),
    ("wrong", "outcome_only", 10, 3),
    ("random", "motion_junk", 10, 3),
    # phase 3 (R-real-1): a clean out-and-back motion; last so the other groups keep their seeds (base + k)
    ("return_home", "clean_variant", 10, 3),
)
CLEAN_ONLY: tuple[tuple[str, str, int, int], ...] = (("clean", "clean", 100, 20),)
CLASSES = ("clean", "motion_junk", "borderline", "outcome_only", "tracked_gap", "clean_variant")

# Per-episode action-noise sigma for every group that draws one (fixed-sigma noise groups and random excepted).
# "low" is the whole-set low-noise variant (plan D3): a smooth leader-arm-like dataset, where the phase 2 `net`
# action_tv_ratio denominator false-flags return_home (the DoD 8 real-data false positive).
NOISE_PROFILES: dict[str, tuple[float, float]] = {"standard": (0.01, 0.04), "low": (0.002, 0.008)}

# Published seed bases (docs/dev/phase2-research-brief.md §4). Group k of a set records with seed base + k.
# QA's private held-out bases live outside the repo and are passed as --seeds file:PATH
# ({"mixed": int, "clean_only": int}).
SEED_SETS: dict[str, dict[str, int]] = {
    "dev": {"mixed": 1000, "clean_only": 1500},
    "heldout": {"mixed": 5000, "clean_only": 5500},
    # R-real-1 regression fixture for the plan D3 power check, NOT a validation set: base 16000 was picked (from
    # 12000/13000/16000, measured 2026-09-25) because the phase 2 `net` formula visibly false-flags its low-noise
    # return_home episodes (net 7/10, split 0/10). Dev/held-out gave 1/10 and 0/10, QA-private 2/10.
    "rreal1": {"mixed": 16000, "clean_only": 16500},
}
MAX_STEPS = 90
