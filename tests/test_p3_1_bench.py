"""P3-1 (QA, bench side): return_home in every mixed set, whole-set low-noise variants, the return_home bar and the
power check that the phase 2 `net` tv formula fails it (plan D3). The full power check is gated: CAPEK_BENCH_FULL=1."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_p2_3_bench import _groups, _toy, small_bench  # noqa: F401  (session fixture shared with the P2-3 tests)

from capek.bench.evaluate import evaluate
from capek.bench.spec import NOISE_PROFILES, SEED_SETS


def test_low_noise_sets_and_return_home_group(small_bench: Path) -> None:  # noqa: F811
    for name, profile in (
        ("mixed", "standard"),
        ("clean_only", "standard"),
        ("mixed_low", "low"),
        ("clean_only_low", "low"),
    ):
        gt = json.loads((small_bench / name / "gt.json").read_text())
        assert gt["noise_profile"] == profile and gt["sigma_range"] == list(NOISE_PROFILES[profile])
        lo, hi = NOISE_PROFILES[profile]
        for row in gt["episodes"].values():
            if row["group"] in ("clean", "nearmiss", "hesitation", "wobble", "stall", "wrong", "return_home"):
                assert lo <= row["params"]["noise"] <= hi, (name, row)
            if row["group"] in ("noise005", "noise010", "noise025"):  # fixed-sigma groups keep their sigma
                assert row["params"]["noise"] in (0.05, 0.10, 0.25)
    for name in ("mixed", "mixed_low"):
        rth = [
            r
            for r in json.loads((small_bench / name / "gt.json").read_text())["episodes"].values()
            if r["group"] == "return_home"
        ]
        assert len(rth) == 3 and all(r["class"] == "clean_variant" for r in rth)
        assert all(0.8 <= r["params"]["out_s"] <= 1.2 and 0.1 <= r["params"]["hold_s"] <= 0.4 for r in rth)
        assert all(r["sim_success"] is False for r in rth)  # it goes back home, so the final frame is not at the target
    # standard and low sets share targets (same seeds), so they differ only in noise
    a = json.loads((small_bench / "mixed" / "dataset" / "meta" / "capek_tags.json").read_text())["episodes"]
    b = json.loads((small_bench / "mixed_low" / "dataset" / "meta" / "capek_tags.json").read_text())["episodes"]
    assert [a[k]["seed"] for k in sorted(a, key=int)] == [b[k]["seed"] for k in sorted(b, key=int)]


def test_return_home_bar_and_binding_by_profile() -> None:
    groups = _groups([0.0] * 20, ["ok"] * 20)
    groups["return_home"] = ("clean_variant", [4.0, 4.0] + [0.0] * 8, ["FLAG", "FLAG"] + ["ok"] * 8)  # 2/10 flagged
    report, gt = _toy(groups)
    clean_report, clean_gt = _toy({"clean": ("clean", [0.0] * 20, ["ok"] * 20)})
    r = evaluate(report, gt, clean_report, clean_gt)
    bar = {b["bar"]: b for b in r["bars"]}["return_home_flagged"]
    assert bar["value"] == pytest.approx(0.2) and not bar["pass"] and not r["all_pass"]
    assert r["counts"]["fp"] == 0  # clean_variant never counts as a clean false positive or in P/R/AUROC
    # low profile: only the return_home bar binds; a failing reported bar does not fail the set
    groups["return_home"] = ("clean_variant", [0.0] * 10, ["ok"] * 10)
    groups["noise010"] = ("motion_junk", [0.0] * 10, ["ok"] * 10)  # would fail noise010_flagged if binding
    report, gt = _toy(groups)
    clean_report, clean_gt = _toy({"clean": ("clean", [0.0] * 20, ["ok"] * 20)})
    gt["noise_profile"] = clean_gt["noise_profile"] = "low"
    r = evaluate(report, gt, clean_report, clean_gt)
    assert r["all_pass"] and {b["bar"] for b in r["bars"] if b["binding"]} == {"return_home_flagged"}


@pytest.mark.skipif(
    os.environ.get("CAPEK_BENCH_FULL") != "1", reason="full power check: set CAPEK_BENCH_FULL=1 (~20 s)"
)
def test_power_check_old_formula_fails_return_home_bar(tmp_path: Path) -> None:
    """The bar has power: on the published R-real-1 regression seeds the phase 2 formula flags >= 2/10 return_home,
    while the current scorer (via `capek score` JSON, judged by the evaluator) flags <= 1/10."""
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    from capek.bench.__main__ import _score, net_formula_power_check
    from capek.bench.build import build_set
    from capek.bench.spec import MIXED

    d = tmp_path / "mixed_low"
    build_set(d, "rreal1", MIXED, SEED_SETS["rreal1"]["mixed"], small=False, private=False, profile="low")
    assert net_formula_power_check(d)["return_home_flagged"] >= 2
    _score(d / "dataset", d / "score.json")
    report = json.loads((d / "score.json").read_text())
    gt = json.loads((d / "gt.json").read_text())
    r = evaluate(report, gt, None, None)
    assert {b["bar"]: b for b in r["bars"]}["return_home_flagged"]["pass"]


def test_record_mix_group_parsing() -> None:
    from capek.bench.build import BenchError, parse_groups

    assert parse_groups("clean:60,noise025:10, random:10") == [("clean", 60), ("noise025", 10), ("random", 10)]
    for bad in ("clean", "clean:0", "nope:3", "clean:x"):
        with pytest.raises(BenchError, match="bad group"):
            parse_groups(bad)
