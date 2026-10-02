"""P2-3 (QA-owned): benchmark v2 builder, DoD 4 evaluator, real-like / corrupt fixtures.

The full dev benchmark + bars run only with CAPEK_BENCH_FULL=1 (about 35 s); QA runs held-out and private sets by hand.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from capek.bench.evaluate import EvalError, auroc, evaluate


# ---- evaluator on hand-checked toy inputs (no sim, no lerobot) -------------------------------------------------
def test_auroc_hand_checked() -> None:
    # positives {0.35, 0.8} vs negatives {0.1, 0.4}: pairs won 3 of 4 -> 0.75
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert auroc([1.0, 1.0], [0, 1]) == 0.5  # a tie counts half
    assert auroc([0.0, math.inf], [0, 1]) == 1.0  # HARD / unscorable episodes rank as +inf
    assert math.isnan(auroc([1.0, 2.0], [1, 1]))


def _toy(groups: dict[str, tuple[str, list[float], list[str]]]) -> tuple[dict, dict]:
    """(score report, gt) from {group: (class, combined scores, qualities)}."""
    eps, gt_eps, i = [], {}, 0
    for g, (cls, scores, quals) in groups.items():
        for c, q in zip(scores, quals, strict=True):
            eps.append({"episode_index": i, "combined": c, "quality": q})
            gt_eps[str(i)] = {"group": g, "class": cls, "policy": "x", "params": {}, "sim_success": True}
            i += 1
    report = {
        "schema_version": 1,
        "dataset": {"total_episodes": i, "total_frames": 90 * i},
        "thresholds": {"flag_z": 3.5},
        "episodes": eps,
    }
    gt = {
        "seed_set": "toy",
        "small": False,
        "dataset": {"total_episodes": i, "total_frames": 90 * i},
        "episodes": gt_eps,
    }
    return report, gt


def _groups(clean_scores: list[float], clean_quals: list[str]) -> dict:
    junk = ([5.0] * 10, ["FLAG"] * 10)
    return {
        "clean": ("clean", clean_scores, clean_quals),
        "noise010": ("motion_junk", *junk),
        "noise025": ("motion_junk", *junk),
        "random": ("motion_junk", [None] * 10, ["HARD"] * 10),
        "hesitation": ("motion_junk", [4.0] * 9 + [1.0], ["FLAG"] * 9 + ["ok"]),
        "nearmiss": ("outcome_only", [0.5] * 10, ["ok"] * 10),
        "stall": ("outcome_only", [0.5] * 10, ["ok"] * 10),
        "return_home": ("clean_variant", [1.0] * 10, ["ok"] * 10),
    }


def test_evaluator_bars_on_toy_reports() -> None:
    report, gt = _toy(_groups([0.0] * 19 + [3.6], ["ok"] * 19 + ["FLAG"]))
    clean_report, clean_gt = _toy({"clean": ("clean", [0.0] * 20, ["ok"] * 20)})
    r = evaluate(report, gt, clean_report, clean_gt)
    bars = {b["bar"]: b for b in r["bars"]}
    assert bars["clean_flagged"]["value"] == pytest.approx(0.05) and bars["clean_flagged"]["pass"]
    assert bars["precision"]["value"] == pytest.approx(39 / 40) and bars["recall"]["value"] == pytest.approx(39 / 40)
    assert bars["random_flagged"]["value"] == 1.0  # HARD counts as flagged
    assert bars["hesitation_auroc"]["value"] == pytest.approx(199 / 200)  # 9 x 4.0 beat all 20 clean; 1.0 beats 19
    assert r["all_pass"] and r["per_group"]["stall"]["flagged"] == 0
    # two flagged clean episodes -> clean bar fails; no clean-only report -> its bar fails, never silently passes
    report2, gt2 = _toy(_groups([0.0] * 18 + [3.6, 3.7], ["ok"] * 18 + ["FLAG"] * 2))
    r2 = evaluate(report2, gt2, None, None)
    failed = {b["bar"] for b in r2["bars"] if not b["pass"]}
    assert failed == {"clean_flagged", "clean_only_flagged"} and not r2["all_pass"]


def test_evaluator_rejects_mismatched_inputs() -> None:
    report, gt = _toy(_groups([0.0] * 20, ["ok"] * 20))
    wrong = json.loads(json.dumps(report))
    wrong["dataset"]["total_frames"] += 1
    with pytest.raises(EvalError, match="wrong dataset"):
        evaluate(wrong, gt, None, None)
    wrong = json.loads(json.dumps(report))
    wrong["episodes"][0]["quality"] = "maybe"
    with pytest.raises(EvalError, match="unknown quality"):
        evaluate(wrong, gt, None, None)


def _write(tmp_path: Path, name: str, obj: object) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return str(path)


@pytest.mark.parametrize(
    "case",
    ["swapped_arguments", "gt_without_dataset", "episode_without_index", "non_default_flag_z", "clean_only_flag_z"],
)
def test_malformed_evaluator_input_exits_2(tmp_path: Path, case: str, capsys) -> None:
    from capek.bench.__main__ import main

    report, gt = _toy(_groups([0.0] * 20, ["ok"] * 20))
    clean_report, clean_gt = _toy({"clean": ("clean", [0.0] * 20, ["ok"] * 20)})
    if case == "gt_without_dataset":
        del gt["dataset"]
    elif case == "episode_without_index":
        del report["episodes"][3]["episode_index"]
    elif case == "non_default_flag_z":
        report["thresholds"]["flag_z"] = 1.0
    elif case == "clean_only_flag_z":
        clean_report["thresholds"]["flag_z"] = 5.0
    r, g = _write(tmp_path, "r.json", report), _write(tmp_path, "g.json", gt)
    cr, cg = _write(tmp_path, "cr.json", clean_report), _write(tmp_path, "cg.json", clean_gt)
    args = [g, r] if case == "swapped_arguments" else [r, g]
    assert main(["evaluate", *args, "--clean-only", cr, cg]) == 2
    err = capsys.readouterr().err
    assert err.startswith("bench: error:") and "Traceback" not in err
    if case == "swapped_arguments":
        assert "arguments swapped" in err
    if "flag_z" in case:
        assert "flag_z" in err


def test_valid_toy_files_exit_0_and_failing_bars_exit_1(tmp_path: Path) -> None:
    from capek.bench.__main__ import main

    report, gt = _toy(_groups([0.0] * 20, ["ok"] * 20))
    clean_report, clean_gt = _toy({"clean": ("clean", [0.0] * 20, ["ok"] * 20)})
    r, g = _write(tmp_path, "r.json", report), _write(tmp_path, "g.json", gt)
    cr, cg = _write(tmp_path, "cr.json", clean_report), _write(tmp_path, "cg.json", clean_gt)
    assert main(["evaluate", r, g, "--clean-only", cr, cg]) == 0
    assert main(["evaluate", r, g]) == 1  # clean-only bar not evaluated -> fails, never silently passes


def test_check_exits_2_when_there_is_nothing_to_score(tmp_path: Path, capsys) -> None:
    from capek.bench.__main__ import main

    assert main(["check", str(tmp_path / "no_benchmark_here")]) == 2
    assert "run `bench build` first" in capsys.readouterr().err
    bad = tmp_path / "broken" / "mixed" / "dataset" / "meta"
    bad.mkdir(parents=True)
    (bad / "info.json").write_text(json.dumps({"codebase_version": "v1.6", "fps": 30}))
    assert main(["check", str(tmp_path / "broken")]) == 2  # capek score itself fails -> bad input, not "bars failed"
    assert "capek score" in capsys.readouterr().err


def test_private_seed_file_must_not_overlap_published(tmp_path: Path) -> None:
    from capek.bench.build import BenchError, resolve_seeds

    f = tmp_path / "seeds.json"
    f.write_text(json.dumps({"mixed": 1003, "clean_only": 777777}))
    with pytest.raises(BenchError, match="overlaps"):
        resolve_seeds(f"file:{f}")
    f.write_text(json.dumps({"mixed": 777000, "clean_only": 777500}))
    assert resolve_seeds(f"file:{f}") == ("private", {"mixed": 777000, "clean_only": 777500}, True)


# ---- small benchmark build (sim + lerobot) ------------------------------------------------------------------
_SMALL_BENCH: dict[str, Path] = {}  # shared by every module that imports small_bench (built once per session)


@pytest.fixture(scope="session")
def small_bench(tmp_path_factory) -> Path:
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    from capek.bench.build import build

    if "path" not in _SMALL_BENCH:
        out = tmp_path_factory.mktemp("bench") / "small"
        build(out, "dev", small=True)
        _SMALL_BENCH["path"] = out
    return _SMALL_BENCH["path"]


def test_small_build_layout_and_gt_matches_capek_tags(small_bench: Path) -> None:
    gt = json.loads((small_bench / "mixed" / "gt.json").read_text())
    tags = json.loads((small_bench / "mixed" / "dataset" / "meta" / "capek_tags.json").read_text())
    assert gt["dataset"] == {"path": "dataset", "total_episodes": 50, "total_frames": 50 * 90}
    assert not (small_bench / "mixed" / "dataset" / "gt.json").exists()  # ground truth stays outside the dataset
    counts: dict[str, int] = {}
    for k, row in gt["episodes"].items():
        counts[row["group"]] = counts.get(row["group"], 0) + 1
        t = tags["episodes"][k]
        assert t["policy_params"]["bench_group"] == row["group"] and t["sim_success"] == row["sim_success"]
    assert counts["clean"] == 20 and all(n == 3 for g, n in counts.items() if g != "clean")
    ramps = [r["params"]["ramp_s"] for r in gt["episodes"].values() if r["group"] == "clean"]
    sig = [r["params"]["noise"] for r in gt["episodes"].values() if r["group"] == "clean"]
    assert 1.0 <= min(ramps) and max(ramps) <= 2.5 and np.std(ramps) > 0.2  # heterogeneous timing (D1)
    assert 0.01 <= min(sig) and max(sig) <= 0.04 and len(set(sig)) == len(sig)
    wrong = [r for r in gt["episodes"].values() if r["group"] == "wrong"]
    assert all(len(r["params"]["wrong_goal_qpos"]) == 6 and not r["sim_success"] for r in wrong)
    assert all("target_sign" not in r["params"] for r in wrong)  # independent goal, not the mirrored target
    stall = [r for r in gt["episodes"].values() if r["group"] == "stall"]
    assert all(
        0.3 * r["params"]["ramp_s"] <= r["params"]["stall_after_s"] <= 0.7 * r["params"]["ramp_s"] for r in stall
    )
    clean_only = json.loads((small_bench / "clean_only" / "gt.json").read_text())
    assert {r["class"] for r in clean_only["episodes"].values()} == {"clean"}


def test_build_is_deterministic(small_bench: Path, tmp_path: Path) -> None:
    from capek.bench.build import _record
    from capek.bench.spec import MIXED
    from capek.session import Session

    _record(tmp_path / "again", MIXED, 1000, small=True)
    a, b = Session.open(small_bench / "mixed" / "session"), Session.open(tmp_path / "again")
    for i in range(len(a.read_metas())):
        xa, xb = a.load_arrays(i), b.load_arrays(i)
        assert all(xa[k].tobytes() == xb[k].tobytes() for k in xa), i


def test_scorer_does_not_depend_on_capek_tags_metadata(small_bench: Path, tmp_path: Path) -> None:
    """capek_tags.json carries bench_group in policy_params; scores must be identical without it (no gt leak)."""
    pytest.importorskip("pyarrow")
    from capek.score.engine import ScoreConfig, score_dataset
    from capek.score.reader import read_dataset

    stripped = tmp_path / "stripped"
    shutil.copytree(small_bench / "mixed" / "dataset", stripped)
    (stripped / "meta" / "capek_tags.json").unlink()
    a = score_dataset(read_dataset(small_bench / "mixed" / "dataset"), ScoreConfig())
    b = score_dataset(read_dataset(stripped), ScoreConfig())
    assert [(e.combined, e.quality, e.reasons) for e in a.episodes] == [
        (e.combined, e.quality, e.reasons) for e in b.episodes
    ]


def test_check_command_runs_capek_score_and_evaluates(small_bench: Path, capsys) -> None:
    pytest.importorskip("pyarrow")
    from capek.bench.__main__ import main

    code = main(["check", str(small_bench), "--json"])
    r = json.loads(capsys.readouterr().out)
    assert code in (0, 1) and set(r) == {"all_pass", "standard", "low"}
    std = r["standard"]
    assert std["small"] and len(std["bars"]) == 11  # small sets: bars computed, not meaningful
    assert std["per_group"]["random"]["flagged"] == 3 and "power_check" in r["low"]


def test_realify_fixtures(small_bench: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from capek.bench.realify import CORRUPT, realify
    from capek.score.engine import ScoreConfig, score_dataset
    from capek.score.reader import read_dataset

    man = realify(small_bench / "mixed" / "dataset", tmp_path / "rl")
    src = read_dataset(small_bench / "mixed" / "dataset")
    base = score_dataset(src, ScoreConfig())

    def summary(r):
        return [(e.quality, sorted(e.reasons), e.hard_flags) for e in r.episodes]

    units = read_dataset(tmp_path / "rl" / "units")
    assert units.state_names[0] == "main_shoulder_pan"
    scale, offset = np.array(man["units"]["scale"]), np.array(man["units"]["offset"])
    np.testing.assert_allclose(units.episodes[4].state, src.episodes[4].state * scale + offset, rtol=1e-6, atol=1e-4)
    assert summary(score_dataset(units, ScoreConfig())) == summary(base)  # DoD 6 (a): identical flags and reasons
    for v, version in (("v21", "v2.1"), ("v20", "v2.0")):
        d = read_dataset(tmp_path / "rl" / v)
        assert d.codebase_version == version and len(d.episodes) == len(src.episodes)
        assert all(np.array_equal(x.state, y.state) for x, y in zip(d.episodes, src.episodes, strict=True))
    v20_info = json.loads((tmp_path / "rl" / "v20" / "meta" / "info.json").read_text())
    assert isinstance(v20_info["features"]["observation.state"]["names"], dict)  # nested {"motors": [...]}
    assert read_dataset(tmp_path / "rl" / "v20").state_names == src.state_names
    vid = read_dataset(tmp_path / "rl" / "video_declared")
    assert (
        "observation.images.front" in vid.info["features"]
        and not (tmp_path / "rl" / "video_declared" / "videos").exists()
    )
    assert summary(score_dataset(vid, ScoreConfig())) == summary(base)
    trunc = read_dataset(tmp_path / "rl" / "truncated")
    cut = man["truncated"]["truncated_episodes"]
    assert len(cut) == round(0.2 * 50) and all(trunc.episodes[e].length < 90 for e in cut)
    assert trunc.tags is not None  # capek_tags fingerprint updated, labels still usable
    score_dataset(trunc, ScoreConfig())  # DoD 6 (b): variable lengths score without error
    for kind, (ep, expected) in CORRUPT.items():
        r = score_dataset(read_dataset(tmp_path / "rl" / kind), ScoreConfig())
        hard = {e.episode_index: [h.split(":")[0] for h in e.hard_flags] for e in r.episodes if e.hard_flags}
        assert hard == {ep: [expected]}, (kind, hard)


@pytest.mark.skipif(os.environ.get("CAPEK_BENCH_FULL") != "1", reason="full benchmark: set CAPEK_BENCH_FULL=1 (~35 s)")
def test_full_dev_benchmark_meets_dod4_bars(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    from capek.bench.__main__ import main

    assert main(["build", "--out", str(tmp_path / "dev"), "--seeds", "dev"]) == 0
    assert main(["check", str(tmp_path / "dev")]) == 0


def test_wrong_goal_distribution_matches_env_targets() -> None:
    pytest.importorskip("mujoco")
    from capek.bench.generators import TARGET_FRACTION
    from capek.sim.so101_reach import TARGET_RANGE_FRACTION

    assert TARGET_FRACTION == TARGET_RANGE_FRACTION
