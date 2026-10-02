"""P3-6: `capek train-bc` and the `bc:` policy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("lerobot.datasets.lerobot_dataset")
pytest.importorskip("pyarrow")

from capek import cli  # noqa: E402
from capek.eval.report import RECIPE_CAVEAT  # noqa: E402


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    """60 clean + 5 noise 0.25 + 5 wrong (bench generators), exported and scored once."""
    from capek.bench.generators import GroupPolicy
    from capek.export.lerobot_writer import export_session, quiet_lerobot
    from capek.record import new_session, record_into
    from capek.sim.registry import make_env

    root = tmp_path_factory.mktemp("bc")
    env = make_env("so101_reach")
    s = new_session(env, root / "session", max_steps=90)
    for k, (group, n) in enumerate([("clean", 60), ("noise025", 5), ("wrong", 5)]):
        record_into(s, env, GroupPolicy(group), n, 31000 + k, 90)
    with quiet_lerobot():
        export_session(root / "session", root / "ds", "local/bc")
    assert cli.main(["score", str(root / "ds"), "--json-out", str(root / "score.json")]) == 0
    return root


def _train(work: Path, name: str, *extra: str) -> dict:
    assert cli.main(["train-bc", str(work / "ds"), "--out", str(work / name), *extra]) == 0
    return json.loads((work / name / "capek_policy.json").read_text())


def test_checkpoint_format_and_speed(work: Path) -> None:
    t0 = time.perf_counter()
    meta = _train(work, "clean", "--keep", "ok-and-success", "--score-json", str(work / "score.json"))
    assert time.perf_counter() - t0 < 10.0
    assert (work / "clean" / "model.pt").is_file()
    assert meta["arch"] == {"inputs": 9, "hidden": [256, 256], "outputs": 6, "activation": "relu",
                            "target": "action - observation.state"}  # fmt: skip
    assert meta["obs_keys"] == ["observation.state", "observation.environment_state"]
    assert meta["train"]["seed"] == 0 and meta["train"]["epochs"] == 40
    assert meta["dataset"]["total_episodes"] == 70 and meta["dataset"]["codebase_version"] == "v3.0"
    assert meta["filter"]["keep"] == "ok-and-success" and meta["filter"]["score_json_sha256"]
    kept = set(meta["kept_episodes"])
    assert not kept & set(range(60, 70))  # noise (scorer) and wrong-goal (outcome) episodes dropped
    assert len(kept) >= 50 and len(meta["weights_sha256"]) == 64
    assert set(meta["normalization"]) == {"x_mu", "x_sd", "y_mu", "y_sd"}


def test_trained_checkpoint_beats_random_and_caveat_for_two_checkpoints(work: Path, capsys) -> None:
    if not (work / "clean").exists():
        _train(work, "clean", "--keep", "ok-and-success", "--score-json", str(work / "score.json"))
    out = work / "vs_random.json"
    assert cli.main(["compare", "random", f"bc:{work / 'clean'}", "--episodes", "60", "--json-out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["paired"]["significant"] and r["paired"]["delta_b_minus_a"] > 0
    assert r["verdict"].startswith("B (bc:") and " is better than A (random)" in r["verdict"]
    assert r["recipe_caveat"] is None
    assert r["B"]["policy"]["kind"] == "bc" and r["B"]["policy"]["weights_sha256"]
    _train(work, "all_quick", "--epochs", "3")
    capsys.readouterr()
    out2 = work / "two_ckpts.json"
    args = ["compare", f"bc:{work / 'all_quick'}", f"bc:{work / 'clean'}", "--episodes", "10", "--json-out", str(out2)]
    assert cli.main(args) == 0
    assert RECIPE_CAVEAT in capsys.readouterr().out
    assert json.loads(out2.read_text())["recipe_caveat"] == RECIPE_CAVEAT


TRAIN_AND_EVAL = textwrap.dedent(
    """
    import json, sys
    from capek import cli
    ds, out, threads, report = sys.argv[1:5]
    assert cli.main(["train-bc", ds, "--out", out, "--epochs", "3", "--threads", threads]) == 0
    assert cli.main(["eval", "bc:" + out, "--episodes", "12", "--threads", threads, "--json-out", report]) == 0
    """
)


def test_same_seed_same_weights_and_rollouts_across_processes(work: Path) -> None:
    # Same seed + same thread count in two separate processes must give identical weights and rollouts.
    # Different thread counts are NOT promised to match: on GitHub's CI runners 1 vs 2 torch threads changed the
    # weights (float reduction order), even though they matched on the crew's machine.
    results = []
    for run in ("a", "b"):
        threads = "1"
        out, report = work / f"det{run}", work / f"det{run}.json"
        r = subprocess.run(
            [sys.executable, "-c", TRAIN_AND_EVAL, str(work / "ds"), str(out), threads, str(report)],
            capture_output=True,
            text=True,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        assert r.returncode == 0, r.stderr
        meta = json.loads((out / "capek_policy.json").read_text())
        rep = json.loads(report.read_text())
        assert meta["train"]["threads"] == int(threads) and rep["machine"]["torch_threads"] == int(threads)
        results.append((meta["weights_sha256"], rep["episodes"]))
    assert results[0] == results[1]


def test_filters_fail_cleanly(work: Path, tmp_path: Path, capsys) -> None:
    import pyarrow.parquet as pq

    ds = work / "ds"
    args = ["train-bc", str(ds), "--out", str(tmp_path / "x")]
    assert cli.main([*args, "--keep", "quality-ok"]) == 1
    assert "needs --score-json" in capsys.readouterr().err
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"dataset": {"path": "/elsewhere", "total_frames": 1}, "episodes": []}))
    assert cli.main([*args, "--keep", "quality-ok", "--score-json", str(other)]) == 1
    assert "was made for /elsewhere" in capsys.readouterr().err
    # a copy without next.success and without labels: outcome is unknown -> success filters refuse
    import shutil

    blind = tmp_path / "blind"
    shutil.copytree(ds, blind)
    (blind / "meta" / "capek_tags.json").unlink()
    for f in blind.glob("data/*/*.parquet"):
        t = pq.read_table(f)
        pq.write_table(t.drop_columns(["next.success"]), f)
    assert cli.main(["train-bc", str(blind), "--out", str(tmp_path / "y"), "--keep", "success"]) == 1
    err = capsys.readouterr().err
    assert "70 episode(s) have no outcome evidence" in err and "Traceback" not in err
    # a filter that keeps nothing
    report = json.loads((work / "score.json").read_text())
    for e in report["episodes"]:
        e["quality"] = "FLAG"
    none = tmp_path / "none.json"
    none.write_text(json.dumps(report))
    assert cli.main([*args, "--keep", "quality-ok", "--score-json", str(none)]) == 1
    assert "leaves no episodes" in capsys.readouterr().err
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_out_dir_and_report_path_protection(work: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "ck"
    assert cli.main(["train-bc", str(work / "ds"), "--out", str(out), "--epochs", "1"]) == 0
    assert cli.main(["train-bc", str(work / "ds"), "--out", str(out), "--epochs", "1"]) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert cli.main(["train-bc", str(work / "ds"), "--out", str(out), "--epochs", "1", "--overwrite"]) == 0
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "notes.txt").write_text("x")
    assert cli.main(["train-bc", str(work / "ds"), "--out", str(busy), "--overwrite"]) == 1
    assert "not a capek checkpoint" in capsys.readouterr().err
    assert cli.main(["eval", f"bc:{out}", "--episodes", "1", "--json-out", str(out / "r.json")]) == 1
    assert "inside" in capsys.readouterr().err
    (out / "model.pt").write_bytes(b"garbage")
    assert cli.main(["eval", f"bc:{out}", "--episodes", "1", "--json-out", str(tmp_path / "r.json")]) == 1
    assert "can't load checkpoint" in capsys.readouterr().err


def test_train_bc_without_torch_is_a_clean_error(work: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    assert cli.main(["train-bc", str(work / "ds"), "--out", str(tmp_path / "z")]) == 1
    assert 'pip install "capek-tech[eval]"' in capsys.readouterr().err


BLOCK_TORCH = textwrap.dedent(
    """
    import sys
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "torch":
                raise ImportError("torch blocked")
    sys.meta_path.insert(0, Block())
    from capek import cli
    out = sys.argv[1]
    assert cli.main(["record", "--episodes", "3", "--out", out + "/s", "-q"]) == 0
    assert cli.main(["tag", out + "/s", "-e", "1", "--label", "fail"]) == 0
    assert cli.main(["list", out + "/s"]) == 0
    assert cli.main(["score", sys.argv[2], "--json-out", out + "/score.json"]) == 0
    assert cli.main(["eval", "scripted", "--episodes", "2", "--json-out", out + "/e.json"]) == 0
    """
)


def test_record_tag_score_and_scripted_eval_run_without_torch(work: Path, tmp_path: Path) -> None:
    r = subprocess.run(
        [sys.executable, "-c", BLOCK_TORCH, str(tmp_path), str(work / "ds")], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


# ---- QA phase 3 review N4-N6 ---------------------------------------------------------------------------------------
def test_failed_overwrite_keeps_the_old_checkpoint(work: Path, tmp_path: Path, monkeypatch) -> None:
    from capek.eval import bc

    out = tmp_path / "ck"
    assert cli.main(["train-bc", str(work / "ds"), "--out", str(out), "--epochs", "1"]) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def boom(model):
        raise RuntimeError("simulated failure after training")

    monkeypatch.setattr(bc, "weights_sha256", boom)
    with pytest.raises(RuntimeError):
        cli.main(["train-bc", str(work / "ds"), "--out", str(out), "--epochs", "1", "--overwrite", "--seed", "3"])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ck"]  # no temp / backup dirs left behind
    monkeypatch.undo()
    assert (
        cli.main(["train-bc", str(work / "ds"), "--out", str(out), "--epochs", "1", "--overwrite", "--seed", "3"]) == 0
    )
    assert json.loads((out / "capek_policy.json").read_text())["train"]["seed"] == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ck"]


def test_wrong_architecture_is_refused_at_load(tmp_path: Path, capsys) -> None:
    import numpy as np

    from capek.eval.bc import build_mlp, weights_sha256

    model = build_mlp(torch, np.zeros(7), np.ones(7), np.zeros(6), np.ones(6))
    ck = tmp_path / "ck7"
    ck.mkdir()
    torch.save(model.state_dict(), ck / "model.pt")
    meta = {"arch": {"inputs": 7, "hidden": [256, 256], "outputs": 6}, "weights_sha256": weights_sha256(model),
            "normalization": {"x_mu": [0.0] * 7, "x_sd": [1.0] * 7, "y_mu": [0.0] * 6, "y_sd": [1.0] * 6}}  # fmt: skip
    (ck / "capek_policy.json").write_text(json.dumps(meta))
    assert cli.main(["eval", f"bc:{ck}", "--episodes", "1", "--json-out", str(tmp_path / "r.json")]) == 1
    err = capsys.readouterr().err
    assert "maps 7 inputs to 6 outputs; so101_reach needs 9" in err and "Traceback" not in err


def test_corrupt_weights_error_is_one_line(work: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "ck"
    assert cli.main(["train-bc", str(work / "ds"), "--out", str(out), "--epochs", "1"]) == 0
    (out / "model.pt").write_bytes(b"garbage")
    capsys.readouterr()
    assert cli.main(["eval", f"bc:{out}", "--episodes", "1", "--json-out", str(tmp_path / "r.json")]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "can't load checkpoint" in err


def test_bc_metadata_carries_kept_episode_indices(work: Path) -> None:
    from capek.eval.policies import load_policy

    if not (work / "clean").exists():
        _train(work, "clean", "--keep", "ok-and-success", "--score-json", str(work / "score.json"))
    meta = json.loads((work / "clean" / "capek_policy.json").read_text())
    md = load_policy(f"bc:{work / 'clean'}").metadata()
    assert md["kept_episode_indices"] == meta["kept_episodes"] and md["kept_episodes"] == len(meta["kept_episodes"])
