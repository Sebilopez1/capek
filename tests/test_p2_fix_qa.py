"""QA phase 2 code review: R1 (frozen joint), R2 (too short), R3 (uniform-junk honesty), N1, N2, N4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from helpers import FakeEnv  # noqa: E402
from lerobot_files import JOINTS, smooth_reach, write_dataset  # noqa: E402

from robot_report_card import cli  # noqa: E402
from robot_report_card.score.engine import RELATIVE_ASSUMPTION, ScoreConfig, score_dataset  # noqa: E402
from robot_report_card.score.reader import ReaderError, read_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
UNIFORM_JUNK = (
    "If every episode shares the same problem (for example a jittery leader arm), few or none will be flagged."
)


def _clean(n: int, seed: int, frames: int = 90) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [smooth_reach(rng, frames, noise=float(rng.uniform(0.005, 0.02))) for _ in range(n)]


# ---- R1 ----------------------------------------------------------------------------------------------------------
def test_held_gripper_is_not_frozen(tmp_path: Path) -> None:
    eps = _clean(30, 1)
    for e in eps[:8]:  # QA B1: gripper unused in 8 episodes -> constant reading AND constant command
        e["observation.state"][:, 5] = 0.3
        e["action"][:, 5] = 0.3
    res = score_dataset(read_dataset(write_dataset(tmp_path / "ds", eps)))
    assert not any("frozen" in h for e in res.episodes for h in e.hard_flags)
    assert all(e.quality != "HARD" for e in res.episodes)


def test_stuck_encoder_with_moving_command_is_hard(tmp_path: Path) -> None:
    eps = _clean(30, 2)
    eps[4]["observation.state"][:, 2] = 0.1  # reading flat, command still ramps (QA corrupt_frozen shape)
    res = score_dataset(read_dataset(write_dataset(tmp_path / "ds", eps)))
    e = res.episodes[4]
    assert e.quality == "HARD"
    assert any(h.startswith(f"frozen joint: {JOINTS[2]}") and "command moves" in h for h in e.hard_flags)
    assert sum(x.quality == "HARD" for x in res.episodes) == 1


def test_tiny_command_jitter_on_a_held_joint_is_not_frozen(tmp_path: Path) -> None:
    eps = _clean(30, 3)
    rng = np.random.default_rng(0)
    eps[0]["observation.state"][:, 5] = 0.3
    eps[0]["action"][:, 5] = 0.3 + rng.normal(0, 1e-4, 90)  # << 1% of the gripper's range
    res = score_dataset(read_dataset(write_dataset(tmp_path / "ds", eps)))
    assert res.episodes[0].hard_flags == []


# ---- R2 ----------------------------------------------------------------------------------------------------------
def test_ten_frame_episode_is_hard_too_short_not_hesitating(tmp_path: Path) -> None:
    eps = _clean(30, 4)
    eps.append({k: v[:10] for k, v in _clean(1, 5)[0].items()})
    res = score_dataset(read_dataset(write_dataset(tmp_path / "ds", eps)))
    e = res.episodes[-1]
    assert ScoreConfig().min_frames(30) == 17
    assert e.quality == "HARD" and e.combined is None and e.reasons == []
    assert e.hard_flags[0] == "too short: 10 frames (0.33 s); need at least 17 (0.57 s) to score motion"
    assert not any("hesitates" in r for x in res.episodes for r in x.reasons)
    ok = {k: v[:17] for k, v in _clean(1, 6)[0].items()}  # exactly the minimum is scored
    res = score_dataset(read_dataset(write_dataset(tmp_path / "ds17", [*_clean(30, 4), ok])))
    assert res.episodes[-1].combined is not None


# ---- R3 ----------------------------------------------------------------------------------------------------------
def test_uniform_junk_sentence_in_summary_json_and_readme(tmp_path: Path, capsys) -> None:
    assert UNIFORM_JUNK in RELATIVE_ASSUMPTION
    root = write_dataset(tmp_path / "ds", _clean(12, 7))
    assert cli.main(["score", str(root), "--json-out", str(tmp_path / "r.json")]) == 0
    assert UNIFORM_JUNK in capsys.readouterr().out
    assert UNIFORM_JUNK in json.loads((tmp_path / "r.json").read_text())["summary"]["assumption"]
    readme = " ".join((REPO / "README.md").read_text().split())
    assert UNIFORM_JUNK in readme
    assert "--overwrite" in readme and "re-run `rrc score`" in readme  # N4


# ---- N1 ----------------------------------------------------------------------------------------------------------
def test_schema_mismatch_across_files_is_a_clean_error(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = write_dataset(tmp_path / "ds", _clean(4, 8), version="v2.1")
    path = root / "data" / "chunk-000" / "episode_000002.parquet"
    t = pq.read_table(path)
    t = t.set_column(0, "observation.state", t.column("observation.state").cast(pa.list_(pa.float64())))
    pq.write_table(t, path)
    with pytest.raises(ReaderError, match=r"different columns or types \(first differing file: data/chunk-000/"):
        read_dataset(root)


def test_success_column_in_some_files_only_is_ignored_with_note(tmp_path: Path) -> None:
    eps = _clean(4, 9)
    for e in eps[1:]:
        del e["next.success"]
    ds = read_dataset(write_dataset(tmp_path / "ds", eps, version="v2.0"))
    assert all(e.success is None for e in ds.episodes)
    assert any("next.success is missing from some data files" in n for n in ds.notes)


# ---- N2 ----------------------------------------------------------------------------------------------------------
def test_append_with_corrupt_session_json_names_the_file(tmp_path: Path, monkeypatch, capsys) -> None:
    from robot_report_card.sim import registry

    monkeypatch.setattr(registry, "make_env", lambda env_id: FakeEnv())
    out = tmp_path / "s"
    assert cli.main(["record", "--episodes", "1", "--max-steps", "4", "--out", str(out), "-q"]) == 0
    (out / "session.json").write_text("{broken")
    assert cli.main(["record", "--append", "--episodes", "1", "--max-steps", "4", "--out", str(out), "-q"]) == 1
    err = capsys.readouterr().err
    assert "session.json is not a valid session.json" in err and str(out) in err and "Traceback" not in err
