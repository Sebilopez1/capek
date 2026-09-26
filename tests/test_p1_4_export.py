"""P1-4: `rrc export` -> LeRobot v3.0 + meta/rrc_tags.json (offline; synthetic sessions)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import FEATURES, make_synthetic_session

from robot_report_card import cli
from robot_report_card.session import Session

N_EPISODES, N_FRAMES = 4, 12


lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
LeRobotDataset = lerobot_dataset.LeRobotDataset
from robot_report_card.export.lerobot_writer import fingerprint_problems, read_tags  # noqa: E402


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("export") / "session"
    make_synthetic_session(root, n=N_EPISODES, num_frames=N_FRAMES)
    assert cli.main(["tag", str(root), "-e", "1", "--label", "fail", "--note", "overshot", "--flag", "jerky"]) == 0
    assert cli.main(["tag", str(root), "-e", "2", "--label", "success"]) == 0
    return root


@pytest.fixture(scope="module")
def exported(session_dir: Path) -> Path:
    out = session_dir.parent / "ds_all"
    assert cli.main(["export", str(session_dir), "--out", str(out), "--repo-id", "local/all"]) == 0
    return out


def test_dataset_loads_with_matching_counts_and_features(exported: Path) -> None:
    ds = LeRobotDataset("local/all", root=exported)
    assert ds.num_episodes == N_EPISODES and ds.num_frames == N_EPISODES * N_FRAMES
    assert ds.fps == 30 and ds.meta.info["codebase_version"] == "v3.0"
    assert ds.meta.info["robot_type"] == "synthetic"
    feats = ds.meta.features
    for key, spec in FEATURES.items():
        assert feats[key]["dtype"] == spec["dtype"]
        assert list(feats[key]["shape"]) == spec["shape"]
        assert feats[key]["names"] == spec["names"]
    assert list(ds.meta.episodes["length"]) == [N_FRAMES] * N_EPISODES
    assert set(ds.meta.tasks.index) == {"Move the gripper tip to the target point."}


def test_sampled_frames_equal_npz(exported: Path, session_dir: Path) -> None:
    ds = LeRobotDataset("local/all", root=exported)
    session = Session.open(session_dir)
    for ep, t in [(0, 0), (1, 5), (3, N_FRAMES - 1)]:
        arrays = session.load_arrays(ep)
        item = ds[int(ds.meta.episodes["dataset_from_index"][ep]) + t]
        assert int(item["episode_index"]) == ep and int(item["frame_index"]) == t
        assert float(item["timestamp"]) == pytest.approx(t / 30, abs=1e-6)
        for key in ("observation.state", "observation.environment_state", "action", "next.reward"):
            np.testing.assert_allclose(item[key].numpy().reshape(-1), arrays[key][t].reshape(-1), rtol=0, atol=1e-7)
        assert bool(item["next.success"]) == bool(arrays["next.success"][t, 0])


def test_tags_snapshot_matches_session_and_info(exported: Path, session_dir: Path) -> None:
    tags = read_tags(exported)
    info = json.loads((exported / "meta" / "info.json").read_text())
    assert fingerprint_problems(exported) == []
    assert tags["schema_version"] == 1 and tags["tool"] == "robot-report-card"
    assert tags["dataset"] == {
        "codebase_version": info["codebase_version"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "episode_lengths": [N_FRAMES] * N_EPISODES,
    }
    for m in Session.open(session_dir).read_metas():
        row = tags["episodes"][str(m.episode_index)]
        expected = m.to_dict()
        expected["source_episode_index"] = expected.pop("episode_index")
        assert row == expected
    assert tags["episodes"]["1"]["label"] == "fail" and tags["episodes"]["1"]["sim_success"] is False


def test_exclude_label_renumbers_and_keeps_source_index(session_dir: Path) -> None:
    out = session_dir.parent / "ds_nofail"
    assert cli.main(["export", str(session_dir), "--out", str(out), "--exclude-label", "fail"]) == 0
    ds = LeRobotDataset("local/ds_nofail", root=out)
    assert ds.num_episodes == N_EPISODES - 1
    tags = read_tags(out)
    assert sorted(tags["episodes"], key=int) == ["0", "1", "2"]
    assert [tags["episodes"][k]["source_episode_index"] for k in ("0", "1", "2")] == [0, 2, 3]
    assert fingerprint_problems(out) == []
    # exported episode 1 is source episode 2
    src = Session.open(session_dir).load_arrays(2)
    item = ds[int(ds.meta.episodes["dataset_from_index"][1])]
    np.testing.assert_allclose(item["action"].numpy(), src["action"][0], atol=1e-7)


def test_existing_root_needs_overwrite(exported: Path, session_dir: Path, capsys) -> None:
    before = (exported / "meta" / "info.json").read_bytes()
    assert cli.main(["export", str(session_dir), "--out", str(exported)]) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert (exported / "meta" / "info.json").read_bytes() == before
    not_a_dataset = session_dir.parent / "precious"
    not_a_dataset.mkdir()
    (not_a_dataset / "keep.txt").write_text("x")
    assert cli.main(["export", str(session_dir), "--out", str(not_a_dataset), "--overwrite"]) == 1
    assert "refusing" in capsys.readouterr().err and (not_a_dataset / "keep.txt").exists()


def test_retag_then_overwrite_updates_snapshot(session_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ds"
    assert cli.main(["export", str(session_dir), "--out", str(out)]) == 0
    assert cli.main(["tag", str(session_dir), "-e", "3", "--label", "fail"]) == 0
    try:
        assert cli.main(["export", str(session_dir), "--out", str(out), "--overwrite"]) == 0
        assert read_tags(out)["episodes"]["3"]["label"] == "fail"
        assert LeRobotDataset("local/ds", root=out).num_episodes == N_EPISODES
    finally:
        cli.main(["tag", str(session_dir), "-e", "3", "--clear-label"])


def test_nothing_to_export_and_stale_fingerprint(exported: Path, session_dir: Path, tmp_path: Path, capsys) -> None:
    argv = ["export", str(session_dir), "--out", str(tmp_path / "x")]
    assert (
        cli.main([*argv, "--exclude-label", "fail", "--exclude-label", "success", "--exclude-label", "unlabeled"]) == 1
    )
    assert "no episodes left" in capsys.readouterr().err and not (tmp_path / "x").exists()
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "meta").mkdir()
    (stale / "meta" / "info.json").write_bytes((exported / "meta" / "info.json").read_bytes())
    tags = read_tags(exported)
    tags["dataset"]["total_frames"] += 1
    (stale / "meta" / "rrc_tags.json").write_text(json.dumps(tags))
    assert any("total_frames" in p for p in fingerprint_problems(stale))


# ---- QA R2: a failed export leaves no --out directory and a re-export then works -----------------------------
@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt])
def test_failed_export_leaves_nothing_and_reexport_works(
    session_dir: Path, tmp_path: Path, monkeypatch, exc: type[BaseException]
) -> None:
    import gc

    out = tmp_path / "ds"
    real_save = LeRobotDataset.save_episode
    calls = {"n": 0}

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:  # episode 0 is fully written (metadata buffered), episode 1 fails
            raise exc("simulated failure")  # a fresh instance, so no test-held object keeps its traceback alive
        return real_save(self, *a, **kw)

    monkeypatch.setattr(LeRobotDataset, "save_episode", flaky)
    raised = False
    try:
        cli.main(["export", str(session_dir), "--out", str(out)])
    except exc:
        raised = True  # the traceback (and the dataset object in its frames) is released after this block
    assert raised
    gc.collect()  # lerobot's __del__ must not resurrect meta/episodes/... after the cleanup
    assert not out.exists()
    monkeypatch.setattr(LeRobotDataset, "save_episode", real_save)
    assert cli.main(["export", str(session_dir), "--out", str(out)]) == 0  # no --overwrite needed
    assert LeRobotDataset("local/ds", root=out).num_episodes == N_EPISODES
    assert fingerprint_problems(out) == []


def test_missing_npz_leaves_no_out_dir_then_reexport_works(tmp_path: Path, capsys) -> None:
    make_synthetic_session(tmp_path / "s", n=3)
    npz = Session.open(tmp_path / "s").npz_path(1)
    moved = tmp_path / "moved.npz"
    npz.rename(moved)
    out = tmp_path / "ds"
    assert cli.main(["export", str(tmp_path / "s"), "--out", str(out)]) == 1
    assert "missing episode file" in capsys.readouterr().err and not out.exists()
    moved.rename(npz)
    assert cli.main(["export", str(tmp_path / "s"), "--out", str(out)]) == 0
    assert LeRobotDataset("local/ds", root=out).num_episodes == 3
