"""0.2.0 renamed Robot Report Card (rrc) to Capek. Files written by 0.1.0 must still be readable."""

from __future__ import annotations

import json
from pathlib import Path

from capek.legacy import POLICY_FILES, TAGS_FILES, find_file, is_our_tool, upgrade_version_key
from capek.report.rules import check_kind
from capek.session import EpisodeMeta, SessionInfo


def test_old_tool_name_is_accepted() -> None:
    assert is_our_tool("capek") and is_our_tool("robot-report-card") and not is_our_tool("other")
    check_kind({"tool": "robot-report-card", "kind": "compare"}, "compare", "old.json")  # no raise


def test_find_file_prefers_new_name_then_old(tmp_path: Path) -> None:
    assert find_file(tmp_path, TAGS_FILES).name == "capek_tags.json"  # neither exists -> current name
    (tmp_path / "rrc_tags.json").write_text("{}")
    assert find_file(tmp_path, TAGS_FILES).name == "rrc_tags.json"
    (tmp_path / "capek_tags.json").write_text("{}")
    assert find_file(tmp_path, TAGS_FILES).name == "capek_tags.json"
    (tmp_path / "rrc_policy.json").write_text("{}")
    assert find_file(tmp_path, POLICY_FILES).name == "rrc_policy.json"


def test_old_session_rows_load() -> None:
    row = {
        "episode_index": 0, "env_id": "so101_reach", "policy_name": "scripted", "policy_params": {}, "seed": 0,
        "fps": 30, "num_frames": 90, "duration_s": 3.0, "sim_success": True, "final_error_m": 0.01,
        "termination_reason": "max_steps", "recorded_at": "2026-09-30T00:00:00Z", "rrc_version": "0.1.0",
    }  # fmt: skip
    assert EpisodeMeta.from_dict(row).capek_version == "0.1.0"
    info = {
        "env_id": "so101_reach",
        "fps": 30,
        "features": {},
        "seeding": "x",
        "created_at": "t",
        "rrc_version": "0.1.0",
    }
    assert SessionInfo.from_dict(info).capek_version == "0.1.0"
    assert upgrade_version_key({"capek_version": "0.2.0"}) == {"capek_version": "0.2.0"}


def test_old_dataset_labels_are_read(tmp_path: Path) -> None:
    from capek.score.reader import _read_tags

    meta = tmp_path / "meta"
    meta.mkdir()
    info = {"total_episodes": 2, "total_frames": 180}
    (meta / "rrc_tags.json").write_text(json.dumps({"tool": "robot-report-card", "dataset": info}))
    notes: list[str] = []
    assert _read_tags(tmp_path, info, notes) is not None and notes == []


def test_old_score_report_passes_kind_check() -> None:
    check_kind({"tool": "robot-report-card", "episodes": [], "summary": {}}, "score", "old.score.json")  # no raise


def test_dataset_exported_by_0_1_0_counts_as_ours(tmp_path: Path) -> None:
    from capek.export.lerobot_writer import written_by_capek

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text("{}")
    assert not written_by_capek(tmp_path)
    (meta / "rrc_tags.json").write_text(json.dumps({"tool": "robot-report-card"}))
    assert written_by_capek(tmp_path)  # so `capek export --overwrite` may replace it
    (meta / "rrc_tags.json").write_text(json.dumps({"tool": "someone-else"}))
    assert not written_by_capek(tmp_path)


def test_bc_checkpoint_with_only_rrc_policy_json_is_found(tmp_path: Path) -> None:
    from capek.eval.bc import POLICY_FILE

    (tmp_path / "rrc_policy.json").write_text("{}")
    assert find_file(tmp_path, POLICY_FILES).name == "rrc_policy.json" and POLICY_FILE == "capek_policy.json"
