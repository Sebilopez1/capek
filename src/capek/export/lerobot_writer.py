"""Export a session to a LeRobot v3.0 dataset with lerobot's own writer, plus the ``meta/capek_tags.json`` snapshot.

Write path (research brief §2): ``LeRobotDataset.create(..., use_videos=False)`` -> ``add_frame`` per frame
(with a constant ``task`` string) -> ``save_episode`` per episode -> ``finalize()``. ``lerobot`` is imported
lazily, only here. Tags never go into ``task``; they live in the sidecar (schema_version 1):

    {"schema_version": 1, "tool": "capek", "capek_version": ...,
     "dataset": {"codebase_version", "total_episodes", "total_frames", "episode_lengths"},
     "episodes": {"<exported index>": {"source_episode_index", ...every episodes.jsonl field}}}
"""

from __future__ import annotations

import contextlib
import json
import shutil
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from capek import __version__
from capek.atomic import atomic_write_json
from capek.features import validate_arrays
from capek.hints import install_hint
from capek.legacy import TAGS_FILES, find_file, is_our_tool
from capek.session import LABELS, EpisodeMeta, Session

TASKS: dict[str, str] = {"so101_reach": "Move the gripper tip to the target point."}
DEFAULT_TASK = "Move the gripper tip to the target point."
ROBOT_TYPES: dict[str, str] = {"so101_reach": "so101_sim"}
TAGS_FILE = "capek_tags.json"  # 0.1.0 wrote rrc_tags.json; see capek.legacy
TAGS_SCHEMA_VERSION = 1


class ExportError(Exception):
    """A user-facing export failure (lerobot missing, root exists, nothing to export...)."""


@dataclass
class ExportResult:
    root: Path
    repo_id: str
    source_indices: list[int]  # position = exported episode_index
    total_frames: int


def import_lerobot_dataset() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ExportError(
            f"`capek export` needs the lerobot extra: {install_hint('lerobot')} (lerobot==0.4.4, numpy<2.4)"
        ) from e
    return LeRobotDataset


@contextlib.contextmanager
def quiet_lerobot() -> Iterator[None]:
    """Hide per-episode `datasets` progress bars and lerobot's bool->uint8 stats warning (CLI output only)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Converting input from bool", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0", category=DeprecationWarning)
        try:
            import datasets

            datasets.disable_progress_bars()
        except ImportError:
            pass
        yield


def select_episodes(metas: list[EpisodeMeta], exclude_labels: Iterable[str] = ()) -> list[EpisodeMeta]:
    excluded = set(exclude_labels)
    unknown = excluded - set(LABELS)
    if unknown:
        raise ExportError(f"unknown label(s) to exclude: {sorted(unknown)}")
    return [m for m in metas if m.label not in excluded]


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def written_by_capek(root: Path) -> bool:
    """True only for a dataset capek exported: ``meta/info.json`` plus our ``meta/capek_tags.json`` marker."""
    if not (root / "meta" / "info.json").is_file():
        return False
    try:
        return is_our_tool(read_tags(root).get("tool"))
    except (OSError, ValueError, AttributeError):
        return False


def check_root(session_dir: Path, root: Path, overwrite: bool) -> None:
    """Validate ``--out`` before anything is written or deleted (no side effects).

    Refuses: a symlinked root; a root that is, contains or lies inside the session; an existing root without
    ``--overwrite``; with ``--overwrite``, anything except an empty dir or a dataset capek itself exported.
    """
    if root.is_symlink():
        raise ExportError(f"{root} is a symlink; refusing to write through it (pass the real path)")
    session_real, root_real = session_dir.resolve(), root.resolve()
    if _is_within(session_real, root_real) or _is_within(root_real, session_real):
        raise ExportError(f"--out {root} overlaps the session {session_dir}; export to a directory outside the session")
    if not root.exists():
        return
    if not overwrite:
        raise ExportError(f"{root} already exists; pass --overwrite to replace it (or pick a new --out)")
    if not root.is_dir():
        raise ExportError(f"{root} exists and is not a directory; refusing to overwrite it")
    if any(root.iterdir()) and not written_by_capek(root):
        raise ExportError(
            f"{root} is not a dataset exported by capek (needs meta/info.json and meta/{TAGS_FILE}); "
            "refusing to delete it - delete it yourself if you really want to replace it"
        )


def prepare_root(session_dir: Path, root: Path, overwrite: bool) -> None:
    """lerobot requires ``root`` not to exist (G2): after ``check_root``, remove a capek-written/empty root."""
    check_root(session_dir, root, overwrite)
    if root.exists():
        shutil.rmtree(root)


def lerobot_features(session_features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Session feature spec -> lerobot ``features`` (lerobot compares shapes as tuples)."""
    return {k: {**v, "shape": tuple(v["shape"])} for k, v in session_features.items()}


def build_tags_snapshot(info: dict[str, Any], exported: list[EpisodeMeta]) -> dict[str, Any]:
    episodes = {}
    for new_index, m in enumerate(exported):
        row = m.to_dict()
        row.pop("episode_index")
        episodes[str(new_index)] = {"source_episode_index": m.episode_index, **row}
    return {
        "schema_version": TAGS_SCHEMA_VERSION,
        "tool": "capek",
        "capek_version": __version__,
        "dataset": {
            "codebase_version": info["codebase_version"],
            "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
            "episode_lengths": [m.num_frames for m in exported],
        },
        "episodes": episodes,
    }


def read_tags(root: Path | str) -> dict[str, Any]:
    return json.loads(find_file(Path(root) / "meta", TAGS_FILES).read_text())


def fingerprint_problems(root: Path | str) -> list[str]:
    """Compare the sidecar fingerprint with ``meta/info.json``; an empty list means it is current."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    tags = read_tags(root)
    fp = tags["dataset"]
    problems = [
        f"{key}: capek_tags.json has {fp[key]!r}, info.json has {info[key]!r}"
        for key in ("codebase_version", "total_episodes", "total_frames")
        if fp[key] != info[key]
    ]
    if len(fp["episode_lengths"]) != info["total_episodes"] or sum(fp["episode_lengths"]) != info["total_frames"]:
        problems.append("episode_lengths do not add up to info.json totals")
    if sorted(tags["episodes"], key=int) != [str(i) for i in range(info["total_episodes"])]:
        problems.append("episodes keys are not 0..total_episodes-1")
    return problems


def export_session(
    session_dir: Path | str,
    root: Path | str,
    repo_id: str,
    exclude_labels: Iterable[str] = (),
    overwrite: bool = False,
) -> ExportResult:
    """Write a finalized LeRobot v3.0 dataset at ``root`` plus ``meta/capek_tags.json``."""
    session = Session.open(session_dir)
    info = session.info()
    selected = select_episodes(session.read_metas(), exclude_labels)
    if not selected:
        raise ExportError("no episodes left to export (all excluded or session empty)")
    root = Path(root)
    check_root(session.root, root, overwrite)  # fail fast, before the slow lerobot import
    missing = [
        str(session.npz_path(m.episode_index)) for m in selected if not session.npz_path(m.episode_index).is_file()
    ]
    if missing:
        raise ExportError(f"session is missing episode file(s): {', '.join(missing)}")
    LeRobotDataset = import_lerobot_dataset()  # before touching root, so a missing extra deletes nothing
    prepare_root(session.root, root, overwrite)

    task = TASKS.get(info.env_id, DEFAULT_TASK)
    ds = None
    try:
        ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=info.fps,
            features=lerobot_features(info.features),
            root=root,
            robot_type=ROBOT_TYPES.get(info.env_id, info.env_id),
            use_videos=False,
        )
        for m in selected:
            arrays = session.load_arrays(m.episode_index)
            n = validate_arrays(arrays, info.features)
            if n != m.num_frames:
                raise ExportError(f"episode {m.episode_index}: {n} frames on disk, {m.num_frames} in episodes.jsonl")
            for t in range(n):
                frame: dict[str, Any] = {k: np.ascontiguousarray(a[t]) for k, a in arrays.items()}
                frame["task"] = task
                ds.add_frame(frame)
            ds.save_episode()
        ds.finalize()
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():  # G7: never let a half-written root reach a loader
            raise ExportError(f"lerobot did not write {info_path}")
        lerobot_info = json.loads(info_path.read_text())
        atomic_write_json(root / "meta" / TAGS_FILE, build_tags_snapshot(lerobot_info, selected))
        problems = fingerprint_problems(root)
        if problems:
            raise ExportError("capek_tags.json does not match info.json: " + "; ".join(problems))
    except BaseException:
        # Close lerobot's parquet writers and flush its metadata buffer *before* deleting: otherwise its
        # __del__ recreates meta/episodes/... after the rmtree and leaves an unloadable root behind (QA B2).
        if ds is not None:
            with contextlib.suppress(Exception):
                ds.finalize()
        shutil.rmtree(root, ignore_errors=True)  # root did not exist before this call, so it is ours
        raise
    return ExportResult(
        root=root,
        repo_id=repo_id,
        source_indices=[m.episode_index for m in selected],
        total_frames=int(lerobot_info["total_frames"]),
    )
