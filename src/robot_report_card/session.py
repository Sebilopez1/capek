"""Local session format (the recorder's output and the tagger's source of truth).

Layout of a session directory::

    <session>/
      session.json                 # env_id, fps, feature spec, seeding scheme (written once by `rrc record`)
      episodes.jsonl               # one EpisodeMeta row per episode; the tag source of truth (atomic rewrites)
      episodes/episode_000000.npz  # per-frame arrays keyed by feature name (see features.py)

This module depends only on numpy and the standard library, so `rrc tag` / `rrc list` work without
mujoco or lerobot installed.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_report_card.atomic import atomic_write, atomic_write_json, atomic_write_jsonl, dump_json_line
from robot_report_card.features import FeatureSpec, validate_arrays

SESSION_SCHEMA_VERSION = 1
SESSION_FILE = "session.json"
EPISODES_FILE = "episodes.jsonl"
EPISODES_DIR = "episodes"

LABELS: tuple[str, ...] = ("success", "fail", "unlabeled")
TERMINATION_REASONS: tuple[str, ...] = ("max_steps", "truncated", "error")


class SessionError(Exception):
    """A user-facing problem with a session directory (missing, corrupt, bad index...)."""


@dataclass
class EpisodeMeta:
    """One row of ``episodes.jsonl``. ``sim_success`` comes from the sim; ``label`` from a human."""

    episode_index: int
    env_id: str
    policy_name: str
    policy_params: dict[str, Any]
    seed: int
    fps: int
    num_frames: int
    duration_s: float
    sim_success: bool
    final_error_m: float
    termination_reason: str
    recorded_at: str
    rrc_version: str
    label: str = "unlabeled"
    notes: str = ""
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")
        if self.termination_reason not in TERMINATION_REASONS:
            raise ValueError(f"termination_reason must be one of {TERMINATION_REASONS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> EpisodeMeta:
        known = set(cls.__dataclass_fields__)
        missing = known - set(row) - {"label", "notes", "flags"}
        if missing:
            raise ValueError(f"episode row is missing fields: {sorted(missing)}")
        return cls(**{k: v for k, v in row.items() if k in known})


@dataclass
class Episode:
    """Metadata plus per-frame arrays keyed by feature name (``observation.state``, ``action``, ...)."""

    meta: EpisodeMeta
    arrays: dict[str, np.ndarray]

    @property
    def num_frames(self) -> int:
        return int(next(iter(self.arrays.values())).shape[0])


@dataclass
class SessionInfo:
    env_id: str
    fps: int
    features: FeatureSpec
    seeding: str
    created_at: str
    rrc_version: str
    schema_version: int = SESSION_SCHEMA_VERSION
    max_steps: int | None = None  # frames per episode (None in sessions written before `--append` existed)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Session:
    """Read/write access to a session directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # ---- paths -------------------------------------------------------------------------------------------
    @property
    def info_path(self) -> Path:
        return self.root / SESSION_FILE

    @property
    def episodes_path(self) -> Path:
        return self.root / EPISODES_FILE

    def npz_path(self, index: int) -> Path:
        return self.root / EPISODES_DIR / f"episode_{index:06d}.npz"

    # ---- lifecycle ---------------------------------------------------------------------------------------
    @classmethod
    def create(cls, root: Path | str, info: SessionInfo) -> Session:
        """Start a new session; refuses to touch a directory that already holds one."""
        s = cls(root)
        if s.root.exists() or s.root.is_symlink():
            if not s.root.is_dir():
                raise SessionError(f"{s.root} exists and is not a directory; choose a new --out directory")
            if s.info_path.exists() or s.episodes_path.exists():
                raise SessionError(
                    f"{s.root} already contains a session; choose a new --out directory (or pass --append)"
                )
            if any(s.root.iterdir()):
                raise SessionError(f"{s.root} is not empty; choose a new (or empty) --out directory")
        try:
            s.root.mkdir(parents=True, exist_ok=True)
            (s.root / EPISODES_DIR).mkdir(exist_ok=True)
        except OSError as e:
            raise SessionError(f"cannot create session directory {s.root}: {e.strerror or e}") from e
        atomic_write_json(s.info_path, {**info.__dict__})
        atomic_write_jsonl(s.episodes_path, [])
        return s

    @classmethod
    def open(cls, root: Path | str) -> Session:
        s = cls(root)
        if not s.episodes_path.is_file():
            raise SessionError(f"{s.root} is not a session (no {EPISODES_FILE}); create one with `rrc record`")
        return s

    def info(self) -> SessionInfo:
        try:
            return SessionInfo.from_dict(json.loads(self.info_path.read_text()))
        except FileNotFoundError as e:
            raise SessionError(f"{self.info_path} is missing") from e
        except (ValueError, TypeError, AttributeError) as e:
            raise SessionError(f"{self.info_path} is not a valid session.json ({e})") from e

    # ---- episode metadata (episodes.jsonl) ----------------------------------------------------------------
    def read_metas(self) -> list[EpisodeMeta]:
        metas = []
        for lineno, line in enumerate(self.episodes_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                metas.append(EpisodeMeta.from_dict(json.loads(line)))
            except (ValueError, TypeError) as e:
                raise SessionError(f"{self.episodes_path}:{lineno}: bad row ({e})") from e
        return metas

    def write_metas(self, metas: list[EpisodeMeta]) -> None:
        """Atomically replace ``episodes.jsonl`` (temp file + rename)."""
        atomic_write_jsonl(self.episodes_path, (m.to_dict() for m in metas))

    # ---- episodes (npz + row) -----------------------------------------------------------------------------
    def append_episode(self, episode: Episode) -> None:
        """Write the episode's ``.npz`` then add its row; indices must be contiguous from 0."""
        metas = self.read_metas()
        if episode.meta.episode_index != len(metas):
            raise SessionError(f"expected episode_index {len(metas)}, got {episode.meta.episode_index}")
        validate_arrays(episode.arrays, self.info().features)
        dump_json_line(episode.meta.to_dict())  # fail on non-JSON metadata (e.g. NaN) before writing the .npz
        if episode.num_frames != episode.meta.num_frames:
            raise SessionError("meta.num_frames does not match the arrays")
        buf = io.BytesIO()
        np.savez(buf, **episode.arrays)
        atomic_write(self.npz_path(episode.meta.episode_index), lambda fh: fh.write(buf.getvalue()))
        self.write_metas([*metas, episode.meta])

    def load_arrays(self, index: int) -> dict[str, np.ndarray]:
        path = self.npz_path(index)
        if not path.is_file():
            raise SessionError(f"missing episode file {path}")
        try:
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile) as e:
            raise SessionError(f"corrupt episode file {path}: {e}") from e

    def load_episode(self, index: int) -> Episode:
        metas = self.read_metas()
        if not 0 <= index < len(metas):
            raise SessionError(f"episode {index} out of range (session has {len(metas)} episodes)")
        return Episode(meta=metas[index], arrays=self.load_arrays(index))
