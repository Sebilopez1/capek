"""Tag episodes in a session (`capek tag`) and render them (`capek list`). Needs only numpy + stdlib.

``sim_success`` is ground truth from the sim and is never edited here; ``label`` is the human judgement.
The two are only linked by an explicit ``--from-sim``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from capek.session import LABELS, EpisodeMeta, Session, SessionError

NOTE_SEPARATOR = "; "


@dataclass
class TagChange:
    label: str | None = None  # new label; "unlabeled" clears it
    note: str | None = None  # appended to notes (or replaces them with replace_notes)
    replace_notes: bool = False
    clear_notes: bool = False
    add_flags: list[str] = field(default_factory=list)
    remove_flags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.label is not None and self.label not in LABELS:
            raise SessionError(f"invalid label {self.label!r}; choose from {', '.join(LABELS)}")
        if self.clear_notes and self.note is not None:
            raise SessionError("--clear-notes cannot be combined with --note")
        if self.replace_notes and self.note is None:
            raise SessionError("--replace-notes needs --note")
        for f in [*self.add_flags, *self.remove_flags]:
            if not f.strip():
                raise SessionError("flags must be non-empty strings")
        if {f.strip() for f in self.add_flags} & {f.strip() for f in self.remove_flags}:
            raise SessionError("the same flag cannot be both added and removed")

    def is_empty(self) -> bool:
        return (
            self.label is None
            and self.note is None
            and not self.clear_notes
            and not self.add_flags
            and not self.remove_flags
        )

    def apply(self, meta: EpisodeMeta) -> None:
        if self.label is not None:
            meta.label = self.label
        if self.clear_notes:
            meta.notes = ""
        elif self.note is not None:
            meta.notes = (
                self.note if (self.replace_notes or not meta.notes) else meta.notes + NOTE_SEPARATOR + self.note
            )
        remove = {f.strip() for f in self.remove_flags}
        flags = [f for f in meta.flags if f not in remove]
        flags += [f.strip() for f in self.add_flags if f.strip() not in flags]
        meta.flags = flags


def resolve_indices(metas: list[EpisodeMeta], indices: list[int] | None) -> list[int]:
    """Validate episode indices (``None`` means all episodes)."""
    if indices is None:
        return [m.episode_index for m in metas]
    valid = {m.episode_index for m in metas}
    bad = sorted({i for i in indices if i not in valid})
    if bad:
        hi = f"0..{len(metas) - 1}" if metas else "none"
        raise SessionError(f"episode index {', '.join(map(str, bad))} out of range (valid: {hi})")
    return sorted(set(indices))


def apply_tags(session: Session, indices: list[int] | None, change: TagChange) -> list[EpisodeMeta]:
    """Apply ``change`` to the selected episodes and atomically rewrite ``episodes.jsonl``."""
    change.validate()
    if change.is_empty():
        raise SessionError("nothing to change: give --label, --note, --clear-notes, --flag or --unflag")
    metas = session.read_metas()
    selected = resolve_indices(metas, indices)
    by_index = {m.episode_index: m for m in metas}
    for i in selected:
        change.apply(by_index[i])
    session.write_metas(metas)
    return [by_index[i] for i in selected]


def labels_from_sim(session: Session, indices: list[int] | None, overwrite: bool = False) -> tuple[int, int]:
    """Copy ``sim_success`` into ``label`` (success/fail). Returns (changed, skipped-because-already-labelled)."""
    metas = session.read_metas()
    selected = set(resolve_indices(metas, indices))
    changed = skipped = 0
    for m in metas:
        if m.episode_index not in selected:
            continue
        if m.label != "unlabeled" and not overwrite:
            skipped += 1
            continue
        new = "success" if m.sim_success else "fail"
        changed += new != m.label
        m.label = new
    session.write_metas(metas)
    return changed, skipped


# ---- rendering -------------------------------------------------------------------------------------------------
def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def format_table(metas: list[EpisodeMeta], notes_width: int = 40) -> str:
    header = ["index", "length", "sim_success", "final_error_m", "label", "policy", "flags", "notes"]
    rows = [
        [
            str(m.episode_index),
            str(m.num_frames),
            "yes" if m.sim_success else "no",
            f"{m.final_error_m:.4f}",
            m.label,
            m.policy_name + (f" (noise {m.policy_params['noise']:g})" if m.policy_params.get("noise") else ""),
            ",".join(m.flags),
            _truncate(m.notes.replace("\n", " "), notes_width),
        ]
        for m in metas
    ]
    widths = [max(len(r[c]) for r in [header, *rows]) for c in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)).rstrip() for r in [header, *rows]]
    labels = Counter(m.label for m in metas)
    lines.append(
        f"{len(metas)} episodes, {sum(m.num_frames for m in metas)} frames; "
        f"sim success {sum(m.sim_success for m in metas)}/{len(metas)}; "
        + ", ".join(f"{lab} {labels.get(lab, 0)}" for lab in LABELS)
    )
    return "\n".join(lines)


def format_json(metas: list[EpisodeMeta]) -> str:
    return json.dumps([m.to_dict() for m in metas], indent=2)
