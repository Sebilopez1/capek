#!/usr/bin/env python3
"""Replace the single release placeholder ``GITHUB_OWNER`` with the real GitHub owner (user or org).

    python tools/set_github_owner.py <owner>          # edits files in place, prints what changed
    python tools/set_github_owner.py <owner> --dry-run  # show what would change
    python tools/set_github_owner.py --check            # exit 1 while the placeholder is still anywhere

The placeholder may only live in ``ALLOWED`` (a test enforces this), so replacing it there is complete.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PLACEHOLDER = "GITHUB_OWNER"
ROOT = Path(__file__).resolve().parents[1]
# files (glob patterns, repo-relative) where the placeholder is allowed and will be replaced
ALLOWED = (
    "pyproject.toml",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "RELEASE.md",
    "docs/launch/*.md",
)
# files that mention the placeholder by name (the tooling itself); never rewritten
EXEMPT = ("tools/set_github_owner.py", "tools/release_check.py", "tests/test_p4_1_packaging.py")
# crew records (plans, briefs, reviews, milestones, STATUS) talk *about* the placeholder; they are history, not
# release links, so they are neither rewritten nor counted
EXEMPT_GLOBS = ("STATUS.md", "docs/phase*.md", "docs/research-brief.md", "docs/reviews/*.md", "docs/milestones/*.md")
OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")  # GitHub user/org name rules


def tracked_files(root: Path = ROOT) -> list[Path]:
    """Files git tracks (plus untracked, non-ignored ones); falls back to a walk outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, check=True,
        ).stdout  # fmt: skip
        return sorted({root / p for p in out.decode().split("\0") if p and (root / p).is_file()})
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", ".venv", "venv", "dist", "build", "__pycache__", ".pytest_cache", ".ruff_cache"}
        return sorted(p for p in root.rglob("*") if p.is_file() and not skip & set(p.relative_to(root).parts))


def is_allowed(rel: str) -> bool:
    return any(Path(rel).match(pattern) for pattern in ALLOWED)


def occurrences(root: Path = ROOT) -> dict[str, int]:
    """repo-relative path -> number of placeholder occurrences (exempt tooling files excluded)."""
    found = {}
    for path in tracked_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in EXEMPT or (not is_allowed(rel) and any(Path(rel).match(g) for g in EXEMPT_GLOBS)):
            continue
        try:
            n = path.read_text(encoding="utf-8").count(PLACEHOLDER)
        except (UnicodeDecodeError, OSError):
            continue
        if n:
            found[rel] = n
    return found


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("owner", nargs="?", help="GitHub user or organization that will own robot-report-card")
    p.add_argument(
        "--check",
        action="store_true",
        help=f"verify only: exit 1 while any {PLACEHOLDER} is left (used by release.yml before publishing)",
    )
    p.add_argument("--dry-run", action="store_true", help="with OWNER: show what would be replaced, change nothing")
    p.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    root = Path(args.root)
    found = occurrences(root)
    stray = sorted(f for f in found if not is_allowed(f))
    if args.check:
        for rel, n in sorted(found.items()):
            print(f"{PLACEHOLDER} x {n} in {rel}")
        if found:
            print(
                f"error: {PLACEHOLDER} is still present; run: python tools/set_github_owner.py <owner>", file=sys.stderr
            )
            return 1
        print(f"no {PLACEHOLDER} left")
        return 0
    if not args.owner:
        p.error("OWNER is required unless --check is given")
    if not OWNER_RE.match(args.owner) or "--" in args.owner or args.owner == PLACEHOLDER:
        print(f"error: {args.owner!r} is not a valid GitHub owner name", file=sys.stderr)
        return 2
    if stray:
        print(f"error: {PLACEHOLDER} found outside the allowed files: {', '.join(stray)}", file=sys.stderr)
        return 1
    for rel, n in sorted(found.items()):
        print(f"{'would replace' if args.dry_run else 'replaced'} {n} x {PLACEHOLDER} in {rel}")
        if not args.dry_run:
            path = root / rel
            path.write_text(path.read_text(encoding="utf-8").replace(PLACEHOLDER, args.owner), encoding="utf-8")
    if not found:
        print(f"no {PLACEHOLDER} left; nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
