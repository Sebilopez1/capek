#!/usr/bin/env python3
"""Pre-release check for robot-report-card. It builds and tests locally and NEVER uploads or publishes anything.

    python tools/release_check.py                 # everything (about 2-4 min, needs network for the venv installs)
    python tools/release_check.py --skip-suite    # skip the light test suite from the sdist
    RRC_REDACTIONS=~/redactions.txt python tools/release_check.py

Steps (each PASS/FAIL in the final table):
  build        sdist + wheel (uv build, or python -m build)
  twine        `twine check` on both artifacts
  metadata     PEP 639: License-Expression Apache-2.0, 3 license files in dist-info/licenses, version matches
  sdist        ships tests, NOTICE and the two research spikes the parity tests need
  base-venv    clean venv + wheel only: size <= 100 MB, `rrc --help`, tag/list on a small session,
               `rrc score` on a dataset prints the PyPI hint for the score extra
  quickstart   clean venv + wheel[score,sim]: record -> tag -> list -> score
  sdist-suite  clean venv + sdist[dev,score,sim]: the light pytest profile, run from the unpacked sdist
  placeholder  no GITHUB_OWNER left (fails until `python tools/set_github_owner.py <owner>`; expected before then)
  redactions   no entry of the redaction denylist (kept outside the repo) appears in any tracked file

The denylist is read from --redactions, else $RRC_REDACTIONS, else ~/.rrc_redactions.txt: one entry per line, `#`
comments allowed, matched case-insensitively. Its entries are never printed: only entry numbers with file:line, or a
file's position when the entry is in the file name, plus a count of history commits that still contain an entry.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from set_github_owner import PLACEHOLDER, occurrences, tracked_files  # noqa: E402

BASE_VENV_LIMIT_MB = 100
DEFAULT_DENYLISTS = ("~/.rrc_redactions.txt",)
SPIKES = ("docs/spikes/phase2_score_proto.py", "docs/spikes/phase3_stats.py")
EXPECTED_LICENSES = ("LICENSE", "NOTICE", "so101/LICENSE")


class CheckFailed(Exception):
    pass


# ---- helpers ----------------------------------------------------------------------------------------------------
def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
        raise CheckFailed(f"`{' '.join(cmd)}` exited {r.returncode}:\n    " + "\n    ".join(tail))
    return r.stdout + r.stderr


def version() -> str:
    for line in (ROOT / "src/robot_report_card/__init__.py").read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip("\"'")
    raise CheckFailed("no __version__ in src/robot_report_card/__init__.py")


def make_venv(path: Path, python: str) -> Path:
    if shutil.which("uv"):
        run(["uv", "venv", "-q", "-p", python, str(path)])
    else:
        run([python, "-m", "venv", str(path)])
    return path / "bin"


def pip_install(bin_dir: Path, *reqs: str) -> None:
    if shutil.which("uv"):
        run(["uv", "pip", "install", "-q", "--python", str(bin_dir / "python"), *reqs])
    else:
        run([str(bin_dir / "pip"), "install", "-q", *reqs])


def dir_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) / 1e6


# ---- steps ------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, work: Path, python: str) -> None:
        self.work, self.python = work, python
        self.dist = work / "dist"
        self.wheel: Path | None = None
        self.sdist: Path | None = None


def step_build(ctx: Ctx) -> str:
    if shutil.which("uv"):
        run(["uv", "build", "-q", "-o", str(ctx.dist), str(ROOT)])
    else:
        run([ctx.python, "-m", "build", "--outdir", str(ctx.dist), str(ROOT)])
    wheels, sdists = list(ctx.dist.glob("*.whl")), list(ctx.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise CheckFailed(f"expected one wheel and one sdist, got {wheels + sdists}")
    ctx.wheel, ctx.sdist = wheels[0], sdists[0]
    return f"{ctx.wheel.name} ({ctx.wheel.stat().st_size / 1e6:.1f} MB), {ctx.sdist.name}"


def step_twine(ctx: Ctx) -> str:
    cmd = ["uvx", "-q", "twine"] if shutil.which("uvx") else [ctx.python, "-m", "twine"]
    out = run([*cmd, "check", "--strict", str(ctx.wheel), str(ctx.sdist)])
    return "PASSED on wheel and sdist" if out.count("PASSED") >= 2 else out.strip()


def step_metadata(ctx: Ctx) -> str:
    assert ctx.wheel
    with zipfile.ZipFile(ctx.wheel) as z:
        names = z.namelist()
        meta = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
    fields = dict(line.split(": ", 1) for line in meta.splitlines() if ": " in line and not line.startswith(" "))
    problems = []
    if fields.get("License-Expression") != "Apache-2.0":
        problems.append(f"License-Expression = {fields.get('License-Expression')!r}")
    if fields.get("Version") != version():
        problems.append(f"Version {fields.get('Version')} != __version__ {version()}")
    if any(line.startswith("Classifier: License ::") for line in meta.splitlines()):
        problems.append("a `License ::` classifier is present (PEP 639 uses License-Expression)")
    lic = [n for n in names if ".dist-info/licenses/" in n]
    for expected in EXPECTED_LICENSES:
        if not any(n.endswith(expected) for n in lic):
            problems.append(f"{expected} missing from dist-info/licenses")
    if any(n.startswith("tools/") or n.startswith("tests/") for n in names):
        problems.append("wheel contains tools/ or tests/")
    if problems:
        raise CheckFailed("; ".join(problems))
    return f"Metadata-Version {fields.get('Metadata-Version')}, Apache-2.0, {len(lic)} license files"


def step_sdist(ctx: Ctx) -> str:
    assert ctx.sdist
    with tarfile.open(ctx.sdist) as t:
        names = {n.split("/", 1)[1] for n in t.getnames() if "/" in n}
    need = [*SPIKES, "NOTICE", "LICENSE", "tests/conftest.py", "src/robot_report_card/__init__.py"]
    missing = [n for n in need if n not in names]
    if missing:
        raise CheckFailed(f"sdist lacks {missing}")
    return f"{len(names)} files incl. tests and the spikes"


def _synthetic_session(bin_dir: Path, where: Path) -> None:
    code = (
        "import sys, numpy as np\n"
        "from robot_report_card.features import make_features\n"
        "from robot_report_card.session import Episode, EpisodeMeta, Session, SessionInfo\n"
        "n = [f'j{i}' for i in range(6)]\n"
        "info = SessionInfo('synthetic', 30, make_features(n, ['x', 'y', 'z'], n), 't', 't', 't')\n"
        "s = Session.create(sys.argv[1], info)\n"
        "for i in range(3):\n"
        "    a = {'observation.state': np.zeros((5, 6), np.float32), 'observation.environment_state': np.zeros((5, 3),"
        " np.float32), 'action': np.zeros((5, 6), np.float32), 'next.reward': np.zeros((5, 1), np.float32),"
        " 'next.success': np.zeros((5, 1), bool)}\n"
        "    m = EpisodeMeta(i, 'synthetic', 'x', {}, 0, 30, 5, 5/30, False, 0.1, 'max_steps', 't', 't')\n"
        "    s.append_episode(Episode(m, a))\n"
    )
    run([str(bin_dir / "python"), "-c", code, str(where)])


def step_base_venv(ctx: Ctx) -> str:
    venv = ctx.work / "venv-base"
    bin_dir = make_venv(venv, ctx.python)
    pip_install(bin_dir, str(ctx.wheel))
    size = dir_mb(venv)
    rrc = str(bin_dir / "rrc")
    run([rrc, "--help"])
    _synthetic_session(bin_dir, ctx.work / "base-session")
    run([rrc, "tag", str(ctx.work / "base-session"), "-e", "1", "--label", "fail", "--note", "release check"])
    listing = run([rrc, "list", str(ctx.work / "base-session")])
    if "release check" not in listing:
        raise CheckFailed("rrc list doesn't show the tag")
    fake = ctx.work / "fake-dataset"  # a real layout; reading it needs pyarrow, which the base install lacks
    (fake / "data" / "chunk-000").mkdir(parents=True)
    (fake / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"")
    (fake / "meta").mkdir()
    (fake / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30}))
    out = run([rrc, "score", str(fake), "--json-out", str(ctx.work / "x.json")], check=False)
    if 'pip install "robot-report-card[score]"' not in out:
        raise CheckFailed(f"`rrc score` without pyarrow didn't print the PyPI hint: {out.strip()}")
    if size > BASE_VENV_LIMIT_MB:
        raise CheckFailed(f"base venv is {size:.0f} MB (> {BASE_VENV_LIMIT_MB} MB)")
    return f"{size:.0f} MB; --help, tag/list, score hint OK"


def step_quickstart(ctx: Ctx) -> str:
    venv = ctx.work / "venv-quickstart"
    bin_dir = make_venv(venv, ctx.python)
    pip_install(bin_dir, f"{ctx.wheel}[score,sim]")
    rrc = str(bin_dir / "rrc")
    q = ctx.work / "quickstart"
    q.mkdir()
    run([rrc, "record", "--env", "so101_reach", "--policy", "scripted", "--episodes", "10", "--seed", "0",
         "--out", "runs/demo", "-q"], cwd=q)  # fmt: skip
    run([rrc, "tag", "runs/demo", "--episode", "3", "--label", "fail", "--note", "overshot"], cwd=q)
    if "overshot" not in run([rrc, "list", "runs/demo"], cwd=q):
        raise CheckFailed("rrc list doesn't show the tag")
    # `rrc score` needs a LeRobot dataset; without the lerobot extra, write a small v3.0 one with pyarrow
    writer = (
        "import json, sys, numpy as np, pyarrow as pa, pyarrow.parquet as pq\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1]); (root / 'meta').mkdir(parents=True); (root / 'data/chunk-000').mkdir(parents=True)\n"
        "rng = np.random.default_rng(0); rows = []\n"
        "for e in range(12):\n"
        "    g = rng.uniform(-1, 1, 6); r = np.clip(np.arange(1, 91) / rng.uniform(30, 70), 0, 1)[:, None]\n"
        "    a = (r * g + rng.normal(0, 0.2 if e == 11 else 0.01, (90, 6))).astype(np.float32)\n"
        "    s = np.vstack([np.zeros((1, 6)), a[:-1]]).astype(np.float32)\n"
        "    rows.append(pa.table({'observation.state': pa.FixedSizeListArray.from_arrays(pa.array(s.ravel()), 6),"
        " 'action': pa.FixedSizeListArray.from_arrays(pa.array(a.ravel()), 6),"
        " 'timestamp': pa.array((np.arange(90) / 30).astype(np.float32)),"
        " 'frame_index': pa.array(np.arange(90)), 'episode_index': pa.array(np.full(90, e))}))\n"
        "pq.write_table(pa.concat_tables(rows), root / 'data/chunk-000/file-000.parquet')\n"
        "feat = {'dtype': 'float32', 'shape': [6], 'names': None}\n"
        "(root / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'fps': 30, 'total_episodes': 12,"
        " 'total_frames': 1080, 'features': {'observation.state': feat, 'action': feat}}))\n"
    )
    run([str(bin_dir / "python"), "-c", writer, str(q / "datasets/demo")])
    out = run([rrc, "score", "datasets/demo"], cwd=q)
    if "Motion quality can't detect" not in out or not (q / "demo.rrc_score.json").is_file():
        raise CheckFailed("rrc score output incomplete")
    return f"{dir_mb(venv):.0f} MB; record -> tag -> list -> score OK"


def step_sdist_suite(ctx: Ctx) -> str:
    venv = ctx.work / "venv-sdist"
    bin_dir = make_venv(venv, ctx.python)
    pip_install(bin_dir, f"{ctx.sdist}[dev,score,sim]")
    src = ctx.work / "sdist-src"
    with tarfile.open(ctx.sdist) as t:
        t.extractall(src, filter="data") if sys.version_info >= (3, 12) else t.extractall(src)  # noqa: S202
    tree = next(src.iterdir())
    t0 = time.perf_counter()
    out = run([str(bin_dir / "python"), "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=tree,
              env={"HF_HUB_OFFLINE": "1"})  # fmt: skip
    summary = [line for line in out.splitlines() if " passed" in line]
    return f"{summary[-1].strip('= ') if summary else 'passed'} ({time.perf_counter() - t0:.0f} s)"


def step_placeholder(ctx: Ctx) -> str:
    found = occurrences(ROOT)
    if found:
        where = ", ".join(f"{f} ({n})" for f, n in sorted(found.items()))
        raise CheckFailed(
            f"{PLACEHOLDER} still in {where}. Expected until the owner is set: python tools/set_github_owner.py <owner>"
        )
    return f"no {PLACEHOLDER} left"


def find_denylist(explicit: str | None) -> Path | None:
    for candidate in (explicit, os.environ.get("RRC_REDACTIONS"), *DEFAULT_DENYLISTS):
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser()
    return None


def denylist_hits(entries: list[str], root: Path = ROOT) -> list[str]:
    """``entry #k: path:line`` for every case-insensitive match (the entry text itself is never returned)."""
    needles = [(k, e.lower()) for k, e in enumerate(entries, 1)]
    hits = []
    for index, path in enumerate(tracked_files(root)):
        rel = path.relative_to(root).as_posix()
        for k, needle in needles:
            if needle in rel.lower():  # the path would reveal the entry, so name only its position
                hits.append(f"entry #{k}: a tracked file name (index {index})")
                rel = f"<tracked file #{index}>"
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            text = path.read_bytes().decode("latin-1")
        for lineno, line in enumerate(text.lower().splitlines(), 1):
            for k, needle in needles:
                if needle in line:
                    hits.append(f"entry #{k}: {rel}:{lineno}")
    return hits


def read_denylist(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def step_redactions(ctx: Ctx, explicit: str | None) -> str:
    path = find_denylist(explicit)
    if path is None:
        raise CheckFailed("no redaction denylist found (pass --redactions PATH or set RRC_REDACTIONS)")
    entries = read_denylist(path)
    if not entries:
        raise CheckFailed(f"denylist {path} has no entries")
    hits = denylist_hits(entries)
    if hits:
        raise CheckFailed("denylisted text in tracked files: " + "; ".join(hits[:20]))
    history = history_commits(entries)
    note = (
        f"; {history} commit(s) in git history still contain an entry (squash-or-keep is Sebi's call)"
        if history
        else ""
    )
    return f"{len(entries)} entries, 0 matches in {len(tracked_files())} tracked files{note}"


def history_commits(entries: list[str], root: Path = ROOT) -> int:
    """How many commits (any branch) add or remove a denylist entry; a count only, never the entries or commits."""
    import re as _re

    commits: set[str] = set()
    for entry in entries:
        try:
            out = subprocess.run(
                ["git", "log", "--all", "--format=%H", "-i", "-G", _re.escape(entry)],
                cwd=root, capture_output=True, text=True, check=True,
            ).stdout  # fmt: skip
        except (OSError, subprocess.CalledProcessError):
            return 0
        commits.update(out.split())
    return len(commits)


# ---- main -------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redactions", help="denylist file (default: $RRC_REDACTIONS, then ~/.rrc_redactions.txt)")
    p.add_argument("--python", default=sys.executable, help="interpreter for the clean venvs (default: this one)")
    p.add_argument("--skip-suite", action="store_true", help="skip the light test suite from the sdist")
    p.add_argument("--keep", action="store_true", help="keep the work directory (printed at the end)")
    args = p.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="rrc-release-check-"))
    ctx = Ctx(work, args.python)
    steps: list[tuple[str, Callable[[], str], tuple[str, ...]]] = [
        ("build", lambda: step_build(ctx), ()),
        ("twine", lambda: step_twine(ctx), ("build",)),
        ("metadata", lambda: step_metadata(ctx), ("build",)),
        ("sdist", lambda: step_sdist(ctx), ("build",)),
        ("base-venv", lambda: step_base_venv(ctx), ("build",)),
        ("quickstart", lambda: step_quickstart(ctx), ("build",)),
        ("sdist-suite", lambda: step_sdist_suite(ctx), ("build",)),
        ("placeholder", lambda: step_placeholder(ctx), ()),
        ("redactions", lambda: step_redactions(ctx, args.redactions), ()),
    ]
    results: dict[str, tuple[str, str]] = {}
    for name, fn, needs in steps:
        if name == "sdist-suite" and args.skip_suite:
            results[name] = ("SKIP", "--skip-suite")
            continue
        if any(results.get(n, ("",))[0] != "PASS" for n in needs):
            results[name] = ("SKIP", f"needs {', '.join(needs)}")
            continue
        t0 = time.perf_counter()
        print(f"[{name}] ...", flush=True)
        try:
            results[name] = ("PASS", f"{fn()} [{time.perf_counter() - t0:.0f} s]")
        except (CheckFailed, OSError, StopIteration, ValueError) as e:
            results[name] = ("FAIL", str(e))
    print("\nrelease check (nothing is uploaded or published)")
    for name, (status, detail) in results.items():
        print(f"  {status:4s}  {name:12s} {detail}")
    failed = [n for n, (s, _) in results.items() if s == "FAIL"]
    if failed == ["placeholder"]:
        print("\nOnly the placeholder check failed: set the owner with tools/set_github_owner.py, then rerun.")
    print(f"\n{'RELEASE CHECK PASSED' if not failed else 'RELEASE CHECK FAILED: ' + ', '.join(failed)}")
    if args.keep:
        print(f"work dir: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
