"""P4-1: packaging metadata, PyPI install hints, the GITHUB_OWNER placeholder and tools/release_check.py helpers."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
SRC = REPO / "src" / "capek"
needs_repo = pytest.mark.skipif(not (TOOLS / "release_check.py").is_file(), reason="tools/ is not in the sdist")


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    return tomllib.loads((REPO / "pyproject.toml").read_text())


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        sys.path.remove(str(TOOLS))
    return mod


def test_pep639_metadata_and_extras() -> None:
    from capek import __version__

    cfg = _pyproject()
    proj = cfg["project"]
    assert __version__ == "0.2.0"
    assert cfg["build-system"]["requires"] == ["hatchling>=1.27"]
    assert proj["license"] == "Apache-2.0"
    for f in proj["license-files"]:
        assert (REPO / f).is_file(), f
    assert not any(c.startswith("License ::") for c in proj["classifiers"])
    for v in ("3.10", "3.11", "3.12"):
        assert f"Programming Language :: Python :: {v}" in proj["classifiers"]
    assert proj["dependencies"] == ["numpy>=1.25"]  # the <2.4 pin only matters for lerobot 0.4.4
    assert "numpy<2.4" in proj["optional-dependencies"]["lerobot"]
    assert set(proj["optional-dependencies"]) == {"sim", "lerobot", "score", "eval", "dev"}
    assert all("/capek" in url for url in proj["urls"].values())
    include = cfg["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    for spike in ("docs/spikes/phase2_score_proto.py", "docs/spikes/phase3_stats.py"):
        assert spike in include and (REPO / spike).is_file()


def test_install_hints_use_the_pypi_form() -> None:
    from capek.hints import install_hint

    assert install_hint("score") == 'pip install "capek-tech[score]"'
    offenders = [
        f"{p.relative_to(REPO)}:{i}"
        for p in SRC.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if "pip install -e" in line or "pip install '." in line
    ]
    assert offenders == []
    users = [p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py") if "install_hint(" in p.read_text()]
    for site in (
        "sim/registry.py",
        "score/reader.py",
        "eval/bc.py",
        "eval/lerobot_adapter.py",
        "export/lerobot_writer.py",
    ):
        assert site in users, site


def test_score_hint_without_pyarrow(tmp_path: Path, monkeypatch, capsys) -> None:
    import json

    from capek import cli

    ds = tmp_path / "ds"
    (ds / "data" / "chunk-000").mkdir(parents=True)
    (ds / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"")
    (ds / "meta").mkdir()
    (ds / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30}))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    assert cli.main(["score", str(ds), "--json-out", str(tmp_path / "r.json")]) == 1
    assert 'pip install "capek-tech[score]"' in capsys.readouterr().err


@needs_repo
def test_placeholder_only_in_allowed_files() -> None:
    owner = _tool("set_github_owner")
    found = owner.occurrences(REPO)
    assert all(owner.is_allowed(f) for f in found), found


@needs_repo
def test_set_github_owner_replaces_everywhere_and_refuses_strays(tmp_path: Path, capsys) -> None:
    owner = _tool("set_github_owner")
    (tmp_path / "pyproject.toml").write_text('Homepage = "https://github.com/GITHUB_OWNER/capek"\n')
    (tmp_path / "docs" / "launch").mkdir(parents=True)
    (tmp_path / "docs" / "launch" / "x-thread.md").write_text("github.com/GITHUB_OWNER/capek GITHUB_OWNER\n")
    (tmp_path / "docs" / "dev").mkdir()
    (tmp_path / "docs" / "dev" / "STATUS.md").write_text("we use a GITHUB_OWNER placeholder\n")  # crew record
    (tmp_path / "src.py").write_text("GITHUB_OWNER = 1\n")  # stray -> refused
    assert owner.main(["acme", "--root", str(tmp_path)]) == 1
    assert "outside the allowed files: src.py" in capsys.readouterr().err
    (tmp_path / "src.py").unlink()
    assert owner.main(["bad name!", "--root", str(tmp_path)]) == 2
    assert owner.main(["acme-labs", "--root", str(tmp_path)]) == 0
    assert "acme-labs/capek" in (tmp_path / "pyproject.toml").read_text()
    assert (tmp_path / "docs/launch/x-thread.md").read_text().count("acme-labs") == 2
    assert "GITHUB_OWNER" in (tmp_path / "docs" / "dev" / "STATUS.md").read_text()
    assert owner.occurrences(tmp_path) == {}


@needs_repo
def test_denylist_check_reports_locations_not_entries(tmp_path: Path, monkeypatch) -> None:
    rc = _tool("release_check")
    (tmp_path / "a.md").write_text("fine\nmentions Secret-Dataset here\n")
    (tmp_path / "b.txt").write_text("nothing\n")
    deny = tmp_path.parent / f"{tmp_path.name}-deny.txt"
    deny.write_text("# comment\nsecret-dataset\n\nother-org\n")
    entries = rc.read_denylist(deny)
    assert entries == ["secret-dataset", "other-org"]
    hits = rc.denylist_hits(entries, tmp_path)
    assert hits == ["entry #1: a.md:2"]
    assert not any("secret" in h.lower() for h in hits)
    monkeypatch.setenv("CAPEK_REDACTIONS", str(deny))
    assert rc.find_denylist(None) == deny
    assert rc.find_denylist(str(tmp_path / "missing.txt")) == deny  # falls through to the env var


@needs_repo
def test_denylist_file_name_hit_does_not_reveal_the_path(tmp_path: Path) -> None:
    rc = _tool("release_check")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "secret-dataset-review.md").write_text("harmless\n")
    hits = rc.denylist_hits(["secret-dataset"], tmp_path)
    assert hits == ["entry #1: a tracked file name (index 0)"]
    assert not any("secret" in h or "notes/" in h for h in hits)
    assert rc.DEFAULT_DENYLISTS == ("~/.capek_redactions.txt", "~/.rrc_redactions.txt")
    assert "/home/claude" not in (TOOLS / "release_check.py").read_text()


@needs_repo
def test_release_check_never_uploads() -> None:
    src = (TOOLS / "release_check.py").read_text()
    calls = re.findall(r"\[[^\]]*\]", src)
    assert not any('"upload"' in c or "'upload'" in c for c in calls)
    assert "gh-action-pypi-publish" not in src and "twine upload" not in src.replace("NEVER uploads", "")


@needs_repo
def test_set_github_owner_check_mode_exits_nonzero_while_placeholder_left(tmp_path: Path, capsys) -> None:
    owner = _tool("set_github_owner")
    (tmp_path / "pyproject.toml").write_text('Homepage = "https://github.com/GITHUB_OWNER/capek"\n')
    assert owner.main(["--check", "--root", str(tmp_path)]) == 1
    assert "still present" in capsys.readouterr().err
    assert owner.main(["acme", "--dry-run", "--root", str(tmp_path)]) == 0
    assert "GITHUB_OWNER" in (tmp_path / "pyproject.toml").read_text()  # dry run changed nothing
    assert owner.main(["acme", "--check", "--root", str(tmp_path)]) == 1  # --check never replaces
    assert owner.main(["acme--x", "--root", str(tmp_path)]) == 2  # GitHub rejects consecutive hyphens
    assert owner.main(["acme", "--root", str(tmp_path)]) == 0
    assert owner.main(["--check", "--root", str(tmp_path)]) == 0
