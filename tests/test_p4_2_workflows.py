"""P4-2: CI + trusted-publishing workflow files (static checks; they have not run on GitHub yet)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"
pytestmark = pytest.mark.skipif(not WF.is_dir(), reason=".github/ is not in the sdist")


def _load(name: str) -> dict:
    doc = yaml.safe_load((WF / name).read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # PyYAML reads the key `on` as boolean True
    return doc


def test_release_uses_trusted_publishing_bound_to_the_pypi_environment() -> None:
    wf = _load("release.yml")
    assert wf["on"]["release"]["types"] == ["published"] and "workflow_dispatch" in wf["on"]
    assert wf["permissions"] == {"contents": "read"}
    jobs = wf["jobs"]
    pypi = jobs["pypi"]
    assert pypi["environment"]["name"] == "pypi" and pypi["if"] == "github.event_name == 'release'"
    assert pypi["permissions"] == {"id-token": "write"} and pypi["needs"] == "build"
    assert any(s.get("uses", "").startswith("pypa/gh-action-pypi-publish@") for s in pypi["steps"])
    test = jobs["testpypi"]
    assert test["if"] == "github.event_name == 'workflow_dispatch'" and test["environment"]["name"] == "testpypi"
    publish_steps = [s for s in test["steps"] if s.get("uses", "").startswith("pypa/gh-action-pypi-publish@")]
    assert publish_steps[0]["with"]["repository-url"] == "https://test.pypi.org/legacy/"
    for name, job in jobs.items():  # id-token only where something is published
        assert ("id-token" in job.get("permissions", {})) == (name in ("pypi", "testpypi")), name


def test_tests_workflow_matrix_and_cpu_torch() -> None:
    wf = _load("tests.yml")
    assert wf["permissions"] == {"contents": "read"} and wf["env"]["HF_HUB_OFFLINE"] == "1"
    light = wf["jobs"]["light"]
    assert light["strategy"]["matrix"] == {"os": ["ubuntu-latest", "macos-14"], "python": ["3.10", "3.11", "3.12"]}
    assert any('".[dev,score,sim]"' in s.get("run", "") for s in light["steps"])
    heavy = wf["jobs"]["heavy"]
    runs = "\n".join(s.get("run", "") for s in heavy["steps"])
    assert '"torch==2.10.*" "torchvision==0.25.*" --index-url https://download.pytorch.org/whl/cpu' in runs
    assert runs.index("download.pytorch.org") < runs.index(".[dev,score,sim,eval,lerobot]")
    assert "schedule" in wf["on"] and any(s.get("env", {}).get("RRC_BENCH_FULL") == "1" for s in heavy["steps"])


@pytest.mark.parametrize("name", ["tests.yml", "release.yml"])
def test_no_secrets_and_actions_pinned_to_major_versions(name: str) -> None:
    text = (WF / name).read_text()
    assert "secrets." not in text and "password" not in text.lower() and not re.search(r"(?<!id-)token:", text)
    assert "has not yet run on GitHub" in text
    for ref in re.findall(r"uses:\s*(\S+)", text):
        assert re.search(r"@(v\d+|release/v\d+)$", ref), ref


def test_release_build_fails_while_the_placeholder_is_present() -> None:
    """QA phase 4 R2: PyPI never allows re-uploading a version, so the build job refuses to run with the placeholder."""
    wf = _load("release.yml")
    steps = wf["jobs"]["build"]["steps"]
    runs = [s.get("run", "") for s in steps]
    gate = next(i for i, r in enumerate(runs) if "tools/set_github_owner.py --check" in r)
    build = next(i for i, r in enumerate(runs) if "uv build" in r)
    assert gate < build
    for job in ("pypi", "testpypi"):
        assert wf["jobs"][job]["needs"] == "build"
