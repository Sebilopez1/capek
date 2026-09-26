# Releasing robot-report-card (checklist for Sebi)

The crew built and rehearsed everything below **except the steps that need your accounts**. Nothing has been uploaded,
published or posted. Work through the steps in order, and don't skip the TestPyPI dry run: **PyPI never lets you
re-upload the same version**, so a mistake in 0.1.0 can only be fixed with a 0.1.1.

You need: a GitHub account, and accounts with 2FA on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
(they're separate sites). Budget about 30–45 minutes.

## 1. Update your checkout

```bash
cd ~/code/robot-report-card && git pull ../robot-report-card.bundle main
```

## 2. Decide: keep the git history, or squash it

The history contains the name of the third-party dataset we checked in phase 2 (commit 263fa31 onwards). The current
files don't. Pick one:

- **Keep history:** do nothing; the name stays visible to anyone reading old commits.
- **Squash (recommended if you're unsure):** publish a fresh single-commit copy and keep your full history private:
  ```bash
  git clone ~/code/robot-report-card ~/code/rrc-public && cd ~/code/rrc-public
  git checkout --orphan release && git commit -q -m "Robot Report Card 0.1.0"
  git branch -D main && git branch -m main && git remote remove origin
  ```
  Do all the remaining steps in `~/code/rrc-public`.

## 3. Set the GitHub owner

The repo contains one placeholder for your GitHub user or org name (in `pyproject.toml`, the README and the other
release docs). Replace it everywhere at once, check it, and commit:

```bash
python tools/set_github_owner.py <your-github-user-or-org>
python tools/set_github_owner.py <your-github-user-or-org> --check     # should report nothing left
git commit -qam "Set GitHub owner"
```

## 4. Run the release check

It builds the sdist and wheel, runs `twine check`, installs the wheel into clean venvs, runs the Quickstart and the
light test suite from the sdist, and checks the placeholder and the redaction denylist. It **never uploads**.

1. Create `~/.rrc_redactions.txt` with the org name and the dataset name of the phase 2 real-data check, one per
   line. You know them; they're deliberately not written here.
2. Then run:
   ```bash
   pip install build twine          # uv is used instead if it's installed
   python tools/release_check.py    # about 2–4 min; every row must say PASS
   ```
3. If you haven't yet, also run the phase 3 Mac check from `docs/milestones/phase3-policy-evaluation.md`.

## 5. Create the GitHub repo and push

1. On GitHub, create an **empty** repository named exactly `robot-report-card`, with no README or license. Making it
   **private** first is a good idea: it lets you watch the first CI run before anyone sees it.
2. Push:
   ```bash
   git remote add origin https://github.com/<owner>/robot-report-card.git
   git push -u origin main
   ```
3. Open the **Actions** tab. `tests` runs for the first time ever: a light matrix on Ubuntu and macOS, plus a heavy
   CPU-torch job. If something fails, send the crew the log before releasing.

## 6. Create the two GitHub environments

Go to repository **Settings → Environments → New environment**:
- `pypi`: under **Required reviewers**, add **yourself**. This is the manual "press publish" gate.
  Optionally, limit deployment branches and tags to `v*` tags.
- `testpypi`: no reviewers are needed.

The names must match exactly: `.github/workflows/release.yml` refers to them.

## 7. Register the pending publishers (trusted publishing, no API tokens)

On **PyPI → Your account → Publishing → Add a new pending publisher → GitHub**:

| Field | Value |
|---|---|
| PyPI project name | `robot-report-card` |
| Owner | your GitHub user or org |
| Repository name | `robot-report-card` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Do the same on **TestPyPI** with the environment name **`testpypi`**. If PyPI says the name isn't available, stop and
tell the crew. The name looked free (404) when we checked, but only this step settles it.

## 8. TestPyPI dry run

1. On GitHub, go to **Actions → release → Run workflow** (on `main`). This builds the package and publishes it to TestPyPI only.
2. Then test the upload in a fresh venv:
   ```bash
   python -m venv /tmp/rrc-test && . /tmp/rrc-test/bin/activate
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "robot-report-card[score,sim]"
   rrc --version && rrc record --policy scripted --episodes 3 --out /tmp/rrc-demo -q && rrc list /tmp/rrc-demo
   ```
3. Check that the project page at https://test.pypi.org/p/robot-report-card renders the README and that its links
   point to your repo.
4. A second dry run of the same version fails with "file already exists". That's expected: TestPyPI doesn't reuse versions either.

## 9. Tag and publish the release

1. In `CHANGELOG.md`, replace `unreleased` with today's date. Commit and push.
2. Tag the release:
   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```
3. On GitHub, go to **Releases → Draft a new release**, choose tag `v0.1.0`, title it "0.1.0", paste the 0.1.0
   section of `CHANGELOG.md`, and click **Publish release**.
4. Under **Actions → release**, the `publish to PyPI` job waits for you. Click **Review deployments → pypi → Approve**.
   Only then does it upload.
5. Verify in a fresh venv: `pip install "robot-report-card[score]"`, then `rrc --version` should print 0.1.0. Check
   https://pypi.org/p/robot-report-card.

**Fallback** (if trusted publishing fails): create a PyPI API token scoped to this project, then run
`python -m build && twine upload dist/*`. Delete the token afterwards.

**If 0.1.0 turns out broken:** you can't overwrite it. "Yank" it on PyPI (project → Manage → Releases), fix the
problem, bump the version in `src/robot_report_card/__init__.py` and in CHANGELOG, and release 0.1.1.

## 10. After publishing

- Make the repository public if it was private.
- Launch drafts are in `docs/launch/` (Discord, HF forum, X). Edit them freely and post them yourself. They name no
  third-party datasets; please keep it that way.
