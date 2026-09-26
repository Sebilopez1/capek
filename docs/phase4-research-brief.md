# Research Brief: Phase 4 Free Release

Researcher, 2026-09-25. **Release-ready only: nothing is published by the crew. Sebi presses publish.** Measured here (Linux, Python 3.11, uv 0.8.17, twine 7.0.0, hatchling 1.32.4) unless marked. **[V]** = checked today; **[M]** = from memory, unverified.

## Recommendations

1. **The name looks free.** `https://pypi.org/pypi/robot-report-card/json` returns **404** (numpy returns 200 as a control), and `pip index versions robot-report-card` finds nothing **[V]**. PyPI can still refuse a name that is too similar to an existing project or was previously deleted **[M]**. Only a real upload settles it, so do the TestPyPI dry run first. TestPyPI is blocked by this proxy (403), so that part is unverified here.
2. **Adopt PEP 639 metadata now.** Use `license = "Apache-2.0"`, `license-files = [LICENSE, NOTICE, so101/LICENSE]`, `hatchling>=1.27`, plus classifiers, keywords and URLs. I built it in a scratch copy and it gives `Metadata-Version 2.5`, `License-Expression: Apache-2.0` and 3 license files in `dist-info/licenses/`. `twine check` PASSED on both artifacts **[V]**. Don't also add a `License ::` classifier (PEP 639 deprecates that alongside an expression) **[M]**.
3. **`pip install robot-report-card` is already light.**
   Wheel 6.0 MB (13 SO-101 STL meshes); base venv **75 MB**, installed in 0.4 s; `[score,sim]` **323 MB**, and from the wheel `rrc score`, `rrc record` and `rrc eval scripted` all worked **[V]**. Keep the extras as they are. Move the `numpy<2.4` pin into the `lerobot` extra, because only lerobot 0.4.4 needs it (research brief G1).
4. **Fix before the first release:**
   - (a) the sdist tests need `docs/spikes/` (**3 failures** from a clean sdist: `test_p2_5_engine` ×2, `test_p3_3_stats`). Ship `docs/spikes/phase2_score_proto.py` and `phase3_stats.py` in the sdist, or skip when they're absent;
   - (b) README relative links (`STATUS.md`, `LICENSE`) break on PyPI, so make them absolute GitHub URLs;
   - (c) the `OWNER` placeholders in the URLs.
5. **Publishing: trusted publishing from GitHub Actions** (OIDC, no stored token) **[V: docs.pypi.org/trusted-publishers]**, triggered by a GitHub release that Sebi creates. Manual `twine upload` with an API token is the fallback. Do the TestPyPI dry run first either way.
6. **CI: two jobs.**
   - **light** (every push, 3.10–3.12 × ubuntu, plus macos-14 arm64; `.[dev,score,sim]`): measured 179 passed / 22 skipped in **25 s** from the sdist, with no torch and no lerobot.
   - **heavy** (Python 3.11 ubuntu; manual, nightly, or the `eval`/`lerobot` paths): torch from the **CPU index first**. The default PyPI torch on Linux is the CUDA build (5.4–10 GB measured in phase 3); the CPU build is about 1.5 GB **[M]**. Set `HF_HUB_OFFLINE=1` everywhere.
7. **Report card = `rrc report`.** It joins existing JSONs (`rrc score`, `rrc eval`, `rrc compare`) into one page. The terminal summary is always printed and Markdown with `--md`. **HTML only via a stdlib template** (cheap, no new dependency). Attribution uses **ordered, evidence-citing rules** that say "consistent with", never "caused by" (§3, superseded by plan D2).
8. **`rrc record --mix`** exposes the bench generators, where `wrong` uses an independent goal. `--policy wrong` switches to the independent goal *before* the first release, and `--wrong-goal mirrored` keeps the old behavior (§4).
9. **Launch** on the LeRobot Discord (invite in lerobot's own source **[V]**) and the HF forum/Hub, with a real-dataset example that names no third-party dataset (§5). HF's own May 2025 post lists community-dataset quality problems, which is our hook **[V]**.

## 1. PyPI packaging (verified builds)

```toml
[build-system]
requires = ["hatchling>=1.27"]            # metadata 2.4 + license-files as globs (hatchling 1.26, 2024-11) [V: changelog]
[project]
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE", "src/robot_report_card/sim/assets/so101/LICENSE"]
keywords = ["robotics", "imitation-learning", "lerobot", "so-101", "dataset-quality", "evaluation"]
classifiers = ["Development Status :: 3 - Alpha", "Intended Audience :: Science/Research",
  "Intended Audience :: Developers", "Operating System :: OS Independent", "Programming Language :: Python :: 3 :: Only",
  "Programming Language :: Python :: 3.10", "Programming Language :: Python :: 3.11", "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Artificial Intelligence"]
[project.urls]
Homepage = "https://github.com/<owner>/robot-report-card"   # Sebi fills in; also Issues / Changelog
[project.optional-dependencies]
lerobot = ["lerobot==0.4.4", "numpy<2.4"]  # move the pin here; base keeps numpy>=1.25
```

- **Vendored assets:**
  - The wheel already ships `sim/assets/so101/LICENSE` and `SOURCE.md` next to the meshes, and NOTICE attributes TheRobotStudio. That meets Apache-2.0 §4 (keep the license and notices) **[M: standard reading]**.
  - With `license-files` the upstream LICENSE also appears in `dist-info/licenses/`, which is where scanners look **[V: built]**.
  - The sdist includes `src/`, `tests/`, README, LICENSE and NOTICE **[V]**.
- **Install hints (QA finding 1):** every user-facing message must say `pip install "robot-report-card[<extra>]"`,
  not the developer form `pip install -e '.[extra]'` (5 sites in `src/`; a test greps for `pip install -e`).
- **Extras (keep):** base = numpy (CLI, record/tag/list logic); `sim` = mujoco; `score` = pyarrow; `eval` = mujoco + torch; `lerobot` = export and the adapter; `dev` = pytest + ruff. README install lines: `pip install "robot-report-card[score]"` for dataset scoring; `[sim,eval]` for policy evaluation.
- **CPU torch guidance** (README + CI): `pip install torch --index-url https://download.pytorch.org/whl/cpu` *before* `pip install "robot-report-card[eval]"`. On macOS, PyPI torch is already CPU/MPS **[M]**. The PyTorch index is blocked by this proxy, so this is unverified here; phase 3 measured 5.4 GB for `.[dev,eval]` with CUDA torch.
- **Build and check** (verified: 10 s; sdist 6.08 MB, wheel 6.05 MB):

```bash
python -m pip install build twine && python -m build && twine check dist/*
python -m venv /tmp/t && /tmp/t/bin/pip install dist/*.whl && /tmp/t/bin/rrc --help      # smoke test
```

- **TestPyPI dry run (Sebi):** (1) create a TestPyPI account and token, then `twine upload -r testpypi dist/*`; (2) `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple robot-report-card[score]`. The extra index is needed because the deps aren't on TestPyPI **[M]**; (3) run the README quickstart; (4) bump the version for any re-upload: PyPI never allows reusing a filename **[M]**.
- **Trusted publishing vs twine:**
  - Trusted publishing (register the repo, workflow and environment as a publisher on PyPI; a workflow with `id-token: write` and `pypa/gh-action-pypi-publish`) avoids long-lived tokens **[V: PyPI docs, packaging guide]**. A "pending publisher" can be registered before the project exists **[M]**.
  - Recommend a `release.yml` triggered on `release: published`, with a `pypi` environment that requires Sebi's approval. That keeps "Sebi presses publish" literal.

## 2. CI (GitHub Actions)

```yaml
name: tests
on: [push, pull_request]
env: { HF_HUB_OFFLINE: "1" }                             # rrc never renders, so no GL setup is needed
jobs:
  light:
    strategy: { matrix: { os: [ubuntu-latest, macos-14], py: ["3.10", "3.11", "3.12"] } }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5                        # caches wheels
      - run: uv venv -p ${{ matrix.py }} && uv pip install -e ".[dev,score,sim]" && uv run ruff check . && uv run pytest -q
  heavy:                                                   # torch + lerobot, CPU only
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv venv -p 3.11 && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      - run: uv pip install -e ".[dev,score,sim,eval,lerobot]" scipy && uv run pytest -q   # ~2.5 min today
```

- Tests already skip cleanly when torch, lerobot or scipy is missing, and gate the full benchmarks behind `RRC_BENCH_FULL=1` **[V: skip reasons listed]**. Add a weekly scheduled job with `RRC_BENCH_FULL=1`.
- **Risk:** in the heavy job the resolver may still pull a CUDA torch if lerobot's torch pin isn't met by the CPU wheel (torch <2.11 required). Pin `torch==2.10.*` in the first step. Unverified here: the PyTorch index is blocked.
- There are no downloaded assets to cache (the MJCF is vendored). Cache uv only. The macOS light job is our only cross-machine check (the phase 3 determinism gap).

## 3. The one-page report card (`rrc report`)

**Inputs:** a dataset score JSON, 1+ eval JSONs, an optional compare JSON, and `--train-seeds N` (default 1).

**Output:** 5 blocks, each with its evidence line: **Data** (flag rate, top reasons, hard count, outcome evidence), **Policy** (k/n, Wilson CI, success definition, eval seeds), **Regression** (Δ, Newcombe CI, McNemar p, MDE, borderline), **Verdict**, and **Can't tell**.

**Attribution rules** (first match wins; the thresholds are proposals, sim-tuned; each output line names its rule):

**Superseded by plan D2 (QA review, 2026-09-25).** The rules originally proposed here over-claimed: R0 called every HARD
episode "corrupt", R1 said "may be uniformly noisy" (uniform junk actually gives few or no flags), R2/R5 assumed the
scored dataset was the training data, R3 treated all motion flags as harmful (hesitation wasn't), and R5 said
"data looks clean". The binding version is in `docs/phase4-plan.md`, section "Report-card rules (D2)":
- typed HARD counts; R0 fires only at HARD ≥ 5% or any non-finite value;
- **linking:** only `bc:` checkpoints whose dataset fingerprint matches the score JSON may be described as trained on
  the scored demos; R5 requires both sides linked;
- R3 cites the measured per-junk-type result (random/high-noise hurt; hesitation didn't);
- R5 says "consistent with training variance or the recipe" and names what the scorer can't see.

**Can't tell (fixed lines, always shown):** uniform junk; wobble / band-limited jitter; smooth failed attempts need
outcome evidence; checkpoint vs recipe (10–15 pts, ≥ 3 seeds); **final-frame success vs LeRobot's any-step rate**;
sim-only eval (`so101_reach`); no cross-dataset comparison; same-machine reproducibility; sim-tuned thresholds.

**Forbidden phrases:** "caused by", "because of", "data looks clean", "data is good", "clean data", "proves",
"real robot will", "reproducible across", any recipe verdict with < 3 seeds, any ACT/LeRobot quality adjective.

**Formats:** terminal text always; `--md FILE` (renders on GitHub and HF cards); `--html FILE`, one self-contained file from a stdlib `string.Template` with inline CSS and no JS (about 100 lines, cheap). Skip charts except ASCII/inline-SVG bars. Put the definitions (success, n, seeds, versions) in a footer. Keep JSON as the source of truth (`report.json`), rendered by all three.

## 4. `rrc record --mix` (design)

- `rrc record --mix clean:60,noise025:10,random:10,hesitation:5,wrong:5 --seed S --out DIR [--noise-profile standard|low] [--append]`.
  - `--mix` is mutually exclusive with `--policy/--episodes/--noise`.
  - Group k records with seed S + k.
  - It is implemented by calling the same `bench.generators.GroupPolicy` and the table parser as `python -m robot_report_card.bench record`, which stays as a thin alias.
  - Test: the two commands produce **byte-identical sessions** for the same args.
- Group names are the bench's (`clean`, `nearmiss`, `noise005/010/025`, `hesitation`, `wobble`, `stall`, `wrong`, `random`, `return_home`). `rrc record --list-groups` prints each with a one-line description and its class (clean / motion junk / outcome-only).
- **`--policy wrong`:** switch to the independent goal now. v0.1 isn't published yet, so no external users break. Add `--wrong-goal mirrored` for the exact phase 1 behavior, and record `policy_params.goal = "independent" | "mirrored"` so old and new sessions stay distinguishable.
  - Old sessions stay readable (the params are free-form).
  - Existing tests pinning the mirrored `wrong` add `--wrong-goal mirrored`.
- Then point the README demo at `rrc record --mix` (the phase 3 backlog item).

## 5. Distribution channels and launch

- **Channels** (reach/fit notes are **[M]** unless marked):
  - **LeRobot Discord** (`discord.com/invite/s3KuuzsPFb`, linked from lerobot's own error messages **[V]**): the core SO-100/101 audience, and the channel to lead with.
  - **HF Hub/forums:** the "LeRobot Community Datasets" blog (May 2025) names empty task strings, too-short or broken episodes and inconsistent features as real problems **[V]**. The LeRobot Worldwide Hackathon org has many SO-101 datasets **[V: org page exists]**.
  - Post on `discuss.huggingface.co` (LeRobot category **[M]**). With the owners' permission, a report on one public dataset makes a strong demo.
  - **r/robotics, r/MachineLearning ("[P]" posts), X/LinkedIn, and robotics-lab mailing lists** **[M]**: a secondary reach. Link back to the GitHub repo.
- **Etiquette:** never publicly name another person's flagged episodes without asking. Use our sim datasets or Sebi's own recordings in public examples.
  This applies to **the repo itself**, not only the posts: publishing the repo publishes STATUS and its git history.
  The DoD 8 dataset name is redacted in tracked files ("a public SO-101 dataset, name withheld"); it remains in git
  history from 263fa31 onwards, and whether to squash history is Sebi's decision (plan, Needs Sebi's call 4).
- **README hero:** one sentence ("Is it your data or your model? A one-page report card for LeRobot/SO-101 imitation learning"); a real terminal block of `rrc score` plus `rrc compare` output; a 3-command quickstart (`pip install "robot-report-card[score]"`, then `rrc score <dataset>`); a "What it can't tell you" box; the license and a citation line.
- **Launch post** (about 150 words): the problem in one line; a concrete result (on a public SO-101 dataset, not named: 1 true positive and 1 false positive (return-to-home); the false positive is fixed **and verified in simulation**, the re-check on the real dataset is pending; and in simulation, one task: two checkpoints trained on the same data, unfiltered vs filtered, scored 0/200 vs 29/200, p = 3.7e-9 — a **checkpoint pair**; the data claim rests on QA's 5-training-seed result, A 0–2/200 vs B 15–39/200, p ≤ 6.1e-5); the install line; honest limits; an explicit ask: "send us a dataset (or the `rrc score` JSON) where it's wrong". Post Tuesday–Thursday, and reply fast in the first 48 h **[M]**.

## Open uncertainties

- Name availability beyond the JSON API (similar-name blocks), TestPyPI, the CPU-torch CI install and the macOS CI job are all **unverified here** (proxy).
- Python 3.12/3.13 aren't tested by us yet; the classifiers claim 3.10–3.12 only once CI passes.
- The attribution thresholds (10% / 5%) are proposals from sim, not tuned on real data.

## QA Review (phase 4)

QA, 2026-09-25. **Verdict: APPROVED WITH CHANGES.** Every packaging number reproduces. The report-card rules need tightening before they're coded, because R0, R1 and R5 over-claim as written. The launch facts need two corrections.

**Packaging, re-run by QA** (scratch copy with the §1 metadata; uv build; `uvx twine`; clean uv venvs):
- Build in 2 s: wheel 6.06 MB, sdist 6.08 MB. `twine check` PASSED on both.
- `Metadata-Version: 2.5`, `License-Expression: Apache-2.0`. The three license files are in `dist-info/licenses/` (the SO-101 one is kept under its `src/...` path).
- **Base wheel venv: 74 MB**, 0.3 s. `rrc --help` works. Without the extras, `rrc record` / `rrc score` print a hint.
- **`[score,sim]`: 322 MB.**
- Moving `numpy<2.4` into the `lerobot` extra works. The base resolves numpy 2.4.6.
- **Light suite from the unpacked sdist** (installed sdist, spikes shipped) passes on all three versions:

  | Python | numpy | Result | Time |
  |---|---|---|---|
  | 3.10.20 | 2.2.6 | 182 passed / 22 skipped | 34 s |
  | 3.11.15 | 2.4.6 | 182 passed / 22 skipped | 26 s |
  | 3.12.3 | 2.5.3 | 182 passed / 22 skipped | 25 s |

  So the 3.10–3.12 classifiers are honest for the light profile. The heavy profile is verified on 3.11 only.

**Findings:**
1. **Install hints are for developers, not PyPI users.** All five say `pip install -e '.[extra]'`: `sim/registry.py:16`, `score/reader.py:78`, `eval/bc.py:45`, `eval/lerobot_adapter.py:65` and `export/lerobot_writer.py:54`. A pip user needs `pip install "robot-report-card[extra]"`. `tests/test_p1_4_export_without_lerobot.py` asserts the old text.
2. **The rules over-claim in places (§3):**
   - **R0:** fires on *any* HARD episode, and calls every such episode "corrupt/unreadable". But HARD also covers ≥ 20% saturation, too-short episodes and frozen joints. A single too-short episode out of 500 would override a significant regression result.
   - **R1:** says "may be uniformly noisy". Phase 2 measured the opposite: uniform junk gives **few or no flags** (an all-σ 0.25 dataset: 0/40 flagged). A ≥ 50% flag rate means mostly-heterogeneous junk.
   - **R2:** "X% of *training* demos failed" assumes the scored dataset is the one the evaluated policy was trained on. Nothing in the rule checks that.
   - **R3:** "consistent with a data problem" for any motion flag. But phase 3 measured that hesitation (flagged by the scorer) **did not** hurt the policy.
   - **R5:** "data looks clean → training/recipe". That breaks §3's own "must NOT claim unflagged data is good", and it assumes both checkpoints were trained on the scored dataset.
3. **The "Can't tell" gaps are only partly covered.** The must-not list covers uniform junk, smooth failed attempts and recipe vs. checkpoint. **Final-frame vs. any-step success isn't mentioned in §3 at all.** Nor are wobble (band-limited jitter), borderline compares or sim-only eval. These must be fixed lines in the block (see the plan review, D2).
4. **Launch facts:**
   - The DoD 8 false positive is "since fixed" **in sim only**. The re-score on the real dataset is still open (STATUS, Needs Sebi's call 1).
   - "0/200 → 29/200, p = 3.7e-9" is **one checkpoint pair** at training seed 0. As a data claim, it's a recipe claim from one seed, which §3 forbids. QA measured the margin over 5 training seeds in P3-8 (A 0–2/200, B 15–39/200, p ≤ 6.1e-5). The drafts must present it as a checkpoint pair and cite the multi-seed result for the data claim.
5. **Third-party data:** STATUS.md names the DoD 8 Hub dataset and describes one of its episodes as "shaky". It's the only file that does, but it's also in git history (263fa31 onwards). Publishing the repo publishes it. §5's etiquette ("never publicly name another person's flagged episodes without asking") therefore applies to the repo itself, not just the posts.
6. **The `--mix` / `wrong` switch is low risk (§4).**
   - The bench already uses the independent goal (9e8b8c1), so the bench bars, P3-8 and the P3-6 tests don't change.
   - The only test that uses `make_policy("wrong")` (`test_p1_2_record.py:118`) asserts that it *fails*, which still holds with an independent goal.
   - The `rrc record --help` text ("heads to the mirrored pose") and the phase 1 README/STATUS wording need updating.

**Required changes:**
- R1: fix the install hints (finding 1), and add a test that no user-facing message contains `pip install -e`.
- R2: adopt the rule rewrite and the fixed Can't-tell lines in the plan's QA review (D2).
- R3: correct the launch facts (finding 4).
- R4: handle the third-party dataset: redact it in STATUS, and give Sebi the history decision (finding 5; plan D6).

### Changes applied (PM, 2026-09-25; Researcher idle, PM owns this round)
- R1: §1 states the PyPI-form install hints and the `pip install -e` grep test.
- R2: the §3 rules table is replaced by a pointer to plan D2 with a summary of why; fixed Can't-tell lines and forbidden phrases added (including final-frame vs any-step, wobble, sim-only).
- R3: the §5 launch facts are corrected: the fix is verified in sim only, and the result is a checkpoint pair, with the 5-seed result backing the data claim.
- R4: §5 etiquette now covers the repo and history. The dataset name is redacted in tracked files, and the history decision goes to Sebi.
