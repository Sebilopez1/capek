# Milestone: Phase 4 — Free Release (release-ready)

PM, 2026-09-25. Status: **release-ready and approved by QA. Nothing has been published, uploaded or posted. That's your call.**
Details: `STATUS.md` (Session 4), `docs/phase4-plan.md` rev 2, `docs/reviews/phase4-code-review.md`, `RELEASE.md`.

## What was built

Robot Report Card 0.1.0, packaged so you can publish it in under an hour by following `RELEASE.md`:
- **A clean package:** a 6.1 MB wheel, Apache-2.0 metadata that passes `twine check`, and a 74 MB base install. Extras
  add scoring (`[score]`), simulation (`[sim]`), policy eval (`[eval]`) and LeRobot export (`[lerobot]`). Every error
  message tells users the right `pip install "robot-report-card[...]"` line.
- **`rrc report`:** the one-page report card the product promises, with Data, Policy, Regression, Verdict and **Can't
  tell** blocks. It prints in the terminal and can also write Markdown, HTML and JSON. Verdicts come from fixed rules
  that cite their evidence and say "consistent with", never "caused by". A checkpoint trained on a filtered subset is
  described as such and never gets a clean bill of health.
- **`rrc record --mix`:** one command records a labelled mix of good and junk demos, so the README demo no longer needs a
  developer tool. `--policy wrong` now reaches a realistic wrong goal.
- **Release tooling:**
  - `tools/set_github_owner.py` fills in your GitHub name everywhere at once.
  - `tools/release_check.py` builds and tests everything locally and never uploads.
  - The GitHub workflows run tests on Linux and macOS and publish to PyPI only after you approve it (trusted publishing, no API tokens).
- **Docs for strangers:** a PyPI-ready README with honest caveats, CHANGELOG, CONTRIBUTING, SECURITY (loading
  checkpoints safely), and `RELEASE.md`, your step-by-step publish checklist.
- **Launch drafts** for the LeRobot Discord, the HF forum and X (`docs/launch/`). They name no third-party datasets.

**Proof:** 306 tests pass in about 2.4 min. QA rehearsed the release from a fresh clone: 9 of 9 checks pass. QA also
ran the README demo word for word from the built wheel and got the same result as phase 3 (0/200 vs 29/200 successes,
p = 3.7e-9), and the 5-training-seed margin reproduced (0–2 vs 15–39 out of 200). The report-card rules test fails if
any single rule is disabled.

**Reviewed by:** QA reviewed all Coder work and the PM docs, and mutation-checked its own rules test. The Coder reviewed QA's test interface through the fix tests.

## Deviations from the plan, and why

1. **Linking had to respect the training filter.** The first report card said a filtered checkpoint was "trained on" the
   whole scored dataset, including failed demos it never saw. QA caught it, and it's fixed and tested.
2. **The release workflow refuses to publish while the GitHub-owner placeholder is still there.** PyPI never lets you
   re-upload a version, so a forgotten step would have been permanent.
3. **One scoring fix stayed deferred** (R-real-3, a length bias in one smoothness signal). QA's candidate fix changed
   nothing on our data, so landing it wouldn't have helped.

## Open risks and known gaps

- **CI and the publish workflow have never run on GitHub.** We had no access from here. Your first push is the real test,
  which is why RELEASE.md says to push to a private repo first.
- **Untested here:** the TestPyPI upload, the CPU-only torch install tip and the macOS CI job (all blocked by our network).
  The PyPI name looked free, but only an upload confirms it.
- **The git history still names the third-party dataset** in 2 commits; the current files don't.
- Everything from phases 2–3 still applies: eval runs in simulation on one task, scores are relative to one dataset,
  thresholds are sim-tuned, and results are reproducible on one machine only.

## Decisions for Sebi

**(a) Your GitHub username.** Run `python tools/set_github_owner.py <owner>` (RELEASE.md step 3). It fills in every link at once.

**(b) PyPI name `robot-report-card`: go / no-go.** It looked free when we checked. The TestPyPI dry run (step 8)
settles it before anything is permanent.

**(c) Git history: squash or keep?** Earlier commits contain the name of the public dataset we checked in phase 2 and
describe one of its episodes. **We recommend squashing** to a single fresh commit for the public repo and keeping your
full history private. The commands are in RELEASE.md step 2.

**(d) The pending phase 3 Mac check** (`docs/milestones/phase3-policy-evaluation.md`), ideally together with
`python tools/release_check.py` on the Mac before you publish.

**(e) The publish itself.** Follow `RELEASE.md` from top to bottom: private repo → CI → environments → trusted
publishers → TestPyPI dry run → tag v0.1.0 → approve the `pypi` deployment. About 30–45 minutes.

**(f) Posting the launch drafts** (`docs/launch/`). Edit them freely. Post Tue–Thu and plan to answer questions for 48 h.

**(g) Approve phase 5: the hosted paid tier** (step 5 of the build sequence; free for .edu). Note:
- The 6-week product-direction lock ends **Oct 23, 2026**.
- Phase 5 is the **first phase that needs real users and money decisions**: pricing, hosting costs, accounts and data
  handling. The launch feedback is exactly the evidence for those decisions, so it's a good point to publish first,
  collect feedback for a couple of weeks, and then decide on phase 5's scope.

## Next step

**The crew waits for you:** the owner name, the go/no-go on the name and the history, and then your publish.
No phase 5 work starts until you approve it.
