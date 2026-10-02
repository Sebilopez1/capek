# Milestone: Phase 3 — Policy Evaluation

PM, 2026-09-25. Status: **built and approved by QA. Two things are still open: the Mac milestone check and Sebi's go-ahead for phase 4.**
Details: `STATUS.md` (Session 3), `docs/phase3-plan.md` rev 2, `docs/reviews/phase3-code-review.md`, `docs/reviews/phase3-qa-work-review.md`.

## What was built

Three commands that answer "how good is this policy?" and "is B really better than A?" honestly:
- `rrc eval POLICY`: 200 repeatable rollouts in the SO-101 reach sim. It reports the success rate with a 95% confidence interval and states the success definition (the arm must be within 2 cm of the target on the final frame).
- `rrc compare A B`: runs both policies on the *same* 200 starting positions. It reports whether the difference is
  real (exact McNemar test), how big it is (with a CI), and the smallest difference this test could have detected.
  A non-result reads "No detectable difference", never "no difference".
- `rrc train-bc DATASET`: trains a small policy on the CPU in about 10 s. It can keep only episodes that `rrc score` rates ok and that succeeded.
- LeRobot checkpoints plug in as `lerobot:<dir>` through a small adapter, tested on a tiny ACT. That's a smoke test only.
- Also fixed: the phase 2 real-data false flag on return-to-home tasks (DoD 8 ep 7), and the `length_z` blow-up.

**The headline result:** we trained two policies on the same mixed dataset (60 good demos, 30 junk). The one trained on all
episodes succeeded 0/200. The one trained only on episodes the scorer rated ok **and** that succeeded reached 29/200
(+14.5 points, CI +9.9 to +20.0, p = 3.7e-9). The worst p over 5 training seeds is 6.1e-5. This is the report card's core
story in numbers: better data, measurably better policy.

**Proof:**
- 234 tests pass in about 2 min 20 s.
- QA checked every statistic against scipy and exact enumeration: CI coverage 0.95 on average and never below 0.91, and power verified by simulation. The Researcher planted 11 formula bugs and the tests caught all 11.
- All phase 2 scoring bars still pass on three seed sets, including QA's private one. Return-to-home episodes are now flagged 0/10 everywhere.
- The README demo was run word for word in a clean install and takes about 50 s.

**Reviewed by:** QA (all Coder work and the README), the Coder (QA's benchmark and end-to-end test), and the Researcher (QA's statistics tests).

## Deviations from the plan, and why

1. **The demo records with `python -m robot_report_card.bench record`** instead of `rrc record`. Only the bench tool has
   hesitation episodes and the fixed "wrong goal" policy. It's a supported, tested command, and folding it into `rrc record` is backlog.
2. **The regression fixture for the return-to-home fix is a hand-picked seed set** where the old formula visibly fails.
   It's labelled as such. On fresh seeds the old bug hit about 2–3 in 10 return-home episodes, and the new formula hits 0 in 60.
3. **`compare` adds a "borderline" sentence** when the exact test and the CI disagree right at the 5% line (158 of the 4,223 tables QA checked), so the output never contradicts itself silently.

## Open risks and known gaps

- **Same-machine only:** results are identical on reruns on one machine. A Mac vs Linux run is a new sample, not a replay.
- **Final-frame success** is stricter than LeRobot's "any step" count. We print both.
- **One checkpoint ≠ one recipe:** retraining the same recipe moved success by 10–15 points. Claims about data or settings need ≥ 3 training seeds per side.
- **ACT is a smoke test only** (0/5 on the tiny fixture). Real Hub checkpoints are untested (the Hub is blocked here).
- **Install size on Linux:** 5.4–10 GB because of CUDA torch. The CPU-only wheel is documented but untested here.

## Decisions for Sebi

**(a) Mac milestone check (about 5 minutes).**
```bash
cd ~/code/robot-report-card && git pull ../robot-report-card.bundle main
pip install -e ".[dev,eval,score,lerobot]"
# then run the README "Evaluate a policy" demo block (bench record → export → score → train-bc ×2 → eval → compare)
rrc score ~/rrc-real/so101 --json-out ~/rrc-real/so101.p3.rrc_score.json
```
Send us:
- the `rrc compare` output (on Linux: A 0/200, B 29/200; the Mac may differ slightly and needn't match bitwise);
- the demo's wall time and install size;
- whether ep 7 in the re-score is now `ok` and ep 9 is still flagged.

**(b) Defaults we applied (say if you want them changed):**
- n = 200 rollouts per policy, with no early stopping;
- success = final frame;
- torch lives in the optional `eval` extra;
- the tiny ACT test checkpoint is trained inside the test instead of checked in.

**(c) Approve phase 4: free release, community distribution** (step 4 of the build sequence). The first prerequisites:
1. **A real Hub checkpoint test** of the LeRobot adapter (state-only, SO-101; run on your Mac, where the Hub works).
2. **`rrc record --mix`** (bench generators and a fixed `wrong` policy in the main CLI), with the README demo moved onto it.
3. **PyPI packaging:** a name check, `pip install robot-report-card`, CPU-torch install guidance, a version and changelog.
4. **A failure-attribution one-pager** that ties `rrc score` and `rrc eval` into the single report the product promises.
5. **Date:** the 6-week product-direction lock ends **Oct 23, 2026**.

## Next step

**The crew waits for Sebi:** the Mac check results and a go/no-go on phase 4. No phase 4 work starts until then.
