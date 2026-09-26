# Milestone: Phase 2 — Dataset Scoring

PM, 2026-09-24. Status: **built and approved by QA. Two things are still open: the real-data check (DoD 8) and Sebi's go-ahead for phase 3.**
Details: `STATUS.md` (Session 2), `docs/phase2-plan.md` rev 3, `docs/reviews/phase2-code-review.md`, `docs/reviews/phase2-bench-review.md`.

## What was built

`rrc score <dataset>` reads any LeRobot dataset on disk and prints one row per episode. Each row shows a quality
verdict (ok / FLAG / HARD), a score, the outcome if the dataset has evidence for it, and plain-language reasons
("looks jittery: action chatter 35× dataset median"). It also writes a JSON report.
- It reads the v3.0, v2.1 and v2.0 layouts with pyarrow alone. No lerobot, torch or simulator is needed
  (`pip install -e ".[dev,score]"`). It never changes the dataset.
- **Motion quality and outcome are kept apart.** Quality looks only at how the arm moved. Outcome comes only from a human
  label or a success column. Otherwise it says `unknown`. Every report says, in plain words, what motion can't detect.
- **Supporting pieces:**
  - `rrc record --append`, so one dataset can mix good and junk episodes;
  - QA's benchmark builder and independent pass/fail judge;
  - "real-like" test copies (degrees, renamed joints, variable lengths, v2.0/v2.1 layouts, missing videos, corrupt files);
  - the phase 1 backlog fixes.

**Proof:** 153 tests pass. QA built the ground truth and the judge, so the scorer's author never saw the answers.
Every bar passes on three seed sets, including a private one QA kept out of the repo:

| Bar | Target | dev | held-out | private |
|---|---|---|---|---|
| Good vs junk-motion ranking (AUROC) | ≥ 0.95 | 0.999 | 1.000 | 1.000 |
| Flagged episodes that really are junk (precision) | ≥ 0.90 | 1.00 | 1.00 | 1.00 |
| Junk episodes caught (recall) | ≥ 0.85 | 0.975 | 0.975 | 0.975 |
| Good episodes wrongly flagged (mixed / all-good set) | ≤ 5% | 0% / 1% | 0% / 0% | 0% / 2% |
| Hesitation ranking (AUROC) | ≥ 0.90 | 0.995 | 1.000 | 1.000 |

**Reviewed by:** QA (all Coder work), Researcher and PM (QA's benchmark and judge), QA (PM's docs).

## Deviations from the plan, and why

1. **"Frozen joint" now means a dead motor:** the reading is flat while its command moves. The first version also flagged
   joints that were simply unused in an episode (e.g. the gripper in a reach), which is normal on a real SO-101.
2. **Episodes under 17 frames (0.57 s) are HARD "too short".** They're too short for the hesitation check to mean anything.
3. **The benchmark's "wrong goal" episodes were redesigned.** The first version pointed at a mirrored target and ran into joint
   limits, so it got "caught" for the wrong reason. Now it reaches smoothly to a random wrong target and is
   correctly *not* flagged, which matches what we promise.
4. Smaller rulings (append creates a new session, a warning under 10 episodes, and others) are in STATUS, all QA-approved.

## Known gaps (stated in the README and in every report)

- **Smooth failed attempts** (wrong target, stopping early, a near-miss) look normal. Only a label or success column shows them.
- **Uniform junk:** if every episode has the same problem (e.g. a jittery leader arm), few or none are flagged,
  because scores are relative to the dataset. The fix is a reference check across datasets (phase 3 backlog).
- **Slow wobble** ranks somewhat worse than clean episodes (AUROC ~0.82) but is not flagged.
- **Watch item:** very short reaches are the likeliest false flag (2% of a wider test set, "dithering commands").

## Decisions for Sebi

**(a) Real-data check (DoD 8): DONE 2026-09-24.** 2/10 flagged on a real SO-101 dataset: ep 9 is really shaky (true positive), ep 7 is a normal return-to-home task (false positive, backlog R-real-1), and ep 3 ok looks decent. Original instructions: In the environment from the Quickstart:
```bash
cd ~/code/robot-report-card && git pull ../robot-report-card.bundle main && pip install -e ".[dev,score]"
hf download lerobot/svla_so101_pickplace --repo-type dataset --include "meta/*" "data/*" --local-dir ~/rrc-real/so101
rrc score ~/rrc-real/so101 --json-out ~/rrc-real/so101.rrc_score.json
```
The dataset name is from memory and may not exist. Any small SO-100/101 dataset works, and your own recordings are even better.
If `hf` isn't found, run `pip install -U huggingface_hub`. If it says a key is missing, add
`--state-key` / `--action-key` with the column names from `meta/info.json`. Send us `meta/info.json`, the JSON
report, and a sentence on whether the flagged episodes look wrong to you when you watch them. We will not retune on it in this phase.

**(b) Defaults we applied (say if you want them changed):**
- pyarrow is an optional install (`score` extra), so record/tag stay light;
- `rrc score` never modifies the dataset (the JSON goes where you run it);
- episode length is reported but not part of the score until we validate it on variable-length data.

**(c) Approve phase 3: policy evaluation** (success rate with confidence intervals from repeatable sim rollouts,
plus the regression check). The dataset-level reference check for uniform junk goes into phase 3's backlog.

## Next step

**The crew waits for Sebi:** the DoD 8 results, and a go/no-go on phase 3. No phase 3 work starts until then.
