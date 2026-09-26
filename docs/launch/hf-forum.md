> **DRAFT — for Sebi to edit and post. The crew posts nothing.** Every number comes from STATUS.md. Name no third-party dataset, org or person.

**Where:** discuss.huggingface.co, in the LeRobot category (or the closest one available).
**Title:** Robot Report Card: dataset scoring and honest policy evaluation for LeRobot / SO-101 (open source)

---

Hugging Face's own post on the LeRobot community datasets listed real quality problems: empty task strings,
broken or too-short episodes, inconsistent features. Behind all of them is a simpler question that anyone training an SO-101 policy runs into:
**when a policy performs badly, is it the data or the model?**

**Robot Report Card** (Apache-2.0) is a small CLI that tries to answer that with evidence instead of guesses:

1. **`rrc score`** reads a LeRobot dataset on disk (v3.0, v2.1 or v2.0; parquet only, no videos needed) and scores every
   episode's motion against the rest of the dataset: jitter, jerk, hesitation, dithering, actions pinned at limits,
   commands the arm doesn't follow. It also shows outcome evidence (success column or labels) separately. It needs
   only pyarrow, and it never writes into the dataset.
2. **`rrc eval` / `rrc compare`** run policies in a simulated SO-101 reach task on the same seeds and report success
   rates with Wilson CIs, an exact McNemar test, a CI on the difference, and the smallest difference the test could
   have detected. "No detectable difference" is never reported as "no difference".
3. **`rrc report`** puts the dataset, policy and regression results on one page, with rule-based verdicts worded
   "consistent with", never "caused by", and a fixed "Can't tell" section.

**What we've measured, with caveats:**
- On a public 10-episode SO-101 dataset (not named here): 1 true positive and 1 false positive, a return-to-home
  motion. The false positive is fixed and verified in simulation; the re-check on the real dataset is still pending.
- On a simulated benchmark with known ground truth, junk motion ranks almost perfectly (AUROC ≥ 0.99). At most 2% of
  good episodes are flagged at standard noise.
- In simulation, on one task: two checkpoints trained on the same mixed data, unfiltered vs filtered to quality-ok
  *and* successful demos, scored **0/200 vs 29/200** (p = 3.7e-9). Over 5 training seeds: unfiltered 0–2/200,
  filtered 15–39/200 (every p ≤ 6.1e-5). In our runs, random and high-noise demos and wrong-goal demos hurt the
  policy, but hesitation didn't.

**What it can't tell you:** failed attempts that move normally (only outcome labels show those); problems shared by
every episode (scores are relative); anything about your real robot (policy eval is simulation only, one task).
The thresholds are tuned in simulation.

```bash
pip install "robot-report-card[score]"
rrc score path/to/your_lerobot_dataset
```

Repo, docs and the full list of limits: https://github.com/Sebilopez1/robot-report-card

**I'd love datasets where it's wrong.** If you share an `rrc score` JSON (no video inside) and tell me which flags
look wrong to you, that's the most useful thing you can do for the project. Please share only data you're allowed to share.
