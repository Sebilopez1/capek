> **DRAFT — for Sebi to edit and post. The crew posts nothing.** Every number comes from STATUS.md. Name no third-party dataset, org or person.

**Format:** 6 posts, each under 280 characters. Attach a screenshot of `rrc score` + `rrc compare` output on our own
sim data (not third-party data) to post 1.

---

**1/6**
Training a robot policy with LeRobot and it performs badly? Is it your data or your model?

I built Robot Report Card, a free, open-source CLI that answers with evidence instead of guesses.

**2/6**
`rrc score` reads a LeRobot dataset and flags episodes whose motion looks off (jitter, hesitation, dithering, commands
the arm doesn't follow), with plain-language reasons.

Laptop CPU, no torch needed, never modifies your data.

**3/6**
`rrc compare` tests "is checkpoint B really better than A?" on the same seeds: exact McNemar, a CI on the difference,
and the smallest difference it could detect.

"No detectable difference" never gets reported as "no difference".

**4/6**
In simulation (one task): two checkpoints trained on the same data, unfiltered vs filtered by score + outcome, got 0/200
vs 29/200 successes (p = 3.7e-9). Across 5 training seeds: 0–2 vs 15–39 out of 200.

**5/6**
On a public real SO-101 dataset: 1 true positive, 1 false positive (a return-to-home motion). The false positive is
fixed in sim; the real re-check is pending.

It can't see failed attempts that move normally. Only labels can.

**6/6**
`pip install "robot-report-card[score]"`
Apache-2.0: https://github.com/Sebilopez1/robot-report-card

Found a dataset where it's wrong? Send the `rrc score` JSON. That's the most useful feedback right now.
