> **DRAFT — for Sebi to edit and post. The crew posts nothing.** Every number comes from docs/dev/STATUS.md. Name no third-party dataset, org or person.

**Where:** LeRobot Discord, the channel for community tools / show-and-tell (pick whichever fits the current layout).
**When:** Tue–Thu; stay around to answer questions for the first 48 h.

---

Hi all! I built **Capek**, a free, open-source CLI for people training SO-100/101 policies with LeRobot.
It tries to answer "is it my data or my model?"

- `capek score <dataset>` reads a LeRobot dataset (v3.0 / v2.1 / v2.0) and flags episodes whose motion looks off
  (jitter, hesitation, dithering, commands the arm doesn't follow), with plain-language reasons. It doesn't need lerobot
  or torch, runs on a laptop CPU, and never modifies your dataset.
- `capek eval` / `capek compare` report success rates with confidence intervals and a paired test for "is B really better than A?".
- `capek report` puts it all on one page, including a "what this can't tell you" section.

**Honest results so far:**
- On a public 10-episode SO-101 dataset: v0.1.0 flags only the one shaky episode (confirmed by watching the video).
  An earlier version also flagged a normal return-to-home motion; that false positive is fixed and re-checked on the real data.
- In simulation, on one task: two checkpoints trained on the same data, unfiltered vs filtered, scored 0/200 vs 29/200
  successes (p = 3.7e-9). The margin held across 5 training seeds.

```
pip install "capek-tech[score]"   # https://pypi.org/p/capek-tech
capek score path/to/your_dataset
```

**The ask:** run it on your dataset. If it flags something that looks fine, or misses an obviously bad episode, send me the
`capek score` JSON (it has no video) or open an issue: https://github.com/Sebilopez1/capek
