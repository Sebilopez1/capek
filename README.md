# Robot Report Card

**Is it your data or your model?** Dataset scoring and honest policy evaluation for imitation learning with
[LeRobot](https://github.com/huggingface/lerobot) and low-cost arms like the SO-100 / SO-101.

You record demos, train a policy, and it performs badly. Robot Report Card turns "eyeball the rollouts and guess" into
evidence: which episodes in your dataset look like junk and why, a success rate with a confidence interval, and whether
checkpoint B really beats checkpoint A. It then puts all of that on a one-page report card that says what the evidence
is *consistent with*, and what it can't tell.

```bash
pip install "robot-report-card[score]"
rrc score path/to/lerobot_dataset
```

```text
90 episodes, 8100 frames, 30 fps, LeRobot v3.0, keys observation.state / action
flagged: 25/90 (27.8%): 25 by motion score > 3.5, 0 with hard flags
top reasons: action_tv_ratio 20, action_hf_energy 20, track_err 20, idle_frac 5
Motion quality can't detect failed attempts that move normally (early stop, wrong goal). Only outcome evidence (success column or labels) can.
```

```text
$ rrc compare bc:ckpt/all bc:ckpt/filtered
A: bc:ckpt/all      | 200 | 0         | 0.0%  | 0.0% – 1.9%   | ...
B: bc:ckpt/filtered | 200 | 29        | 14.5% | 10.3% – 20.0% | ...
B (bc:ckpt/filtered) is better than A (bc:ckpt/all) (Δ = +14.5 pts, 95% CI +9.9 to +20.0; McNemar p = 3.7e-09).
```

Everything runs on a laptop CPU. No GPU, no account, and no network calls after install. **Status: 0.1.0, alpha.**

## Install

Python 3.10–3.12. Pick the extras for what you want to do:

| You want to… | Install |
|---|---|
| Score an existing LeRobot dataset | `pip install "robot-report-card[score]"` (about 75 MB base + pyarrow) |
| Record simulated SO-101 episodes | `pip install "robot-report-card[sim]"` |
| Export recordings to LeRobot format | `pip install "robot-report-card[sim,lerobot]"` |
| Evaluate / compare / train small policies | `pip install "robot-report-card[sim,eval]"` (+ `score` for `train-bc`) |
| Everything in this README | `pip install "robot-report-card[score,sim,eval,lerobot]"` |

- **torch on Linux:** pip's default torch wheel is the CUDA build. The `eval` extra is then about **5.4 GB**, and all
  extras about **10 GB**. Everything here runs on the CPU, so install the CPU-only wheel first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`. (We document this but haven't tested it: that
  index is blocked in our build environment.) On macOS the PyPI torch wheels are CPU/MPS only and much smaller. A full
  install on an Apple Silicon MacBook Air was about 1.8 GB.
- **numpy:** the `lerobot` extra pins `numpy<2.4`, because numpy 2.4 crashes lerobot 0.4.4 when it saves an episode.
- The heavy extras (`eval`, `lerobot`) are tested on Python 3.11. The rest is tested on 3.10, 3.11 and 3.12.

## Score a dataset you already have

`rrc score` reads a LeRobot dataset on disk (v3.0, v2.1 or v2.0) with pyarrow alone. It needs no lerobot, no torch
and no simulator, and **it never writes into the dataset**.

```bash
pip install "robot-report-card[score]"
hf download <user>/<dataset> --repo-type dataset --include "meta/*" "data/*" --local-dir ~/rrc-data/mine   # parquet only, no videos
rrc score ~/rrc-data/mine                        # table + summary; JSON report written to ./mine.rrc_score.json
rrc score ~/rrc-data/mine --only-flagged --json-out reports/mine.json --overwrite
```

Each row shows `ep | frames | quality (ok/FLAG/HARD) | score | outcome (+source) | why`.
- **Quality** is about motion only: jitter, jerk, hesitation, dithering, actions pinned at their limits, commands the arm
  doesn't follow. A robust z-score is computed per signal against *this* dataset, and an episode is flagged when its
  largest z is above 3.5 (`--threshold`). **HARD** means a recording with a hard defect: NaN/inf values, a joint whose reading never
  changes while its command moves, dropped frames, ≥ 20% saturated actions, or an episode that's too short.
- **Outcome** comes only from evidence: a human label in `meta/rrc_tags.json`, otherwise the last-frame `next.success`,
  otherwise `unknown`.
- Column names differ between datasets? Use `--state-key` / `--action-key`. The JSON report is never silently
  replaced: to re-run `rrc score` on the same dataset, pass `--overwrite` (or pick another `--json-out`).

## Record, tag and export simulated episodes

A simulated SO-101 reach task in MuJoCo: 90 frames (3 s) at 30 fps, with the same joint names as real SO-101 data.

```bash
pip install "robot-report-card[sim,lerobot]"
rrc record --policy scripted --episodes 10 --seed 0 --out runs/demo
rrc tag runs/demo --episode 3 --label fail --note "overshot"
rrc list runs/demo
rrc export runs/demo --out datasets/demo --repo-id local/demo
```

- `rrc record --mix` records a labelled mix of good and junk episode types into one session, for example
  `--mix clean:60,noise025:10,random:10,hesitation:5,wrong:5`. `rrc record --list-groups` lists the types.
- `--policy wrong` reaches an independent wrong goal (clean motion, failed outcome). `--wrong-goal mirrored` restores
  the old mirrored-pose behavior.
- Tags live in the session (`runs/demo/episodes.jsonl`) and are copied to `datasets/demo/meta/rrc_tags.json` on export.
  Nothing is uploaded.

## Evaluate a policy

`rrc eval` rolls a policy out in the simulated reach task and reports its success rate. `rrc compare` runs two
policies on the **same** seeds and tests whether B differs from A. `rrc train-bc` trains a small behavior-cloning MLP on
an exported dataset (on the CPU, in seconds), so there's a real A vs B to compare. The full demo (about 35–50 s):

```bash
pip install "robot-report-card[score,sim,eval,lerobot]"
rrc record --mix clean:60,noise025:10,random:10,hesitation:5,wrong:5 --seed 30000 --out runs/mix
rrc export runs/mix --out datasets/mix --repo-id local/mix
rrc score datasets/mix --json-out mix.score.json
rrc train-bc datasets/mix --out ckpt/all                                                        # A: every episode
rrc train-bc datasets/mix --out ckpt/filtered --keep ok-and-success --score-json mix.score.json  # B: quality ok AND succeeded
rrc eval bc:ckpt/filtered
rrc compare bc:ckpt/all bc:ckpt/filtered --json-out compare.json
rrc report --score mix.score.json --compare compare.json --md card.md --html card.html
```

On our Linux machine, A succeeds 0/200 and B 29/200 (McNemar p = 3.7e-9). Policies are `scripted`, `random`,
`bc:<checkpoint dir>` or `lerobot:<pretrained_model dir>`. Defaults are 200 episodes and `--eval-seed 900000`, and
episode i always starts from the same target.

**Success definition:** the gripper tip is within 2 cm of the target **on the final frame**, so the arm has to hold
there. LeRobot's own eval counts success if it happens at any step. We report that too (the "any-step rate"), and the
two numbers differ.

**How to read `rrc compare`:**
- **Rate with a Wilson 95% CI** for each policy: the range the true success rate plausibly lies in. At n = 200 it's
  about ±7 points near 50%.
- **Paired counts** (both / A only / B only / neither): only the episodes where the two policies disagree carry information.
- **Exact McNemar p** tests whether "B only" and "A only" are unbalanced. The verdict names the direction when p < 0.05.
- **Δ with a 95% CI** (Newcombe) is the size of the difference, in points of success rate.
- **MDE:** "this test detects differences of about X points or more with 80% power", based on how much the two
  policies disagreed. *"No detectable difference"* means the difference is smaller than this test can see at this n.
  It does **not** mean the policies are equal. If the exact test and the CI disagree right at the 5% line, the output
  says the result is borderline.

**Checkpoint vs recipe:** `compare` tests two *fixed* checkpoints, and its CI covers rollout noise only. Retraining the
same data and settings with another seed moved success by **10–15 points** on this task. A claim like "filtered data
trains better policies" needs at least 3 training seeds per side (`--seed`) with a consistent direction.

**LeRobot checkpoints** (`lerobot:<dir>/pretrained_model`, needs the `lerobot` extra) run through a small adapter. So far
this is **only a smoke test**: we ran it on a tiny state-only ACT trained locally and make no claim about ACT's success
rate. Policies that need inputs the simulator doesn't provide (cameras, other robots) are refused with a message naming
the missing input. See [SECURITY.md](https://github.com/Sebilopez1/robot-report-card/blob/main/SECURITY.md) before
loading checkpoints you didn't train.

## The report card

`rrc report` joins the JSON from `rrc score`, `rrc eval` and `rrc compare` into one page with five blocks:
**Data**, **Policy**, **Regression**, **Verdict** and **Can't tell**. The card is always printed. `--md` and `--html`
also write it as Markdown or a self-contained HTML page, and `rrc_report.json` holds everything the renderings show.

- The verdict comes from fixed, ordered rules that cite their evidence. It says what the evidence is **consistent
  with**, and it never names a cause. For the demo above:
  ```text
  [R2] 25.6% of demos in mix failed (outcome source: sim 90 of 90 episodes). Consistent with a data problem.
  A (bc:ckpt/all) learned from all 90 episodes. B (bc:ckpt/filtered) learned from a filtered subset
  (keep = ok-and-success, 59 of 90 episodes), of which 0% failed. In our sim study, training on failed
  wrong-goal demos cost ~24 points.
  ```
- A `bc:` checkpoint counts as trained on the scored dataset only when its dataset fingerprint matches **and** it was
  trained on every episode (`--keep all`). A filtered checkpoint is reported as a subset of the scored dataset, and
  it never gets a clean verdict.
- `--train-seeds N` tells the card how many training seeds stand behind each side of a comparison. Below 3, it won't
  draw a conclusion about the training recipe.
- The thresholds (for example, 10% failed demos) are proposals tuned in simulation, and the card says so.

## What it can't tell you

- **Failed attempts that move normally** (stopping early, reaching for the wrong goal) look fine to the motion score.
  Only outcome evidence (a success column or labels) shows them.
- **Uniform problems:** scores are relative to one dataset and assume most episodes are good. If every episode shares
  the same problem (for example a jittery leader arm), few or none will be flagged. Scores aren't comparable across datasets.
- **Slow wobble** (band-limited jitter) isn't flagged.
- **Simulation, not your robot:** `rrc eval` runs in a simulated SO-101 reach task. It says nothing directly about your
  real robot or other tasks.
- **Same machine only:** reruns on one machine are bit-for-bit identical. A rerun on another machine is a new sample,
  not a replay.
- **Thresholds are sim-tuned.** They haven't been calibrated on real teleoperation data yet.

## What we've measured so far

- **Scoring (simulation benchmark, three seed sets including a private one):** junk motion (random actions,
  high-noise actions, hesitation) ranks almost perfectly (AUROC ≥ 0.99). At the default threshold, flag precision is
  1.00 and recall 0.975–1.00. At most 2% of good episodes are flagged at standard noise (2–5% at low noise).
- **Scoring (one public real SO-101 dataset, 10 episodes):** 1 true positive and 1 false positive (a return-to-home
  motion). The false-positive pattern is fixed and verified in simulation; the re-check on the real dataset is pending.
- **Evaluation (simulation, one task):** two checkpoints trained on the same mixed dataset, unfiltered vs filtered to
  quality-ok *and* successful episodes, scored 0/200 vs 29/200 (p = 3.7e-9). Over 5 training seeds, unfiltered scored
  0–2/200 and filtered 15–39/200 (every p ≤ 6.1e-5).
- **What hurt the policy (simulation, MLP, 3 training seeds):** random actions, high-noise actions and wrong-goal demos
  did. Hesitation didn't. The scorer removes the first two; only outcome evidence removes wrong-goal demos.
- **Statistics:** the CIs and tests are checked against scipy and exact enumeration (Wilson coverage 0.95 on average,
  never below 0.91 for n = 20–200 and success rates between 5% and 95%).

Details: [STATUS.md](https://github.com/Sebilopez1/robot-report-card/blob/main/STATUS.md) and the
[milestone reports](https://github.com/Sebilopez1/robot-report-card/tree/main/docs/milestones).

## Contributing, security, changelog

- [CONTRIBUTING.md](https://github.com/Sebilopez1/robot-report-card/blob/main/CONTRIBUTING.md): dev install, test
  profiles, how to report a dataset where the score is wrong. CI has not yet run on GitHub.
- [SECURITY.md](https://github.com/Sebilopez1/robot-report-card/blob/main/SECURITY.md): loading checkpoints safely.
- [CHANGELOG.md](https://github.com/Sebilopez1/robot-report-card/blob/main/CHANGELOG.md).
- Found a dataset where `rrc score` is wrong? Please
  [open an issue](https://github.com/Sebilopez1/robot-report-card/issues) with the `rrc score` JSON. The JSON
  contains no video.

## Roadmap

1. Data logger: record, tag, export *(done)*
2. Dataset scoring: `rrc score` *(done)*
3. Policy evaluation: `rrc eval`, `rrc compare`, `rrc train-bc`, `rrc report` *(done)*
4. Free open-source release *(this release)*
5. Hosted team dashboard (paid; free for .edu)

## License

Apache License 2.0. See [LICENSE](https://github.com/Sebilopez1/robot-report-card/blob/main/LICENSE) and
[NOTICE](https://github.com/Sebilopez1/robot-report-card/blob/main/NOTICE). The bundled SO-101 robot model comes
from TheRobotStudio's SO-ARM100 project (Apache-2.0).
