<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/banner-dark.svg">
    <img alt="Robot Report Card: is it your data or your model? Dataset scoring and honest policy evaluation for LeRobot and low-cost arms like the SO-101." src="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/banner-light.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/robot-report-card/"><img alt="PyPI" src="https://img.shields.io/pypi/v/robot-report-card?color=0969da"></a>
  <a href="https://pypi.org/project/robot-report-card/"><img alt="Python 3.10 to 3.12" src="https://img.shields.io/pypi/pyversions/robot-report-card"></a>
  <a href="https://github.com/Sebilopez1/robot-report-card/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/Sebilopez1/robot-report-card/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/Sebilopez1/robot-report-card/blob/main/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> ·
  <a href="#-score-a-dataset-you-already-have">Score</a> ·
  <a href="#-evaluate-and-compare-policies">Evaluate</a> ·
  <a href="#-the-report-card">Report card</a> ·
  <a href="#-what-it-cant-tell-you">Limits</a> ·
  <a href="#-roadmap">Roadmap</a>
</p>

---

You record demos, train a policy, and it performs badly. Was it the data or the model?
**Robot Report Card (`rrc`)** turns "eyeball the rollouts and guess" into evidence: which episodes in your
[LeRobot](https://github.com/huggingface/lerobot) dataset look like junk and why, a success rate with a confidence
interval, and whether checkpoint B really beats checkpoint A. It puts all of it on a one-page report card that says
what the evidence is *consistent with*, and what it can't tell.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/result-dark.svg">
    <img alt="Bar chart: policy A trained on all 90 demos succeeds 0 of 200 times; policy B trained on the 59 demos rrc kept succeeds 29 of 200 (14.5%). Difference +14.5 points, 95% CI +9.9 to +20.0, McNemar p = 3.7e-9. One task, in simulation." src="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/result-light.svg" width="92%">
  </picture>
</p>

## ✨ Highlights

- 🔎 **Find junk episodes**: `rrc score` ranks every episode on jitter, jerk, hesitation, dithering, saturated actions
  and commands the arm doesn't follow, and says *why* each one was flagged. It reads the parquet files directly and
  **never writes into your dataset**.
- 📊 **Honest A vs B**: `rrc compare` runs two policies on the same seeds and reports Wilson intervals, an exact McNemar
  test, the size of the difference, and the smallest difference the test could have detected.
- 🧾 **One-page report card**: `rrc report` joins everything into Data / Policy / Regression / Verdict / Can't tell,
  in the terminal, Markdown or HTML.
- 🦾 **Built for low-cost arms**: SO-100 / SO-101 joint names, LeRobot v3.0, v2.1 and v2.0 datasets, and a bundled
  MuJoCo SO-101 reach task.
- 💻 **Laptop-only**: runs on a CPU. No GPU, no account, no network calls after install.

## 🚀 Quickstart

```bash
pip install "robot-report-card[score]"
rrc score path/to/lerobot_dataset --only-flagged
```

Want the whole loop in simulation (record → score → train two policies → compare → report card)? That takes about
a minute on a laptop; see [the full demo](#-evaluate-and-compare-policies).

## 🧭 How it fits together

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/pipeline-dark.svg">
    <img alt="Pipeline: rrc record (simulated demos) → rrc export (LeRobot format) → rrc score (find junk episodes, the main feature) → rrc train-bc (teach a policy) → rrc eval and compare (test it, A vs B) → rrc report (one-page card). Real-robot users start at rrc score." src="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/pipeline-light.svg" width="100%">
  </picture>
</p>

| Command | What it does | Needs |
|---|---|---|
| `rrc score` | Score every episode of a LeRobot dataset, flag the junk | `[score]` |
| `rrc record` / `tag` / `list` / `export` | Record simulated SO-101 demos, label them, export to LeRobot format | `[sim,lerobot]` |
| `rrc train-bc` | Train a small behavior-cloning policy on the CPU, in seconds | `[score,eval]` |
| `rrc eval` / `compare` | Success rate with CIs; paired A vs B test | `[sim,eval]` |
| `rrc report` | The one-page report card | base |

## 📦 Install

Python 3.10–3.12. Pick the extras for what you want to do:

| You want to… | Install |
|---|---|
| Score an existing LeRobot dataset | `pip install "robot-report-card[score]"` (about 75 MB base + pyarrow) |
| Record simulated SO-101 episodes | `pip install "robot-report-card[sim]"` |
| Export recordings to LeRobot format | `pip install "robot-report-card[sim,lerobot]"` |
| Evaluate / compare / train small policies | `pip install "robot-report-card[sim,eval]"` (+ `score` for `train-bc`) |
| Everything in this README | `pip install "robot-report-card[score,sim,eval,lerobot]"` |

<details>
<summary><b>Install notes</b> (torch size on Linux, numpy pin, tested Pythons)</summary>

- **torch on Linux:** pip's default torch wheel is the CUDA build. The `eval` extra is then about **5.4 GB**, and all
  extras about **10 GB**. Everything here runs on the CPU, so install the CPU-only wheel first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu` (this is what our CI does). On macOS the PyPI
  torch wheels are CPU/MPS only and much smaller. A full install on an Apple Silicon MacBook Air was about 1.8 GB.
- **numpy:** the `lerobot` extra pins `numpy<2.4`, because numpy 2.4 crashes lerobot 0.4.4 when it saves an episode.
- The heavy extras (`eval`, `lerobot`) are tested on Python 3.11. The rest is tested on 3.10, 3.11 and 3.12, on Linux
  and macOS.

</details>

## 🔎 Score a dataset you already have

`rrc score` reads a LeRobot dataset on disk (v3.0, v2.1 or v2.0) with pyarrow alone. It needs no lerobot, no torch
and no simulator, and **it never writes into the dataset**.

```bash
pip install "robot-report-card[score]"
hf download <user>/<dataset> --repo-type dataset --include "meta/*" "data/*" --local-dir ~/rrc-data/mine   # parquet only, no videos
rrc score ~/rrc-data/mine                        # table + summary; JSON report written to ./mine.rrc_score.json
rrc score ~/rrc-data/mine --only-flagged --json-out reports/mine.json --overwrite
```

<p align="center">
  <img alt="Terminal output of rrc score --only-flagged on a 90-episode simulated dataset: 25 of 90 episodes flagged, each with its score, outcome and the reasons, such as action chatter and commands not followed." src="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/score.svg" width="92%">
</p>

Each row shows `ep | frames | quality (ok/FLAG/HARD) | score | outcome (+source) | why`.

- **Quality** is about motion only: jitter, jerk, hesitation, dithering, actions pinned at their limits, commands the arm
  doesn't follow. A robust z-score is computed per signal against *this* dataset, and an episode is flagged when its
  largest z is above 3.5 (`--threshold`). **HARD** means a recording with a hard defect: NaN/inf values, a joint whose
  reading never changes while its command moves, dropped frames, ≥ 20% saturated actions, or an episode that's too short.
- **Outcome** comes only from evidence: a human label in `meta/rrc_tags.json`, otherwise the last-frame `next.success`,
  otherwise `unknown`.
- Column names differ between datasets? Use `--state-key` / `--action-key`. The JSON report is never silently
  replaced: to re-run `rrc score` on the same dataset, pass `--overwrite` (or pick another `--json-out`).

## 🧪 Evaluate and compare policies

`rrc eval` rolls a policy out in the simulated SO-101 reach task and reports its success rate. `rrc compare` runs two
policies on the **same** seeds and tests whether B differs from A. `rrc train-bc` trains a small behavior-cloning MLP on
an exported dataset, so there's a real A vs B to compare. The full demo (about 35–50 s):

```bash
pip install "robot-report-card[score,sim,eval,lerobot]"
rrc record --mix clean:60,noise025:10,random:10,hesitation:5,wrong:5 --seed 30000 --out runs/mix
rrc export runs/mix --out datasets/mix --repo-id local/mix
rrc score datasets/mix --json-out mix.score.json
rrc train-bc datasets/mix --out ckpt/all                                                        # A: every episode
rrc train-bc datasets/mix --out ckpt/filtered --keep ok-and-success --score-json mix.score.json  # B: quality ok AND succeeded
rrc compare bc:ckpt/all bc:ckpt/filtered --json-out compare.json
rrc report --score mix.score.json --compare compare.json --md card.md --html card.html
```

<p align="center">
  <img alt="Terminal output of rrc compare: A 0 of 200, B 29 of 200, paired counts, delta +14.5 points with 95% CI +9.9 to +20.0, exact McNemar p = 3.7e-9, minimum detectable difference about 9 points." src="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/compare.svg" width="92%">
</p>

Policies are `scripted`, `random`, `bc:<checkpoint dir>` or `lerobot:<pretrained_model dir>`. Defaults are 200
episodes and `--eval-seed 900000`, and episode i always starts from the same target.

<details>
<summary><b>How to read <code>rrc compare</code></b></summary>

- **Rate with a Wilson 95% CI** for each policy: the range the true success rate plausibly lies in. At n = 200 it's
  about ±7 points near 50%.
- **Paired counts** (both / A only / B only / neither): only the episodes where the two policies disagree carry information.
- **Exact McNemar p** tests whether "B only" and "A only" are unbalanced. The verdict names the direction when p < 0.05.
- **Δ with a 95% CI** (Newcombe) is the size of the difference, in points of success rate.
- **MDE:** "this test detects differences of about X points or more with 80% power", based on how much the two
  policies disagreed. *"No detectable difference"* means the difference is smaller than this test can see at this n.
  It does **not** mean the policies are equal. If the exact test and the CI disagree right at the 5% line, the output
  says the result is borderline.
- **Success definition:** the gripper tip is within 2 cm of the target **on the final frame**, so the arm has to hold
  there. LeRobot's own eval counts success if it happens at any step. We report that too (the "any-step rate"), and
  the two numbers differ.

</details>

> [!IMPORTANT]
> **Checkpoint vs recipe.** `compare` tests two *fixed* checkpoints, and its CI covers rollout noise only. Retraining
> the same data and settings with another seed moved success by **10–15 points** on this task. A claim like "filtered
> data trains better policies" needs at least 3 training seeds per side (`--seed`) with a consistent direction.

**LeRobot checkpoints** (`lerobot:<dir>/pretrained_model`, needs the `lerobot` extra) run through a small adapter. So
far this is **only a smoke test**: we ran it on a tiny state-only ACT trained locally and make no claim about ACT's
success rate. Policies that need inputs the simulator doesn't provide (cameras, other robots) are refused with a
message naming the missing input. See [SECURITY.md](https://github.com/Sebilopez1/robot-report-card/blob/main/SECURITY.md)
before loading checkpoints you didn't train.

## 🧾 The report card

`rrc report` joins the JSON from `rrc score`, `rrc eval` and `rrc compare` into one page with five blocks:
**Data**, **Policy**, **Regression**, **Verdict** and **Can't tell**. The card is always printed; `--md` and `--html`
also write it as Markdown or a self-contained HTML page, and `rrc_report.json` holds everything the renderings show.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/card-dark.png">
    <img alt="The HTML report card for the demo: Data (25 of 90 flagged, 23 failed), Policy (A 0/200, B 29/200 with CIs), Regression (B minus A +14.5 points, p = 3.7e-9, with the checkpoint-vs-recipe caveat) and Verdict R2: consistent with a data problem." src="https://raw.githubusercontent.com/Sebilopez1/robot-report-card/main/docs/assets/card-light.png" width="80%">
  </picture>
</p>

- The verdict comes from fixed, ordered rules that cite their evidence. It says what the evidence is **consistent
  with**, and it never names a cause.
- A `bc:` checkpoint counts as trained on the scored dataset only when its dataset fingerprint matches **and** it was
  trained on every episode (`--keep all`). A filtered checkpoint is reported as a subset of the scored dataset, and
  it never gets a clean verdict.
- `--train-seeds N` tells the card how many training seeds stand behind each side of a comparison. Below 3, it won't
  draw a conclusion about the training recipe.
- The thresholds (for example, 10% failed demos) are proposals tuned in simulation, and the card says so.

## 🎬 Record, tag and export simulated episodes

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

## ⚠️ What it can't tell you

- **Failed attempts that move normally** (stopping early, reaching for the wrong goal) look fine to the motion score.
  Only outcome evidence (a success column or labels) shows them.
- **Uniform problems:** scores are relative to one dataset and assume most episodes are good. If every episode shares the same problem (for example a jittery leader arm), few or none will be flagged. Scores aren't comparable across datasets.
- **Slow wobble** (band-limited jitter) isn't flagged.
- **Simulation, not your robot:** `rrc eval` runs in a simulated SO-101 reach task. It says nothing directly about your
  real robot or other tasks.
- **Same machine only:** reruns on one machine are bit-for-bit identical. A rerun on another machine is a new sample,
  not a replay.
- **Thresholds are sim-tuned.** They haven't been calibrated on real teleoperation data yet.

## 📏 What we've measured so far

| Claim | Evidence |
|---|---|
| Junk motion ranks almost perfectly | Sim benchmark, three seed sets incl. a private one: AUROC ≥ 0.99; flag precision 1.00, recall 0.975–1.00; ≤ 2% of good episodes flagged at standard noise (2–5% at low noise) |
| It works on real data (early) | One public real SO-101 dataset, 10 episodes: 0.1.0 flags 1 episode, a true positive. An earlier false positive (a return-to-home motion) is fixed |
| Filtering helps, in sim | Unfiltered vs filtered to quality-ok *and* successful: 0/200 vs 29/200 (p = 3.7e-9); over 5 training seeds, 0–2/200 vs 15–39/200 (every p ≤ 6.1e-5) |
| What hurt the policy | Random actions, high-noise actions and wrong-goal demos did; hesitation didn't (MLP, 3 seeds). The scorer removes the first two; only outcome evidence removes wrong-goal demos |
| The statistics are right | CIs and tests checked against scipy and exact enumeration (Wilson coverage 0.95 on average, never below 0.91 for n = 20–200, rates 5–95%) |

Details: [STATUS.md](https://github.com/Sebilopez1/robot-report-card/blob/main/STATUS.md) and the
[milestone reports](https://github.com/Sebilopez1/robot-report-card/tree/main/docs/milestones).

## 🗺️ Roadmap

- [x] Data logger: record, tag, export
- [x] Dataset scoring: `rrc score`
- [x] Policy evaluation: `rrc eval`, `rrc compare`, `rrc train-bc`, `rrc report`
- [x] Free open-source release (0.1.0, alpha)
- [ ] Calibrate thresholds on more real teleoperation datasets
- [ ] Hosted team dashboard (paid; free for .edu)

## 🤝 Contributing

- **Found a dataset where `rrc score` is wrong?** That's the most useful thing you can send.
  [Open an issue](https://github.com/Sebilopez1/robot-report-card/issues) with the `rrc score` JSON (it contains no
  video).
- [CONTRIBUTING.md](https://github.com/Sebilopez1/robot-report-card/blob/main/CONTRIBUTING.md): dev install and test
  profiles. [SECURITY.md](https://github.com/Sebilopez1/robot-report-card/blob/main/SECURITY.md): loading checkpoints
  safely. [CHANGELOG.md](https://github.com/Sebilopez1/robot-report-card/blob/main/CHANGELOG.md).

## 📄 License

Apache License 2.0. See [LICENSE](https://github.com/Sebilopez1/robot-report-card/blob/main/LICENSE) and
[NOTICE](https://github.com/Sebilopez1/robot-report-card/blob/main/NOTICE). The bundled SO-101 robot model comes
from TheRobotStudio's SO-ARM100 project (Apache-2.0).
