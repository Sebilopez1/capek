# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/). Before 1.0, minor versions may change the CLI.

## [0.1.0] — 2026-10-01

First public release (alpha).

### Added
- **Data logger:** `rrc record` (simulated SO-101 reach task in MuJoCo, 90 frames at 30 fps, seeded and reproducible),
  `rrc tag`, `rrc list`, and `rrc export` to LeRobot dataset v3.0 (via `lerobot==0.4.4`), with tags in `meta/rrc_tags.json`.
- `rrc record --mix GROUP:N,...` and `--list-groups`, which record a labelled mix of good and junk episode types
  (clean, nearmiss, noise005/010/025, hesitation, wobble, stall, wrong, random, return_home). **These group names and
  their definitions are a public API from 0.1.0.** `--noise-profile standard|low`; `--append` to extend a session.
- **Dataset scoring:** `rrc score` reads LeRobot v3.0 / v2.1 / v2.0 datasets with pyarrow (no lerobot or torch). It
  gives each episode a per-dataset robust motion-quality score with plain-language reasons, hard flags, and outcome
  evidence kept separate. It writes a JSON report and never modifies the dataset.
- **Policy evaluation:** `rrc eval` (success rate with a Wilson CI, final-frame and any-step success) and `rrc compare`
  (paired seeds, exact McNemar, Newcombe CI on the difference, minimum detectable effect, borderline detection).
- `rrc train-bc`: a small CPU behavior-cloning MLP with `--keep all|quality-ok|success|ok-and-success` filters; the
  checkpoint records its data fingerprint and a weights hash.
- `lerobot:<pretrained_model dir>` policy adapter (state-only; smoke-tested on a tiny ACT, no quality claim).
- **Report card:** `rrc report` builds a one-page card (terminal, `--md`, `--html`, JSON) from score, eval and compare
  JSON. It has rule-based "consistent with" verdicts and a fixed "Can't tell" block.
- The bundled SO-101 robot model from TheRobotStudio SO-ARM100 (Apache-2.0).

### Changed (relative to pre-release development builds)
- `rrc record --policy wrong` now reaches an **independent** wrong goal. `--wrong-goal mirrored` restores the old
  mirrored-pose behavior, and `policy_params.goal` records which one was used.
- `numpy<2.4` is pinned only in the `lerobot` extra (lerobot 0.4.4 needs it); the base install allows newer numpy.
- `action_tv_ratio` uses a split denominator, so return-to-home motions are no longer false-flagged.

### Known limitations
- Policy evaluation runs in simulation (one reach task), not on a real robot.
- Scores are relative to one dataset. Uniform problems, slow wobble and smooth failed attempts aren't flagged.
- Thresholds are tuned in simulation.
- Results are reproducible on one machine only.
- CI and the release workflow have not yet run on GitHub.

[0.1.0]: https://github.com/Sebilopez1/robot-report-card/releases/tag/v0.1.0
