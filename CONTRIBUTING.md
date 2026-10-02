# Contributing

Thanks for helping. The most useful contribution right now is **a dataset where `capek score` is wrong**: an episode it
flags that looks fine to you, or a bad one it misses. Please
[open an issue](https://github.com/Sebilopez1/capek/issues/new/choose) (pick "capek score is wrong on my dataset") with the `capek score` JSON report, the episode
indices and what you saw. Only share data you're allowed to share. Please don't post other people's datasets or name
their episodes without their permission.

## Development install

```bash
git clone https://github.com/Sebilopez1/capek.git && cd capek
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,score,sim]"                         # light profile: no torch, no lerobot
pip install -e ".[dev,score,sim,eval,lerobot]" scipy      # full profile (on Linux, install CPU torch first:
                                                          #   pip install torch --index-url https://download.pytorch.org/whl/cpu)
```

## Tests and lint

```bash
ruff check . && ruff format --check .
pytest -q                     # light profile: about 30 s; tests needing torch / lerobot / scipy skip cleanly
pytest -q                     # full profile: about 2.5 min
```

Slow checks are gated behind environment variables:

| Variable | What it runs |
|---|---|
| `CAPEK_BENCH_FULL=1` | full benchmark builds and the scoring bars on every seed set |
| `CAPEK_E2E_FULL=1` | the full-size data → policy end-to-end study |
| `CAPEK_ACT_FULL=1` | the 300-step ACT adapter run |

Tests set `HF_HUB_OFFLINE=1`; nothing in the suite needs the network.

**CI** (`.github/workflows/tests.yml`) has a light matrix (Ubuntu and macOS × Python 3.10–3.12), a heavy CPU-torch job
on `main`, and a weekly full-benchmark run. Pull requests must pass the light matrix before review.

## Ground rules

- Be kind. Everyone here follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- Every change is reviewed by someone other than its author before it's merged.
- Scoring changes must keep the benchmark bars passing (`CAPEK_BENCH_FULL=1`). Don't tune thresholds on the held-out seed sets.
- User-facing text must stay honest. Say "consistent with", never "caused by". Don't claim that unflagged data is
  good, or that simulation results transfer to real robots. `tests/test_p4_5_report_rules.py` enforces this for the report card.
- Install hints in messages use the PyPI form: `pip install "capek-tech[<extra>]"`.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
