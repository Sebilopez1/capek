## What this changes

<!-- One or two sentences, and link the issue it fixes (e.g. "Fixes #12"). -->

## How it was tested

<!-- Commands you ran and what you saw. -->

## Checklist

- [ ] `ruff check . && ruff format --check .` pass
- [ ] `pytest -q` passes (say which profile: light or full)
- [ ] Scoring changes keep the benchmark bars passing (`CAPEK_BENCH_FULL=1`)
- [ ] User-facing text stays honest: "consistent with", never "caused by"; no claims that sim results transfer to real robots
- [ ] `CHANGELOG.md` updated if users will notice the change
