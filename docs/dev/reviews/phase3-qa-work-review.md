## Coder review (P3-1 bench, P3-8)

Coder, 2026-09-25. Scope: 261bc70 (P3-1 bench: return_home, low-noise sets, R-real-1 bar, power check) and
8ee8daf (P3-8 headline e2e + `bench record`), checked against `docs/phase3-plan.md` rev 2 (DoD 9, DoD 10, D3, D4).
Reran here: `RRC_BENCH_FULL=1 pytest tests/test_p3_1_bench.py tests/test_p3_8_e2e.py` → 5 passed, 1 skipped
(`RRC_E2E_FULL`), e2e 36.8 s, gated power check 14.6 s; full default suite with 94aeac4: 234 passed, 3 skipped, 144 s.

**Verdict: APPROVED** (P3-1 bench and P3-8). No required changes; non-blocking notes below.

### P3-1 bench (261bc70)

- **return_home generator** (`ReturnHomeReach`): out over U(0.8, 1.2) s, hold U(0.1, 0.4) s, back over the same time,
  as D3/brief §6 specify. The longest case is 36 + 12 + 36 = 84 frames, so it always completes inside the 90-frame
  episode. Both legs use the same gravity offset as `ScriptedReach`, and the out leg starts from the raw state as
  `ScriptedReach` does, so it is "clean motion" by our own definition, and white noise comes from the profile's σ.
  `sim_success` is false by construction (it ends at home), and the class is `clean_variant`, which the evaluator
  keeps out of P/R/AUROC and clean-FP counts (tested). Correct.
- **Seed hygiene:** return_home is appended last in `MIXED` (group k = 10), so every existing group keeps its seed
  base + k. Moving `sigma` above the ramp draw doesn't change any draw: `r_sigma` and `r_ramp` are separate spawned
  streams, and `r_sigma.uniform` was already evaluated eagerly for fixed-σ groups. The unchanged phase 2 bar
  numbers in QA's table confirm this.
- **Low-noise variant:** every σ-drawing group (clean, nearmiss, hesitation, wobble, stall, wrong, return_home) uses
  σ ~ U(0.002, 0.008); the fixed-σ noise groups keep 0.05/0.10/0.25 and random is untouched. Matches D3 exactly, and
  the test checks it per group.
- **Binding:** standard profile = every DoD 4 bar plus return_home ≤ 1/10; low profile = only return_home binds, the
  rest (including low-noise clean-only) is reported. Matches D3.
- **`rreal1` seed set: honest as labelled.** `spec.py` says it is a regression fixture, not a validation set, that it
  was picked from three candidates because the old formula visibly fails there, and it lists the unselected results
  (dev 1/10, held-out 0/10, private 2/10). The phase 3 code review adds six fresh bases (0, 5, 6, 2, 0, 3 of 10). Two
  wording points so nobody over-reads it later (non-blocking, for STATUS/milestone text):
  - "net 7/10" is a *selected* number. The fair estimate of how often the phase 2 formula false-flags return_home on
    a low-noise set is the unselected ones: about 2–3/10 on average, 0–6/10 per set.
  - The fixture shows the bar *can* fail (power), not how often the bug bites. DoD 10's power clause is already met
    without it (QA-private 2/10).
- **Power check enforcement (non-blocking):** `bench check` prints the power check but its PASS/FAIL doesn't depend on
  it. It is enforced only by the gated `test_power_check_old_formula_fails_return_home_bar` (`RRC_BENCH_FULL=1`).
  That is fine for a benchmark-validity check; just say in the acceptance notes that it runs under the gate.
- **Non-blocking:** the standard and low-noise twins share seeds (same targets, same timing draws), so they are two
  views of one sample, not independent evidence. That is the right design for isolating the noise effect; don't
  count them as separate replications.
- **Non-blocking:** the build docstring now says 160 mixed episodes (60 + 10 × 10); the phase 2 plan text and brief
  still say 150. Fine as long as STATUS says the mixed set grew by the 10 return_home episodes.

### P3-8 e2e and `bench record` (8ee8daf)

- **Pinned mix matches D4/DoD 9:** `clean:60,noise025:10,random:10,hesitation:5,wrong:5`, bench generators, fixed seed
  base 30000, one session, then the installed `rrc`: export → score (`--json-out`) → `train-bc --keep all` and
  `--keep ok-and-success --score-json`, both `--seed 0` → `compare … --episodes 200 --eval-seed 900000`.
- **Assertions match DoD 9 and the wording DoDs:** p < .01; B > A with the CI above 0; verdict starts with
  "B (bc:ckpt_filtered) is better than A (bc:ckpt_all)"; no "No detectable difference"; recipe caveat with
  "10–15 points" and "≥ 3 training seeds"; success definition (final frame, any step); MDE sentence present; 200
  seeds starting [900000, 0]; train seed 0 on both sides. The filter check (kept episodes = quality ok ∩ sim
  success, and no random/noise/wrong-goal episode kept) is a good extra.
- My earlier independent measurement of the same recipe with seed base 20000 (P3-6) gave A 3/200 vs B 39/200,
  p = 5.6e-9; QA's base 30000 gives 0/200 vs 29/200, p = 3.7e-9, and worst 6.1e-5 over train seeds 0–4. The margin
  is robust, not a lucky seed.
- **Compatibility with the R1 fix (94aeac4):** the MDE sentence is still printed on significant results and says
  "checkpoints" for two `bc:` checkpoints, so `report["mde_statement"] in text` still holds; a borderline sentence
  can't appear here (p ≪ .05 and the CI excludes 0). Verified by rerunning the test.
- **`bench record`:** group parsing rejects unknown names, zero or non-numeric counts (tested); group k records with
  seed + k via the library `record_into` path (the same code as `rrc record --append`). Note, non-blocking: the plan's
  P3-8 scope says "via the installed rrc … (bench generators, --append)". The recording step uses
  `python -m robot_report_card.bench record` rather than `rrc record --append`, because bench generators aren't CLI
  policies. Every later step goes through the installed `rrc`. That's the only possible reading; say so in STATUS.
- **Suite time (non-blocking):** the e2e is the largest default test (≈ 37 s); the full default suite is now 144 s
  against the 150 s budget. If README tests or anything else lands, gate this test (as QA's own review proposes)
  rather than shrinking n = 200.


## Researcher review (P3-4)

Researcher, 2026-09-25. Scope: `tests/test_p3_4_stats_validation.py` (commit `6d49fb1`) against plan rev 2, P3-4 (a)–(e), plus mutation testing of `src/robot_report_card/eval/stats.py`. I made scratch copies of `src/` only; no repo code was modified.

**Verdict: APPROVED.** The suite tests the right quantities against independent references, the bars match plan rev 2, and it caught every formula bug I planted (11/11 mutants).

**Right quantities, right bars (plan P3-4):**
- (a) Wilson and Clopper-Pearson are compared with `scipy.stats.binomtest(...).proportion_ci("wilson" / "exact")` at a 1e-9 tolerance. ✔
- (b) `mcnemar_exact` is compared with `binomtest(b, b + c, .5).pvalue` over the full 0–60 × 0–60 square plus large pairs, and b = c = 0 gives 1.0. ✔
- (c) Newcombe method 10:
  - it matches QA's own implementation written from the paper on 2,007 tables, including degenerate ones, with a tolerance of 1e-12;
  - exact multinomial enumeration at n = 50/200 is held to mean 0.94–0.96 and worst ≥ 0.90, matching the plan. ✔
- (d) Power at the reported MDE is simulated independently (2,000 multinomial draws through `mcnemar_exact`) and must be ≥ 0.78 at 3 grid points. There's also a check that MDE − 0.005 is underpowered. ✔
- (e) Wilson exact coverage on a 1e-4 grid over [0.05, 0.95], n = 20–200, requires mean 0.945–0.955 and worst ≥ 0.91, which is plan rev 2 R2. ✔
- Runtime: 15 passed in 8.3–9.6 s (bar < 20 s). ✔

**Mutation test.** Each change was applied to a scratch copy of the module; the suite was pointed at it via `PYTHONPATH`, and the unmutated copy passes 15/15:

| mutant | result |
|---|---|
| M1 Wilson z = 1.96 instead of 1.959964 | **caught** (2 failed) |
| M10 Wilson without the z²/4n² term (Wald-like) | **caught** (5 failed) |
| M3 Clopper-Pearson with α instead of α/2 | **caught** (1) |
| M2 one-sided McNemar (no ×2) | **caught** (1) |
| M11 McNemar CDF off by one | **caught** (1) |
| M4 Newcombe φ sign flipped | **caught** (1) |
| M5 Newcombe φ ignored (φ = 0) | **caught** (1) |
| M6 Newcombe lower/upper term swapped | **caught** (1) |
| M7 power computed at α = .10 | **caught** (3) |
| M8 MDE target power 0.70 | **caught** (3) |
| M9 MDE reported 1 pt too small | **caught** (3) |

**Notes (non-blocking):**
- N1 (c):
  - The exact-coverage test enumerates QA's reference `newcombe10`, not `stats.newcombe_paired_diff`. It is linked to the library only by the equality test. That's fine, since M4–M6 are caught there.
  - A *shared* misreading of the paper by both implementations would still pass. Adding one published worked example from Newcombe (1998) would close that gap. I don't have the table values verified, so I can't supply them.
  - The φ grid covers only non-negative correlations. That's realistic for checkpoint pairs, but φ < 0 is untested.
- N2 (a):
  - The plan says "k, n ≤ 200". Wilson is checked for all k with n = 1–60 plus 7 larger n; Clopper-Pearson only for 11 values of n.
  - The docstring says the full grids are "run by hand", but no gated test exists. Add an `RRC_STATS_FULL=1` gated test so "by hand" is reproducible.
- N3 (d): the grid points use lose = 0.0 and 0.10 only. `compare` will usually print the MDE at the *observed* disagreement (D2; e.g. 0.22 gives 12.5 pts at n = 200). Consider adding (200, 0.555, 0.22) as a 4th point. It's cheap.
- N4 (d): the "MDE − 0.005 is underpowered" check uses the library's own `mcnemar_power`, so it isn't independent. It's still adequate, because the power at the MDE itself is checked by independent simulation.
