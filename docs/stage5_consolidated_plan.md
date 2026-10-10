# Stage 5 analysis plan — consolidated summary (as of 2026-10-10)

**This is a derived summary, not a registered document.** It exists to make
`stage5_analysis_plan.md`'s rules easy to find without reading the full amendment history in
order. **Wherever this summary and `stage5_analysis_plan.md`'s dated amendments differ, the
amendments control** — this document adds no rule that is not already stated in an amendment, and
if a discrepancy is ever found, it means this summary is wrong, not that the plan has changed.
Every rule below cites the amendment (or original section) that sets it, and the supersession
table at the end traces every case where a later amendment corrected an earlier one.

## Research questions

RQ1 (x86-only, extends cross-platform after M1 data), RQ2 (same scope), RQ3 (cross-platform,
after M1 data) — verbatim from the proposal, §1. RQ1 split into RQ1a (coverage) and RQ1b
(estimation mode) — A7 point 10. RQ3 redefined to a within-CPU `state × platform` interaction test
as primary, Kendall's tau across states descriptive only — A7 point 11, extended with the INT8
backend confound (x86=fbgemm, M1=qnnpack) — A7r1(i).

## Blocks

Three blocks: **x86-GPU** (supplementary), **x86-CPU** and **M1-CPU** (both primary/confirmatory)
— A7 point 1, per the researcher's 2026-10-09 decision (`deviation_log.md` D5 update). No M1-GPU
block exists or was ever planned — A7r2's D5r1. Comparisons are within-block only; cross-block
comparisons are descriptive, never confirmatory — A7 point 1.

**Prospective registration differs by block:** the x86-GPU block's data were collected before A7
existed (2026-10-06 through 2026-10-08, vs. A7 dated 2026-10-09) — not prospectively registered.
The two CPU blocks are prospectively registered, **conditional on the freeze tag and Zenodo
deposit existing before the first CPU-block session is collected** — A7r3(j), extended by A7r4(f).

## Baselines

- **x86-GPU block:** FP32-**eager** (not TorchScript — the six main sessions never collected a
  TS baseline) — A7r1(a), correcting A7 point 2 for this block only.
- **x86-CPU and M1-CPU blocks:** FP32-**TorchScript**, regardless of what the D2 runtime-confound
  check concludes — A7 point 2 (as originally registered, for these two blocks), reaffirmed by
  A7r3(b).

## Primary estimator

Geometric mean, over sessions, of the within-session ratio `r` (compressed ÷ same-block baseline,
same session); 95% t-interval on `log r`, `df = sessions − 1`; **session is the replicate unit**,
not the rep or window — A7 point 3. Mann-Whitney U is within-session/exploratory only, never the
primary cross-session test — A7 point 4, resolving the contradiction between §4 (D14) and
`stage4b-design.md` §5.

## Confirmatory family

**The confirmatory family is RQ2's H2 test (β < 1), one per primary block (x86-CPU, M1-CPU) — two
tests total — Holm-Bonferroni across those two only.** Every per-state ratio is estimation
(interval, no p-value); every ±5% equivalence check is reported resolved/unresolved, not as a
significance test. This is decision rule **P-CF** — A7r1(b), full statement; renamed from the
originally-mislabelled "D3" once A7r2 corrected the audit's actual D1-D5 wording (see Decision
rules below). **Supersedes A7 point 6**, which registered Holm-Bonferroni "across the full family"
of ~30 per-state contrasts.

## RQ2 model and test

`log(r) ~ β·log(MAC ratio) + model (fixed, 3 levels) + (1|session)`, fit via `lmerTest::lmer`,
REML, Satterthwaite df. **Primary β fit uses zero-finetune pruned states only** (pruned30/50/70);
`_bnrecal` states form a separate sensitivity fit (same MAC ratio — pooling would double-count) —
A7r3(f), correcting the pooled fit A7r1(d) originally specified. This is decision rule **P-MAC** —
A7r1(d), full statement; renamed from the originally-mislabelled "D4."

**Test:** one-sided H2 (β < 1), `t = (β̂ − 1)/SE`, left-tail p-value against the Satterthwaite df —
computed explicitly, not `lmerTest`'s default (which tests against 0) — A7r3(f). **Always report
β̂ with its 95% CI alongside the p-value; no claim stronger than the interval supports** — A7r6(c).

**Input unit:** exactly one row per `(session, model, state)` — session is the replicate unit (A7
point 3); rep-level rows are rejected, enforced directly in `analysis/rq2_model.R` — A7r7(e).

**Singular-fit handling:** estimates from singular fits are kept and flagged, never discarded or
replaced after seeing the result — A7r6(a). **Non-convergence fallback** (an outright fitting
failure, not merely a singular fit): refit with `session` as a fixed factor, OLS, `df = n_obs −
(1 + 1 + (n_models−1) + (n_sessions−1))` (45 for the registered design) — A7r6(b).

**Analysis environment:** R 4.3.3, lme4 1.1.35.1, lmerTest 3.1.3 — `docs/analysis_environment.md`,
confirmed in force by A7r7(c) (correcting A7r4(g)'s now-stale "none of this exists yet").
**Scripts (`analysis/`) are pre-freeze, validated on synthetic data only** (Block O/P simulations,
summarised in A7r6(d)) — no real Stage 4b data has been fit by this pipeline as of this summary.

## Decision rules D1r1-D5r1 (the audit's D1-D5, corrected)

A7r1(f) first wrote out D1-D5 as a reconstruction from memory — later found to diverge from the
audit's actual wording in two ways (D1/D2/D5 right topic/wrong branch detail; "D3"/"D4" were the
wrong topic entirely, renamed P-CF/P-MAC above). **Corrected versions, A7r2:**

- **D1r1 (wall meter):** does `delta(NVML power-usage + RAPL package)` agree with the wall-derived
  delta within ±5%? YES → keep backend, report as validation. NO + cumulative counter agrees →
  Stage 4b GPU numbers need a rerun (data after D24 only — raw traces before D24 store one API).
  NO + neither agrees → wall energy primary for system-level claims. Cannot be evaluated until the
  P1 wall-meter protocol is run (exists in draft, `docs/p1_wall_meter_protocol.md` — A7r7(a)); not
  evaluable on sessions 1-6 regardless (pre-D24 data).
- **D2r1 (runtime):** is FP32-TS within ±5% (TOST, df per block) of FP32-eager? YES → keep
  eager-baseline ratios, disclosed. NO → **CPU blocks:** switch directly to the FP32-TS baseline
  (already measured in every CPU-block session). **GPU block:** use the bridged ratio — `ratio_eager
  × (eager/TS factor from series R)`, log-scale variance sum, Welch-Satterthwaite df combining the
  GPU block's df (5) and series R's df (3) — A7r3(a)/A7r4(e)/A7r7(b) (the latter clarifying the
  Welch-Satterthwaite inputs are variances of the *means*, not raw sample variances). **This
  bridged-ratio construction is the only one used for the GPU block** — supersedes A7r2's earlier
  "TS sourced from series R as a direct substitute" practical note, found unsafe given D17's
  measured between-session drift (−12.20% to +3.70%).
  - CPU-block equivalence check: df = 5 (6 sessions), SD threshold ≈ 0.0593, evaluated per
    architecture — a block counts "bounded" only if all three are — A7r3(b).
  - GPU-block/series-R SD threshold ≈ 0.0415 (df = 3, 4 sessions) — necessary, not sufficient, for
    equivalence; SD above it mechanically triggers NO — A8r2, extended by the BLOCK J append.
- **D3r1 (weight dependence):** are `_bnrecal` and zero-finetune energies within ±5% at the same
  architecture, for the 6 (architecture, prune-level) combinations the CPU-block design actually
  measures (3 excluded as known-collapsed, per `scripts/run_cpu_block_session.py`'s
  `BNRECAL_SKIP`)? YES only if all 6 are equivalent, else NO with every combination reported —
  A7r4(d). **Regardless of outcome, A5's fine-tuned models are always measured directly** —
  `_bnrecal` changes only BN statistics, fine-tuning changes all weights, so D3r1 informs
  discussion but never substitutes for A5's own measurement — A7r3(d). Marked: proposal pending
  researcher confirmation.
- **D4r1 (regime, 224×224):** do ratios keep the 32×32 ordering and β < 1 conclusion? **Inactive
  unless A6 is amended first** (a dated revision, before any 224×224 data exists, registering the
  β<1 model as confirmatory for A6's pass — A6 currently registers only a descriptive/exploratory
  use). Until then, A7r1(j)'s appended limitation applies: regime dependence stated as a
  limitation, no generalisation claim made — A7r3(e). Marked: proposal pending researcher
  confirmation.
- **D5r1 (scope):** original gate — can M1-CPU + x86-CPU finish in ~4 weeks? **Superseded** by the
  researcher's 2026-10-09 decision (M1-CPU in Paper 1, x86-GPU supplementary, no M1-GPU) —
  A7r2. Fallback if M1-CPU can't be completed: the original NO branch (Paper 1 narrows to x86 +
  RQ2/P-MAC + RQ1a; RQ3/RQ1b to Paper 2) — recorded now, not triggered by anything currently known.

## Session counts

x86-GPU: 6 (already collected). x86-CPU: 6. M1-CPU: 6. All three fixed, none open, no conditional
add/drop — A7r1(e)/A4's existing no-stopping-rule precedent.

**Each CPU-block session contains:** FP32-TS (baseline), FP32-eager (for D2r1), INT8, pruned30/50/70
(each paired with `_bnrecal` where meaningful), no FP16 (feasibility-ruled-out) — A7r1(e).
**Seeds:** x86-CPU `3000+N`, M1-CPU `4000+N` — A7r4(h), implemented in
`scripts/run_cpu_block_session.py`.

## Exclusion and void/rerun rules

**x86-GPU block exclusion rule (registered, A7r5(a)):** any condition whose measurement window
overlaps a verified battery/discharging interval is excluded, affected conditions listed and
disclosed, no session added or dropped. **Current coverage: zero conditions** — checked against
UPower history (2026-10-03 through 2026-10-10), the full boot journal, and all 270 conditions'
timeline — A7r5(b). Limitations: UPower sample spacing measured (median 60s within discharge
episodes, far wider during steady periods); only 4/30 conditions per session discriminate via GPU
regime; "expected-to-pin" is empirically self-referential; short blips could be missed — A7r5(c)
(whose second bullet's source attribution to "A7r1's Block M" is corrected by A7r7(d) — the actual
source is the 2026-10-10 integrity investigation).

**CPU-block void-and-rerun criteria (A7r3(h)):** AC guard abort → void entire session, rerun.
Crash → void that condition only (whole session if the cause implies contamination). Thermal-
throttle flag (x86: mean P-core freq < 80% of max despite `performance` governor — **marked
assumption, unverified**; M1: `pmset -g therm` above nominal or an uncalibrated clock-drop
threshold) → void that condition only. Interrupted session → void entire session. **Hard gate:** no
primary CPU-block session may start until the thermal-throttle guard exists, has unit tests, and
calibrated thresholds are registered in a dated amendment (calibration itself requires a
non-energy run) — A7r4(b).

**Rerun mechanics (A7r4(c)):** a voided condition is rerun appended at the end of the same
session's order, flagged; >2 voided conditions in a session voids the whole session; a rerun
session reuses the same planned seed.

## Specification curve

Axes: instrument, boundary, runtime, regime, gross/net — A7 point 9. **Per-block grid, with
availability** (A7r4(a)): x86-GPU has all 5 axes populated (instrument and boundary limited to
data collected after D24, runtime's TorchScript/CUDA-Graphs levels available only via series R —
see `stage5_analysis_plan.md` A7r4(a)'s table for the exact level counts per axis); x86-CPU has
instrument × runtime × gross/net available (2×2×2 = 8 points); M1-CPU has runtime × gross/net only
(2×2 = 4 points) — boundary and regime are not applicable/available on a CPU-only block. **Two
conditional axes, inactive until their data exist** — wall-meter boundary (after P1 is run) and
estimator (hardware/CodeCarbon-native/CodeCarbon-fallback, for the block RQ1b is collected on) —
A7r7(a). `SD_spec` on the 4- and 8-point CPU/M1 grids is descriptive only, even once populated —
A7r7(a).

**Headline metrics:** `headline_1 = SD_spec / w*` (primary interval half-width); `headline_2 =
SD_spec / max(|log r*|, ln 1.05)` (denominator floored at the ±5% margin, replacing A7r3(g)'s
undivided version) — A7r4(a). Both marked descriptive, no significance threshold attached.

## Analysis tooling

`analysis/rq2_model.R`, `analysis/helpers.R` (D2 TOST, SD threshold, bridged ratio, spec-curve
metrics, Holm adjustment — base R's `p.adjust`, not reimplemented), `analysis/simulate_rq2.R` +
`analysis/simulate_rq2_block_p.R` (simulation validation), `analysis/tests/` (unit tests) — written
and tested on synthetic data only, Blocks O-Q. Any post-freeze change to these scripts needs a
dated amendment, the same discipline as the prose plan — `docs/analysis_environment.md`.

## "Looked at" statement

Descriptive-only throughout: per-condition means in `session_log.md` files; the pinned-vs-unpinned
data-quality check (D19); the `_bnrecal`-vs-zero-finetune check (D18 — an **energy** comparison, not
accuracy-only as A7 point 13 originally said — corrected by A7r3(c)); two spot checks (pinned vs.
unpinned +0.6%, range −1.6% to +3.5%; `_bnrecal` vs. zero-finetune ~1% apart) **run by the
supervisor review, not the researcher** (corrected attribution, A7r3(c), fixing A7r1(g)'s
mislabel); and the 2026-10-10 integrity investigation's own look at per-session GPU power/regime
and `windows.csv`/`environment.json` schema — A7r3(c). **No ratio, interval, or hypothesis test has
been computed on any Stage 4b energy data at any point up to this summary.**

## Open prerequisites (not yet satisfied, as of this summary)

- A7 and every dated revision (A7r1-A7r7), A8 and its revisions (A8r1-A8r2) — all still **DRAFT**,
  pending researcher/supervisor confirmation.
- No wall meter chosen; P1 protocol exists in draft only, not run.
- No M1 hardware ever accessed; `docs/m1_check_commands.md`'s checks not yet run.
- Thermal-throttle guard: not implemented, no tests, no calibrated thresholds.
- Series R (A8): not collected — D2r1 cannot be evaluated for the GPU block without it.
- A5's fine-tune arm: drafted (`docs/a5_rewrite_draft.md`), not adopted or run.
- A6: not amended to activate D4r1.
- Freeze tag and Zenodo deposit: not created — CPU blocks' prospective-registration claim depends
  on this happening before the first CPU session (A7r4(f)).
- The curvature finding (A7r6(d)): mild, well-justified curvature in the simulated MAC-ratio
  relationship produced a large, persistent bias in the linear RQ2 model's β̂ with zero CI coverage
  — flagged as an open gap, no curvature check currently registered.

## Supersession table

| Older text | Amendment that replaces/corrects it |
|---|---|
| A7 point 2 (baseline = FP32-TS, all blocks) | A7r1(a) — GPU block only: baseline = FP32-eager |
| A7 point 6 (Holm over ~30 contrasts) | A7r1(b) — confirmatory family narrowed to P-CF (RQ2 H2, 2 tests) |
| A7 point 7 (RQ2 model, `(1\|model)` random) | A7r1(d) — model as fixed effect (P-MAC); further corrected by A7r3(f) — primary fit zero-finetune only |
| A7r1(f)'s "D1-D5" (reconstructed from memory) | A7r2 — corrected D1r1-D5r1, audit's actual wording; "D3"/"D4" renamed P-CF/P-MAC |
| A7r2's D2r1 practical note (TS sourced from series R directly) | A7r3(a) — bridged ratio is the only GPU-block construction (cross-session substitution unsafe per D17) |
| A7 point 13 / A7r1(g) ("looked at" statement, attribution + scope) | A7r3(c) — attribution corrected (supervisor, not researcher), battery-check's own look added |
| A7r3(g) (`headline_2`, undivided denominator) | A7r4(a) — denominator floored at `max(\|log r*\|, ln 1.05)` |
| A7r4(g) ("analysis environment: none of this exists yet") | A7r7(c) — environment now exists (R 4.3.3/lme4/lmerTest) |
| A7r5(c) (source mislabelled "A7r1's Block M") | A7r7(d) — actual source is the 2026-10-10 integrity investigation |
| A7r4(e) (Welch-Satterthwaite inputs, ambiguous) | A7r7(b) — clarified as variances of the means, not raw sample variances |
| §6(iii) tag-push tension (flagged, unresolved) | `a7_freeze_checklist.md` §8 — freeze tag is pushed by name; wildcard push commands never used |
