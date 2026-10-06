# Stage 4b design — multi-session replication (DRAFT, NOT EXECUTED)

**Status: drafted 2026-10-06 at the supervisor's request, for review only. No part of this design
has been run.** It exists to close the gap D17 found: a single later rerun of 12 conditions differed
from the primary matrix by -12.20% to +3.70% on `gross_j_per_image_mean`, confounded with run order
(one fixed sequence, run strictly after the primary matrix) — so it's not yet known whether that gap
is genuine between-session variability, an order/practice effect, or both. This design is built to
separate those.

## 1. Why a redesign, not just "run it again"

- D14: reps within one session are serial, not independent — already registered as a §4 limitation,
  handled by treating effect sizes as primary and p-values as within-session-descriptive only.
- D17: a second source of variance exists *between* sessions, and the one comparison available
  (dip-stability check vs. primary matrix) can't separate it from order, because there was only one
  rerun and it always ran later, in one fixed order.
- Consequence: the primary Stage 4 matrix is **one session's worth of data** for every one of its 18
  (now 21, with D13) conditions. Nothing in the data collected so far tells us how that one session's
  numbers would vary if collected again, in a different session, in a different order.

## 2. Design

- **≥3 fresh-boot sessions.** Each session: reboot, confirm `uptime` genuinely low before starting
  (same discipline as the primary matrix's own Task 0), then run.
- **Each session runs all 21 conditions** (18 compression states + the 3 D13 FP32-CPU baselines) **in
  a freshly randomised order, different per session** — directly targets D17's order confound. The
  random permutation (and the seed used to draw it) is logged in that session's own run metadata, not
  reused across sessions.
- **~10 reps per condition per session** (not 31) — deliberately smaller than the primary matrix's
  per-condition rep count, because the replicate unit here is the **session**, not the rep. 10 reps
  is enough to get a stable within-session mean per condition (the primary matrix's own `pairs>=3`
  floor for a usable `summarize()` row is far below 10); the statistical power this design buys comes
  from ≥3 independent sessions, not from more reps inside any one of them.
- **Session is the replicate unit, stated explicitly so no one reads this as a 3×10=30-rep design
  with the same statistical properties as the primary matrix's 31-rep design.** It is not — it trades
  within-session rep count for between-session replication, which is the thing D17 found was never
  measured.

## 3. Three energy boundaries

Using item 3's harness patch (`--concurrent-cpu-package`, `pilot.py` commit `b66349d`):

- **GPU-only** (`energy_j`, NVML `nvmlDeviceGetPowerUsage`): the primary matrix's existing boundary,
  for the 15 GPU-measured conditions (fp32/fp16/pruned30/50/70 × 3 models).
- **CPU package** (`cpu_package_energy_j` on GPU runs; `energy_j` on the 6 CPU-measured conditions
  INT8×3 + D13 FP32-CPU×3, now package-0-only per D16): CPU-side draw concurrent with a GPU run, or
  the sole reading for CPU-measured conditions.
- **GPU+CPU** (sum of the above): the closest this harness gets to whole-system energy for a GPU
  run, still excluding RAM/peripherals/other rails (no claim of true whole-system energy).

**The 6 CPU-measured conditions only have one real boundary (CPU package) — "GPU-only" is undefined
for them (no GPU used) and "GPU+CPU" reduces to the same CPU package number.** The three-boundary
design applies fully only to the 15 GPU-measured conditions; stated here so the eventual results
table doesn't imply three independent readings exist where only one does.

## 4. Analysis

- **Primary: per-session compressed/baseline energy ratios, with between-session CIs.** For each
  session and each compressed state, compute that session's own compressed-vs-baseline energy ratio
  (using that session's own ~10 reps for both, same-session so the ratio isn't cross-session-
  contaminated). Across the ≥3 sessions, report the mean ratio and a between-session CI (bootstrap
  over the session-level ratios, session as the resampling unit — not over individual reps, which
  would understate the real uncertainty by treating session as a fixed effect).
- **Mixed model, session as a random effect.** A linear mixed-effects model on (log-)energy, fixed
  effect = compression state, random intercept (and, if the data supports it, random slope) for
  session. This is the model that actually accounts for both variance sources at once — within-
  session serial correlation (D14) and between-session variance (D17) — rather than either ignoring
  between-session variance (as a naive pooled-rep analysis would) or ignoring within-session structure
  (as treating each session as a single point estimate with no uncertainty would).
- **Mann-Whitney U kept only as a within-session descriptive check** — e.g. "did this state's reps
  clearly separate from baseline's reps, within this one session" — not as the cross-session
  confirmatory test. This follows directly from D14 (independent-reps reasoning holds within a
  session) and D17 (that reasoning doesn't extend across sessions without the mixed-model treatment
  above).
- Each of the three energy boundaries (§3) gets this same analysis independently — they are not
  pooled into one number.

## 5. Wall-time estimate, from real timestamps

Per-rep time is consistently ~32-34 s/rep across every condition type measured so far in this
project (GPU and CPU alike, with or without CodeCarbon, with or without the item-3 patch) —
confirmed repeatedly: the primary matrix's own 16.6-17.5 min at 31 reps (≈32.1-33.9 s/rep), the
CodeCarbon feasibility checks (32.67 s/rep, twice), and the item-3 validation runs (same range).

- Per condition at ~10 reps: ~10 × 33 s + a few seconds of fixed setup (checkpoint/dataset load) ≈
  **~5.5-6 min/condition**.
- Per session (21 conditions): 21 × ~5.5-6 min ≈ **~116-126 min (≈1.9-2.1 hours)** of actual
  measurement, plus session-start overhead (reboot, `uptime` verification, confirming the random
  order was drawn and logged) — call it **~2.0-2.3 hours/session** all-in.
- **3 sessions: ~6.0-6.9 hours of measurement time total**, necessarily spread across ≥3 separate
  reboots (likely separate work sessions on different days, matching how the primary matrix's own
  sessions were spaced).

Not executed. Awaiting approval.
