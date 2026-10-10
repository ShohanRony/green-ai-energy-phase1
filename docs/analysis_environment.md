# Analysis environment (R, for RQ2's mixed-effects test)

Recorded directly from the installed environment, not asserted from memory. Exact commands used:

```
$ which Rscript
/usr/bin/Rscript

$ Rscript -e 'cat(R.version.string, "\n"); cat("platform:", R.version$platform, "\n")'
R version 4.3.3 (2024-02-29)
platform: x86_64-pc-linux-gnu

$ Rscript -e 'cat("lme4:", as.character(packageVersion("lme4")), "\n")'
lme4: 1.1.35.1

$ Rscript -e 'cat("lmerTest:", as.character(packageVersion("lmerTest")), "\n")'
lmerTest: 3.1.3
```

- **R:** 4.3.3 (2024-02-29), platform `x86_64-pc-linux-gnu`.
- **lme4:** 1.1.35.1.
- **lmerTest:** 3.1.3.

Installed by the researcher directly (no sudo used by the coding assistant, per the standing
instruction). This satisfies the reproducibility-information requirement
`stage5_analysis_plan.md` A7r5(g) registered — the versions above are the required fields for the
freeze manifest once a freeze actually happens.

## Status of the analysis scripts in `analysis/`

**The scripts under `analysis/` are pre-freeze and have been tested only on synthetic data**
(`analysis/simulate_rq2.R` and the unit tests in `analysis/tests/`) — no real Stage 4b data has
been read, opened, or referenced by any script in this directory, consistent with this block's own
scope restriction. **They must run unchanged on real data** — the registered model specification
(`analysis/rq2_model.R`) implements `stage5_analysis_plan.md` A7r3(f) exactly as written, with no
option invented beyond what that entry specifies. **Any change to these scripts after the A7
freeze (`docs/a7_freeze_checklist.md`) requires a dated amendment**, the same append-only
discipline this project applies to the registered plan itself — a script is as much a part of the
registered analysis as the prose describing it, and silently editing it after freezing would defeat
the freeze's purpose.

## Simulation results (updated after each re-run; synthetic data only)

All figures below come from `analysis/simulate_rq2.R`, `analysis/simulate_rq2_block_p.R`, and
`analysis/simulate_rq2_diagnostics.R` (BLOCK R), 1000 replicates per scenario, seeds in the
`9000 + scenario_id` family. **This table describes the analysis code's own behaviour on
simulated data built from fixed, known-true parameters — not a finding about real Stage 4b
energy data, which none of these scripts has touched.** `stage5_analysis_plan.md` A7r6(d)
summarises the earlier (Block O/P) homogeneous, near-boundary, and single-gamma rows; this table
adds BLOCK R's curvature-sweep and LRT-diagnostic rows.

### Curvature sweep (`analysis/simulate_rq2_diagnostics.R`, scenario_id 200-203)

β_true = 0.8, session_sd = 0.04, residual_sd = 0.02 (matching the earlier single-gamma curvature
scenario for comparability). `blp_slope` is the population best-linear-projection slope the
primary (linear) model is actually estimating, computed over the real design points — this
equals β_true exactly when γ=0, and diverges from it as curvature increases.

| γ | BLP slope | bias vs. β_true | bias vs. BLP slope | coverage vs. β_true | coverage vs. BLP slope | rejection rate | quad. LRT detection | n_singular | diag n_singular |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.8000 | 0.0001 | 0.0001 | **0.951** | 0.951 | 1.000 | 0.066 | 0 | 0 |
| 0.01 | 0.7706 | **-0.0293** | 0.0001 | **0.000** | **0.950** | 1.000 | 0.219 | 0 | 0 |
| 0.03 | 0.7117 | **-0.0883** | 0.0000 | **0.000** | **0.964** | 1.000 | 0.905 | 1 | 0 |
| 0.06 | 0.6233 | **-0.1766** | 0.0000 | **0.000** | **0.992** | 1.000 | 1.000 | 1 | 0 |

**The core finding:** at every nonzero γ, `beta_hat` is essentially unbiased **against the BLP
slope** (bias ≤ 0.0001 in every row, coverage 0.950-0.992) while being badly biased **against
β_true** (up to -0.1766, zero coverage from γ=0.01 upward). The registered linear model correctly
estimates its own linear-projection estimand; it does not — and was never going to — recover the
underlying nonlinear truth's coefficient once real curvature exists. This is the basis for A7r8's
"not a power-law exponent" definition. **The quadratic LRT's detection rate at γ=0 is 0.066**,
somewhat above the nominal 0.05 (consistent with ordinary small-sample LRT behaviour, not a
severe miscalibration) — detection power rises quickly with γ (0.219 → 0.905 → 1.000).

### Heterogeneous slopes and homogeneous null (`analysis/simulate_rq2_diagnostics.R`, scenario_id 204-206)

γ = 0, session_sd = 0.04, residual_sd = 0.02. `beta_true` for the heterogeneous rows is the
simple pooled mean of the three per-architecture true slopes (matching A7r6(d)'s existing
convention) — **not** a BLP-style variance-weighted projection; that refinement was not computed
for these rows.

| Scenario | True per-arch. slopes | β_true (pooled mean) | bias | coverage | rejection rate | interaction LRT detection | n_singular | diag n_singular |
|---|---|---|---|---|---|---|---|---|
| Heterogeneous, pooled 0.8 | 0.6 / 0.8 / 1.0 | 0.8 | -0.0195 | 1.000 | 1.000 | **1.000** | 531 | 1 |
| Heterogeneous, pooled 1.0 | 0.9 / 1.0 / 1.1 | 1.0 | -0.0102 | 1.000 | **0.003** | **1.000** | 51 | 1 |
| Homogeneous null | 1.0 / 1.0 / 1.0 | 1.0 | 0.0000 | 0.948 | 0.053 | 0.075 | 1 | 1 |

**The interaction LRT reliably detects both heterogeneous cases (100% of replicates)**, including
the one where the primary H2 test itself is barely powered (0.3% rejection rate for the
pooled-1.0 case) — confirming the diagnostic adds real information the primary test's p-value
alone doesn't carry. **In the homogeneous null, the interaction LRT fires at 0.075**, somewhat
above nominal 0.05 — the same mild-inflation pattern as the quadratic LRT at γ=0, not a separate
issue. The primary model's own type I error in the homogeneous null (0.053) and coverage (0.948)
both land close to nominal, consistent with A7r6(d)'s earlier finding.
