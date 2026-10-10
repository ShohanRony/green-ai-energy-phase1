# Stage 5 analysis plan, v1.0

**Status: v1.0 DRAFT, awaiting researcher confirmation.** Becomes authoritative only through a
dated confirmation entry appended to `docs/stage5_analysis_plan.md` (amendment A10). From that
date this document is the single authoritative plan. Amendments A1-A9, A7r1-A7r9 and A8-A8r2
remain in `docs/stage5_analysis_plan.md` as history. Where this document and an earlier amendment
differ, this document controls from the confirmation date.

## 1. Scope

| Block | Role | Status |
|---|---|---|
| x86-CPU (i5-13450HX) | **Single confirmatory block** | Not collected |
| M1-CPU (MacBook Air M1) | Registered replication: identical analysis, reported whether or not it replicates; never blocks Paper 1 | Not collected |
| x86-GPU (RTX 3050 Laptop) | Supplementary | Six sessions collected 2026-10-07 to 2026-10-09, before this plan; **not prospectively registered** |

No M1-GPU block. Comparisons are within block only; cross-block comparisons are descriptive.

**Prospective registration:** the two CPU blocks count as prospectively registered only if the
freeze tag and the Zenodo deposit exist before the first CPU-block session is collected.

**Deferred or not adopted (no data, no claims):** series R (A8 family); A5 fine-tune arm; A6
224x224 pass; RQ1b (CodeCarbon fallback mode, to Paper 2); P8 checkpoints (not used for any
claim).

## 2. Research questions

- **RQ1a (coverage):** under steady loads, how much of the wall-meter-derived energy change does
  each in-device instrument or boundary capture (NVML power-usage, NVML cumulative, RAPL
  package-0, RAPL psys, CodeCarbon in its normal hardware-counter mode, logged during the same
  P1 loads)? Descriptive, Bland-Altman style, via the P1 protocol
  (`docs/p1_wall_meter_protocol.md`). No extra session.
- **RQ1b (estimation mode): deferred to Paper 2.** CodeCarbon's fallback estimation mode is not
  collected in this study.
- **RQ2 / H2 (confirmatory):** MAC reduction overstates the energy saving of structured pruning.
  Test in section 4.
- **RQ3 (secondary, not confirmatory):** does the effect of compression state on energy differ
  between x86-CPU and M1-CPU? Test in section 6. Evaluated only once M1-CPU data exist; if they do
  not exist when Paper 1 is ready, RQ3 moves to Paper 2. Paper 1 never waits for it.

## 3. Measurement

- **Outcome:** gross energy per image (J), batch 1, 32x32 inputs. Net-of-idle is a sensitivity
  axis (section 7).
- **Boundary:** x86-CPU = RAPL package-0 (psys logged, secondary). M1-CPU = `powermetrics` CPU
  power per `docs/m1_harness_design.md`. x86-GPU = NVML power-usage + RAPL package-0 (system),
  with GPU-only and CPU-package-only as secondary boundaries.
- **Per condition per session:** mean of warm repetitions (the first, cold repetition is always
  excluded). **Ratio** r = J/image(state) / J/image(block baseline), same session.
- **Baselines:** CPU blocks = FP32-TorchScript, measured in every session. x86-GPU = FP32-eager
  (the only baseline collected); every GPU ratio carries the eager-baseline caveat.
- **Conditions in every CPU-block session:** FP32-TorchScript (baseline), FP32-eager (for D2r1),
  INT8 (x86: fbgemm; M1: qnnpack), pruned30/50/70 zero-finetune, and their `_bnrecal` versions
  except the combinations excluded as known-collapsed in `scripts/run_cpu_block_session.py`
  (`BNRECAL_SKIP`). No FP16 on CPU. Harness settings and repetitions as implemented in
  `scripts/run_cpu_block_session.py` at the freeze commit.
- **Sessions:** six per block, fixed; no stopping rule; no session added or dropped based on
  results. Session is the replicate unit. Fresh boot per session; random condition order per
  session from seed **3000+N** (x86-CPU) and **4000+N** (M1-CPU). (x86-GPU used 1000+N.)
- **Interpreter:** every Python test and run uses `measurement_env/bin/python3` (D28).

## 4. Confirmatory test (RQ2, H2)

- **Quantity:** for each of the 9 zero-finetune pruned cells (ResNet-18, MobileNetV3-Small,
  EfficientNet-B0 x pruned30/50/70) in each session, d = log(r) - log(MAC ratio), using the
  realised MAC ratios from Stage 2. d > 0 means energy fell by less than MACs fell.
- **Primary test:** average the 9 d values within each session; one-sample t-test across the 6
  session means, df = 5, one-sided, H2: mean d > 0, alpha = 0.05. The only confirmatory test; no
  multiplicity adjustment.
- **Completeness rule:** the primary test requires all 9 cells in all 6 sessions after the
  void/rerun procedure (section 5). If any cell is missing, the primary becomes the cell-fixed
  mixed model `d ~ 0 + cell + (1|session) + (1|session:model)` (REML, Satterthwaite), tested on
  the equal-weight contrast of the 9 cell means. The trigger is data completeness only.
  Implemented in `analysis/rq2_gap_model.R` (`fit_h2_primary`).
- **Headline:** exp(mean d) with its 95% CI, stated as "MAC counts overstate the energy saving by
  a factor of X". Per-cell d with 95% CIs, descriptive, no p-values.
- **A non-rejection is reported as inconclusive**, never as "energy is proportional to MACs".
  Simulation (synthetic data): at a true mean d of 0.03 the test detects the effect about 45% of
  the time with 6 sessions.
- **Cross-check, always reported:** the cell-fixed mixed model above. Its singular or
  non-converged flags are reported and never change the primary result. The old non-convergence
  fallback (A7r6(b), A7r9(a)) no longer applies to the confirmatory test.
- **Secondary, descriptive:** anchored per-architecture slopes through the FP32 anchor (known to
  be biased under an energy floor).
- **Sensitivity:** the free-intercept slope model `log r ~ log_mac + model + (1|session) +
  (1|session:model)`, the `_bnrecal` cells as a separate fit, and the A7r8 curvature and
  heterogeneity diagnostics. Framing: the pooled slope is a Frisch-Waugh-weighted average of the
  per-architecture slopes (weights ResNet-18 0.4088, EfficientNet-B0 0.3100, MobileNetV3-Small
  0.2812); slope heterogeneity causes power loss, not type I error inflation.

## 5. Void, rerun and exclusion rules

- **Hard gate:** no CPU-block session starts until the thermal-throttle guard exists, has unit
  tests, and its calibrated thresholds are registered in a dated amendment. The x86 candidate rule
  (mean P-core frequency below 80% of maximum under the `performance` governor) is an unverified
  assumption until calibrated in a non-energy run. M1: `pmset -g therm` above nominal, threshold
  to be calibrated the same way.
- **AC guard abort or interrupted session:** void the whole session; rerun it with the same seed.
- **Crash or thermal flag:** void that condition only (the whole session if the cause implies
  contamination); rerun it at the end of the same session's order, flagged.
- **More than 2 voided conditions in a session:** void the whole session.
- **Exclusions (counted and reported):** the cold-start repetition; plausibility-guard trips;
  governor not `performance`; concurrent GPU process; M1 runs that swapped memory. x86-GPU:
  conditions overlapping a verified battery interval (currently zero, A7r5).

## 6. Secondary analyses

- **RQ3:** `log r ~ state * platform + (1|session)` over the CPU states measured on both
  platforms; the state x platform interaction is the test (likelihood-ratio test). Kendall's tau
  and pairwise rank reversals are descriptive. The INT8 backend difference (fbgemm vs qnnpack) is
  a stated confound.
- **Per-state ratios (all blocks):** geometric mean over sessions of r, 95% t-interval on log r,
  df = sessions - 1. Estimation only, no p-values. Mann-Whitney U only as within-session
  description.

## 7. Decision rules

- **D1r1 (wall meter):** does the change in NVML power-usage + RAPL package-0 agree with the
  wall-derived change within +/-5%? Yes: keep the backend and report it as validation. No, and
  the cumulative counter agrees: GPU numbers need re-collection with both APIs logged (only data
  after D24 carry both). No, and neither agrees: wall energy becomes primary for system-level
  claims. Also pre-registered: regress NVML cumulative on NVML power-usage across load levels
  (slope and intercept), to separate an additive offset from a scaling factor. Unresolved if the
  CI exceeds +/-5%. Not evaluable on the x86-GPU sessions (pre-D24).
- **D2r1 (runtime, CPU blocks):** TOST of FP32-TorchScript vs FP32-eager per architecture,
  margin +/-5%, df = 5 (SD threshold 0.0593). Reported as bounded or unbounded. The CPU baseline
  stays FP32-TorchScript regardless. x86-GPU: not evaluable (series R deferred).
- **D3r1 (weight dependence):** are `_bnrecal` and zero-finetune energies within +/-5% for the
  measured combinations? Yes only if all are. Informs discussion only; never substitutes for
  measuring any other model.
- **D4r1 (input size):** inactive (A6 not adopted). Results are not generalised beyond 32x32,
  batch 1; stated as a limitation.

## 8. Specification curve (descriptive)

Axes, as available per block: x86-CPU = instrument (package-0, psys) x runtime (TorchScript,
eager) x gross/net (8 points); M1-CPU = runtime x gross/net (4 points); x86-GPU = boundary
(system, GPU-only, CPU-package-only) x gross/net (6 points; instrument and runtime axes are not
available). The wall-meter boundary is added once P1 data exist. Metrics: headline_1 = SD_spec /
w* and headline_2 = SD_spec / max(|log r*|, ln 1.05), both descriptive.

## 9. Accuracy rule (unchanged from section 6 of the original plan)

A compressed state is deployable if its top-1 accuracy on the CIFAR-10 test set (n = 10,000) is
at least 99% of its own model's FP32 accuracy; the strict tier is 99.9%. Report 95% Wilson
intervals and McNemar's test against the same model's FP32 predictions; label a state
"borderline" if its interval straddles the threshold. Every state appears on the Pareto plot.
Single training seed (2026), disclosed. Separately, `_bnrecal` CPU-vs-CUDA functional
equivalence uses McNemar not significant at 0.05 and |accuracy difference| <= 0.05 percentage
points (all 9 pass); this is not the deployability rule.

## 10. "Looked at" statement

Up to this plan's confirmation, no ratio, interval or hypothesis test has been computed on any
Stage 4b energy data. Viewed descriptively only: per-condition means in `session_log.md` files,
the pinned-vs-unpinned check (D19), the `_bnrecal`-vs-zero-finetune spot check (D18), and the
2026-10-10 integrity investigation (GPU power regime, file schema). No CPU-block or M1 data
exist. This statement is re-checked at the freeze commit.

## 11. Analysis environment and code

R 4.3.3, lme4 1.1.35.1, lmerTest 3.1.3; Python per `measurement_env`. Code in `analysis/`,
validated on synthetic data only (reproducible: `analysis/simulate_h2_gap.R` regenerates
`analysis/sim_output/summary_h2_gap.csv` byte-identically). Any change after the freeze needs a
dated amendment. The manuscript states the use of AI coding assistants in its Methods.

## 12. What this plan supersedes

| Earlier rule | Replaced by |
|---|---|
| Confirmatory family of two H2 slope tests with Holm (A7r1(b)) | One test on x86-CPU (section 4) |
| RQ2 slope model `log r ~ log_mac + model + (1|session)`, beta < 1 (A7r3(f)) | Gap test on d (section 4); slope model kept as sensitivity with `(1|session:model)` added |
| Non-convergence fallback for the primary (A7r6(b), A7r9(a)) | Session-mean t with the completeness rule (section 4) |
| M1-CPU as a confirmatory block (A7 point 1) | Registered replication (section 1) |
| GPU bridged ratio via series R (A7r3(a), A8 family) | Deferred; GPU ratios use the eager baseline with caveat |
| D5r1 scope gate | Removed; scope fixed in section 1 |
| RQ1b as part of this study (A7 point 10) | Deferred to Paper 2 (section 2) |
| "Type I error far below nominal" (A7r6(d), A7r8(d)) | Power loss under slope heterogeneity (section 4) |
| "Bias" under heterogeneous slopes (A7r6(d)) | Frisch-Waugh design weighting (A7r9(b), section 4) |
| bnrecal exact-count equivalence rule | Tolerance rule (section 9) |
