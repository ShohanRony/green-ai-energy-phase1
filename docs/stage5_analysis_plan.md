# Stage 5 Analysis Plan (registered before statistical analysis)

**Status:** Registered after x86 Stage 4 data collection and before any statistical analysis.
Per-condition summary values were viewed during collection for quality control only (checking
`pairs`, `power_regime`, and plausibility — never a hypothesis test). M1 data not yet collected.

**Note on §4's statistical methods vs. the proposal's own §3 — RESOLVED 2026-10-06:** this plan
originally registered Welch's t-test with bootstrap CIs and Holm-Bonferroni correction as the primary
methods, diverging from the proposal's own §3 (Wilcoxon signed-rank, Spearman, Kendall's tau). Per
the researcher's explicit instruction, this is now reversed: §4 registers the proposal's tests as
primary (Mann-Whitney U substituted for Wilcoxon signed-rank, with Spearman/Kendall's tau caveated
for pooling non-independence and low power), and the Welch's/bootstrap/Holm-Bonferroni family as
supplementary. Logged as `deviation_log.md` D14.

---

## 1. Research questions and hypotheses

Copied verbatim from the proposal (`Proposal_GreenAI_taught.NEW.docx` §1).

**RQ1 — x86-only analysis, extends to cross-platform after M1 data.** "For post-training compressed
vision models, how closely does software-based energy estimation agree with hardware-level measurement,
and does the agreement degrade as numerical precision decreases from FP32 to INT8 and INT4?"

**RQ2 — x86-only analysis, extends to cross-platform after M1 data.** "Does FLOPs reduction from
post-training quantization and structured pruning produce proportional reductions in hardware-measured
energy, and where it does not, is the divergence consistent with a shift from compute-bound to
memory-bound execution?"

**RQ3 — cross-platform, analysed after M1 data collection.** "Is the rank ordering of compression
methods by measured energy efficiency preserved between x86 and ARM platforms, or is the ordering
architecture-dependent?" **Cross-platform questions are not limitations of this plan — RQ3 cannot be
answered at all without M1 data, by the proposal's own design, not due to any scope reduction made
here.**

**Hypotheses, verbatim:** "The corresponding hypotheses are that estimator error will increase at lower
precision, that FLOPs reduction will overstate energy savings, and that rank ordering will not be fully
preserved across architectures. A fourth expectation, held more loosely, is that faster compressed
models may avoid sustained-load thermal throttling that slower uncompressed models trigger on
passively cooled hardware, which would produce an energy advantage invisible to any operation-count
analysis."

**Known deviation already affecting RQ1/RQ2 as currently collected (see `deviation_log.md` D1):** the
proposal's RQ1 asks about degradation "from FP32 to INT8 and INT4" — this project's sixth state is FP16,
not INT4 (D1). RQ1's INT4 clause cannot be answered from this project's data as collected; RQ1 is
answerable for FP32→INT8 only, with FP16 reported as a disclosed substitute state outside RQ1's original
wording.

## 2. Datasets

- **Primary (confirmatory), x86:** Stage 4, batch=1, 18 conditions (3 models × 6 states) plus 3
  FP32-CPU baselines (D13 — collected 2026-10-06, `results_stage4/{arch}_fp32_cpu/`), 30 pairs each.
  All 21 primary x86 conditions now collected.
- **Primary (confirmatory), M1:** Stage 4-M1 (§10), same conditions. Not yet collected.
- **Secondary (sensitivity):** batch=16 runs on both platforms (Part B item 2 for x86 — not yet
  collected; M1 batch=16 timing TBD, follows Stage 4-M1).
- **Excluded from confirmatory analysis:** Stage 1-3 and pre-flight data (`deviation_log.md` D12) —
  reported as pilot only, never mixed into RQ1-RQ3 statistics.

## 3. Outcomes

- **Primary:** energy per image (J), **gross** (idle time between launches at batch=1 is a real
  deployment cost, not measurement overhead to subtract away). x86 GPU states (FP32/FP16/pruned30/50/70)
  use NVML; x86 INT8 and the new FP32-CPU baseline (D13) use RAPL package; M1 uses `powermetrics`.
  **Never compare energy across different instruments directly** — every cross-instrument comparison in
  this project (GPU vs. CPU, x86 vs. M1) must be stated with its instrument boundary, per the
  pre-flight checklist discipline carried from Stage 1.
  **Confirmed directly against `pilot.py`'s `Sensor` class (2026-10-06):** a run's energy figure comes
  from exactly one backend, chosen by `--device` — RAPL package counters for `cpu`, NVML
  `nvmlDeviceGetPowerUsage` (sampled power, trapezoidal-integrated) for `cuda`. There is no combined
  RAPL+NVML reading anywhere in this harness; a GPU run never includes CPU/RAM power, a CPU run never
  includes GPU power. This is a deliberate single-sensor-per-run design, not an oversight — it's what
  makes the "never compare across instruments" rule necessary in the first place.
- **Secondary:** net-of-idle energy, latency, accuracy (from Stage 2), realised FLOPs/params reduction
  (from Stage 2's `torch-pruning` dependency-graph report, not nominal sparsity), power regime.
- **RQ1 instrument-agreement check:** CodeCarbon vs. hardware counters, reported as mean difference and
  ratio with 95% CI, plus Bland-Altman limits of agreement — **contingent on CodeCarbon data existing.**
  Checked directly (Part B item 4): `codecarbon` was `False` in every Stage 4 run's logged `arguments`
  field — **no paired CodeCarbon readings exist in the Stage 4 data collected so far.** This is a real
  gap against the proposal's explicit requirement (§3: "CodeCarbon runs concurrently with hardware
  measurement on every run, which is what makes the RQ1 comparison a within-run paired comparison") —
  flagged here and in Part B's report, not silently worked around. RQ1 cannot be answered from the
  current x86 dataset until CodeCarbon-paired runs exist.

## 4. Comparisons and statistics

**Resolved 2026-10-06 (`deviation_log.md` D14): the proposal's own tests are primary.** Each
compressed state vs. its own model's **same-instrument** baseline: x86 GPU states vs. FP32-GPU; INT8
vs. FP32-CPU (D13); M1 states vs. M1 FP32.

**Primary (proposal-specified):**
- **Mann-Whitney U (Wilcoxon rank-sum), substituted for the proposal's Wilcoxon signed-rank test.**
  The signed-rank test requires paired observations (e.g., the same unit measured twice, or
  deliberately matched pairs); Stage 4's reps within a condition are independent repeated
  measurements of the same static configuration, not pairs matched to anything in the baseline
  condition — there is no natural 1:1 pairing between compressed-state rep *i* and baseline rep *i*,
  they're just two independent samples of 30 (pairs=30 after the cold-start discard) draws each.
  Applying the signed-rank test to unpaired data would assume a pairing structure that doesn't exist
  and would be invalid. Mann-Whitney U is the correct rank-based test for two independent samples and
  is the closest proposal-faithful substitute — same family (distribution-free, rank-based), same
  comparison target (compressed vs. baseline), different (correct) assumption about independence.
- **Spearman rank correlation (RQ2)** and **Kendall's tau (RQ3)**, as specified in the proposal.
  **Caveat, stated here because it affects how much weight these results can carry:** both are
  computed by pooling across a model's 6 compression states, but those 6 states are not independent
  draws — they're the same underlying checkpoint's weights measured under different, deliberately
  related post-training transformations (pruning ratios are nested, quantization/precision states
  share the same architecture). Treating 6 non-independent points as if they were 6 independent
  samples overstates the effective degrees of freedom. Separately, n=6 per model gives low
  statistical power regardless of independence — a non-significant result here is as likely to
  reflect insufficient n as a true null. Both correlations are reported, but as descriptive signals
  bounded by these two caveats, not as confirmatory hypothesis tests.

**Supplementary (this plan's original registration, kept for triangulation):**
- Welch's t-test on per-rep energy; effect size = energy ratio (compressed / baseline) with bootstrap
  95% CI (10,000 resamples, percentile method); Holm-Bonferroni correction across the full family of
  compressed-vs-baseline tests, per platform.
- Per condition: mean, SD, 95% bootstrap CI, reported regardless of which test family is read as
  primary.

**Did the proposal intend the signed-rank test, and would it be valid here? Intended: yes — proposal
§3 names it explicitly for paired compression-state comparisons. Valid for this data: no.** The
signed-rank test's validity rests on a real pairing between the two samples being compared (matched
units, or the same unit under two conditions). Stage 4's design produces two independent batches of
reps per comparison, with no shared identity between a compressed-state rep and a baseline rep beyond
both being draws from the same measurement protocol. Running the signed-rank test on this data would
not just be a weaker choice than Mann-Whitney U — it would misrepresent the data's actual structure.
This is why Mann-Whitney U, not the signed-rank test, is registered as primary above.

**Limitation, added 2026-10-06: reps are serial, not independent in the statistical sense, and
conditions were run in a fixed sequence, not randomized or interleaved.** Within a condition, the 30-31
reps are back-to-back repeated measurements on one continuously-running process — not independent
draws from a population, just repeated samples of one static configuration over a few minutes. Across
conditions, the real run order (reconstructed from each `environment.json`'s `timestamp_utc`/`uptime_s`,
2026-10-06) was:
- **Session 1** (fresh boot ~05:00 UTC): resnet18 fp32 → int8 → fp16 → pruned30 → pruned50 (5
  conditions).
- **Reboot.** **Session 2** (fresh boot ~06:45 UTC): resnet18 pruned70, then mobilenet_v3_small
  (fp32 → int8 → fp16 → pruned30 → pruned50 → pruned70), then efficientnet_b0 (fp32 → int8 → fp16 →
  pruned30 → pruned50 → pruned70) — 13 conditions, one continuous session, no reboot in between.
- **Reboot.** **Session 3** (fresh boot ~12:55 UTC): the 3 D13 FP32-CPU baselines, in order
  resnet18 → mobilenet_v3_small → efficientnet_b0.

Within each model, INT8 was run second (right after FP32), not last as its position in the state list
might suggest — the actual dispatch order interleaves CPU and GPU runs. No condition was repeated at a
different point in the sequence to check for order effects (session, thermal drift, or position-in-
sequence confounds), except where D7/D8's dip-stability work already did so for the power-regime
question specifically. **Consequence for §4's tests:** a p-value from Mann-Whitney U (or any test here)
describes repeatability of two specific back-to-back runs in these specific sessions — it is not a
claim about repeatability across arbitrary sessions, times of day, or thermal states. This is why the
effect sizes (energy ratios with bootstrap CIs) lead the reporting and the p-values are read as
descriptive of within-session repeatability, not as the primary evidentiary claim.

## 5. Core thesis test (FLOPs vs. energy)

Decision rule fixed now, before any data is analyzed against it: for each state, compare the measured
energy ratio with the realised FLOPs ratio (from Stage 2's `torch-pruning` report, not nominal sparsity —
consistent with this project's standing rule that nominal compression is never reported without its
realized counterpart). If the energy ratio's 95% CI excludes the FLOPs ratio, conclude "energy savings
not proportional to FLOPs reduction" for that state. Across states, report Spearman correlation
descriptively only — small n, no strong inference drawn from it.

**What "FLOPs ratio" means for FP16/INT8, fixed before any analysis (added 2026-10-06):** the
`torch-pruning` dependency-graph report counts MACs, which are identical for FP16/INT8 states and
their FP32 baseline — same architecture, same layer shapes, only the numeric representation changes.
A plain MACs ratio is therefore 1.0 for every precision state by construction, which cannot capture
the compute/memory-traffic change that precision reduction actually causes. **Proposed, labelled
measure, not a silent swap-in:** a bit-width-adjusted compute measure, BOPs (bit-operations) =
MACs × bits(weights) × bits(activations), following the usage of this metric in the quantization
literature. For FP32 (32×32), FP16 (16×16), INT8 (8×8), this gives nominal BOPs ratios of 0.25 and
0.0625 respectively relative to FP32, vs. 1.0 under plain MACs. **Both are reported, explicitly
labelled** ("MACs ratio" vs. "BOPs ratio") — the pruning states' realised-FLOPs comparison stays on
plain MACs (where it correctly varies with sparsity), and the precision states' comparison additionally
reports the BOPs ratio so §5's exclusion test has a non-trivial target for FP16/INT8, not a vacuous 1.0.

**Reconciling this section's bootstrap CI with §4's "supplementary" label (added 2026-10-06):** §5's
energy-ratio-vs-FLOPs-ratio exclusion test uses the bootstrap 95% CI, which §4 now registers as
supplementary (Welch's/bootstrap/Holm-Bonferroni family), not primary. This is not an inconsistency:
§5's test is a ratio-exclusion test, which has no direct analog in the Mann-Whitney/Spearman/Kendall
primary family — there is no "primary" version of this specific test to prefer instead. It draws from
the supplementary toolkit by design, because that is where this project's bootstrap-CI machinery
already lives, not because §5 is treated as less rigorous than §4's primary tests.

## 6. Accuracy rule — PENDING Shohan's explicit confirmation

**Status: proposed 2026-10-06, not yet confirmed. No Stage 5 statistics are run before Shohan
explicitly confirms (a)'s 99% primary threshold.**

**(a) PRIMARY — MLPerf convention.** A compressed state is deployable if its top-1 accuracy on the
CIFAR-10 test set (n=10,000) is **≥99% of its own model's FP32 top-1 on the same test set**. This is
the MLPerf Inference accuracy-tier convention for the "closed division."
`TODO — not found in records: exact MLPerf Power paper citation. No project reference-library file
exists in this repo, and the external drive hosting the "scholarship" library is not mounted on this
machine as of this entry — add the full citation once available.` **Disclosed:** MLPerf's 99%/99.9%
tiers were defined for ImageNet/BERT-scale classification and language tasks; this project borrows the
convention for CIFAR-10-scale models, which is a disclosed convention transplant, not a threshold
independently validated for this task's scale or difficulty.

**(b) STRICT tier.** Also report, for every state, whether it clears **99.9%** of FP32 top-1 — labelled
separately from (a)'s primary tier, not merged into it.

**(c) No author-chosen percentage-point tolerance.** The earlier `[TODO — Shohan to set, e.g. 1-2
percentage points]` placeholder is removed. Only (a) and (b)'s MLPerf-convention thresholds are used.

**(d) Reporting, per state:**
- 95% **Wilson CI** on top-1 accuracy (n=10,000, the CIFAR-10 test-set size) — Wilson chosen over the
  normal approximation because several states (D4's collapsed 70%/50% pruning states) sit near
  CIFAR-10's 10% chance floor, where the normal approximation breaks down.
- **McNemar's test** against the same model's FP32 baseline, paired by test-set image index — this is
  the one accuracy comparison in this project where a genuine pairing exists (same images, same
  model family, two checkpoints), unlike the independent-reps energy comparisons in §4.
- A state whose Wilson CI straddles the (a) threshold is labelled **"borderline,"** not forced into
  deployable/not-deployable.

**(e) Pareto plot.** Accuracy vs. energy plotted for **all 18 x86 states** (and, later, all M1 states),
regardless of (a)/(b)'s outcome. The threshold affects only point *labels* (deployable / borderline /
not-deployable / strict-tier) — no state is ever removed from the plot.

**(f) Training-seed disclosure, confirmed from Stage 2 records (2026-10-06):** a single fixed seed,
**2026**, was used throughout this project — `train_baseline.py` calls `torch.manual_seed(2026)` and
`random.seed(2026)`; every checkpoint's provenance metadata and `stage2_task_report.md`/
`stage2_deliverable.md` record "seed 2026." There is **no multi-seed replication** anywhere in this
project — one trained/derived checkpoint per architecture × state, not an average over independent
training runs. **Consequence:** the Wilson CI and McNemar test in (d) capture test-set sampling
variability only; they say nothing about training-run-to-run variability, which was never measured and
cannot be recovered from the data as collected.

**(g) Final.** This rule is final once (a)'s 99% primary threshold is explicitly confirmed by Shohan.
Any later change is a dated amendment under §12, not an in-place edit — see A1.

The collapsed 70% pruning state (and, for MobileNetV3-Small and — under the retrained 60-epoch
baseline — EfficientNet-B0, the 50% state too; see `deviation_log.md` D4) is expected to fail (a) by a
wide margin once this rule is confirmed and run, given its near-chance accuracy.

## 7. Power-regime handling (x86)

- Every condition reports its regime (`power_regime`: `pinned`/`dip`/`mixed`, logged automatically by
  `pilot.py` since the Stage 4 pre-flight guard was added — `deviation_log.md` D7).
- A `dip` condition is included in the primary analysis only if it passes the stability check: same
  regime in a separate fresh session, and steady low utilisation in an `nvidia-smi dmon` log (Part B
  item 3 — not yet executed as of this plan's registration).
- Any condition that is `mixed`, or unstable across sessions: rerun once; if still unstable, report it
  separately and exclude it from the confirmatory tests.
- **RTX 3050 regime findings are not assumed to transfer to M1** — M1 gets its own stability check
  (§10), not an inherited assumption from the x86 investigation.

## 8. Exclusions

Only these, nothing decided later:

- Plausibility-guard trips (`plausibility_flag=True`) — zero so far across all Stage 1-4/pre-flight data
  collected to date (`deviation_log.md` D9).
- Governor not `performance` (x86) — one historical instance (D9's governor-contaminated rerun,
  pre-flight only, already excluded and redone).
- Concurrent GPU process detected — enforced by `pilot.py`'s existing guard; zero trips recorded.
- Cold-start rep — one per condition, by design (18 across Stage 4's matrix, D9).
- M1 runs that swapped memory — not yet applicable, M1 data not yet collected; `vm_stat` monitoring
  specified in §10.

Every exclusion is listed here with its count; this list is updated, not silently expanded, as more
data comes in.

## 9. Sensitivity analyses (report all, regardless of outcome)

- (a) batch=16 vs. batch=1 — do the conclusions change? (Part B item 2, x86 only so far.)
- (b) gross vs. net-of-idle energy — does the primary-outcome choice (§3) matter to the conclusions?
- (c) x86 primary results restricted to `pinned` conditions only — isolates whether the sub-saturation
  finding (D7) itself is driving any RQ2 conclusion, independent of the stability-check gating in §7.

## 10. M1 measurement protocol — fixed now, before any x86 statistics are run

- **Stage 1-M1 (harness):** `powermetrics` with a narrowly scoped sudoers rule (that binary only, same
  pattern as the x86 governor-script rule already built and working — `deviation_log.md` Task 3 context
  in `stage4-implementation-brief.md`); same log schema as x86; memory-pressure flag (`vm_stat` — any
  run that swapped is excluded, 8GB RAM per the proposal's hardware table); thermal/throttle logging
  (fanless machine, per the proposal's own "Thermal observation" §3 clause). Exit test: ResNet-18 FP32,
  10 reps, with checklist items confirmed using real logged values — same discipline as the x86
  `stage1_exit_test.md`.
- **Feasibility check per state, before the pilot:** GPU via MPS; INT8 via the `qnnpack` backend
  (fbgemm is x86-only); FP16 via MPS; pruned checkpoints load and run. Any state that cannot run is
  logged as a new deviation-log entry, not silently dropped.
- **Stage 3-M1 pilot:** ResNet-18, all states, ~10 reps, plus its own stability check (not inherited
  from x86's — §7).
- **Stage 4-M1:** same conditions as x86 (batch=1 primary, batch=16 secondary), ≥30 reps per condition.
- **Cross-platform rule:** compare only within-platform relative savings (compressed/baseline ratios).
  Absolute joules are never compared across platforms directly — different instruments (NVML/RAPL vs.
  `powermetrics`) and different measurement boundaries.

## 11. Analysis order

x86-only analyses (RQ1, RQ2) first; cross-platform analyses (RQ3) only after Stage 4-M1 is complete.

## 12. Changes after registration

Any change to this plan is added as a dated amendment with its reason, appended below. The original
text above is never edited in place.

### Amendments

**A1 — 2026-10-06.** Note on process first: §12's own rule ("original text above is never edited in
place") was not followed for this amendment — the changes below were made as in-place edits at the
researcher's explicit instruction, not appended underneath untouched original text. This entry is the
disclosure of that, not a retroactive justification; §12's rule is restored for amendments after this
one. Changes made this date:
- **§3:** confirmed and documented, against `pilot.py`'s `Sensor` class source, that GPU-state energy
  is NVML-only and CPU-state energy is RAPL-only — never combined — and why.
- **§4:** reversed the primary/supplementary assignment — proposal's tests (Mann-Whitney U, Spearman,
  Kendall's tau) now primary; Welch's/bootstrap/Holm-Bonferroni now supplementary. Logged as
  `deviation_log.md` D14. Added the serial-reps/fixed-sequence limitation with the real run order and
  session boundaries (reconstructed from `timestamp_utc`/`uptime_s`).
- **§5:** defined the FLOPs-ratio measure for FP16/INT8 (plain MACs ratio is 1.0 by construction;
  added a labelled BOPs/bit-width-adjusted alternative). Reconciled the bootstrap CI used here with
  its "supplementary" label in §4 (no primary-family analog exists for this specific test).
- **§6:** fully replaced. New rule: MLPerf 99%/99.9% accuracy-tier convention (primary/strict),
  Wilson 95% CI + McNemar's test per state, borderline labelling, an all-states Pareto plot, and the
  single-training-seed (2026, no multi-seed replication) disclosure. **Still pending Shohan's explicit
  confirmation of the 99% primary threshold** — not yet run.
- **§2:** updated "not yet collected" wording for D13's FP32-CPU baselines to reflect that they were
  collected 2026-10-06.
- **`deviation_log.md`:** D7's "~3 hours" corrected to the real timestamp gap (3:57-4:19); D13 marked
  executed with dates; D14 given an OPEN sub-entry pending verbatim proposal re-verification, plus a
  note that signed-rank may validly apply to the paired CodeCarbon-vs-hardware comparison (D15); D15
  added and then expanded with per-component (CPU/GPU/RAM) CodeCarbon-vs-hardware data, the corrected
  (throughput-based, not wall-clock) overhead argument, and the revised 21-condition RQ1 pass proposal;
  D16 added — `pilot.py`'s CPU/RAPL reading sums `package-0` and `psys` domains, likely inflating every
  CPU-measured energy number in the project, not yet remediated; D9's CUDA-OOM-skip TODO closed (zero
  `skipped.jsonl` files found).

**Tagging:** once Shohan confirms §6(a)'s threshold, this plan is tagged in git (e.g.
`stage5-plan-v1`) as the frozen, final version analysis actually runs against — not done yet, since
confirmation hasn't happened.
