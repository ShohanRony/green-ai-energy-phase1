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
  FP32-CPU baselines (D13, Part B — not yet collected as of this plan's registration), 30 pairs each.
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

## 5. Core thesis test (FLOPs vs. energy)

Decision rule fixed now, before any data is analyzed against it: for each state, compare the measured
energy ratio with the realised FLOPs ratio (from Stage 2's `torch-pruning` report, not nominal sparsity —
consistent with this project's standing rule that nominal compression is never reported without its
realized counterpart). If the energy ratio's 95% CI excludes the FLOPs ratio, conclude "energy savings
not proportional to FLOPs reduction" for that state. Across states, report Spearman correlation
descriptively only — small n, no strong inference drawn from it.

## 6. Accuracy rule — threshold not yet set

**Shohan sets the threshold before analysis runs.** States whose accuracy drop vs. baseline exceeds
`[TODO — Shohan to set, e.g. 1-2 percentage points, with a stated reason]` are reported but labelled "not
deployable," and their energy savings are not presented as efficiency gains. The collapsed 70% pruning
state (and, for MobileNetV3-Small and — under the retrained 60-epoch baseline — EfficientNet-B0, the
50% state too; see `deviation_log.md` D4) falls under this rule once the threshold is set. **No Stage 5
statistics are run until this threshold is confirmed, per explicit instruction.**

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

*(none yet)*
