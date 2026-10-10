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
  use NVML; M1 uses `powermetrics`.
  **Confirmed directly against `pilot.py`'s `Sensor` class (2026-10-06):** a run's energy figure comes
  from exactly one backend, chosen by `--device` — NVML `nvmlDeviceGetPowerUsage` (sampled power,
  trapezoidal-integrated) for `cuda`; for `cpu`, RAPL, but **summed across two domains, not one** (see
  below). There is no combined RAPL+NVML reading anywhere in this harness; a GPU run never includes
  CPU/RAM power. GPU-state primary energy is confirmed **NVML-only** — the GPU `Sensor`'s `read()`
  returns exactly one value per sample and every GPU run's `trace` field carries a single-element list,
  checked directly. **Never compare energy across different instruments directly** — every
  cross-instrument comparison in this project (GPU vs. CPU, x86 vs. M1) must be stated with its
  instrument boundary, per the pre-flight checklist discipline carried from Stage 1.
  **CPU boundary corrected 2026-10-06 (`deviation_log.md` D16) — this entry previously said "RAPL
  package"; that was wrong.** `pilot.py`'s CPU `Sensor` globs `/sys/class/powercap/intel-rapl:*/energy_uj`
  one colon deep, which on this machine matches **both** `intel-rapl:0` (`package-0`) and
  `intel-rapl:1` (`psys`), and `integrate()` sums both domains' deltas into one number. Confirmed
  empirically (2026-10-06, 15s idle + 30s 16-core CPU load, fresh reads with explicit domain names):
  `psys` reports **more** than `package-0` at both idle (40.93 W vs. 12.31 W) and under load (133.37 W
  vs. 84.80 W) — consistent with `psys` being a superset platform-power domain (package + chipset +
  voltage regulators), not an independent additive rail. This directly contradicts this project's own
  `README.md` §"RAPL domain in use," which states the harness reads `package-0` only, "not `psys`
  (whole-system)," as a deliberate, disclosed choice — the code does not match that documented intent.
  **Every CPU-measured energy number in this project (all 3 INT8 states, all 3 D13 FP32-CPU baselines,
  plus Stage 2/3/pre-flight CPU pilot runs) used the summed package+psys value, confirmed roughly
  2.8x the package-only figure for `resnet18_int8`'s feasibility-check windows** (pilot.py combined:
  736.02 J; pilot.py package-only recomputed from the same raw trace: 266.21 J — which matches
  CodeCarbon's own independent package-only RAPL read, 266.40 J, to within 0.1%).
  **Resolved 2026-10-06: offline recompute + harness patch, no rerun.**
  `scripts/recompute_cpu_package_energy.py` recomputed package-0-only energy from each affected
  directory's own `raw.jsonl` trace (never modified) and wrote `summary_package.csv` alongside the
  original, untouched `summary.csv`, for all 10 affected directories (9 produced a summary; the
  10th, `pilot_proof_int8`, has only 1 rep). **`summary_package.csv`'s figures are now the corrected
  primary CPU energy for every affected condition; `summary.csv`'s combined (package+psys) figure
  and the `psys`-only component are both secondary.** The original `summary.csv` INT8/FP32-CPU values
  were viewed during Stage 4 collection for QC only (pairs/regime/plausibility checks, never a
  hypothesis test, per this document's own opening status note) — this correction was made before
  any Stage 5 statistics ran on CPU data, not after. CPU-side analysis (INT8 vs. FP32-CPU, or any
  comparison involving either) is unblocked using `summary_package.csv`.
- **Secondary:** net-of-idle energy, latency, accuracy (from Stage 2), realised FLOPs/params reduction
  (from Stage 2's `torch-pruning` dependency-graph report, not nominal sparsity), power regime.
- **RQ1 instrument-agreement check — component-wise comparison is PRIMARY, total-vs-total is
  secondary (added 2026-10-06, after the codecarbon_check feasibility data showed why):** CodeCarbon
  reports `cpu_energy`, `gpu_energy`, and `ram_energy` as separate fields, not a single number, and
  they measure fundamentally different things:
  - **GPU runs:** compare CodeCarbon's `gpu_energy` against `pilot.py`'s NVML power-integrated
    reading. CodeCarbon's GPU path (confirmed in installed `codecarbon` v3.3.1's `core/gpu_nvidia.py`)
    calls `pynvml.nvmlDeviceGetTotalEnergyConsumption` — the cumulative counter this project's own
    Stage 1 validation work found over-reports active-phase power ~30% versus the validated
    `nvmlDeviceGetPowerUsage` default `pilot.py` uses. A direct GPU-vs-GPU comparison isolates this
    known, already-characterised discrepancy from everything else CodeCarbon adds on top.
  - **CPU runs:** compare CodeCarbon's `cpu_energy` against `pilot.py`'s RAPL reading — **but only
    once D16 is resolved and `pilot.py`'s own CPU reading is package-only**, since CodeCarbon's
    `cpu_energy` is already package-only by default (`rapl_prefer_psys=False`; confirmed in
    `core/resource_tracker.py`'s `_setup_rapl()`).
  - **CodeCarbon's `ram_energy` is not a measurement — confirmed from source
    (`external/ram.py`): a fixed-wattage heuristic** (`RAM_SLOT_POWER_X86 = 5` Watts per estimated
    DIMM slot, from total detected RAM size, not read from any sensor — no consumer hardware exposes
    real-time RAM power telemetry). It should never be compared against a hardware "RAM energy" figure
    because no such hardware figure exists anywhere in this project.
  - **CodeCarbon tracks all detected hardware on the machine, not just what the measured workload
    uses — confirmed from source** (`core/resource_tracker.py`'s `_run_full_hardware_setup`: a `RAM`
    tracker is always appended; a `GPU` tracker is appended whenever any NVIDIA GPU is detected,
    independent of `pilot.py`'s own `--device` argument). This is why the CPU-only `resnet18_int8`
    feasibility-check run still logged a nonzero `codecarbon_gpu_energy_j` (98.10 J) — the idle RTX
    3050 was tracked the whole time even though no CUDA code ran.
  - **Total-vs-total** (CodeCarbon's aggregate `energy_consumed` vs. `pilot.py`'s single-device total)
    is kept as a secondary, whole-system-vs-single-device sanity check, not the primary RQ1 comparison
    — it conflates all of the above into one number and is harder to attribute.
  **Contingent on CodeCarbon data existing.** Checked directly (Part B item 4): `codecarbon` was
  `False` in every Stage 4 run's logged `arguments` field — **no paired CodeCarbon readings exist in
  the Stage 4 data collected so far.** This is a real gap against the proposal's explicit requirement
  (§3: "CodeCarbon runs concurrently with hardware measurement on every run, which is what makes the
  RQ1 comparison a within-run paired comparison") — flagged here and in Part B's report, not silently
  worked around. RQ1 cannot be answered from the current x86 dataset until CodeCarbon-paired runs
  exist, and the CPU half of it is additionally blocked on D16.

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

**Relabeling, added 2026-10-06: "Pruned30/50/70" names a channel ratio, not a sparsity or FLOPs
figure.** The number passed to `torch_pruning.pruner.MagnitudePruner` as `pruning_ratio` is a
fraction of output channels removed per prunable layer — calling it "30% pruning" without
qualification invites reading it as 30% of FLOPs, params, or weights removed, none of which is true.
The realised MACs/params reduction (from Stage 2's dependency-graph report, `stage2_deliverable.md`
§4) is always larger than the nominal channel ratio, because removing one layer's output channels
cascades to remove the corresponding input channels of every downstream layer it feeds:

| Model | Channel ratio | Realised MACs reduction | Realised params reduction |
|---|---|---|---|
| ResNet-18 | 30% | 51.6% | 51.1% |
| ResNet-18 | 50% | 74.8% | 75.0% |
| ResNet-18 | 70% | 91.0% | 91.1% |
| MobileNetV3-Small | 30% | 46.7% | 50.1% |
| MobileNetV3-Small | 50% | 69.1% | 73.6% |
| MobileNetV3-Small | 70% | 86.8% | 90.0% |
| EfficientNet-B0 | 30% | 48.4% | 50.0% |
| EfficientNet-B0 | 50% | 71.2% | 73.5% |
| EfficientNet-B0 | 70% | 88.1% | 89.8% |

"Pruned30/50/70" is kept as the checkpoint/condition name (matches `checkpoints/*.pt` filenames and
every prior stage's own naming) but is reported alongside its realised MACs figure wherever accuracy
or energy is discussed, not alone.

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

## 6. Accuracy rule — CONFIRMED by Shohan (2026-10-06)

**Status: confirmed by Shohan on 2026-10-06. The 99% primary / 99.9% strict relative-to-FP32 accuracy rules are approved and active.**

**(a) PRIMARY — MLPerf convention.** A compressed state is deployable if its top-1 accuracy on the
CIFAR-10 test set (n=10,000) is **≥99% of its own model's FP32 top-1 on the same test set**. This is
the MLPerf Inference accuracy-tier convention for the "closed division." **Citation, verified by the
supervisor session against the project-library PDF (page 1 and section V-D), 2026-10-06:** Tschand et
al., "MLPerf Power: Benchmarking the Energy Efficiency of Machine Learning Systems from µWatts to
MWatts for Sustainable AI," 2025 IEEE HPCA, arXiv:2410.12032 — the paper's own example cited is
BERT-99.0, requiring 99% of the original FP32 accuracy, the same convention applied here.
**Disclosed, two separate caveats:**
- MLPerf's 99%/99.9% tiers were defined for ImageNet/BERT-scale classification and language tasks;
  this project borrows the convention for CIFAR-10-scale models, which is a disclosed convention
  transplant, not a threshold independently validated for this task's scale or difficulty.
- **MLPerf Tiny's own CIFAR-10 ResNet benchmark does not use this relative convention at all** — it
  sets an **absolute** accuracy target (~85% top-1) rather than a percentage of an FP32 reference.
  MLPerf itself does not have one single convention for CIFAR-10-scale models; (a) above picks the
  Inference-benchmark relative convention over the Tiny-benchmark absolute convention, which is a
  choice, not the only available option.
- **Single-seed variance near the threshold was not measured and cannot be estimated from data as
  collected.** (f) below confirms every checkpoint in this project comes from one fixed seed (2026),
  with no multi-seed replication — so there is no empirical estimate of how much a state's accuracy
  would move under a different seed, which matters most exactly for states landing close to the 99%/
  99.9% boundary, where a different seed could plausibly flip a "deployable" call to "not deployable"
  or vice versa. The Wilson CI in (d) does not cover this; it only covers test-set sampling variability
  for the one seed actually trained.

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

**Three further disclosures, added 2026-10-06, all affecting how comparable the 18 accuracy numbers
above actually are to each other:**
- **INT8 pipeline differs by model.** ResNet-18's INT8 uses torchvision's official quantized path;
  MobileNetV3-Small's and EfficientNet-B0's use a manual FX graph-mode PTQ pipeline with cross-layer
  equalisation deliberately skipped (full detail: `deviation_log.md` D3). The three models' INT8
  accuracy-cost numbers are not from a uniform procedure.
- **30 vs. 60 training epochs.** ResNet-18's FP32 baseline (and everything derived from it) was
  trained for 30 epochs and never retrained — it was "already plateaued" (loss 0.0288) when checked.
  MobileNetV3-Small's and EfficientNet-B0's baselines were retrained to 60 epochs after the original
  30-epoch versions were found not plateaued (`stage2_retrain_report.md`). Any comparison of accuracy
  behaviour *across* the three models (not within one model's own states) compares models trained to
  different convergence depths, not a controlled variable.
- **No validation split.** Training uses only a train/test split (`train_baseline.py`); `test_acc` is
  computed exactly once, after the last epoch, never used inside the per-epoch loop (early stopping
  and the LR schedule use only `train_acc`/loss). There is no held-out validation set distinct from
  the test set. This means: (1) there's no guard against the single reported `test_acc` being a
  favourable-or-unfavourable draw from one specific epoch rather than a selected best-epoch checkpoint;
  (2) the 60-epoch retrain decision itself was made by inspecting the training-accuracy trend
  (`stage2_deliverable.md`: "+1.97pt and +2.05pt over their last 5 epochs"), not test accuracy, so this
  specific decision did not use the test set — but the structural absence of a validation split means
  no decision in this pipeline ever could, without touching the test set directly.

**(g) Final.** This rule is final once (a)'s 99% primary threshold is explicitly confirmed by Shohan.
Any later change is a dated amendment under §12, not an in-place edit — see A1.

**Projected outcome using item 1's measured full-test-set accuracies (`results_accuracy/summary.csv`,
2026-10-06) — a preview, not a ruling: (a)'s 99% threshold is still pending Shohan's confirmation, and
none of this is final until it is.**

| Model | State | Accuracy | Ratio to own FP32 | 99% tier | 99.9% tier |
|---|---|---|---|---|---|
| ResNet-18 | INT8 | 93.12% | 100.03% | PASS | PASS |
| ResNet-18 | FP16 | 93.08% | 99.99% | PASS | PASS |
| ResNet-18 | Pruned30 | 64.82% | 69.63% | **FAIL** | **FAIL** |
| ResNet-18 | Pruned50 | 17.61% | 18.92% | **FAIL** | **FAIL** |
| ResNet-18 | Pruned70 | 10.35% | 11.12% | **FAIL** | **FAIL** |
| MobileNetV3-Small | INT8 | 83.89% | 97.04% | **FAIL** | **FAIL** |
| MobileNetV3-Small | FP16 | 86.47% | 100.02% | PASS | PASS |
| MobileNetV3-Small | Pruned30 | 17.16% | 19.85% | **FAIL** | **FAIL** |
| MobileNetV3-Small | Pruned50 | 10.00% | 11.57% | **FAIL** | **FAIL** |
| MobileNetV3-Small | Pruned70 | 10.00% | 11.57% | **FAIL** | **FAIL** |
| EfficientNet-B0 | INT8 | 87.18% | 98.10% | **FAIL** | **FAIL** |
| EfficientNet-B0 | FP16 | 88.84% | 99.97% | PASS | PASS |
| EfficientNet-B0 | Pruned30 | 60.29% | 67.84% | **FAIL** | **FAIL** |
| EfficientNet-B0 | Pruned50 | 13.35% | 15.02% | **FAIL** | **FAIL** |
| EfficientNet-B0 | Pruned70 | 10.00% | 11.25% | **FAIL** | **FAIL** |

**The consequence worth flagging explicitly: under this convention, every pruned state fails,
including Pruned30 for all three models** — not just the previously-identified "collapsed" (≤15%
accuracy) states from D4. D4's old ad-hoc collapse threshold was far more permissive than the MLPerf
convention; Pruned30 was never flagged as collapsed (64.82%/17.16%/60.29% are nowhere near chance) but
still fails a 99%-of-FP32 bar badly. Only the three FP16 states and ResNet-18's INT8 pass either tier;
MobileNetV3-Small's and EfficientNet-B0's INT8 both fail the 99% tier (97.04%, 98.10%) despite not
looking obviously degraded in absolute terms.
**Pruned-state caveat (`results_pruned_controls/bn_recalibration.csv`, 2026-10-06):** several of
these pruned-state failures are, at least in part, a BatchNorm-statistics artifact, not a pure
capacity limit — BN-stats-only recalibration (no weight updates, forward passes on 2000 train images)
recovers ResNet-18 Pruned50 from 17.61%→71.02% and EfficientNet-B0 Pruned50 from 13.35%→46.17%, while
MobileNetV3-Small's Pruned50/70 and EfficientNet-B0's Pruned70 show zero recovery (true collapse: the
model predicts one class for all 10,000 test images, before and after recalibration). **This plan's
accuracy rule evaluates the checkpoints as measured by `pilot.py` (zero-finetune, no BN recalibration)
— the control data above is reported so the deployability table doesn't read as "pruning this much
necessarily destroys the model," when for some states it's a fixable statistics mismatch, not a
capacity limit. Whether BN recalibration counts as still "post-training only" (and so whether a
recalibrated number should ever be used instead of the as-measured one) is Shohan's call, not resolved
here.**

## 7. Power-regime handling (x86)

- Every condition reports its regime (`power_regime`: `pinned`/`dip`/`mixed`, logged automatically by
  `pilot.py` since the Stage 4 pre-flight guard was added — `deviation_log.md` D7).
- **Executed 2026-10-06 (`e2df9c1`): all 12 `dip` conditions reproduced their exact original regime
  in a separate fresh session (12/12 match, zero flips)** — the stability-check gate below is
  satisfied for every `dip` condition in the primary matrix.
- **`nvidia-smi dmon` finding, corrected 2026-10-06 (see `deviation_log.md` D7's mechanism
  correction): "steady low utilisation" is not what the log shows.** During active windows, SM
  occupancy is continuously elevated (57-58% for `mobilenet_v3_small_fp32`'s dip; 89-96% for
  `resnet18_pruned70`'s dip) with clock boosted to near-max the entire time — not low, and not
  idling between dispatches. The 0%-utilization stretches in the log correspond entirely to the
  harness's own `idle_before`/`idle_after` phases and inter-rep warmup, not to gaps within active
  compute. The gating rule below is kept (fresh-session reproducibility is still required), but is
  no longer read as confirming "steady low utilisation" — it confirms regime stability, which is a
  different and still-valid thing to check.
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

**A2 — 2026-10-06.** Same process note as A1: made as in-place edits, not appended-only, at explicit
instruction; this entry discloses that rather than hiding it. Covers every in-place edit made after
A1, across two supervisor-review rounds the same day:
- **§3:** CPU boundary corrected a second time — D16's root cause (package+psys summed, not
  package-0 alone) confirmed, then marked resolved once the offline recompute + harness patch landed;
  component-wise (GPU-vs-NVML, CPU-vs-RAPL-package) comparison made primary for the RQ1 instrument-
  agreement check, total-vs-total demoted to secondary; documented that CodeCarbon's `ram_energy` is a
  fixed-wattage heuristic, not a measurement, and that it tracks all detected hardware (idle GPU
  included) regardless of the measured workload's actual device.
- **§5:** added the "channel ratio, not sparsity/FLOPs" relabeling for Pruned30/50/70, with the
  realised-MACs table alongside.
- **§6:** citation TODO filled (Tschand et al., 2025 IEEE HPCA, MLPerf Power — provenance later
  corrected to supervisor-verified against the project-library PDF, p.1 and §V-D); added the MLPerf
  Tiny absolute-target (not relative-to-FP32) caveat and the single-seed-near-threshold-variance gap;
  added the full 18-state projected pass/fail table against the 99%/99.9% tiers using item 1's
  measured accuracies (still a preview, (a)'s threshold still unconfirmed); added the pruned-state
  BN-recalibration caveat (several "failures" are a statistics artifact, not a pure capacity limit,
  per `results_pruned_controls/`); added the INT8-pipeline-differs-by-model, 30-vs-60-epoch, and
  no-validation-split disclosures.
- **§7:** marked the dip-stability check executed (12/12 reproduced); corrected "steady low
  utilisation" — the real `dmon` finding is continuously elevated (not low) occupancy during active
  windows, with the 0%-utilization stretches matching the harness's own idle phases exactly, not GPU
  idling mid-computation (`deviation_log.md` D7, D8).
- **`deviation_log.md`:** D7 and D8 both corrected with the `dmon`-confirmed mechanism (occupancy
  scales with workload size while continuously active and fully clock-boosted; not a dispatch-idling
  effect); D9's note already closed stands; D15's stale "not yet remediated" line for D16 fixed; D16
  completed (offline recompute executed, harness patched, both committed) and reclassified from
  "likely inflated" to root-cause-confirmed with three independent lines of evidence; D17 added
  (between-session drift, -12.20% to +3.70% on the 12 dip-stability conditions vs. the primary
  matrix, confounded with run order — not yet separable from a genuine between-session effect).
- **`stage4-implementation-brief.md`:** §2's M1-access line corrected to match D5; §5's "~3 hours"
  corrected to "~4 hours" (same real-timestamp correction as D7); §9's citable finding given the same
  `dmon`-confirmed mechanism correction as D7/D8.
- **`phase1-execution-plan.md`:** the three psys-vs-package "check remaining" lines (§1, §3 Stage 1,
  §5) all closed, noting the check was flagged from the start but never acted on until D16.
- **`results_stage3/stage3_deliverable.md`:** INT8's instrument label corrected from "RAPL package
  energy" to "RAPL package+psys," matching D16, with a note that the underlying 0.18395 J/image figure
  itself is not recomputed (Stage 3 is pilot-only, D12; no `summary_package.csv` generated for it).
- **`README.md`:** "perf-events" corrected to "sysfs" (RAPL access was always sysfs-direct);
  `requirements.txt` versions corrected to match what's actually installed (every pinned version was
  stale except `codecarbon`), `torch-pruning` added (was missing despite being a real dependency since
  Stage 2); the stray root-level `test_pilot.py` this session created was merged into the existing
  `tests/test_pilot.py` (duplicate filename, wrong location — the project's real test suite has always
  lived under `tests/`) and the root copy deleted.

**A3 — 2026-10-06/07.** Explicitly records Shohan's formal confirmation (2026-10-06) of §6's accuracy
rule (99% primary / 99.9% strict relative-to-FP32 deployability threshold). Notes that the initial
Stage 4b design draft (`85d966c`) is superseded by the finalized Stage 4b specification in `stage4b-design.md`,
substituting the 9 `_bnrecal` conditions for energy measurement and designating the 9 `_ft` recovery models as
accuracy-only controls.

**A4 — 2026-10-08. DRAFT, pending supervisor review.** Six confirmatory sessions instead of four
(appended only; no registered text above is edited).
- **Change:** the Stage 4b confirmatory matrix (`stage4b-design.md`) is set at **exactly six**
  `main` sessions, not four. The count is fixed now. There is no stopping rule, so no session is
  added or dropped based on results.
- **Reason:** four sessions give 3 degrees of freedom (95% t-multiplier 3.18). Six give 5 (t = 2.57),
  so intervals are about 19% narrower. The original "four" was set by a time budget, which no longer
  applies.
- **What has been looked at before this amendment:** only descriptive per-condition means in the
  per-session `session_log.md` files (Sessions 1 and 2). No inferential statistic (test, interval,
  equivalence bound) has been computed on Stage 4b data.
- **Session status:** Session 1 (pinned re-run, D19) and Session 2 are valid. The first attempts at
  Sessions 3 and 4 are void (D20; kept under `results_stage4b/main_session{3,4}_aborted/`). Sessions
  3 and 4 are re-run, then 5 and 6 are added. Order seeds stay at `1000 + N`.
- **Protocol:** each session runs after a fresh reboot, with no other use of the laptop. The six
  sessions span **at least 3 calendar days**.

**A5 — 2026-10-08. DRAFT, pending supervisor review.** A real prune + fine-tune arm.
- **Change:** the zero-finetune arm and the `_bnrecal` arm are kept unchanged. The 3-epoch `_ft`
  arm (A3: accuracy-only control) is superseded by one **20–30 epoch fine-tune from the pruned
  weights** for `pruned30` and `pruned50` (`pruned70` optional) on all three architectures. It is
  labelled "prune + fine-tune (not post-training)" everywhere it appears. Exact epochs/LR schedule are
  fixed in the training script before training starts and logged; accuracy uses §6's rule unchanged.
- **Energy:** measured, not inferred. One extra session runs once the models exist, with the same
  harness and settings as `main`. It is reported as its own arm, not pooled into the six-session
  matrix.
- **Reason:** no pruned state currently reaches 99% of FP32 (D18), so the energy-vs-accuracy
  comparison has no deployable pruned point to test.

**A6 — 2026-10-08. DRAFT, pending supervisor review.** Realistic-regime sensitivity pass
(energy only).
- **Change:** a separately labelled pass at **224×224 input, batch 1 and batch 16**, for the same
  three architectures and compression states where the format allows. Weights may be random or
  ImageNet-initialised because energy does not depend on weight values. The pass makes **no accuracy
  claims** and is **never pooled** with the 32×32 data.
- **Reason:** 32×32 at batch 1 on an RTX 3050 may be launch- or host-bound. This pass tests whether
  the energy ratios hold when the GPU is not.
- **Analysis:** descriptive and exploratory under §9. It adds no confirmatory hypothesis.

Planned but not yet written as amendments (each gets its own dated entry when done): wall-meter
ground truth (supervisor D-E), Zenodo deposit of the registered design before any Stage 5 statistic
(D-F), and the methodology-study framing of the thesis (D-A).

**A7 — 2026-10-09. DRAFT, pending supervisor review. Analysis freeze, per the researcher's
2026-10-09 Paper-1 framing decision.** No outcome statistic has been computed on Stage 4b data
in drafting this amendment — see the "what has been looked at" statement below.

1. **Blocks, per the researcher's 2026-10-09 decision (`deviation_log.md` D5 update):** **x86-CPU**
   and **M1-CPU** are primary (same protocol on both); **x86-GPU** is supplementary. Comparisons
   are made *within* a block; any comparison *across* blocks is descriptive only, never a
   confirmatory claim — this directly addresses the phase-1 audit's C5 (RQ3 mixing CPU and GPU
   devices inside one "ranking") by making the device boundary the block boundary itself, not an
   afterthought.
2. **Per-block baseline: FP32-TorchScript**, not FP32-eager. This directly addresses the audit's
   C1 (baseline eager, compressed states traced, confounding runtime with compression) — every
   ratio in every block now compares like-runtime to like-runtime. (A8, below, is the dedicated
   check for whether this choice itself matters.)
3. **Primary estimator: geometric mean, over sessions, of the within-session ratio `r`**
   (compressed state ÷ same-block FP32-TorchScript baseline, both from the same session), with a
   **95% t-interval on `log r`** (`df = sessions − 1`). **Session is the replicate unit** — not
   the rep, not the window. This is the same estimator the phase-1 audit's own upgraded analysis
   plan proposed (its "D1-D5" decision framework, `P0` step), adopted here directly rather than
   re-derived.
4. **Mann-Whitney U is within-session/exploratory only — never the primary cross-session test.**
   This **resolves the direct contradiction the audit flagged as C3**: `stage5_analysis_plan.md`
   §4 (D14) still names Mann-Whitney U on within-session reps as primary; `stage4b-design.md` §5
   registered per-session ratios as primary instead, without ever reconciling the two. A7 is that
   reconciliation: §4's Mann-Whitney registration is superseded by this amendment for every
   confirmatory claim going forward (§4's text is left as-is per §12's append-only rule — this
   paragraph is the correction of record, not an edit to §4 itself). Mann-Whitney stays useful
   and reported, but strictly as a within-session repeatability description, consistent with how
   D14 always reasoned about independence within one session.
5. **Uptime covariate, pre-declared now, before any session's data is analyzed:** session uptime
   at measurement time is included as a covariate in the mixed-model robustness check (point 8)
   specifically because the randomised condition order (already registered) makes uptime a
   source of unmodeled noise otherwise — pre-declaring it now means it can't be added or dropped
   later based on whether it changes the result.
6. **Multiplicity:** one confirmatory family — the compressed-vs-FP32-TorchScript ratios, per
   block — with Holm-Bonferroni applied across it. Everything else in this plan is estimation,
   reported without a multiplicity-adjusted p-value.
7. **RQ2 model, per block:** `log(r) ~ β·log(MAC ratio) + (1|model) + (1|session)`, hypothesis
   **H2: β < 1**. Run separately per block (x86-CPU, M1-CPU, and — supplementary — x86-GPU), not
   pooled across blocks, consistent with point 1's within-block-only comparison rule.
8. **Gross vs. net-of-idle energy: which is primary is declared here, before any session's data
   is computed against it** — **gross** stays primary (unchanged from §3's existing registration),
   net-of-idle remains a declared secondary decomposition, not chosen after seeing which one tells
   a cleaner story.
9. **Specification curve, reported for every primary ratio:** recomputed across instrument (now
   possible for every GPU condition without re-measuring, since D24 logs both NVML interfaces and
   both RAPL domains every window already), boundary (GPU-only / CPU-package / GPU+CPU, same D24
   data), runtime (eager vs. TorchScript vs. CUDA Graphs — A8), regime, and gross/net. Presented
   as a distribution against the compression effect, per the audit's own novelty-positioning
   recommendation — this is the paper's actual differentiator, not a sensitivity-analysis
   afterthought.
10. **RQ1 split:** **RQ1a (coverage)** — how much of a wall-meter-derived (once available, P1 in
    the audit's roadmap) or best-available system energy figure each tool/boundary actually
    captures. **RQ1b (estimation mode)** — CodeCarbon with hardware counters available vs. forced
    into its documented TDP/CPU-load fallback (`docs/feasibility_x86_cpu_block.md` §d confirms the
    forcing mechanism exists and is supported) compared against the same reference. This directly
    resolves the audit's C4 (RQ1 as previously specified compared a RAPL-backed counter against
    itself) — RQ1b is where CodeCarbon is actually *estimating*, which is what the proposal's H1
    is about.
11. **RQ3, redefined to what can actually be answered:** a **within-CPU** interaction test,
    `state × platform` (x86-CPU vs. M1-CPU), as primary — not a six-state ranking that silently
    mixes CPU and GPU devices (the audit's C5). Kendall's tau across states remains descriptive
    only, per the original plan's own n=6/model low-power caveat (§6(a) area).
12. **Session count per block is stated here, before any block's data exists, and will not be
    changed after looking:** carried over from Stage 4b's existing registration (6 sessions,
    `main_session1`-`main_session6`) for the x86-GPU supplementary block. **x86-CPU and M1-CPU
    primary-block session counts are not yet fixed** — this is listed as open, not silently
    defaulted to 6, since no CPU-block session has been run yet (`docs/feasibility_x86_cpu_block.md`
    §c: zero pruned/TorchScript-FP32 states have ever been measured on either CPU). Fixing this
    count is a precondition for running any primary-block session, not an afterthought.
13. **"What has been looked at" statement, in full, as the audit's own framework requires before
    any amendment claiming an analysis freeze:** only descriptive per-condition means in each
    session's own `session_log.md` (quality-control level, same discipline as every earlier stage);
    the pinned-vs-`main_session1_unpinned` comparison (D19, a data-quality check, not an outcome
    comparison); and the `_bnrecal`-vs-zero-finetune accuracy check (D18, an accuracy-only
    comparison, no energy ratio involved). **No ratio, interval, or hypothesis test has been
    computed on any Stage 4b energy data at any point before this amendment was drafted** —
    consistent with this review's standing order.

**A8 — 2026-10-09. DRAFT, pending supervisor review. Runtime-baseline series R.** Answers A7
point 2 empirically: does the FP32-eager-vs-TorchScript choice actually matter, and does
TorchScript vs. a CUDA-Graphs-style compiled runtime matter further.

- **Design:** (a) FP32-eager vs. FP32-TorchScript, CPU and CUDA, all 3 architectures (6
  conditions); (b) model × {FP32, FP16, `pruned50_bnrecal`} × {eager, TorchScript, CUDA Graphs}
  (27 conditions). Matches the audit's own P3 "runtime ablation" step.
- **Protocol:** fresh-boot sessions (same discipline as Stage 4b), seeds **2000 + N** (a third
  seed family, distinct from this project's standing seed 2026 and the P8 campaign's
  1001-1003 — chosen specifically so series R's provenance is never confusable with either),
  **7 reps**, **randomised order**, same guards as the main matrix (governor, platform profile,
  fresh-boot check, concurrent-GPU guard).
- **Output location: `results_stage4b_R/`** — kept entirely separate from `results_stage4b/`,
  never pooled with the main confirmatory matrix.
- **Not executed as part of this review** — draft only, per the researcher's "no data collection"
  order.

**A7r1 — 2026-10-10. Dated revision of A7 (A7 remains DRAFT; its text above is unchanged — this
revision stands alongside it, per §12's own append-only rule, which this entry follows unlike A1/A2).
No outcome statistic has been computed on Stage 4b data in drafting this revision — see (g) below.**

**(a) GPU-block baseline is FP32-eager, not FP32-TorchScript, as actually collected — A7 point 2
corrected for the GPU block only.** Checked directly: every `main_session1`-`main_session6` (and
`main_session1_unpinned`, `main_session3_aborted`, `main_session4_aborted`) directory has only
`{model}_fp32` conditions, no `{model}_fp32_ts` or equivalent — the x86-GPU block as actually run
never collected a TorchScript GPU baseline. A7 point 2's "FP32-TorchScript, not FP32-eager" baseline
registration is correct for the CPU blocks (each x86-CPU/M1-CPU session, per (e) below, explicitly
includes its own FP32-TS condition) but was never true for the x86-GPU block's six sessions as they
already exist. **Decision, effective immediately for the GPU block:**
- GPU-block ratios use **FP32-eager** as the within-session baseline, every ratio for that block
  labelled **runtime-confounded (the phase-1 audit's C1)** — the same caveat A7 was written to
  resolve for the CPU blocks, still open for the supplementary GPU block.
- A **bridged version** is also reported: `bridged ratio = eager-baseline ratio × (eager/TS factor
  from series R)`, with the factor's own between-session uncertainty (A8r1 below) propagated into
  the bridged ratio's interval (error propagation on the product of two independent ratios: relative
  variances add).
- **Decision rule D2** (full statement in (f) below) determines which version is primary for the GPU
  block, once series R exists: if FP32-TS is within ±5% (TOST-equivalent) of FP32-eager, the confound
  is bounded and the existing eager-baseline ratio stays primary with the bound noted; otherwise the
  bridged version becomes primary for the GPU block.
- CPU blocks are unaffected by this correction — they already carry their own FP32-TS baseline inside
  every session (e), so no bridging is needed there regardless of what D2 concludes.

**(b) Confirmatory family narrowed; Holm-Bonferroni over ~30 per-state contrasts removed — A7 point 6
corrected.** A7 point 6 registered Holm-Bonferroni "across [the] full family" of compressed-vs-baseline
ratios per block — in practice, roughly 30 per-state contrasts (CPU block: FP32-TS, FP32-eager, INT8,
pruned30/50/70, `_bnrecal` variants, × 3 architectures). That is replaced:
- **The confirmatory family is RQ2's H2 test (β < 1, see (d)) — one test per primary block (x86-CPU,
  M1-CPU) — two tests total.** Holm-Bonferroni is applied across those two, not across per-state
  contrasts. This is **decision rule D3** (full statement in (f)).
- **Every per-state ratio (compressed state ÷ same-block baseline) is estimation, not hypothesis
  testing:** reported as the geometric-mean ratio with its 95% t-interval (df = sessions − 1, per A7
  point 3), no p-value attached, no multiplicity adjustment applied to it individually.
- **The ±5% equivalence question** (used by D2 above, and anywhere else a state's ratio is checked
  against a bound) is reported as **resolved** or **unresolved** only — whether the state's own
  t-interval falls entirely inside ±5%, entirely outside, or straddles it — not as a p-value, and not
  folded into the Holm-adjusted family.
- x86-GPU (supplementary) is descriptive only throughout, consistent with A7 point 1 — it was never
  part of the confirmatory family and still is not.

**(c) Instrument axis: checked directly whether any session 1-6 artifact records the cumulative NVML
counter — none does; A7 point 9 corrected for the affected data.**
- Checked every `environment.json` under `results_stage4b/` (all 9 directories, all conditions):
  `"codecarbon": false` on every single run — no CodeCarbon-paired reading exists anywhere in
  sessions 1-6, confirming §3's existing "no CodeCarbon gap" finding still holds for this data too,
  not just the earlier Stage 4 data it was originally checked against.
  `results_stage4b/main_session5/resnet18_fp32/environment.json` and seven sibling files.
- Checked the `windows.csv` **schema** for a pre-D24 session (`main_session1`) against a session that
  ran before D24's commit but is otherwise representative (`main_session6`, last condition start
  2026-10-08T19:05:42Z — D24 committed 2026-10-09T21:57:54+06:00, after every session 1-6 directory
  was fully collected): identical 22-column header in both, with exactly one GPU-energy field
  (`gpu_energy_j`) and no second NVML-interface column anywhere. **No session 1-6 artifact of any
  kind — CodeCarbon or raw window data — records the cumulative NVML counter
  (`nvmlDeviceGetTotalEnergyConsumption`).** D24's dual-interface logging genuinely applies to new
  data only.
- **A7 point 9 corrected:** its specification-curve "instrument axis... now possible for every GPU
  condition without re-measuring, since D24 logs both NVML interfaces... every window already" is
  true only for runs collected after D24's commit (`d82cd95`, 2026-10-09T21:57:54+06:00). For
  sessions 1-6, the instrument axis of the specification curve cannot be computed post hoc — it
  applies to new data collected under D26/D24 going forward, not retroactively to the existing
  confirmatory matrix. This is **decision rule D1** (full statement in (f)).

**(d) RQ2 model specification — A7 point 7 corrected: model is a FIXED effect, not random; β
estimated from pruned states only; MAC counting stated.**
- Corrected model: `log(r) ~ β·log(MAC ratio) + model + (1|session)`, **model as a fixed effect with
  3 levels** (ResNet-18, MobileNetV3-Small, EfficientNet-B0), session random — not
  `(1|model) + (1|session)` as A7 point 7 had it. With only 3 architectures, treating model as random
  estimates a variance component from 3 groups, which is poorly identified; a fixed effect with 3
  levels is the better-justified choice here and is what is actually registered now.
- **β is estimated from the pruned states only** (pruned30/50/70, and their `_bnrecal` counterparts
  where meaningful per (e)) — FP16 and INT8 are excluded from the β fit because their MAC ratio is
  1.0 by construction (§5: MACs are identical across precision states at fixed architecture; only the
  bit-width changes), which would contribute rows with zero variance on the predictor and distort the
  slope estimate rather than inform it. FP16/INT8 are still reported (ratio, CI, equivalence call per
  (b)) — just not used to fit β.
- **How MACs are counted, stated here as the order requires:** the realised-MACs figures from
  Stage 2's `torch-pruning` dependency-graph report (§5's table — e.g. ResNet-18 pruned30 = 51.6%
  MACs reduction), not the nominal channel ratio and not a re-derivation — the same source §5 already
  designates as authoritative for the FLOPs-ratio exclusion test, reused here for consistency rather
  than computed a second way.
- This is **decision rule D4** (full statement in (f)).

**(e) Session counts fixed now, before any primary-block session is run — A7 point 12 resolved.**
- **x86-CPU: 6 sessions. M1-CPU: 6 sessions. x86-GPU: 6 sessions** (already collected,
  `main_session1`-`main_session6`). All three blocks now have a fixed count; none is open.
- **Each CPU-block session (x86-CPU and M1-CPU) contains:** FP32-TorchScript (block baseline),
  FP32-eager (collected specifically to let D2 be evaluated on the CPU blocks too, not just
  bridged-via-GPU-series-R), INT8, and pruned30/50/70 — each pruned state paired with its `_bnrecal`
  counterpart **where meaningful**, i.e. where `_bnrecal` recalibration is expected to change the
  checkpoint's behavior at all (not for states already at or near chance accuracy regardless of
  recalibration — carried over from the existing `_bnrecal` collapse findings,
  `docs/bnrecal_cpu_equivalence.md`).
- **No FP16 on CPU** — `docs/feasibility_x86_cpu_block.md`'s own feasibility finding (54-65× slower,
  software-emulated, not a real CPU FP16 execution path) rules it out; this was already established,
  restated here only so the fixed condition list in this amendment is self-contained.
- **No conditional add/drop of sessions** — the count is fixed now, before any primary-block session
  exists, matching A4's existing "no stopping rule" precedent for the GPU block.

**(f) The audit's D1-D5 decision rules, written out in full.** A7 point 3 referenced "the phase-1
audit's own upgraded analysis plan... its 'D1-D5' decision framework" as the source of the
session-as-replicate-unit estimator, but no document in this repository has ever stated D1-D5 in
full — they are written out here, synthesized from what A7/A7r1 and the researcher's own decisions
establish, since the original external document is not part of this repository. **These are
decision rules — pre-declared criteria that resolve a specific open methodological question once
the relevant data exists — distinct from `deviation_log.md`'s own D-numbered entries (D1-D27),
which record deviations/incidents, not decision rules. The name collision is inherited from the
audit's own terminology, not introduced here.**
- **D1 (instrument axis).** If the wall-derived energy delta (once the P1 wall-meter protocol exists,
  (j) below) agrees within ±5% (after adapter efficiency) with `delta(NVML power-usage + RAPL
  package)` — this project's existing validated backend — **keep the backend**, report the agreement
  as the instrument-axis finding. If the cumulative NVML/RAPL-psys counter agrees with the wall meter
  instead, **Stage 4b's GPU numbers need a rerun** on the cumulative interface. If **neither** agrees
  within ±5%, **wall energy is primary** for any system-level claim going forward, and every
  instrument this project uses is reported as a disclosed estimator of it, not as ground truth.
  Applies to data collected after D24/D26 only, per (c) — cannot be evaluated on sessions 1-6.
- **D2 (runtime confound).** If FP32-TorchScript is within ±5% (TOST equivalence, (f) of A8r1 below)
  of FP32-eager, for a given block, the eager-vs-TS confound is bounded and the existing baseline for
  that block stays primary (with the bound disclosed). Otherwise, the bridged ratio ((a) above) is
  primary for that block. Evaluated once series R (A8/A8r1) exists; CPU blocks can also be checked
  directly, since each CPU session carries both FP32-TS and FP32-eager itself (e).
- **D3 (confirmatory family).** The confirmatory family is RQ2's H2 (β < 1) test, one per primary
  block (x86-CPU, M1-CPU), Holm-Bonferroni across those two tests only. Every per-state ratio is
  estimation (interval, no p-value); every ±5% equivalence check is reported resolved/unresolved,
  not as a significance test. Full statement: (b) above.
- **D4 (RQ2 model and MAC counting).** `log(r) ~ β·log(MAC ratio) + model + (1|session)`, model fixed
  (3 levels), session random, β fit from pruned states only (MAC ratio ≡ 1 for FP16/INT8 by
  construction, excluded from the fit), MACs from Stage 2's realised-MACs dependency-graph report.
  Full statement: (d) above.
- **D5 (block/platform scope) — the researcher's 2026-10-09 decision, restated here as the fifth
  decision rule, not a new decision:** x86-CPU and M1-CPU are primary (confirmatory); x86-GPU is
  supplementary/optional, descriptive only; **no M1-GPU block exists in this project at all** — M1
  measurement (§10) is CPU-only via `powermetrics`, there was never a planned M1-GPU (MPS) energy
  measurement arm. Matches A7 point 1 and `deviation_log.md`'s D5 update referenced there; stated in
  full here per this block's instruction, not newly decided.

**(g) "Looked at" statement, corrected and extended — A7 point 13 revised.** The researcher's own
descriptive spot checks, not previously recorded in this plan, are added, and one prior claim is
corrected:
- **Added:** the researcher's descriptive energy spot check of Session 1 — pinned vs. unpinned mean
  system J/image, **+0.6%**, range **−1.6% to +3.5%** across the compared conditions; and a
  `_bnrecal`-vs-zero-finetune **energy** comparison, **~1% apart**. Both are descriptive spot checks
  at the same quality-control level as everything else this statement already lists (per-condition
  means, never an interval or test) — recorded here so the "what has been looked at" inventory is
  complete, not because either changes the analysis-freeze status.
- **Corrected:** A7 point 13 characterised the `_bnrecal`-vs-zero-finetune check as "an accuracy-only
  comparison, no energy ratio involved" — **this was wrong.** The researcher's own spot check above
  is an energy comparison between the two arms. The accuracy-only characterisation in A7 is left
  as-is per §12's append-only rule (not edited in place); this paragraph is the correction of record.
- **With this addition, the complete "looked at" inventory remains descriptive-only: per-condition
  means in `session_log.md` files; the pinned-vs-unpinned data-quality check (D19); the
  `_bnrecal`-vs-zero-finetune accuracy check (D18); and now the two spot checks above. No ratio,
  interval, or hypothesis test has been computed on Stage 4b energy data at any point up to and
  including this revision** — consistent with A7's and this block's standing order.

**(h) Net-of-idle baseline, defined before any computation, per `pilot.py`'s actual implementation
(`summarize()`, confirmed by reading the source directly, not re-derived):** for a given active-phase
window (`a1` or `a2` of a given rep), idle power is the mean of that **same rep's own two flanking
idle windows** (`idle_before` and `idle_after` — energy ÷ duration for each, then averaged), **not** a
single session-wide idle baseline and **not** idle windows borrowed from a different rep. Net-of-idle
energy for that active window = active-window gross energy − (idle power × active-window duration).
This is computed per rep, per active phase, then averaged within a condition for the condition-level
`above_idle_j_mean`/`above_idle_fraction_mean` figures already in every `summary.csv`. Stated here,
before any net-of-idle computation under this revision, so the definition isn't chosen after seeing
which one tells a cleaner story — the gross-vs-net sensitivity analysis (§9(b)) uses this exact
definition as already implemented, no new computation invented for this revision.

**(i) RQ3 backend note — A7 point 11 extended.** x86's INT8 states use the `fbgemm` backend; M1's
INT8 states use `qnnpack` (`fbgemm` is x86-only, §10). The `state × platform` interaction test (A7
point 11) therefore has its INT8 cell confounded with a backend change, not platform alone — stated
here so the interaction test's INT8 result is read as "platform and quantization backend changed
together," not attributed to platform (architecture) alone. No other state in the matrix carries this
confound (FP32/FP16/pruned states use the same computational backend family on both platforms).

**(j) Prerequisites for any new primary-block run, effective immediately.** Before any x86-CPU or
M1-CPU primary-block session (or any new GPU-block session) is collected under this plan:
1. **D26's AC guard** (`pilot.py`'s `check_ac_power()`) must be active in the harness used — already
   true for the current `pilot.py`, stated here as a standing precondition, not a new build step.
2. **The P1 wall-meter protocol must exist and be specified** before it can serve as D1's arbitration
   reference. It does not exist yet — §12's own list already carries it as "planned but not yet
   written" (wall-meter ground truth, supervisor D-E). This revision does not write that protocol; it
   registers it as a gate: D1 cannot be evaluated, and no claim depending on D1's outcome can be made,
   until it exists. No session is blocked on D1 having been *evaluated* — only on the harness-level
   AC guard — but D1 itself stays open until the wall-meter protocol is written and run.

**Appended 2026-10-10 (BLOCK J):** if the 224×224 regime pass (A6/D4r1) is not run, regime
dependence is stated as a limitation and no generalisation claim is made — RQ2's 32×32 finding is
reported as established only for 32×32, with 224×224 generalisation left explicitly open, not
assumed, until D4r1 is actually evaluated.

**A8r1 — 2026-10-10. Dated revision of A8 (A8 remains DRAFT; its text above is unchanged — this
revision stands alongside it). Fixes series R's session count and states the D2 equivalence
computation. Not executed — draft only, per this block's "no data collection" order, same as A8
itself.**

- **Session count fixed: 4 fresh-boot sessions**, each running the full **33-condition** design A8
  already specifies — (a) 6 conditions (FP32 eager/TorchScript × CPU/CUDA × 3 architectures) + (b) 27
  conditions (3 architectures × {FP32, FP16, `pruned50_bnrecal`} × {eager, TorchScript, CUDA Graphs})
  = 33, matching A8's own (a)+(b) breakdown exactly; A8 left the session count open, fixed here.
  Seeds **2000 + N** for N = 1..4 (A8's existing seed-family convention, extended to a concrete count).
- **Why 4, not more:** 4 sessions gives **3 degrees of freedom** for the between-session t-interval
  on the eager-vs-TS log-ratio, which is what D2 (A7r1(f)) needs — the minimum that lets a
  between-session SD and a t-based equivalence interval be computed at all (3 sessions would give
  df=2, wider and more fragile; this follows the same reasoning A4 used to move from 4 to 6 main
  sessions, applied here to the smaller series-R design instead of re-litigated from scratch).
- **D2's TOST computation, stated in full:** per session, compute the mean `log(FP32-TS energy /
  FP32-eager energy)` across that session's reps, for the relevant block (CPU or CUDA) and
  architecture — this is the within-session replicate A7 point 3's estimator already uses elsewhere,
  reused here. Across the 4 sessions, compute the mean and SD of these per-session log-ratios (n=4,
  df=3). Equivalence margin: ±5%, i.e. ±ln(1.05) ≈ **±0.04879** in log units. **Two One-Sided Tests
  (TOST):** construct the **90% CI** on the between-session mean log-ratio (90%, not 95% — the
  standard TOST convention: two one-sided tests at α=0.05 each correspond to a 90% two-sided
  interval), using `t(0.95, df=3) ≈ 2.3534`. **Declare equivalence (D2 "bounded") if this 90% CI
  falls entirely within [−0.04879, +0.04879]; otherwise D2 concludes "not bounded"** and the bridged
  ratio (A7r1(a)) becomes primary for that block/architecture.
- **Between-session SD threshold for equivalence to even be detectable at n=4, stated before any
  series-R data exists:** the 90% CI half-width is `t × SD / sqrt(n) = 2.3534 × SD / 2`. For this
  half-width to fit inside the ±0.04879 margin, `SD ≤ 0.04879 × 2 / 2.3534 ≈ 0.0415` (about **4.1%**
  in log units). **This is a precision/feasibility bound, not a prediction of the outcome:** if
  series R's actual between-session SD of the log-ratio exceeds ~4.1%, the TOST cannot conclude
  equivalence at n=4 even if the true mean ratio is exactly 1.0 — the interval would be too wide to
  fit inside the margin regardless of where it's centred. This is disclosed now so a "not bounded"
  result arising from insufficient precision (wide SD) is not misread as evidence of a real eager/TS
  difference without checking which of the two actually happened.

**A7r2 — 2026-10-10. Dated revision correcting A7r1(f)'s D1-D5 wording (A7r1's text above is
unchanged — this stands alongside it, per §12's append-only rule).**

**Provenance, for each of D1-D5 as A7r1(f) wrote them, stated honestly:** A7r1(f) said plainly that
it was "synthesized from what A7/A7r1 and the researcher's own decisions establish, since the
original external document is not part of this repository" — **all five were this session's own
reconstruction. None were quoted from any audit document (none exists in this repository) or from
the exact wording of any prior supervisor block.** That disclosure was accurate as far as it went,
but the reconstruction itself was checked against the audit's actual wording (given directly in this
block) and found to diverge — in two different ways:
- **D1, D2, D5: right topic, wrong or incomplete branch content.** The reconstruction correctly
  identified what each decision rule was *about*, but got specific branch outcomes wrong (D1), used a
  mechanism the audit didn't specify (D2's "bridged ratio" instead of a direct baseline switch), or
  reframed the question as already-resolved instead of stating the original conditional gate (D5).
- **D3, D4: wrong topic entirely.** The reconstruction substituted this block's own open questions —
  confirmatory-family scope (A7r1(b)) and the RQ2 model specification (A7r1(d)) — for what the
  audit's actual D3 and D4 cover (weight-dependence of pruned-state energy, and 224×224-regime
  generalisation respectively). These are real, useful decision rules in their own right — restated
  below as this project's own rules, not the audit's D3/D4 — but they were never D3/D4.

**Collision note:** A7r1(b) and A7r1(d) each contain a cross-reference reading "This is decision
rule D3" / "This is decision rule D4" — those cross-references are now known to point to the wrong
label. A7r1's own text is left unedited per §12; the correction is: **A7r1(b)'s confirmatory-family
rule and A7r1(d)'s RQ2-model rule are this project's own pre-registered decision rules, not part of
the audit's D1-D5 framework.** They are referred to from here on as **P-CF** (confirmatory family,
A7r1(b)) and **P-MAC** (RQ2 model/MAC counting, A7r1(d)) to avoid further collision with the
corrected D3/D4 below. Both remain fully in force exactly as A7r1(b)/(d) stated them — only the
label changes.

**Corrected D1-D5, using the audit's wording as given in this block, verbatim where quoted:**

- **D1r1 (wall meter).** *"Does `delta(NVML power-usage + RAPL package)` agree with the
  wall-derived system delta within ±5% after adapter efficiency?"*
  - **YES:** keep the backend, report the agreement as **validation** (A7r1(f)'s D1 said "report
    the agreement as the instrument-axis finding" — corrected to the audit's own word,
    "validation").
  - **NO, and the cumulative counter agrees instead:** Stage 4b's GPU numbers need a rerun on the
    cumulative interface. **Raw traces store one API per window — both-API logging must exist
    before any such rerun.** For data collected after D24's commit, this prerequisite is already
    satisfied (D24 logs both NVML interfaces as separate columns every window). Sessions 1-6
    predate D24 and store only the power-usage interface in their raw traces — a rerun for them
    would be a fresh collection, not a reprocessing of existing logs.
  - **NO, and neither agrees:** wall energy is the primary boundary for system-level claims; every
    instrument this project uses is reported as a disclosed estimator relative to it.
  - **Correction from A7r1(f):** the earlier text mentioned "RAPL-psys" as part of the cumulative
    counter — **wrong axis.** D1 is about the GPU's NVML interface choice (power-usage vs.
    cumulative-energy counter); RAPL package-vs-psys is a separate, already-resolved CPU-side
    question (D16), not part of D1 at all. The "psys" mention is dropped entirely here.
- **D2r1 (runtime).** *"Is FP32-TorchScript within ±5% (TOST) of FP32-eager?"*
  - **YES:** keep eager-baseline ratios, with disclosure.
  - **NO:** use FP32-TorchScript as the baseline for every ratio; report runtime as a factor in the
    specification curve (A7 point 9) — **corrected from A7r1(f)'s "bridged ratio" formulation**,
    which invented a multiplicative correction mechanism the audit does not specify. The audit's
    actual NO branch is a direct baseline switch, not a correction factor.
  - **Practical note, not part of the audit's own wording — needed because of a real data gap this
    project has:** the x86-GPU block's six main sessions never measured FP32-TorchScript
    directly (A7r1(a)) — there is no direct GPU-block FP32-TS baseline to switch to. **For the GPU
    block only, if D2 resolves NO, the baseline switch uses FP32-TorchScript sourced from series R
    (A8/A8r1)** — not a multiplicative bridge applied to the eager ratio, but the TS baseline
    itself, taken from the one place it was actually measured. CPU blocks are unaffected: each
    CPU-block session already measures its own FP32-TS directly (A7r1(e)), so switching requires no
    substitute source there.
- **D3r1 (weight dependence) — the audit's actual D3; A7r1(f)'s "D3" was a different rule,
  renamed P-CF above.** *"Are `_bnrecal` and zero-finetune energies within ±5% at the same
  architecture?"*
  - **YES:** infer A5's fine-tuned (`_ft`) models' energy from architecture (reuse the
    zero-finetune/`_bnrecal` figures for that architecture as a stand-in), with disclosure that this
    is an inference, not a direct measurement.
  - **NO:** measure A5's fine-tuned models' energy directly (A5's own design, a dedicated extra
    session); report data-dependent power (energy depending on the actual trained weight values, not
    just architecture/sparsity pattern) as a finding in its own right — this would mean the
    project's standing working assumption, that energy tracks FLOPs/architecture and not weight
    values, does not universally hold.
  - **Relevant existing data, read-only, not a ±5% computation (a Stage 4b-style ratio/interval is
    out of scope for this block):** A7r1(g)'s "looked at" statement already records the
    researcher's own descriptive spot check — `_bnrecal`-vs-zero-finetune energies ~1% apart, for
    the one comparison already looked at. Well inside the ±5% band D3r1 asks about, but this is one
    architecture's descriptive spot check, not the full per-architecture ±5% check D3r1 specifies —
    suggestive, not a resolution.
- **D4r1 (regime, 224×224) — the audit's actual D4; A7r1(f)'s "D4" was a different rule, renamed
  P-MAC above.** *"Do ratios keep the 32×32 ordering and the same β < 1 conclusion?"*
  - **YES:** RQ2 generalises across regimes — the 32×32 finding is not an artifact of an
    unrealistically small input size.
  - **NO:** regime-dependence is the headline finding itself; both the 32×32 and 224×224 regimes are
    reported as primary, neither subordinate to the other.
  - **Status, checked against A6 as currently drafted:** A6 ("Realistic-regime sensitivity pass,"
    `stage5_analysis_plan.md`) is explicitly registered as "descriptive and exploratory under §9...
    adds no confirmatory hypothesis." D4r1 requires A6's data to be run through the same β<1 model
    as RQ2/P-MAC a second time, at 224×224 — a confirmatory use A6 does not currently register. This
    gap is noted, not resolved, here; A6 itself is left as-is per §12.
- **D5r1 (scope, original gate) — restated in full per this block's instruction, then superseded.**
  *"Can the M1-CPU arm plus the x86-CPU arm finish within approximately 4 weeks?"*
  - **YES:** one paper (all RQs, cross-platform, combined).
  - **NO:** **Paper 1** = x86 measurement sensitivity + RQ2 + RQ1a; **Paper 2** = cross-platform
    RQ3 + RQ1b. No placeholder RQ3 in Paper 1.
  - **Superseded, 2026-10-09, not by the gate's own YES/NO mechanics:** the researcher's decision
    (`deviation_log.md` D5 update; A7 point 1) directly supersedes this time-budget gate: **M1-CPU
    is in Paper 1**, **x86-GPU is supplementary**, **no M1-GPU arm exists at all** (never planned,
    not a casualty of the time budget).
  - **Fallback if the M1-CPU arm cannot be completed — recorded per this block's explicit
    instruction, decided by the researcher if and when it arises, not triggered automatically by
    session-count or scheduling issues:** the fallback is the **original D5r1 NO branch** — Paper 1
    narrows to x86 measurement sensitivity + RQ2/P-MAC + RQ1a (dropping M1-CPU and RQ3 from Paper
    1), cross-platform RQ3 + RQ1b deferred to a Paper 2. Recorded now, before any such incompleteness
    has occurred or been found — nothing in this review indicates the M1-CPU arm is at risk.

**A8r2 — 2026-10-10. Dated revision adding a pre-registered sentence to A8r1's D2 discussion
(A8r1's text above is unchanged — this stands alongside it).**

**Pre-registered now, before series R exists:** if the between-session SD of the log(FP32-TS ÷
FP32-eager) ratio exceeds **0.0415** (the best-case bound: true difference = 0, 90% CI, df = 3),
equivalence cannot be concluded, and this is treated directly as **D2r1's NO branch — a mechanical
consequence of the threshold, not a judgment call made after seeing the data.** Consequence, per
D2r1 above: **bridged version primary for the GPU block** (TS sourced from series R itself, applied
via A7r1(a)'s bridging construction, since the GPU block has no main-session TS baseline to switch
to directly); **FP32-TorchScript as the direct baseline for the CPU blocks** (each CPU-block session
already measures its own FP32-TS, A7r1(e), so no bridging is needed there).

**Derivation:** the 90% CI half-width at df = 3 is `t(0.95, 3) × SD / sqrt(4)`. For this half-width
to fit inside the ±5% equivalence margin (`ln(1.05) ≈ 0.04879` in log units) even in the best case —
centred exactly on 0, i.e. no bias to overcome:

```
t(0.95,3) × SD / sqrt(4) ≤ ln(1.05)
2.3534 × SD / 2 ≤ 0.04879
SD ≤ 0.04879 × 2 / 2.3534
SD ≤ 0.04146  ≈  0.0415
```

Any observed between-session SD above `0.0415` makes a 90% CI that cannot fit inside ±5% no matter
where it is centred — the NO branch follows mechanically from the SD alone, independent of where the
mean actually falls. This is the same number A8r1 already derived descriptively ("about 4.1%"); this
revision's addition is pre-registering it as a direct trigger for D2r1's NO branch, not merely a
precision caveat to weigh after the fact.

**Appended 2026-10-10 (BLOCK J):** the `0.0415` threshold is a **necessary condition for
equivalence, not a sufficient one** — it is the best-case bound (true difference = 0) on how small
the between-session SD must be for a 90% CI to even fit inside ±5%. An observed SD at or below
`0.0415` does **not** by itself establish equivalence; it only means equivalence remains possible —
the actual 90% CI (centred on the observed mean, not assumed to be 0) must still be computed and
checked against the ±5% margin per D2r1 before concluding YES. SD above `0.0415` is sufficient to
conclude NO (as already stated above); SD at or below it is not sufficient to conclude YES.

**A7r3 — 2026-10-10. Dated revision (A7/A7r1/A7r2's text above is unchanged — this stands alongside
them).**

**(a) GPU-block D2 NO branch: one construction only — the bridged ratio.** A7r2's D2r1 "practical
note" (*"the baseline switch uses FP32-TorchScript sourced from series R... not a multiplicative
bridge... but the TS baseline itself"*) is **superseded by this entry** for the GPU block. The
**only** construction used for the GPU block's D2 NO branch, from here on, is:

```
bridged ratio = ratio_eager × (eager/TS factor from series R)
```

with the eager/TS factor's own between-session uncertainty propagated into the bridged ratio's
interval (relative variances add, since the two factors are independent ratios from different
measurement sources).

- **Reason this supersedes A7r2's practical note:** that note proposed substituting series R's own
  FP32-TS reading *directly* as the GPU block's baseline — i.e. dividing a main-session eager
  measurement by a *different session's* TS measurement. That constructs a **cross-session ratio**,
  which D17 (`deviation_log.md`) already found unsafe in this exact setting: between-session drift
  on this GPU, measured directly, ranges from **−12.20% to +3.70%** on repeated conditions — larger
  than D2's own ±5% equivalence margin. A cross-session substitution could introduce more noise than
  the runtime confound it was meant to fix. The bridged-ratio construction avoids this: the eager/TS
  *factor* from series R is a **ratio of two conditions measured within series R's own sessions**,
  and is then applied multiplicatively to the GPU block's own within-session eager ratio — drift
  affects the factor's own uncertainty (propagated, disclosed) rather than silently contaminating
  the GPU block's numerator with a foreign session's value.
- **This is an adaptation for the GPU block, not the audit's own wording** — restated plainly, same
  disclosure A7r2 already made for the practical note it replaces: the audit's literal D2 NO branch
  (*"use FP32-TorchScript as the baseline for every ratio"*) is a direct baseline switch, which
  works for the CPU blocks (own within-session TS, (b) below) but not for the GPU block (no
  within-session TS to switch to). The bridged ratio is this project's own bridging solution for
  that specific gap, not a transcription of the audit.
- **A8r2's consequence line is also superseded accordingly:** A8r2 already described the GPU
  consequence as "applied via A7r1(a)'s bridging construction," which is consistent with this entry,
  not A7r2's "TS sourced from series R itself" framing — A8r2's wording stands as the correct one;
  A7r2's practical-note sentence is what this entry withdraws.

**(b) CPU blocks: FP32-TS stays the primary baseline regardless of D2's outcome — A7 point 2
reaffirmed.** D2 is still evaluated on the CPU blocks, but for **disclosure only** — it does not
change which baseline is primary there (unlike the GPU block, where D2's outcome is decision-
relevant per (a)).
- **Degrees of freedom: 5** (6 sessions − 1, per the CPU blocks' fixed 6-session count, A7r1(e)).
- **SD limit for the CPU blocks' own D2 check:** `ln(1.05) × sqrt(6) / t(0.95, 5)`. Computed:
  `0.04879 × 2.4495 / 2.0150 ≈ 0.0593` — **about 0.059** (in log units, ≈5.9%), looser than the GPU
  block's 4-session 0.0415 bound, as expected with one more degree of freedom and one more
  replicate.
- **D2 is evaluated per architecture, not pooled across the three** — each CPU-block session
  already measures FP32-TS and FP32-eager for all three architectures (A7r1(e)), giving three
  separate per-architecture equivalence checks per block. **A block counts as "bounded" only if all
  three architectures are** — a single architecture's disagreement is enough to call the block's D2
  result "not bounded" overall, disclosed per-architecture rather than averaged away.

**(c) "Looked at" statement — attribution and scope corrected, A7r1(g) revised.** Two corrections:
- **Attribution:** A7r1(g) (and A7 point 13 before it) attributed the Session 1 spot checks (pinned
  vs. unpinned mean system J/image, **+0.6%**, range **−1.6% to +3.5%**; `_bnrecal`-vs-zero-finetune,
  **~1% apart**) to "the researcher." **This was wrong — both were run by the supervisor review, not
  the researcher.** Corrected here; A7r1(g)'s text is left as-is per §12.
- **Scope:** both spot checks are **comparisons of energy between runs**, not merely per-condition
  means as A7r1(g)'s surrounding language implied — stated precisely so "looked at" isn't read as
  weaker than it actually was.
- **Added to the inventory:** the battery-check work (`docs/stage4b_battery_check.md`) looked at,
  per-session, the mean and minimum GPU power and power-regime classification of `resnet18_fp32` in
  all 9 session directories, plus inspection of `windows.csv`'s column schema and
  `environment.json`'s recorded fields (confirming no power-source field exists pre-D26). **None of
  this constitutes a state-vs-baseline ratio, interval, or hypothesis test** — it is single-condition
  (`resnet18_fp32` only, not compared against any other state) descriptive power/regime inspection,
  for a data-integrity purpose (AC-vs-battery corroboration), not an outcome comparison. Stated
  explicitly so the "looked at" inventory remains complete and accurate after this review's own work
  is folded in.

**(d) D3r1, computation specified in full. Marked: proposal pending researcher confirmation.**
- **Per architecture and prune level** (9 combinations: 3 architectures × {30, 50, 70}): per
  CPU-block session, compute `log(bnrecal energy / zero-finetune energy)` for that architecture and
  prune level, using that session's own paired measurements (both conditions exist in every
  CPU-block session, A7r1(e)).
- **Mean over the 6 sessions**, **90% interval** (TOST-equivalent, same convention as D2), **df = 5**
  (6 sessions − 1) — same session-as-replicate-unit logic as every other confirmatory/decision-rule
  estimator in this plan.
- **Margin: ±5%** (log units, `ln(1.05) ≈ 0.04879`), same convention as D1/D2.
- **Added, a scope clarification not previously stated:** regardless of what D3r1 concludes, **A5's
  fine-tuned models' energy is measured directly in every case** — D3r1 is not a substitute for
  measuring A5's energy, because `_bnrecal` only changes BatchNorm statistics (a handful of
  parameters, no gradient-based weight update) while fine-tuning changes **all** weights; the two
  are not the same kind of post-pruning intervention, and D3r1's YES branch's own "infer from
  architecture" language (A7r2) is read narrower than it may have first appeared: **D3r1 informs the
  discussion of whether weight-dependent power effects exist at all, it does not license skipping
  A5's own direct measurement.**
- **Marked: proposal pending researcher confirmation** — the per-architecture/prune-level
  computation above is specified so it is ready to run once CPU-block data exists; it is not itself
  confirmed as the final method by the researcher as of this entry.

**(e) D4r1: inactive unless A6 is amended. Marked: proposal pending researcher confirmation.**
Restated sharper than A7r2's "status" note: **D4r1 is inactive** — not evaluated, not a pending
computation waiting on data — **unless A6 is itself amended first**, by a **dated revision, written
before any 224×224 data exists**, that explicitly registers the β<1 model (RQ2/P-MAC) as a
confirmatory test to run against A6's 224×224 pass, not merely the "descriptive and exploratory...
adds no confirmatory hypothesis" use A6 currently registers. **Until that amendment exists, D4r1 is
not evaluated at all, and A7r1(j)'s appended limitation sentence applies** (regime dependence stated
as a limitation, no generalisation claim made) — this is the default, active state of this plan
right now, not a fallback contingent on a future negative result. **Marked: proposal pending
researcher confirmation.**

**(f) RQ2 primary β fit: zero-finetune pruned states only; `_bnrecal` is a sensitivity fit —
P-MAC (A7r1(d)) corrected.** A7r1(d)'s model pooled pruned30/50/70 **and** their `_bnrecal`
counterparts into one β fit. Corrected: **the primary fit uses the zero-finetune pruned states
only** (pruned30/50/70, one MAC-ratio value per architecture/ratio pair). The `_bnrecal` states are
excluded from the primary fit and instead form a **separate sensitivity fit**, because a `_bnrecal`
state has the **identical MAC ratio** to its zero-finetune counterpart (recalibration changes no
weight shapes, no channels — `_bnrecal` is BatchNorm-statistics-only, (d) above) — pooling both into
one fit would **double-count** each architecture/ratio pair's MAC-ratio value under two different
energy observations, inflating the effective sample size on the predictor axis without adding an
independent data point.
- **The test, stated precisely:** a **one-sided test of H2: β < 1** (not β ≠ 0 — the hypothesis is
  about the slope being below unity, i.e. energy under-reduces relative to MACs, not merely nonzero).
- **Software and degrees-of-freedom method, named:** fit in **R**, via **`lme4`/`lmerTest`**
  (`lmerTest::lmer`, REML), with **Satterthwaite-approximated degrees of freedom** for the fixed
  effect — the standard small-sample df correction for mixed models, and the reason R is named here
  rather than this project's existing Python toolchain: no equivalent Satterthwaite/Kenward-Roger
  implementation exists in this project's current Python dependencies (`statsmodels.MixedLM` does
  not provide one). This is a disclosed, additional piece of analysis tooling, separate from the
  measurement harness itself (which stays Python/`pilot.py`, unchanged).
- **One-sided p-value, computed explicitly (not read directly off `lmerTest`'s default output,
  which tests against 0, not against 1):** test statistic `t = (β̂ − 1) / SE(β̂)`, left-tailed
  p-value against the Satterthwaite df `lmerTest` reports for that fixed effect.
- **Reported: β̂ with its 95% CI** (symmetric t-interval, same Satterthwaite df), alongside the
  one-sided p-value above — both the point estimate/interval and the directional test, not the test
  alone.

**(g) Specification-curve headline metric, defined before any data exists. Marked: descriptive.**
A7 point 9 names the specification-curve's axes (instrument, boundary, runtime, regime, gross/net)
but never defined a single summary number for "how much the ratio moves across those axes." Defined
here, for a given compressed state, against the set `{log r_s}` of log-ratios computed under every
specification combination `s`:

```
SD_spec = standard deviation of {log r_s} across all specification combinations

headline_1 = SD_spec / w*      (w* = half-width of the primary ratio's own 95% log-r interval)
headline_2 = SD_spec / |log r*|  (r* = the primary-specification ratio itself)
```

- `headline_1` compares the across-specification spread to the primary estimate's own sampling
  uncertainty — a value well above 1 means specification choice moves the ratio more than ordinary
  sampling noise would.
- `headline_2` compares the spread to the effect size itself — a value well above 1 means the
  specification-dependence is large enough to plausibly flip the qualitative conclusion (not just
  add noise around an otherwise-stable effect).
- **Marked descriptive** — consistent with A7 point 9's own framing ("presented as a distribution...
  not a sensitivity-analysis afterthought," never registered as a hypothesis test); no p-value or
  significance threshold is attached to either headline number.

**(h) Void-and-rerun criteria, pre-declared, independent of outcomes.** For any x86-CPU or M1-CPU
primary-block session (not yet run):
1. **AC guard abort** (D26's `check_ac_power()` raising, at condition start or mid-run) — **void the
   entire session**, rerun entirely on a fresh boot. Consistent with D26's own no-override design:
   an AC loss is treated as a session-level integrity failure, not a single-condition exclusion.
2. **Crash** (any `pilot.py` invocation exiting non-zero for a reason other than an AC abort —
   unhandled exception, OOM, killed process) — **void and rerun that condition only**, unless the
   crash's cause itself implies session-wide contamination (e.g. the concurrent-GPU guard tripped,
   §8), in which case void the whole session.
3. **Thermal-throttle flag, a measurable definition proposed for each platform (neither currently
   implemented in the harness — disclosed, not assumed built):**
   - **x86:** mean P-core `scaling_cur_freq` during an active window falls below **80% of
     `cpuinfo_max_freq`** despite the `performance` governor being active (mirrors the GPU's own
     `classify_power_regime` convention — a threshold against a known maximum, not an absolute
     number invented from nothing). **Not yet implemented** — §8 currently has no thermal-throttle
     exclusion category for x86 at all; this is a new gap this entry identifies, not a pre-existing
     guard being restated.
   - **M1:** `pmset -g therm` reporting above `nominal`, **or** a clock-frequency drop relative to
     that session's own early-rep baseline exceeding a threshold **not yet calibrated** (needs a
     real throttle event captured first, per `docs/m1_harness_design.md` §4's own disclosure — this
     entry proposes the *comparative* structure of the check, not a calibrated number). Extends
     §10's existing "thermal/throttle logging (fanless machine)" mention, which named the logging
     requirement but never a measurable flag.
   - **Both:** void and rerun the affected condition only, not the whole session, consistent with
     crash handling above (a throttle event is condition-local, not a session-wide integrity
     failure the way an AC loss is).
4. **Interrupted session** (process killed externally, machine rebooted mid-session, or any power
   loss not caught by the AC guard, e.g. a battery-swap edge case) — **void the entire session**,
   rerun entirely.
- **Consistency with §8:** §8's existing exclusions (plausibility-guard trips, governor-not-
  performance, concurrent-GPU-detected, cold-start rep, M1 swapped-memory) predate D26 and the CPU
  blocks; they remain exclusion criteria (data points dropped from analysis) for the already-
  collected GPU block, where A4's "no conditional add/drop of sessions" rule means no rerun ever
  follows an exclusion there. For the **new** CPU blocks, the criteria above are **additional**,
  not a replacement — §8's existing list still applies as exclusions where a rerun criterion above
  doesn't independently apply (e.g. a cold-start rep is still simply excluded, not void-and-rerun).
- **Consistency with §10:** M1's swapped-memory exclusion (§10, `vm_stat` monitoring) is unchanged
  and is an exclusion, not a void-and-rerun trigger, by the same logic — swapping degrades one
  condition's numbers without indicating a session-wide integrity failure.
- **Seeds, assigned here:** **x86-CPU: `3000 + N`**; **M1-CPU: `4000 + N`** — two new families,
  each distinct from every other seed family this project uses (2026, 1001-1003, 2000+N for series
  R), so no session-order seed is ever confusable with another family's provenance. (`3000+N` is
  already the condition-order seed implemented in `scripts/run_cpu_block_session.py`, confirmed
  consistent with this assignment, not a new choice contradicting existing code.)

**(i) Stale cross-references fixed, dated correction, not an in-place edit:**
- **A7r1(j) point 2** said the P1 wall-meter protocol "does not exist yet." **Stale** — it now
  exists in draft form, `docs/p1_wall_meter_protocol.md` (written since A7r1 was drafted). Current,
  accurate status: **the protocol document exists; no meter has been chosen; the protocol has not
  been run.** D1 remains unevaluated for the same reason as before (no meter, no data), just via an
  updated status, not the original "doesn't exist" gap.
- **A7 point 5** cross-references "the mixed-model robustness check (point 8)" — **wrong point
  number.** The mixed-model robustness check is **point 7** (the RQ2 model); point 8 is the
  gross-vs-net-of-idle primary-outcome declaration, an unrelated point. A7's own text is left as-is
  per §12; this is the correction of record.

**(j) Disclosure: prospective registration status differs by block.** The **x86-GPU block's** six
sessions (`main_session1`-`main_session6` and the unpinned/aborted directories) were **collected
before A7 existed** (A7 is dated 2026-10-09; the GPU sessions' collection dates are earlier,
`docs/stage4b_battery_check.md`'s own per-session timestamps, 2026-10-06 through 2026-10-08) — **the
GPU block's data are not prospectively registered**, regardless of how its analysis is now specified
post hoc. **The two CPU blocks (x86-CPU, M1-CPU) are prospectively registered** — their conditions,
session counts, seeds, and void-and-rerun criteria are all fixed in this plan before any session has
been run. This distinction is stated plainly because it affects how strong a confirmatory claim each
block's results can support, independent of anything else this plan registers about them.
