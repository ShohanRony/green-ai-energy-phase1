# Stage 4 Supervisor Review: whole-repository audit

**Repo:** `ShohanRony/green-ai-energy-phase1`, reviewed at commit `70f5d1d` on 2026-10-06. 48 commits.
**Method:**
- I read the harness (`pilot.py`) line by line.
- I recomputed key Stage 4 numbers independently from `raw.jsonl`.
- I compared the fresh-session stability reruns against the primary runs.
- I parsed the `nvidia-smi dmon` logs.
- I verified the MLPerf Power citation against the PDF in the project library.
- Two delegated audits covered the training, compression and accuracy pipeline and documentation consistency. I re-checked the critical ones myself; the items marked "inferred" below were not run.

**Bottom line:** the measurement *engineering* is careful and honest. The harness logic is sound, the D16 fix is correct, and the deviation log is unusually candid. The *experimental design* has gaps that would decide whether a reviewer believes the results:

- Between-session drift is about the same size as several of the compression effects.
- Session is confounded with condition order.
- GPU energy excludes the host CPU, which is the bottleneck at batch=1.
- The accuracy side is incomplete. FP16 was never measured, and the per-image predictions needed for McNemar were never saved.

None of this is fatal. All of it is fixable in roughly 2–3 evenings of machine time. But Stage 4 is **not done**, and no statistics should run yet.

---

## 1. What is solid (independently verified)

- **Harness logic (`pilot.py`).**
  - Phase separation works: idle_before → a1 → a2 → idle_after.
  - The cold first rep is discarded but kept in the raw data.
  - NVML sampled-power trapezoid integration is correct.
  - RAPL counter wraparound is handled.
  - The guards are in place: interval floor and ceiling, concurrent GPU job, platform profile, fresh boot, and no-overwrite.
  - TF32 is disabled. Seeds are fixed.
- **D16 fix is correct.**
  - Package-0 is now primary and psys is logged separately.
  - My own recompute of `resnet18_int8` from the raw trace gives 0.080085 J/img, the same as `summary_package.csv`.
  - The package-vs-psys column assignment is cross-validated against CodeCarbon's independent package read (ratio 0.999).
- **Stage 4 primary matrix is complete as data.**
  - 21 directories (18 states plus 3 FP32-CPU), each with `pairs=30`.
  - Every run has governor `performance` and `codecarbon=False`.
  - There are no OOM skips and no plausibility flags in Stage 3, Stage 4 or pre-flight.
- **Calibration hygiene.** INT8 calibration uses training images, not the test set, so there is no test leakage.
- **Documentation honesty.** D1–D16 record results-informed decisions as AFTER, including my own batch-size inconsistency. That is the right standard.
- **The MLPerf citation is correct.** Tschand et al., *MLPerf Power*, IEEE HPCA 2025, arXiv:2410.12032. The 99% / 99.9% of FP32 accuracy tiers are stated in §V-D. The same section gives MLPerf Tiny's CIFAR-10 ResNet an absolute 85% target.

---

## 2. Critical issues

### C1. Between-session drift is as large as the effects, and session is confounded with order (verified)

The 12 "dip stability" reruns were judged only on the regime *label* (12/12 still `dip`). Comparing their **energy** with the primary runs:

| Condition | Primary uptime → rerun uptime | Power | Throughput | **J/image** |
|---|---|---|---|---|
| eff_pruned70 | 286 → 26 min | −5.1% | +8.0% | **−12.1%** |
| eff_pruned50 | 269 → 24 | −3.9% | +6.0% | **−9.3%** |
| mbv3_pruned70 | 153 → 16 | −4.0% | +2.9% | **−6.7%** |
| mbv3_fp16 | 103 → 9 | +0.7% | +6.7% | **−5.6%** |
| eff_fp32 | 171 → 18 | −3.7% | +1.9% | **−5.5%** |
| mbv3_fp32 | 40 → 7 | +1.0% | +6.4% | **−5.1%** |
| eff_pruned30 | 252 → 22 | −8.0% | −11.3% | **+3.7%** |
| resnet18_pruned50 / 70 | 70 / 23 → 3 / 5 | ≈ −1% | ≈ −1% | **≈ 0%** |

(Reruns have only 3 pairs each, so this is indicative, not precise.)

- **Size of the drift.** Session-to-session differences reach −12% to +4% in J/image. Within-run CVs are about 1%, so within-run SDs understate the real uncertainty by up to roughly 10×.
- **Size of the effects being compared.** Some of the differences you want to compare are of similar size. EfficientNet pruned30, 50 and 70 differ by about 9% per step. MobileNetV3-Small pruned30, 50 and 70 are within 5% of each other.
- **Order confound.** Within each model, FP32 was measured first and the compressed states later, at steadily increasing uptime. EfficientNet went from 171 min at FP32 to 286 min at pruned70. Any uptime- or session-linked drift is therefore confounded with compression state.
- **Further evidence.** The CodeCarbon check v1 and v2 runs (same protocol, 1h45m apart) also moved by about 6% on two of three conditions.
- **Session separation for CPU states.** INT8 (session 1–2) and FP32-CPU (session 3, about 7 hours later) were measured in different boots.
- **Consequence.** Mann-Whitney U on 30 serial reps will give tiny p-values for differences that are partly session drift. This is pseudo-replication: the real replicate unit is the session, and there is one session per condition.

### C2. GPU energy boundary excludes the host CPU, which is the bottleneck at batch=1 (verified for boundary; mechanism partly inferred)

- GPU-state energy is NVML only; the CPU package is not measured in GPU runs.
- Throughput shows the batch=1 workloads are launch-bound (CPU-bound):
  - ResNet-18 pruned50 runs at 1,768 img/s and pruned70 at 1,684 img/s. Cutting MACs further brought no speed-up.
  - MobileNetV3-Small pruned30, 50 and 70 all run at about 1,280–1,310 img/s.
- In launch-bound workloads the host CPU's energy per image is roughly constant and large. Excluding it inflates relative savings, so RQ2's central conclusion depends on this boundary choice.
- MLPerf Power (§III-C, "Myth #1") states explicitly that measuring only the accelerator is insufficient.
- **Fix:** read RAPL package-0 alongside NVML in every GPU run, and report GPU-only, CPU-package, and GPU+CPU as three boundaries.

### C3. Accuracy side is incomplete (verified)

- **FP16 accuracy was never measured.** `fp16_speedup_check.py` and `materialize_checkpoints.py` use random inputs only. `stage2_deliverable.md` reports FP16 = FP32 "numerically identical", which is an assumption presented as a result.
- **No per-image predictions are saved anywhere.** The plan's McNemar test ("paired by test-set image index") cannot be computed.
- **The accuracy-evaluated objects are not the energy-measured objects.** Pruned and FP16 accuracy came from live modules, but `pilot.py` measures TorchScript traces, which were never re-evaluated.
- **Most compressed states fail the 99% rule.** With the measured numbers, 11 of the 15 compressed states fail. Only ResNet-18 INT8 (93.14 vs 93.09) passes, plus the three FP16 states, which are unmeasured. MobileNetV3-Small INT8 reaches 97.2% of FP32 and EfficientNet-B0 INT8 reaches 97.7%. **Every pruned state fails:**
  - ResNet-18: p30 64.84%, p50 17.58%.
  - MobileNetV3-Small: p30 17.13%.
  - EfficientNet-B0: p30 60.18%.
- **Plan §6 understates this.** It says only the 70% states (and some 50% states) will fail.

### C4. The pruning arm produces non-deployable models (verified; design judgement)

- **Zero fine-tuning is a valid "post-training only" choice (D4), but its consequence is severe.** 30% channel pruning already destroys accuracy, so the pruned-state energy numbers describe models no one would deploy.
- **The labels mislead.** "30/50/70%" is a per-layer *channel* ratio. Realised MAC reduction (`task4_full_grid.json`) is 52%, 75% and 91% for ResNet-18, 47%, 69% and 87% for MobileNetV3-Small, and 48%, 71% and 88% for EfficientNet-B0. The tables need to say "channel ratio" and give realised MACs.
- **A cheap control is missing:** recalibrating BatchNorm statistics without gradients. A reviewer will ask whether the collapse is just stale BN statistics, especially since several states sit at exactly 10.00% (one predicted class). Show the predicted-class histogram too.
- **Decision for you:** keep zero-finetune as the primary arm, and add a clearly labelled *brief-recovery* arm (D4 already shows 3 epochs takes ResNet-18@70% from 10.35% to 84.24%). Architecture, and therefore compute, is identical, so energy should change little. That is an assumption to check with 2–3 short runs.

---

## 3. Major issues

- **M1. The mechanism claim is overstated, and my stability check was underspecified.**
  - dmon for ResNet-18 pruned70 shows SM busy 89–96% during active windows at 1,965 MHz.
  - That is not "the GPU idles between launches". Low per-kernel occupancy is the more likely cause (inferred).
  - MobileNetV3-Small FP32 (58% active SM) does show gaps.
  - Only 2 of 12 conditions had dmon logs, and none were EfficientNet.
  - Pre-flight dips in *heavy* configs (FP16@b64, pruned70@b32) are still unexplained, so "not a mystery" (brief §5.3, D7) is too strong.
  - My Part B spec asked only whether the regime label reproduced. It should have required energy agreement within a stated tolerance. That is my error.
- **M2. INT8 pipelines differ across models (verified).**
  - ResNet-18 uses eager PTQ with 1,280 calibration images, no AdaRound or bias correction, and no seed.
  - MobileNetV3-Small and EfficientNet-B0 use FX PTQ with layer-wise AdaRound and bias correction on 512 images.
  - Cross-model INT8 comparisons are confounded by this.
  - *Inferred, not run:* AdaRound is applied before BN folding, so its benefit may be lost. Bias correction only hits convs that have a bias. An ablation against plain PTQ would settle it.
- **M3. Training design (verified).**
  - ResNet-18 trained for 30 epochs, the other two for 60.
  - One seed, with cuDNN not deterministic.
  - There is no validation split, and decisions (the retrain, the pruning policy) were made after seeing test accuracy.
  - Disclose all of this as forking paths.
- **M4. Pre-registration hygiene (verified via `git diff 0a358c6 HEAD`).**
  - Amendment A1 records the first batch of edits. Later commits (`abed585`, `6d361b5`) edited §3 and §6 in place with no A2.
  - A1's own text still says D16 is "not yet remediated".
  - No git tag exists yet.
  - D13 and the stability data were collected *after* registration but are listed as primary.
- **M5. Stale or contradictory status lines.**
  - D8 and plan §7 say the stability reruns were "not yet executed", but they ran in `e2df9c1`.
  - D15 still says D16 is "not yet remediated".
  - Brief §2 still says "no MacBook access", and brief §5(c) still says "~3 hours".
  - The execution plan still lists "psys vs package: one-line check" as open. That unchecked item is D16's root cause.
  - `stage3_deliverable.md` still labels INT8 energy as "RAPL package".
- **M6. Stage 4 deliverables are missing.** `stage4_deliverable.md`, `stage4_task_report.md`, the Task 2 plausibility pass and the Task 3 checklist do not exist. Under our rules Stage 4 cannot be called done.
- **M7. CPU runs use `--threads 4`, unpinned, on a hybrid P-core/E-core CPU (i5-13450HX).** The scheduler can place threads on different core types run to run. That is a plausible variance source (inferred). Pin with `taskset` to P-cores and log it.

## 4. Minor issues

- `requirements.txt` pins torch 2.5.1 and torchvision 0.20.1, but the checkpoints were made with 2.7.1 and 0.22.1. `torch_pruning` is missing from it.
- Data paths are hard-coded to `/home/shohan/...`.
- README errors:
  - It says RAPL is read via "perf-events"; the code reads sysfs powercap.
  - It calls the harness "paired RAPL+NVML"; each run uses one backend.
  - It says to run `unittest discover -s tests`, which skips the real tests in root `test_pilot.py`.
- Docs quote 736.02 / 266.21 J, which include the cold rep. The summary values are 736.30 / 266.24. The conclusion is unchanged.
- Plan §8 says "zero plausibility flags across Stage 1–4", but the Stage 1 exit test has one documented flag.
- Every batch=1 window re-runs the same single test image. That is acceptable for dense compute, but disclose it.
- The `approx_mde` and `candidate_for_confirmation` fields use within-rep A/A differences. They understate real uncertainty (see C1), so don't cite them.
- **Not verifiable from the repo:**
  - The proposal `.docx`, so D14 and the "verbatim" RQs cannot be re-checked.
  - `resnet18_pruned50_rerun1`, which was deleted.
  - The powersave run that "landed pinned" (D7).
  - CodeCarbon check v1, which was overwritten.
  - The data file for the psys idle/load check.

---

## 5. Status

**Independently verified:**
- Harness correctness.
- The D16 diagnosis and fix.
- 21/21 primary runs complete with consistent settings.
- CodeCarbon absent from Stage 4.
- Train-set calibration.
- The MLPerf citation.
- The accuracy table, which matches its JSON sources.

**Open / unresolved:**
- C1 drift and order confound; C2 measurement boundary.
- C3 FP16 accuracy, per-image predictions, and evaluation of the traced artifacts.
- C4 pruning BN control and the fine-tuning arm decision.
- M1 mechanism wording; M2–M3 disclosures; M4–M6 documentation; M7 thread pinning.
- RQ1 (no paired CodeCarbon data).
- Batch=16 sensitivity.
- The proposal text (D14).
- The 99% threshold confirmation.

**Deferred by your explicit choice (not broken):** the M1 stages (after x86), and batch=1 as the headline. The batch-size sweep is a secondary sensitivity analysis.

---

## 6. Recommended path (nothing runs until you approve)

1. **Accuracy completion** (minutes of compute, no energy runs).
   - Evaluate all 18 TorchScript artifacts that `pilot.py` actually loads, including FP16, on the 10,000 test images, on one device per comparison.
   - Save per-image predictions.
   - Run the BN-recalibration control and predicted-class histograms for the pruned states.
2. **Harness patch.**
   - Dual sensor: RAPL package-0 alongside NVML in GPU runs.
   - `taskset` pinning for CPU runs.
   - Log img/s per window.
   - Unit-test it and validate it on one short run.
3. **Stage 4b: replicated, counterbalanced matrix** (the confirmatory dataset).
   - At least 3 fresh-boot sessions on different days.
   - Each session runs all 21 conditions in a randomised order, with about 10 reps each.
   - Session is the replicate unit, and the analysis uses per-session compressed/baseline ratios.
   - The cost is about the same total machine time as Stage 4 (around 6–7 hours), spread across sessions.
   - The existing Stage 4 matrix becomes exploratory (one ordered session), not discarded.
4. **Register the analysis for 4b before running it.**
   - Ratios with between-session CIs, and a mixed model with session as a random effect.
   - Mann-Whitney is kept only as within-session description.
   - Equivalence margins.
   - Three energy boundaries.
5. **Then** run the RQ1 CodeCarbon pass (separate, because of the 5.8% throughput hit) and the batch=16 sensitivity run, both on the patched harness.
6. **Decide the pruning arm** (zero-finetune only, or zero-finetune plus brief recovery) before step 3, so the recovered checkpoints can be included.
7. **Documentation:**
   - An A2 amendment.
   - Fix the status lines.
   - The Stage 4 deliverables.
   - Disclosure of the INT8 pipeline differences, epochs, seed, and channel-ratio labelling.
   - Commit the proposal text, or quote it verbatim into the repo.
   - Tag the plan only after steps 4 and 6.

**Honest note on my own role:** I recommended the batch=1, sequential, one-session-per-condition design, and I specified the stability check in terms of the regime label rather than energy. C1 and part of M1 trace back to those choices.

## 7. Decisions needed from Shohan

1. Approve Stage 4b (replicated, counterbalanced) as the confirmatory design. *Recommended.*
2. Pruning arm: zero-finetune only, or add a labelled brief-recovery arm. *Recommended: add it.*
3. Confirm the 99% / 99.9% accuracy rule.
4. Provide the proposal's statistics paragraph and the date M1 became available.
