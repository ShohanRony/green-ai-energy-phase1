# Stage 4b: Confirmatory x86 Dataset — Registered Design

**Status: registered and finalized (2026-10-07). Tagged in git as `stage4b-registered`.**
Supersedes the initial draft (`85d966c`), which remains in git history (see `stage5_analysis_plan.md` amendment A3).

---

## 1. Conditions (30 total)

Each session executes all 30 conditions in a freshly drawn random sequence:

For each model (`resnet18`, `mobilenet_v3_small`, `efficientnet_b0`):
1. `fp32` (cuda)
2. `fp16` (cuda)
3. `pruned30` zero-finetune (cuda)
4. `pruned50` zero-finetune (cuda)
5. `pruned70` zero-finetune (cuda)
6. `pruned30_bnrecal` BN-recalibrated (cuda)
7. `pruned50_bnrecal` BN-recalibrated (cuda)
8. `pruned70_bnrecal` BN-recalibrated (cuda)
9. `int8` (cpu)
10. `fp32` (cpu)

*Note on brief-recovery (`_ft`) models:* The 9 brief-recovery models (`checkpoints/{model}_pruned{N}_ft.pt`) are evaluated for accuracy only (`results_accuracy/`). Energy draw is assumed equivalent to the corresponding zero-finetune and `_bnrecal` architectures, conditional on the D2 hypothesis test (verifying whether energy depends on weight/stat values at fixed architecture within ±5%).

---

## 2. Fixed Settings

* **Harness parameters:** `--sizes 32 --batches 1 --windows 5 --interval 0.4 --warmup 3 --threads 4 --cpu-affinity pcores`
* **System control:** CPU governor locked to `performance`, ACPI platform power profile locked to `performance`, CodeCarbon disabled.
* **Concurrent RAPL:** `--concurrent-cpu-package` enabled on all CUDA runs.
* **Repetitions:** **7 repeats** per condition per session. The first repeat (repeat 0) is discarded as cold-cache/cold-thermal per checklist item 3, leaving 6 warm pairs (12 active windows).

---

## 3. Sessions and Presentation Order

* **Number of sessions:** **4**, each executed in a **separate fresh boot**, spread across at least 2 calendar days.
  * Session 1 executes in the current boot with uptime recorded.
  * Sessions 2–4 each start within 15 minutes of a clean system reboot.
  * The session count is fixed at 4 a priori; no sessions are added or removed conditionally.
* **Presentation order:** Drawn independently per session via `random.Random(1000 + session_id)`. The permutation is written to `results_stage4b/session{N}/order.txt` prior to execution.
* **Replication unit:** The **session** is the primary unit of replication. Within-session repeats describe within-session repeatability only.

---

## 4. Energy Boundaries

* **Primary boundary:**
  * **GPU states:** **system energy = NVML GPU power + RAPL CPU package-0**, adhering to the MLPerf Power full-system principle (Tschand et al., IEEE HPCA 2025, §III-C).
  * **CPU states:** **RAPL CPU package-0**.
  * Direct energy comparisons between GPU states and CPU states are never made.
* **Secondary boundaries (GPU states):**
  * GPU-only (NVML sampled power).
  * CPU-package-only (RAPL `package-0`).

---

## 5. Primary Outcome and Statistical Analysis

* **Primary outcome:** Gross energy per image in Joules (`gross_j_per_image_mean`).
* **Primary comparisons:** Within-session ratio of compressed state to same-model, same-device FP32 baseline:
  $$r = \frac{\text{J/image}(\text{state})}{\text{J/image}(\text{FP32})}$$
* **Primary estimation:**
  * Per-comparison geometric mean of $r$ across the 4 independent sessions.
  * 95% confidence interval derived from the Student's $t$-distribution on $\log r$ ($df = 3$).
  * Confirmatory linear mixed-effects model robustness check:
    $$\log(\text{J/image}) \sim \text{condition} + (1 \mid \text{session})$$
* **Smallest Effect of Interest (SESOI):** $\mathbf{\pm 5\%}$ (anchored to observed between-session drift; D17):
  * Equivalence established via Two One-Sided Tests (TOST) on $\log r$ with margin $\log(1.05)$. A ratio whose 95% CI falls entirely within $[0.952, 1.050]$ is classified as "practically equivalent."
  * A CI spanning $\pm 5\%$ that fails to exclude 1.0 is reported as "unresolved" rather than "no effect."
* **RQ2 (FLOPs vs. Energy):** Realized MAC reduction ratio vs. energy ratio for pruned states; bit-width adjusted BOPs for FP16 and INT8.

---

## 6. Accuracy and Deployability Criteria

* Accuracy evaluated over the full CIFAR-10 test set ($n = 10,000$) from `results_accuracy/` with 95% Wilson score CIs.
* Deployability threshold confirmed by Shohan (2026-10-06):
  * **Primary (MLPerf closed division tier):** Top-1 accuracy $\ge 99.0\%$ of same-model same-device FP32 baseline.
  * **Strict tier:** Top-1 accuracy $\ge 99.9\%$ of baseline.
* McNemar tests performed against same-device FP32 using per-image prediction arrays.
* Full Pareto frontiers presented with all evaluated states shown and labeled without exclusion.

---

## 7. Execution and Exclusion Rules

* **Interruption handling:** If a condition fails or is interrupted by power loss, it is re-run once at the end of the session. If it fails a second time, it is marked failed and excluded from that session.
* **Exploratory status of Stage 4:** The initial Stage 4 dataset is formally designated exploratory (D17) and is never pooled with Stage 4b confirmatory data.
* **Secondary / Follow-up passes (same harness version, not pooled):**
  * RQ1 CodeCarbon pass: 1 session, all 30 conditions with `--codecarbon`.
  * Batch=16 sensitivity pass: 1 session, all 30 conditions with `--batches 16`.

---

## 8. Amendments (Dated Additions)

The following amendments reflect structural adjustments required to elevate Phase 1 to a Q1–Q2 publication standard, explicitly recorded prior to Stage 5 inferential analysis.

### A4. Six Measurement Sessions (Added 2026-10-07)
**Rationale:** Four sessions provide 3 degrees of freedom, requiring a 95% t-multiplier of 3.18 (resulting in wide intervals). Six sessions provide 5 degrees of freedom (t-multiplier 2.57), narrowing intervals by ~19%.
**Change:** The confirmatory matrix is expanded from exactly four sessions to exactly six. The count is fixed at six prior to any inferential analysis (no stopping rule).
- Sessions 1 and 2 (already run) are valid.
- Sessions 3, 4, 5, and 6 will be run following the established fresh-boot and time protocols.
- The six sessions must span at least 3 calendar days to capture real-world temporal variability.
- Seeds for added sessions follow the existing logic: `1000 + session_number`.

### A5. Proper Pruning Pipeline (Added 2026-10-07)
**Rationale:** Pruning without fine-tuning, or with merely a 3-epoch recovery, rarely achieves deployable accuracy (99% of FP32).
**Change:** The matrix includes a formal "prune + fine-tune" arm. For `pruned30` and `pruned50` states (with `pruned70` as optional), a 20–30 epoch fine-tuning run will be conducted. This replaces the preliminary 3-epoch recovery (`_ft`) arm and is explicitly labeled as not post-training. The energy of these fine-tuned models will be empirically measured in one additional measurement session once the models exist.

### A6. Realistic-Regime Sensitivity Pass (Added 2026-10-07)
**Rationale:** To preempt reviewer objections that 32×32 inputs at batch 1 on an RTX 3050 measure host/launch overhead rather than true model efficiency.
**Change:** A separate, energy-only measurement pass using 224×224 inputs at batch sizes 1 and 16 will be run for the three architectures. This pass makes no accuracy claims (random or ImageNet weights are acceptable) and its data will **not** be pooled with the main 32×32 matrix.

### A7. Physical Ground Truth (Added 2026-10-07)
**Rationale:** NVML and RAPL are software models of power. Relying on them exclusively without wall-socket validation risks rejection on methodological grounds.
**Change:** A plug-in power meter (logging at 1 Hz or faster) will be used to measure wall power on the charger during idle and sustained load (10 minutes per level). The correlation and offset against the software-reported system energy (NVML + RAPL) will be reported. Absolute equality is not claimed (due to PSU/fan/display losses).

### A8. Pre-Registration via Zenodo (Added 2026-10-07)
**Rationale:** Establishing a timestamp outside the project's control guarantees the design was fixed prior to analysis.
**Change:** The registered design and amendments will be deposited on Zenodo to secure a DOI before any Stage 5 statistics are computed.
