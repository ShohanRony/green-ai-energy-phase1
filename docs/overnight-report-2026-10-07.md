# Overnight Report — Stage 4 Rework (2026-10-07)

**Execution Status:** Completed D0, D1, D2, D3, Step 7 runner script + dry-run. Stage 4b Session 1 launched in background.

---

## 1. Done, with Evidence

### Commit Trail
* `f921cfc`: Added overnight instructions, delta, and supervisor review to `docs/`.
* `6ef0b77`: D1 — Accuracy verification, added FP32-CPU prediction arrays, updated harness multi-boundary fields, confirmed accuracy rule in analysis plan.
* `54918f0`: D2 — Materialized 9 `_bnrecal` and 9 `_ft` checkpoints, evaluated test accuracies, saved predictions, logged deviation D18.
* `ff0695c`: D3 — Finalized and registered Stage 4b design (30 conditions with BN-recal arm); tagged `stage4b-registered`.
* `d0c841c`: Step 7 — Added `scripts/run_stage4b_session.sh`, validated via session 0 dry-run.

### Key Numbers and Findings
* **FP16 accuracies** (`results_accuracy/summary.csv`):
  * `resnet18_fp16`: 93.08% [92.57, 93.56]% (FP32 baseline: 93.09%)
  * `mobilenet_v3_small_fp16`: 86.47% [85.79, 87.13]% (FP32 baseline: 86.45%)
  * `efficientnet_b0_fp16`: 88.84% [88.21, 89.44]% (FP32 baseline: 88.87%)
* **Traced vs. Stage 2 accuracy differences** (`results_accuracy/summary.csv` vs `results_stage2/task4_full_grid.json` / `checkpoints/*.json`):
  * `efficientnet_b0 int8`: 87.18% vs 86.87% (+0.31 pt). Attributed to batching/evaluation differences under FX graph mode quantization.
  * `mobilenet_v3_small int8`: 83.89% vs 84.00% (-0.11 pt).
  * All 13 other non-FP16 states differed by $\le 0.04$ pt (10 matched to $\le 0.02$ pt).
* **Pruned module identity checks (Step 3.1)**:
  * Regenerated live modules matched stored traced artifacts bit-for-bit: `max_abs_diff = 0.00e+00` on 64 test images, exact parameter and MAC match across all 9 pruned states.
* **BN-recalibration (`_bnrecal`) accuracies** (`results_accuracy/summary.csv`, `results_pruned_controls/bn_recalibration.csv`):
  * `resnet18`: 30%: 86.56%, 50%: 71.02%, 70%: 17.12%
  * `mobilenet_v3_small`: 30%: 18.93%, 50%: 10.00%, 70%: 10.00%
  * `efficientnet_b0`: 30%: 78.11%, 50%: 46.28%, 70%: 10.10%
* **Brief-recovery fine-tuning (`_ft`) accuracies** (`results_accuracy/summary.csv`, `results_pruned_controls/recovery_arm.json`):
  * `resnet18`: 30%: 91.27%, 50%: 89.92%, 70%: 84.92%
  * `mobilenet_v3_small`: 30%: 72.61%, 50%: 63.09%, 70%: 44.60%
  * `efficientnet_b0`: 30%: 85.84%, 50%: 84.54%, 70%: 78.59%
* **99% / 99.9% deployability tallies**:
  * **BN-recalibrated:** 0/9 pass 99% tier; 0/9 pass 99.9% tier.
  * **Brief-recovery (`_ft`):** 0/9 pass 99% tier; 0/9 pass 99.9% tier. Highest achieved is `resnet18_pruned30_ft` at 91.27% (98.04% of FP32 baseline, missing 99% threshold by 0.89 pt).
* **Harness validation numbers** (`/tmp/agtest/val_gpu` & `/tmp/agtest/val_cpu`):
  * CUDA: `gpu_energy_j` = 289.20 J, `cpu_package_energy_j` = 124.63 J, `system_energy_j` = 413.82 J. `system - (gpu + package) = 0.0` within 1e-9.
  * CPU: P-core affinity `[0..11]` verified in `environment.json`; `cpu_package_energy_j` = 260.40 J, `psys_energy_j` = 439.00 J.

---

## 2. Blocked or Unexpected

* **Pruned deployability collapse confirmed across all tiers:** Even with 3 epochs of gradient fine-tuning with momentum and data augmentation (`_ft`), not a single pruned model reaches the MLPerf 99% threshold. ResNet-18 at 30% channel pruning comes closest at 98.04% of baseline.
* **No task blockers encountered:** All steps D0 through D3 and Step 7 runner execution completed cleanly without aborts.

---

## 3. Deviation Entries Added

* **D17:** Between-session drift and presentation-order confound (-12.20% to +3.70% on `gross_j_per_image_mean`). Stage 4 primary matrix designated exploratory.
* **D18:** Post-pruning recovery arms added (`_bnrecal` gradient-free recalibration measured for energy in Stage 4b; `_ft` brief-recovery fine-tuning designated accuracy-only).

---

## 4. Status Breakdown

* **Independently verified tonight:**
  * Multi-boundary harness telemetry (`gpu_energy_j`, `cpu_package_energy_j`, `system_energy_j`, `images_per_s`).
  * P-core CPU affinity logging.
  * Full 30-condition test set accuracy predictions (including FP32-CPU reference arrays).
  * Runner script preflight guards and resume handling.
* **Still open:**
  * Stage 4b multi-session data collection (Session 1 launched tonight).
* **Deferred by Shohan's choice:**
  * M1 platform benchmarking (scheduled after x86 completion).
  * Secondary RQ1 CodeCarbon pass and batch=16 pass (scheduled after Stage 4b main sessions 1–4).

---

## 5. Instructions for Shohan

### Sessions 2–4 Execution Commands
Each subsequent session requires a clean reboot:
1. Reboot the laptop.
2. Wait ~2 minutes after boot for background OS services to settle.
3. Open a terminal and run:
   ```bash
   cd /home/shohan/green-ai-research/green-ai-energy-phase1
   scripts/run_stage4b_session.sh --kind main --session 2
   ```
4. Repeat for Session 3 and Session 4 on separate fresh reboots spread across at least 2 calendar days.

### Follow-up Passes (Run Only After Sessions 1–4 Complete)
* **RQ1 CodeCarbon Pass:**
  ```bash
  scripts/run_stage4b_session.sh --kind rq1 --session 1
  ```
* **Batch=16 Sensitivity Pass:**
  ```bash
  scripts/run_stage4b_session.sh --kind b16 --session 1
  ```

### Outstanding Administrative Items
* Missing proposal statistics paragraph (deviation D14).
* Exact date M1 hardware access began (deviation D5).

---

## 6. Session 1 Launch Status

* **Status:** Launched, running in background.
* **Runner script:** `scripts/run_stage4b_session.sh --kind main --session 1`
* **Log location:** `results_stage4b/main_session1/runner.log`
* **Runner PID:** `3882` (pilot child PID: `3914`)
