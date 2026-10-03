# Stage 1 exit test (regenerated)

Uncompressed ResNet-18 (CIFAR stem), x86 platform (Lenovo LOQ 15IRX9), CUDA/RTX 3050,
10 repetitions, CodeCarbon-paired, harness commit `2b81df2` (all four pre-flight
guards in place: NVML floor, RAPL ceiling, concurrent-GPU, platform-profile).
Run: `python pilot.py --device cuda --data /home/shohan/green-ai-research/data --out
results/stage1_exit_test --sizes 32 --batches 16 --windows 5 --repeats 10 --interval 0.4
--warmup 3 --codecarbon`
Raw data: `results/stage1_exit_test/{raw.jsonl,windows.csv,summary.csv,environment.json}`.

This supersedes the previous exit test (committed under `d2c38b9`), which only had
13 checklist items and ran before the RAPL ceiling, concurrent-GPU guard, and
platform-profile guard existed.

**M1/ARM platform: still not run.** No access to the MacBook Air this session.
Task 6 (powermetrics setup) remains an open item.

| # | Checklist item (plan §2) | Status | Observed value |
|---|---|---|---|
| 1 | Instrument + accuracy class + measurement boundary stated for every number | ✅ | Every row carries `backend: "NVML cumulative energy"`; `environment.json` records GPU (RTX 3050 6GB Laptop) + driver (595.84). CPU/RAPL boundary (package-0 vs psys) documented in README. |
| 2 | Idle baseline measured before **and** after each block, reported, not silently subtracted | ✅ | `idle_before`/`idle_after` logged every rep (rep 0: 161.1J / 145.8J). `gross_j_per_image_mean` in `summary.csv` is not idle-adjusted; `above_idle_j_mean` is a separate column. |
| 3 | First run of every configuration discarded (cold-cache/cold-thermal) | ✅ | `summary.csv` shows `pairs=9` for 10 logged reps — one discarded by `summarize()`'s cold-run guard (fixed in `694b7d6`). |
| 4 | Sampling-rate rules: RAPL/CPU ≤100Hz; NVML/GPU ≥0.3–0.5s | ✅ **both halves now enforced in code** | GPU side confirmed from real timestamps: `max_sample_gap_s` is 0.404–0.409s across all 40 rows. CPU/RAPL ≤100Hz ceiling, missing in the previous exit test, is now enforced via `check_interval_floor()` (commit `b23e774`) — not exercised by this GPU-only run, but verified separately with a real `--device cpu --interval 0.005` invocation (rejected by default, proceeds with a logged warning under `--override-fast-rapl-interval`). |
| 5 | Batch size recorded; vary 2–3 values as a declared control | ⚠️ partial | `batch: 16` recorded every row. Not varied — single batch by design for this harness-correctness check; sweep is Stage 3/4's job. |
| 6 | CPU governor fixed to performance mode, every run, both platforms | ✅ | `environment.json` logs `cpu_governors: ["performance"]`, confirmed on all 16 logical CPUs before this run. |
| 7 | Driver/library/runtime versions pinned and logged | ✅ | `torch: 2.7.1+cu118`, `torchvision: 0.22.1+cu118`, `nvidia_driver_version: 595.84`. |
| 8 | Nominal compression reported with realized-compression evidence | N/A | Out of scope — uncompressed FP32 baseline; applies from Stage 2. |
| 9 | Thermal state and session length logged on the M1 | N/A | M1 not run this session. |
| 10 | ≥30 runs per condition; mean, SD, 95% CI reported | ❌ expected at this stage | 10 reps run (9 after cold-discard) — exit test is explicitly scoped to x10, not the full ≥30. `candidate_for_confirmation: False` because `pairs=9<30`, correctly. Explicit 95% CI column still not computed (mean/SD are); add when scaling to Stage 3/4. |
| 11 | Phase separation: inference energy isolated from data-loading/post-processing | ✅ | Unchanged from prior audit (Task 5) — data loading happens once outside the phase loop; `a1`/`a2` windows bracket only `model(x)`. |
| 12 | Raw traces retained, not just summary statistics | ✅ | `raw.jsonl` rows carry the full per-sample `trace`; only the in-memory aggregation copy drops it. |
| 13 | No concurrent GPU jobs during any measurement run | ✅ **now automated, not just manually observed** | `check_no_concurrent_gpu()` (commit `0082612`) runs at the top of every `window()` call and did not abort — confirmed clean by the run completing (exit 0) and by `nvidia-smi --query-compute-apps` showing no other process immediately after. Previously this was a manual-only check; now it's enforced in code on every window, every run. |
| 14 | Power profile locked to performance via ACPI `platform_profile`, verified and logged per run | ✅ **new this session** | `check_platform_profile()` (commit `2b81df2`) read `/sys/firmware/acpi/platform_profile`. Verified it actually gates: the run aborted earlier today when the profile was `balanced`, and proceeded only after you set it to `performance` via sudo. `environment.json` logs `platform_profile: "performance"` for this run. |

## CodeCarbon pairing (Task 4) reconfirmed in this run

Every `a1`/`a2` row carries both numbers for the identical window, e.g. rep 0:

| phase | energy_j (NVML) | codecarbon_energy_j |
|---|---|---|
| a1 | 420.7 | 644.2 |
| a2 | 453.3 | 668.7 |

Same ~50% gap as the previous exit test — consistent instrument disagreement, not noise.

## Recurring anomaly: last rep's final idle window reads high

`rep=9, idle_after`: 526.3J this run, vs ~140–170J for every other idle reading.
**This is the same pattern as the previous exit test**, where `rep=9, idle_after` was
329.7J against a ~140–170J baseline — anomalously high in the same position (the very
last window of the run) both times. Two occurrences in the same slot is enough to call
this a pattern worth investigating before Stage 3 scales up (candidate causes: something
tearing down at end-of-run, a delayed GPU clock/power-state transition, OS background
activity triggered by run completion), not random noise. Not fixed or excluded here —
flagged as-is for the next session to chase down.

## Net: all 14 items checked against real logged values

- **10/14 ✅** (including both items added this session: 13's automation, 14 new).
- **2 N/A** (compression evidence, M1 — out of scope this session).
- **1 ❌ expected** (≥30 reps/95% CI — correctly deferred to Stage 3/4).
- **1 ⚠️ partial** (batch-size sweep — single value by design, not yet varied).
- M1 (Task 6, and the ARM half of items 9/13/14) remains fully deferred.
- One recurring anomaly flagged (last-rep idle_after), not yet explained.
