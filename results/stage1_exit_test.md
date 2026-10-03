# Stage 1 exit test

Uncompressed ResNet-18 (CIFAR stem), x86 platform (Lenovo LOQ 15IRX9), CUDA/RTX 3050,
10 repetitions, CodeCarbon-paired, ported harness commit `694b7d6`.
Run: `python pilot.py --device cuda --data "<cifar10 root>" --out results/stage1_exit_test
--sizes 32 --batches 16 --windows 5 --repeats 10 --interval 0.4 --warmup 3 --codecarbon`
Raw data: `results/stage1_exit_test/{raw.jsonl,windows.csv,summary.csv,environment.json}`.

**M1/ARM platform: not run.** Deferred by explicit decision this session — this environment
has no access to the MacBook Air M1. Task 6 (powermetrics setup) is an open item.

| # | Checklist item (plan §2) | Status | Observed value |
|---|---|---|---|
| 1 | Instrument + accuracy class + measurement boundary stated for every number | ✅ | Every row in `raw.jsonl` carries `backend: "NVML cumulative energy"`; `environment.json` records which GPU (RTX 3050 6GB Laptop) and driver (595.84). CPU/RAPL boundary (package-0 vs psys) documented in README from Task 1. |
| 2 | Idle baseline measured before **and** after each block, reported, not silently subtracted | ✅ | `idle_before`/`idle_after` are separate logged phases every rep (e.g. rep 0: 165.5J / 134.6J). `summary.csv`'s primary metric `gross_j_per_image_mean` is NOT idle-adjusted; `above_idle_j_mean` is a distinct, separately-reported column. |
| 3 | First run of every configuration discarded (cold-cache/cold-thermal) | ✅ (fixed this session) | `summarize()` previously included every repeat — a real gap, found during this audit. Fixed in commit `694b7d6`. Confirmed in output: 10 reps logged in `raw.jsonl`, but `summary.csv` shows `pairs=9` (one discarded), not 10. |
| 4 | Sampling-rate rules: RAPL/CPU ≤100Hz; NVML/GPU ≥0.3–0.5s | ⚠️ partial | GPU side confirmed from real log timestamps: `max_sample_gap_s` is 0.404–0.409s across all 40 rows (ran with `--interval 0.4`, and the floor added in Task 2 rejects anything <0.3 without override). CPU/RAPL ≤100Hz is **not enforced in code** — `pilot.py` only checks `interval>0`, no ceiling. Not exercised by this run (GPU-only); flagging as an open gap for the CPU path before Stage 4. |
| 5 | Batch size recorded for every figure; vary 2–3 values as a declared control | ⚠️ partial | `batch: 16` recorded on every row. Not varied — this exit test used one batch size by design (it's a harness-correctness check, not the real matrix). Batch-size sweep is Stage 3/4's job. |
| 6 | CPU governor fixed to performance mode, every run, both platforms | ✅ | Was `powersave` at session start (checked, flagged, you fixed it via `sudo cpupower frequency-set -g performance`). `environment.json` now logs `cpu_governors: ["performance"]` — confirmed set on all 16 logical CPUs before this run. |
| 7 | Driver/library/runtime versions pinned and logged | ✅ (gap found + fixed this session) | `environment.json` was missing the NVIDIA driver version — added in commit `694b7d6`. Now logs `torch: 2.7.1+cu118`, `torchvision: 0.22.1+cu118`, `nvidia_driver_version: 595.84`. |
| 8 | Nominal compression reported with realized-compression evidence | N/A | Out of scope — this is the uncompressed FP32 baseline; applies starting Stage 2. |
| 9 | Thermal state and session length logged on the M1 | N/A | M1 not run this session (Task 6 deferred). |
| 10 | ≥30 runs per condition; mean, SD, 95% CI reported | ❌ expected at this stage | Only 10 reps run (9 after cold-discard) — exit test is explicitly scoped to x10 by the brief, not the full ≥30. `summary.csv` correctly reports `candidate_for_confirmation: False` because `pairs=9<30`. Mean/SD are reported (`idle_j_sd`, `total_j_sd`, `paired_difference_sd`); an explicit 95% CI column is not yet computed — add when scaling to Stage 3/4's ≥30-rep runs. |
| 11 | Phase separation: inference energy isolated from data-loading/post-processing | ✅ | Confirmed by code + real trace timestamps (Task 5): data loading happens once outside the phase loop; `a1`/`a2` windows bracket only `model(x)`. |
| 12 | Raw traces retained, not just summary statistics | ✅ | `raw.jsonl` rows carry the full per-sample `trace` list (timestamp, sensor value pairs); `windows.csv`/`summarize()` only drop `trace` from the in-memory aggregation copy, not from the file. |
| 13 | No concurrent GPU jobs during any measurement run | ✅ (observed, not yet automated) | `nvidia-smi --query-compute-apps` showed no other process during this run. The harness doesn't log this automatically — worth adding an automatic process-count check to `environment.json` before Stage 4, so it's not just a manually-observed claim. |

## CodeCarbon pairing (Task 4) confirmed in this same run

Every `a1`/`a2` row carries both numbers for the identical window, e.g. rep 0:

| phase | energy_j (NVML) | codecarbon_energy_j |
|---|---|---|
| a1 | 407.3 | 625.0 |
| a2 | 419.0 | 636.3 |

CodeCarbon's estimate runs consistently ~50% higher than the hardware counter across all 10
reps — a real, measured instrument gap, not an artifact of this run; left as observed data per
§1.6's disclosure-not-hide convention.

## One anomaly observed, not explained away

`rep=9, idle_after`: 329.7J, roughly double every other idle reading (~140–170J) in this run.
Logged as-is, included in `idle_j_sd` (41.8J — visibly wider than it would be without this
point). Worth a look before Stage 3 scales this up, but not fixed/excluded here — that would be
retroactively cleaning data, which the brief's own idle-baseline rule (item 2) argues against.

## Net: Stage 1 exit test is clean enough to call done, with two items carried forward

- Item 4 (RAPL ≤100Hz ceiling) and item 13 (automatic concurrent-GPU-job check) are small,
  real code gaps, not yet fixed — neither was exercised by this GPU-only exit test.
- Item 10 is intentionally not satisfied yet (10 reps, not 30) — that's Stage 3/4's job, not a
  Stage 1 defect.
- M1 (Tasks 6, 9, and the ARM half of item 9/13) is fully deferred, not attempted.
