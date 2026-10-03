# Stage 1 exit test (3rd version — corrected backend)

Uncompressed ResNet-18 (CIFAR stem), x86 platform (Lenovo LOQ 15IRX9), CUDA/RTX 3050,
10 repetitions, CodeCarbon-paired, harness commit `672ce93` — **default CUDA backend
is now `nvmlDeviceGetPowerUsage`** (sampled power, trapezoidal-integrated), not the
cumulative counter used in the two prior exit tests.

Run: `python pilot.py --device cuda --data /home/shohan/green-ai-research/data --out
results/stage1_exit_test --sizes 32 --batches 16 --windows 5 --repeats 10 --interval 0.4
--warmup 3 --codecarbon`
Raw data: `results/stage1_exit_test/{raw.jsonl,windows.csv,summary.csv,environment.json}`.

**This supersedes both prior exit tests** (commits `d2c38b9`, `e2a1798`), which used the
now-known-unreliable `nvmlDeviceGetTotalEnergyConsumption` backend. See
`results/active_power_baseline_investigation.md` for the full correction.

**M1/ARM platform: still not run.** No access to the MacBook Air this session.

| # | Checklist item (plan §2) | Status | Observed value |
|---|---|---|---|
| 1 | Instrument + accuracy class + measurement boundary stated for every number | ✅ | `environment.json`: `backend: "NVML sampled power"`, `gpu_power_cap_w: 60.0`, driver 595.84. |
| 2 | Idle baseline measured before **and** after each block, reported, not silently subtracted | ✅ | `idle_before`/`idle_after` logged every rep (rep 0: 12.65W / 12.34W — both now tightly clustered 11.2-12.65W across all 10 reps, no outliers). |
| 3 | First run of every configuration discarded (cold-cache/cold-thermal) | ✅ | `pairs=9` in `summary.csv` for 10 logged reps. |
| 4 | Sampling-rate rules: RAPL/CPU ≤100Hz; NVML/GPU ≥0.3–0.5s | ⚠️ **re-examined this session, floor's necessity now in question** | Interval still enforced in code (`check_interval_floor`, unchanged). But a fresh sweep with the *corrected* backend (20ms-500ms, same GPU) shows **no interval-dependence at all**: 12.62-12.98W flat across the entire range. The original counter-telescoping finding (20ms→~300W) was specific to `nvmlDeviceGetTotalEnergyConsumption`; it does not reproduce with `nvmlDeviceGetPowerUsage`. The floor is left enabled in code (not asked to remove it this session) but may no longer be necessary — flagging, not deciding. |
| 5 | Batch size recorded; vary 2–3 values as a declared control | ⚠️ partial | `batch: 16` recorded every row. Single value by design for this exit test. |
| 6 | CPU governor fixed to performance mode, every run, both platforms | ✅ | `cpu_governors: ["performance"]`. |
| 7 | Driver/library/runtime versions pinned and logged | ✅ | `torch: 2.7.1+cu118`, `torchvision: 0.22.1+cu118`, `nvidia_driver_version: 595.84`. |
| 8 | Nominal compression reported with realized-compression evidence | N/A | Uncompressed FP32 baseline; applies from Stage 2. |
| 9 | Thermal state and session length logged on the M1 | N/A | M1 not run. |
| 10 | ≥30 runs per condition; mean, SD, 95% CI reported | ❌ expected at this stage | 9 pairs after cold-discard, `candidate_for_confirmation: False`. Correctly deferred to Stage 3/4. |
| 11 | Phase separation: inference energy isolated from data-loading/post-processing | ✅ | Unchanged from Task 5's audit. |
| 12 | Raw traces retained, not just summary statistics | ✅ | Full per-sample traces in `raw.jsonl`. |
| 13 | No concurrent GPU jobs during any measurement run | ✅ | Enforced by `check_no_concurrent_gpu`; confirmed clean by completion + `nvidia-smi` check. |
| 14 | Power profile locked to performance via ACPI `platform_profile`, verified and logged per run | ✅ | `platform_profile: "performance"`. |

## The headline number changed, and got far more precise

| | this run (corrected) | prior exit test (cumulative counter, superseded) |
|---|---|---|
| `gross_j_per_image_mean` | **0.0340** | 0.0507 |
| `idle_j_sd` | **2.50** | 88.85 |
| `total_j_sd` | **0.34** | 19.98 |
| active-phase implied power | **~54-60W** (pinned at the real 60W cap) | ~75-82W (fabricated) |

The energy-per-image figure dropped ~33%, matching the previously-measured ~30%
counter inflation. More strikingly, **the noise (SD) collapsed by roughly
35-60x** — removing the erratic counter artifact didn't just correct the mean,
it made every number in this dataset far more precise and trustworthy.

## The `idle_after` anomaly is gone

All 10 reps' `idle_after` readings: 11.71, 12.54, 12.19, 11.64, 12.20, 11.85,
12.43, 11.71, 12.42, 12.09W — tightly clustered, **no last-rep spike** (compare
to the two prior exit tests' rep-9 jumps to 65.88W and 105.18W). This is strong
further confirmation that the `idle_after` anomaly investigated separately
(`results/idle_after_anomaly_investigation.md`) was itself downstream of the
same cumulative-counter defect, not an independent phenomenon.

## Plausibility flag: only 1/40 rows, and it's boundary noise

`rep=9, a2`: 60.03W — 0.03W over the 60.00W cap, the only flagged row in this
entire run. That's sensor-noise-scale, not a real violation; everything else
sits cleanly at or under the cap.

## CodeCarbon gap, now larger relative to the corrected ground truth

CodeCarbon's per-window estimate (534-595J over a 5s active window, ~107-119W
implied) is now **~2x** the corrected hardware reading (~60W), a bigger
relative gap than the ~50% seen against the old, inflated hardware number.
This isn't a regression — it's the ground truth improving while CodeCarbon's
estimate stayed the same; the instrument-vs-instrument comparison this
proposal's RQ1 depends on is now measuring against an accurate reference.

## Net

All 14 items checked against real logged values from the corrected backend.
13/14 behave as expected or better (✅/N/A/expected-❌); item 4 needs a
decision (keep the floor as a conservative default, or relax it now that its
original justification doesn't reproduce with the corrected backend) before
Stage 3/4. M1 remains the only fully open thread.
