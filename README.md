# green-ai-energy-phase1

> **⚠️ Superseded-data notice (2026-10-03):** every GPU energy number collected
> in this project before commit `672ce93` — both prior Stage 1 exit tests, the
> original NVML interval sweep, the `idle_after` and concurrent-GPU
> investigations' raw data, and **the original `research/energy-pilot` pilot's
> `0.0447 J/image` finding** — was measured via `nvmlDeviceGetTotalEnergyConsumption`
> (the cumulative-energy counter), which is now confirmed to over-report
> active-phase GPU power by ~30% on this hardware (not real boost overshoot —
> a counter-reporting artifact). The harness's default CUDA backend is now
> `nvmlDeviceGetPowerUsage`, validated against live telemetry to within
> -0.02W. **None of the pre-`672ce93` numbers should be cited as validated
> measurements** — kept for the evidence trail only (each affected
> `results/*/` directory has its own `DEPRECATED.md`). Full explanation:
> `results/active_power_baseline_investigation.md`.
>
> **⚠️ Superseded-data notice (2026-10-06, D16):** every CPU energy number in
> this project's `energy_j`/`summary.csv` (`total_j_mean` etc.) — all 3 INT8
> states, all 3 D13 FP32-CPU baselines, and the Stage 1-3/pre-flight CPU pilot
> runs — sums **two** RAPL domains, `package-0` and `psys`, not `package-0`
> alone, contradicting this file's own "RAPL domain in use" section below
> (written as if the code already did the right thing; it didn't). `psys` is
> a superset platform-power domain, confirmed empirically to report more than
> `package-0` at both idle and under CPU load — summing both inflates reported
> CPU energy by roughly 2.8x for the one condition checked in detail. **Use
> `summary_package.csv` (package-0 only, recomputed offline from each
> directory's own `raw.jsonl` by `scripts/recompute_cpu_package_energy.py`,
> which never modifies `raw.jsonl` or `summary.csv`) as the corrected primary
> CPU energy figure; `summary.csv`'s combined figure and `psys` are secondary.**
> The original INT8/FP32-CPU values in `summary.csv` were viewed during
> collection for QC only (pairs/regime/plausibility checks, never a hypothesis
> test) — this fix was made before any Stage 5 statistics ran on CPU data, not
> after. See `docs/deviation_log.md` D16.

## Terminology note

Where project documents refer to a 'supervisor', 'supervisor review' or 'supervisor decision', this means advice from an AI assistant used by the researcher. There is no human supervisor; all decisions were made and applied by the researcher.

## Current status (2026-10-16)

- **Analysis plan:** `docs/stage5_plan_v1.md` (sha256 `ace75a6b9bd9d38c947a44dde7aba8af9856986dbacfb4cbe5d7fb35a17b7434`), confirmed by amendment A10; frozen under tag `stage5-a7-frozen-20261012` (`70f84f0`). Plan archive: Zenodo `10.5281/zenodo.23357783` (deposited 2026-10-16, after data collection; see D34).
- **Data:** x86-CPU confirmatory block, six sessions in `results_stage5/cpu/session_01..06` (2026-10-13 to 2026-10-14); x86-GPU supplementary block in `results_stage4b/main_session1..6`.
- **Analysis outputs:** `analysis/results/` (output text summaries): `accuracy_table_output.txt`, `exclusion_counts.txt`, `gpu_supplementary_output.txt`, `h2_primary_output.txt`, `pruned_cpu_equivalence.txt`, `sensitivity_part1_output.txt`, `sensitivity_part2_output.txt`.
- **Deviations:** `docs/deviation_log.md` (D1-D40).
- **Paper checklist:** `docs/paper_must_include.md`.
- **Pending:** M1-CPU replication (not collected), P1 wall-meter test (not run).

RAPL+NVML paired energy-measurement harness for Phase 1 of this research project
(see `docs/phase1-execution-plan.md`, `docs/stage1-implementation-brief.md`).
Ported from `research/energy-pilot` (2026-10-03) — that repo stays with the other,
earlier proposal (calibration/reliability angle); this repo is this proposal's own
harness going forward.

## RAPL domain in use

This machine exposes two domains:

```
intel-rapl:0 -> package-0
intel-rapl:1 -> psys
```

Both are world-readable without sudo via `/etc/udev/rules.d/51-rapl-permissions.rules`
(`chmod -R a+r /sys%p` on powercap add). **Intent vs. actual (corrected 2026-10-06, D16):**
this section previously claimed the harness reads `package-0` only, "not `psys`
(whole-system)," as a deliberate, disclosed choice. That was wrong — the glob matching
`intel-rapl:*` one colon deep caught both `package-0` and `psys`, and both were summed into
every `energy_j` this harness ever logged for a CPU run. The intent stated here (package-0
only, because `psys` is a superset including DRAM/VRM/peripheral rails, not an independent
rail to add on top) was correct; the code didn't implement it. Fixed offline (not by
rerunning) via `scripts/recompute_cpu_package_energy.py`, which recomputes package-0-only
energy from each affected run's existing trace data and writes `summary_package.csv`
alongside the original, untouched `summary.csv`. `pilot.py` itself is patched going forward
(see the harness-patch commit) to log `package`/`psys` as separate columns for any new CPU
run, with package-0 as primary.

## NVML energy backend (corrected 2026-10-03) and sampling interval

Default CUDA backend is `nvmlDeviceGetPowerUsage` (sampled instant power,
trapezoidal-integrated), **not** `nvmlDeviceGetTotalEnergyConsumption`
(cumulative counter). The cumulative counter over-reports active-phase power
by ~30% on this hardware — confirmed against live, independently-logged
telemetry (`results/active_power_baseline_investigation.md`). The old
counter remains available via `--legacy-cumulative-counter`, purely to
reproduce/cite pre-correction numbers; it logs a warning when used.

The originally-reported counter-telescoping artifact (20ms -> ~300W against a
60W cap, converging to plausible only at 0.3-0.5s) was swept again
2026-10-03 **with the corrected backend**: flat 12.6-13.0W across the entire
20ms-500ms range, no interval-dependence at all. That artifact was specific
to the cumulative counter, not a general GPU/polling-rate limitation.

The 0.3-0.5s floor is still enforced in code (unchanged this session — not
yet decided whether to relax it now that its original justification doesn't
reproduce with the corrected backend). `--interval` defaults to 0.4s; any
`--device cuda` run with `--interval < 0.3` raises unless
`--override-fast-interval` is passed, which prints a warning and proceeds anyway.
Enforced in `pilot.py`'s `check_interval_floor()`, called from `main()`.

## RAPL sampling-rate ceiling

Checklist item 4's other half: RAPL/CPU reads (via sysfs, `/sys/class/powercap/
intel-rapl:*/energy_uj` — corrected 2026-10-06; this previously said "perf-
events," which `pilot.py` has never used) must never sample faster than 100Hz
(interval < 0.01s). Enforced the same way, same function:
`pilot.py`'s `check_interval_floor()`, called a second time from `main()` for
`--device cpu`. Default rejects; `--override-fast-rapl-interval` allows it and
logs a warning.

## Concurrent-GPU-job guard

Checklist item 13: a second GPU job contaminates the measurement (confirmed
~2x slowdown for both jobs in `thesis_arch_results`' VGG-19/ResNet-18 contention
finding). Enforced in `pilot.py`'s `check_no_concurrent_gpu()`, called at the top
of every `window()` call when `--device cuda`. Queries
`nvmlDeviceGetComputeRunningProcesses` and aborts if any PID besides the
harness's own is using the GPU; `--allow-concurrent-gpu` permits it and logs a
warning. No-op for `--device cpu`.

## Platform power-profile guard

Checklist item 14: the OS power profile is a controlled variable for the whole
project. Setup: `echo performance | sudo tee /sys/firmware/acpi/platform_profile`
— standard ACPI interface, no `legion_laptop` kernel module needed/available on
this kernel build (7.0.0-31-generic). Enforced in `pilot.py`'s
`check_platform_profile()`, called from `main()` before any measurement run
starts: aborts (no override) unless `/sys/firmware/acpi/platform_profile` reads
exactly `performance`. The value is logged into `environment.json` for every
run, alongside the NVIDIA driver version and CPU governor.

## What's here

- `pilot.py` — RAPL+NVML paired measurement logic, idle-baseline capture, the
  interval floor/ceiling and concurrent-GPU guard above.
- `tests/test_pilot.py` — unit tests: the original 4 (counter wrap, power
  integration, counter-reset rejection, idle-duration/detection) plus the
  interval-floor, concurrent-GPU-guard, and platform-profile-guard tests added
  2026-10-03, plus the D16/item-3 tests added 2026-10-06 (RAPL domain-split
  correctness, `find_rapl_domain()`, `--cpu-affinity`'s `p_core_set()`) — 107
  tests total from `measurement_env/bin/python3 -m unittest discover tests/`: 97
  passed, 10 failed/errored (3 failures, 7 errors), 0 skipped. Tests needing the
  performance governor fail when run outside an active measurement session where
  the governor is set to powersave.
- `scripts/` — `recompute_cpu_package_energy.py` (D16 offline recompute),
  `eval_artifacts.py` (full-test-set accuracy via `pilot.py`'s own loading
  path), `pruned_controls.py` (pruned-state class histograms + BN-stats-only
  recalibration).
- `analysis/` — statistical models, sensitivity analysis, diagnostic fits, and specification curve scripts in R and Python (`analysis/results/` stores output text summaries).
- `results_stage5/` — confirmatory x86-CPU measurement sessions (`results_stage5/cpu/session_01..06`).
- `docs/` — the Phase 1 execution plan and Stage 1 implementation brief this repo
  is built against.
- `docs/stage5_plan_v1.md` — frozen confirmatory and exploratory research plan (Stage 5 Plan v1).
- `docs/deviation_log.md` — full audit trail of project deviations (D1-D40).
- `docs/paper_must_include.md` — checklist of required disclosures, results, and limitations for the manuscript.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python pilot.py --device cuda --data <cifar10-root> --out results/run1 --sizes 32 --batches 16 --windows 2 --repeats 30
```
