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

RAPL+NVML paired energy-measurement harness for the Masaryk Green AI proposal's
Phase 1 (see `docs/phase1-execution-plan.md`, `docs/stage1-implementation-brief.md`).
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
  correctness, `find_rapl_domain()`, `--cpu-affinity`'s `p_core_set()`) — 40
  tests total, all passing via `python -m unittest discover -s tests`.
- `scripts/` — `recompute_cpu_package_energy.py` (D16 offline recompute),
  `eval_artifacts.py` (full-test-set accuracy via `pilot.py`'s own loading
  path), `pruned_controls.py` (pruned-state class histograms + BN-stats-only
  recalibration).
- `docs/` — the Phase 1 execution plan and Stage 1 implementation brief this repo
  is built against.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python pilot.py --device cuda --data <cifar10-root> --out results/run1 --sizes 32 --batches 16 --windows 2 --repeats 30
```
