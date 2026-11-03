# M1 thermal guard calibration evidence — 2026-11-04

**This is guard-calibration data only. It has not been analysed for energy
and is not deposited/registered confirmatory data.** It exists to document
and let anyone reproduce the derivation of `T_mhz` in
`config/thermal_thresholds_m1.json` (referenced from there via
`"source": "calibration/m1_thermal_20261104"`).

## What's here

- `test0_resnet18_fp32_ts/`, `test0_resnet18_int8/` — first calibration run:
  15 minutes of continuous batch-1 32x32 inference per condition, 180
  consecutive 5s windows each (`windows.json`, `windows_raw.jsonl`,
  `arrivals.csv`, `run_meta.json`), plus the raw `powermetrics` output
  (`raw_powermetrics.txt.gz`). FP32-TS run first, INT8 second.
- `test0_ambient.json` — AC/thermal guard text and uptime at Test 0's start.
- `test1_resnet18_int8/`, `test1_resnet18_fp32_ts/` — order-reversed repeat
  of the same 15-minute-per-condition protocol: INT8 first, FP32-TS second,
  with a >=10-minute idle cool-down enforced before starting and between
  legs.
- `test2_dutycycle/` — 45-minute session-like duty-cycle check: 14
  alternating blocks (7 INT8, 7 FP32-TS), each block 6 repeats x (idle 3s
  settle + 5s idle, active 3s warm-up + 5s window, active 3s + 5s, idle 3s +
  5s) — 168 active 5s windows total (`active_windows.json`,
  `active_windows_raw.jsonl`, raw file).
- `idle_check_pre_test1/` — the >=10-minute idle confirmation run before
  Test 1 started. This was a continuous passive monitor, not inference, and
  was never divided into windows -- `samples.csv` is per-sample (1537 rows,
  not per-window), parsed here from the raw file after the fact (it was not
  saved as a samples table by the run itself). Not part of the reference
  set (see below) -- background/pre-check evidence only.
- `compute_threshold.py` — recomputes `T_mhz` from the JSON files in this
  directory alone (no other inputs). Reproduces
  `config/thermal_thresholds_m1.json`'s `T_mhz = 2881.11` exactly.

## Reference set / excluded set

**Reference set (n=528):** FP32-TorchScript continuous windows from Test 0
(n=180) and Test 1 (n=180), plus all Test 2 duty-cycle active windows
(n=168, both INT8 and FP32-TS blocks). Pooled 5th percentile of these
windows' per-window mean of CPU 4-7 frequency: **P5 = 3201.231731 MHz**.
**T = 0.90 x P5 = 2881.108558 MHz** (stored rounded to 2dp in
`config/thermal_thresholds_m1.json`).

**Excluded from the reference set:** the INT8 continuous runs (Test 0
n=180, Test 1 n=180). Both show a statistically significant sustained
negative frequency trend over their 15-minute duration (Test 0:
-31.25 MHz/min, 95% CI entirely negative; Test 1: -38.06 MHz/min, 95% CI
entirely negative) -- consistent with sustained-load throttling, not a
nominal/idle-adjacent operating state, so they are not used to characterize
what "not throttled" looks like.

**Not part of either set:** `idle_check_pre_test1/` (pre-check, not an
active-inference window) and Test 2's duty-cycle INT8 windows specifically
are *included* in the reference set as stated above (all 168 Test 2 active
windows, both conditions, are in the reference set -- the duty-cycle INT8
windows do NOT show the continuous-run throttling pattern, 0/84 fall below
T, unlike INT8 run continuously).

## Discrimination check (fraction of windows below T=2881.108558 MHz)

| Set | Below T | Fraction |
|---|---|---|
| Test 0 FP32-TS continuous (n=180) | 0/180 | 0.00% |
| Test 0 INT8 continuous (n=180) | 154/180 | 85.56% |
| Test 1 INT8 continuous (n=180) | 135/180 | 75.00% |
| Test 1 FP32-TS continuous (n=180) | 0/180 | 0.00% |
| Test 2 duty-cycle, FP32-TS active windows (n=84) | 0/84 | 0.00% |
| Test 2 duty-cycle, INT8 active windows (n=84) | 0/84 | 0.00% |

First window below T, elapsed time into the run:
- Test 0 INT8 continuous: window 26, 130.04s (2.17 min) elapsed, 2871.09 MHz.
- Test 1 INT8 continuous: window 44, 220.10s (3.67 min) elapsed, 2878.21 MHz.

## Ambient conditions, as recorded

- Lid: open.
- Surface: laptop stand.
- Power: charger plugged in (AC), verified via `pmset -g batt` at each run's
  start (100% charged / charging, "Now drawing from 'AC Power'").
- Room air conditioning: off.
- Room temperature: not measured.
- No location/address recorded.
- These conditions were reported by the user/operator for Test 0; the same
  physical setup (lid open, on AC, laptop stand) was in effect for Test 1,
  Test 2, and the idle check, run back-to-back in the same session without
  moving the machine.

## Two crash incidents, disclosed

1. **Test 0, first attempt (superseded, not used):** `run_one()`'s
   `proc.communicate(timeout=30.0)` was a fixed 30s timeout that did not
   account for the remaining `powermetrics` margin (~178s left to run after
   the 15-minute FP32-TS inference finished). The inference itself
   completed cleanly (180 windows, confirmed in stdout), but the script
   crashed with `subprocess.TimeoutExpired` before window boundaries (held
   only in that process's memory) were written to disk. That attempt's
   window data is unrecoverable and was not used -- not fabricated, not
   reconstructed after the fact. The underlying `powermetrics` process
   continued running orphaned (harmless) until it self-exited naturally;
   this was waited out, not killed, before the clean rerun that produced
   the `test0_*` data actually archived here. Fix applied before any
   further runs: each window is appended to `windows_raw.jsonl` on disk
   immediately as it completes, and the `communicate()` timeout is computed
   from the actual remaining margin + 60s buffer, not a fixed value.

2. **Test 2, duty-cycle run:** all 14 blocks (44.8 min) and all 168 active
   5-second windows executed completely, with every window boundary written
   incrementally to `active_windows_raw.jsonl` during the run (the fix from
   incident 1, already in place). At the end of the duty cycle,
   `powermetrics` took approximately 81.2s to finish flushing its sample
   buffer, exceeding the script's `communicate()` timeout buffer at the
   time (~80.7s) by a small margin. This was **not** a data-loss event: the
   underlying `powermetrics` process completed cleanly with all 6782/6782
   requested samples captured (`test2_dutycycle_20261103T215722Z_73464.txt`,
   18.2 MB uncompressed), and all 168 window boundaries were already safely
   on disk from the incremental-write fix, independent of whether the
   Python orchestrator process itself then errored on the
   `communicate()` call. `active_windows.json` was compiled directly from
   these live-recorded raw window boundaries and the completed
   `powermetrics` sample blocks -- nothing was lost, fabricated, or
   reconstructed.

## Raw `powermetrics` file provenance (sha256 of the original, uncompressed file)

| File (gzipped copy committed here) | sha256 (original .txt) |
|---|---|
| `test0_resnet18_fp32_ts/raw_powermetrics.txt.gz` | `6c58bd2ee6a5272a099be2fc3f5ae2a9aee8fc7fbc9930fa562d4bf1d2e1e0c9` |
| `test0_resnet18_int8/raw_powermetrics.txt.gz` | `7c081d72fa2c55db41d1c2490817dd1870ce46b880ffcb7b6d1e77a30ee830f2` |
| `test1_resnet18_fp32_ts/raw_powermetrics.txt.gz` | `96e85d384e6d9623cfa9d552c92e669346ad0f4d01a75ec912ffcbd019689e1b` |
| `test1_resnet18_int8/raw_powermetrics.txt.gz` | `7170bb641255aa6567a41a923b89a3aa2c85728fa1982bc358e02528602e9874` |
| `test2_dutycycle/raw_powermetrics.txt.gz` | `3c887deb063ca47f88d793fe0758bec78e52b33cd2a56ae47466a06e62ffad8b` |
| `idle_check_pre_test1/raw_powermetrics.txt.gz` | `698a992bfba014da8b2f9d439c510d248e5e48754dbf9aae580c87afcb557c57` |

All six gzipped raw files are small enough (largest 1.04 MB compressed, from
an 18.2 MB original) to commit directly -- none exceeded the 10 MB
compressed threshold that would have required listing sha256 only instead.
