# Recurring `idle_after` anomaly investigation (2026-10-03)

Prior observation: in both Stage 1 exit-test runs, the last rep's `idle_after`
window read 2-3.5x the baseline (330J/526J vs ~150J). Investigated per the
four-step plan below; no harness code was changed except where Step 3's
boundary-bug hypothesis was confirmed (it wasn't).

## Step 1 — Reproduce and localize

Four runs, same machine/session, pre-flight clean (`platform_profile=performance`,
governor=performance, no concurrent GPU process):

| run | windows | warmup | repeats | result |
|---|---|---|---|---|
| A | 1.5s | 1.5s | 5 | No discrete jump. `idle_after` drifted mildly and monotonically: 28.85 → 30.71 → 34.32 → 37.36 → 38.04 W. |
| B | 1.5s | 1.5s | 15 | No discrete jump. Noisy, bounded 23.76–40.80W across all 15 reps; rep 14 (last) at 40.13W — unremarkable, within the noise band. |
| C | 5s | 3s | 10 (exact repro of the original exit test) | **Jump reproduced — but not only at the last rep.** Reps 0–3: 26.44–33.19W (flat, matches historical baseline). Reps 4–9: 85.14, 103.24, 73.62, 112.09, 47.04, 119.13W — all elevated, to varying degrees, starting mid-run. |
| (historical, for reference) | 5s | 3s | 10 | Both prior exit tests: reps 0–8 flat (26–34W), **only** rep 9 jumped (65.88W, then 105.18W in the second test). |

**Conclusion on Step 1: this is not "last window of N."** Varying `--repeats`
(5 vs 15) at short window/warmup produced no discrete jump at all, ruling out a
pure rep-count trigger. Reproducing the exact original window/warmup config
(5s/3s) did reproduce elevated readings — but this time spread across 6 of 10
reps (4 through 9), not isolated to rep 9 as in the two historical runs. The
phenomenon is real and reproducible under the original parameters, but its
*position* within a run is not fixed — sometimes only the last rep, sometimes
most of the back half. This already argues against a structural "last window"
code-position trigger and points toward something that becomes more likely to
occur as the run progresses (time- or heat-accumulation-linked), without a
hard rep-count or last-window threshold.

## Step 2 — Correlate with real GPU state, not just energy

Ran `nvmlDeviceGetClockInfo` (SM/mem), `nvmlDeviceGetPowerUsage`, and
`nvmlDeviceGetPerformanceState` independently at 100ms resolution throughout
run C, then sliced that log against the exact window boundaries of a low rep
(1, implied 26.44W) and two elevated reps (4, 85.14W; 9, 119.13W):

| rep | implied W (harness's cumulative-energy reading) | live power.draw range during the SAME window | live SM clock | pstate |
|---|---|---|---|---|
| 1 (normal) | 26.44 | 11.59 – 15.77W | 1492 MHz, flat | P0 |
| 4 (elevated) | 85.14 | 11.66 – 16.64W | 1492 MHz, flat | P0 |
| 9 (most elevated) | 119.13 | 11.72 – 18.93W | 1492 MHz, flat | P0 |

**The live power sensor and clock are indistinguishable between the "normal"
and "elevated" windows.** No downclock transition, no power-state change, no
boost-clock residue — all three windows show the GPU sitting at the same idle
clock (1492MHz) and the same ~12-19W instantaneous power band throughout. If
the GPU were actually drawing 85-119W during these windows, the independent
power sensor would show it. It doesn't.

## Step 3 — Rule out a boundary/teardown bug

Checked `pilot.py` directly:

```python
def window(sensor, seconds, interval, work=None, sync=lambda: None, allow_concurrent_gpu=False):
    check_no_concurrent_gpu(sensor, allow_concurrent_gpu)
    sync(); trace=[]; errors=[]; stop=threading.Event()
    ...
    finally:
        sync(); stop.set(); thread.join(); sample()   # <- runs for every phase, every rep, identically
```

```python
for phase in ['idle_before','a1','a2','idle_after']:
    ...
    result=window(sensor,seconds,a.interval,work if phase.startswith('a') else None,sync,a.allow_concurrent_gpu)
    row=dict(...); rows.append(...); 
    with (out/'raw.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
del x                                               # <- after ALL 4 phases, not during any of them
...
finally:
    summary=summarize(rows); csv_write(out/'summary.csv',summary); csv_write(out/'windows.csv',rows)
```

`window()`'s own `sync()`/`sample()` teardown runs identically after every
single phase of every single rep — not specially for `idle_after`, not
specially for the last rep. `del x` and the `csv_write`/`summarize` calls run
once per grid-cell iteration, strictly after all 4 phases (including
`idle_after`) have already returned and been logged. **No code executes
during a window that wouldn't also execute during every other window.**

This is also supported empirically: if a boundary bug caused this, the
anomaly's position *within* the window should be consistent (e.g. always the
final sample, right where `finally`'s extra `sample()` call sits). It isn't —
rep 4's elevated deltas are concentrated in the **first** ~2 seconds of the
window (151→178W for 5 straight 0.4s steps, then dropping to normal for the
remaining 3s), while rep 9's elevated deltas are scattered/alternating
throughout the whole window, with the single largest value right at the end.
Two different reps, two different positions. A fixed code-location bug would
produce a fixed position.

**Step 3 conclusion: not a window-boundary bug.** No fix applied.

## What it actually is

Pulled the harness's own raw per-sample deltas (the `nvmlDeviceGetTotalEnergyConsumption`
readings `pilot.py` itself takes at the enforced-safe 0.4s interval — not a
fast-polling artifact, we're well inside the 0.3-0.5s floor):

```
rep=4 idle_after, consecutive 0.4s deltas:
  151.09W  158.26W  165.21W  171.74W  177.79W   <- first ~2s: sustained, smoothly ramping "high"
   22.58W   12.12W   18.62W   24.97W   31.55W   37.83W   43.36W   <- remaining ~3s: normal, matches rep 1's pattern

rep=9 idle_after, consecutive 0.4s deltas:
   47.09W  112.49W  118.85W  125.61W  131.69W   79.31W  145.01W  151.46W   26.16W   33.10W  149.72W  178.23W  443.85W
   <- irregular, no clean "high-then-drops" shape, ends on the single largest value of the whole investigation
```

Both patterns are completely absent from the live power sensor at the same
timestamps (Step 2). **The elevated numbers exist only in what
`nvmlDeviceGetTotalEnergyConsumption` itself reports — not in the GPU's actual,
independently-measured power draw.** This is consistent with (and extends) the
earlier NVML investigation's open question
(`results/nvml_aggregation_investigation.md`): that investigation showed the
*fast-polling* artifact (20ms-interval inflation) tracks call frequency, and
left open whether the mechanism was genuine extra power draw or a
driver/firmware reporting quirk. This investigation adds direct evidence
toward the **reporting-quirk** side of that question specifically: even at the
enforced-safe 0.4s interval, with live telemetry confirming nothing physically
unusual is happening, the cumulative counter can still intermittently
over-report — more often later in a longer-duration run (reps 4-9 of a
~320s/10-rep run; never in two shorter 60-190s runs), but not on any fixed
schedule tied to rep count or window position.

## Conclusion (per Step 4)

**Genuine instrument artifact, not a bug in `pilot.py`, and not a real GPU
power-state transition.** No code fix applies here — there's nothing in the
harness to fix. This is a property of `nvmlDeviceGetTotalEnergyConsumption` on
this GPU/driver that becomes more likely to manifest later in a longer run.

**For the methodology section:** state plainly that the NVML cumulative-energy
counter on this RTX 3050 occasionally reports energy deltas with no
corresponding change in live power/clock telemetry, more often later in
longer sessions; raw per-sample traces (already retained, checklist item 12)
make these identifiable and excludable case-by-case, but the counter itself
cannot be trusted blindly for every single window.

**On the suggested cooldown-before-`idle_after` mitigation:** not applicable
here — a cooldown delay exists to let a genuine clock/power-state transition
settle before measuring, but Step 2 showed there is no clock/power-state
transition to settle. A cooldown would add time without addressing anything.

**One concrete, narrow follow-up worth considering (not implemented — flagging
for a decision, not a silent fix):** since the GPU's physical power cap is
60W, any window whose implied power exceeds that is already known-impossible
and could be auto-flagged (not silently corrected) in `summarize()` or at
write-time, the same way `integrate()` already rejects a negative counter
delta as a hard error. That would catch this specific artifact mechanically
in future runs instead of relying on manual inspection. Left unimplemented
pending your decision — it's a new validation rule, not the fix this
investigation was scoped to find.
