# Does the "active 71-78W" baseline exceed the real 60W cap, or is it counter over-reporting?

Resolves the question `results/idle_after_anomaly_investigation.md`'s plausibility-flag
tally raised: 139/158 flagged rows were ordinary active-phase (a1/a2) windows,
not the `idle_after` anomaly — calling into question the "active 71-78W,
physically plausible" baseline used since the first September validation
sprint. This investigation answers it directly, using the same method that
worked for `idle_after`: correlate the NVML cumulative-energy counter against
independently-logged live instant telemetry, this time during active windows.

**No harness code or default changed.** Investigation only, per instruction.

## Data used

Reused `results/anomaly_investigation_repro10/` rather than running a new
test — it already matches the requested config exactly (uncompressed
ResNet-18, 5s windows, 3s warmup, 10 repeats, 0.4s interval — identical to
the Stage 1 exit test) and already has a concurrent GPU-state monitor
(`results/anomaly_investigation_gpu_state_monitor.py`) that ran at 100ms
resolution for the entire session, fully covering all 20 active (a1/a2)
windows (monitor range 16094.06–16417.75; active windows span
16107.94–16409.38 — full coverage, no gaps). Reusing already-collected real
data avoids any risk of cherry-picking a new run toward a preferred answer.

## Every active window, counter vs. live telemetry, side by side

| rep | phase | counter-implied W | live mean W | live max W | SM clock range (MHz) | pstate |
|---|---|---|---|---|---|---|
| 0 | a1 | 78.97 | 59.93 | 60.04 | 1605–1650 | P0 |
| 0 | a2 | 81.04 | 59.92 | 59.98 | 1627–1650 | P0 |
| 1 | a1 | 76.73 | 59.95 | 60.02 | 1627–1657 | P0 |
| 1 | a2 | 77.11 | 59.88 | 60.01 | 1627–1680 | P0 |
| 2 | a1 | 77.83 | 59.91 | 60.03 | 1612–1642 | P0 |
| 2 | a2 | 76.41 | 59.91 | 59.98 | 1620–1627 | P0 |
| 3 | a1 | 78.36 | 59.91 | 60.06 | 1612–1642 | P0 |
| 3 | a2 | 76.02 | 59.92 | 59.98 | 1620–1627 | P0 |
| 4 | a1 | 79.42 | 59.90 | 60.10 | 1605–1642 | P0 |
| 4 | a2 | 77.68 | 59.92 | 59.98 | 1612–1627 | P0 |
| 5 | a1 | 77.11 | 59.91 | 60.08 | 1612–1642 | P0 |
| 5 | a2 | 77.76 | 59.91 | 60.00 | 1612–1635 | P0 |
| 6 | a1 | 76.94 | 59.91 | 60.02 | 1605–1627 | P0 |
| 6 | a2 | 75.68 | 59.90 | 59.95 | 1620–1627 | P0 |
| 7 | a1 | 79.02 | 59.90 | 59.98 | 1605–1635 | P0 |
| 7 | a2 | 79.34 | 59.92 | 59.99 | 1612–1627 | P0 |
| 8 | a1 | 79.63 | 59.91 | 60.05 | 1597–1642 | P0 |
| 8 | a2 | 78.16 | 59.92 | 59.95 | 1612–1627 | P0 |
| 9 | a1 | 81.49 | 59.92 | 60.05 | 1597–1635 | P0 |
| 9 | a2 | 75.92 | 59.90 | 59.96 | 1582–1627 | P0 |

**Aggregates across all 20 windows:**

| | mean | min | max |
|---|---|---|---|
| counter-implied W | **78.03** | 75.68 | 81.49 |
| live instant mean W (within-window) | **59.91** | 59.88 | 59.95 |
| live instant max W (within-window) | **60.01** | 59.95 | 60.10 |

**Gap: 18.12W, a 30.2% inflation** of the counter's reading over the live
mean, present in every single window, with essentially zero variance.

## Does live telemetry ever actually reach 71-78W?

**No. Not once, in any of the 20 windows.** Live mean power sits at
59.88-59.95W throughout — effectively pinned at the enforced cap, not below
it with headroom, not above it. Live *max* power (the single highest 100ms
sample within each 5-second window) tops out at 59.95-60.10W — a few
hundredths of a watt either side of the cap, consistent with sensor
quantization noise, not a real boost excursion. There is no window where live
telemetry gets anywhere near 71W, let alone 78W.

## Brief boost spike, or sustained? Neither — there's no spike to characterize

The question "are excursions brief boost spikes or sustained" presumes live
telemetry shows excursions above 60W somewhere. It doesn't, meaningfully.
`live_max` exceeding 60.0W by up to 0.10W in 10/20 windows is noise-scale, not
a boost event. There is no gap between "live max" and "live mean" either
(both sit in a ~0.1W band) — if this were brief boosting followed by
clamping, we'd expect live max to run visibly higher than live mean within
the same window. It doesn't.

## Independent corroboration from clock behavior

SM clock during these windows runs 1582-1680MHz — well *below* the
~1972MHz single-job boost clock this exact hardware was shown to reach under
*uncontended* conditions in `research/thesis_arch_results` (MobileNetV3-Small
clean run, confirmed hitting full 1972MHz boost). A GPU power-limited to 60W
clamps its own clock down from its boost target to stay within that power
envelope — that's exactly the mechanism NVIDIA's "enforced power limit"
implements, and it's exactly what's observed here: reduced clock (~1600-1680MHz
vs. ~1972MHz available), pstate P0 throughout (not meaningfully informative on
its own for a consumer card, but consistent), and power pinned at the cap.
Two independent signals — the clock reduction and the live power sensor —
agree with each other and point the same direction.

## Conclusion

**Hypothesis (b) confirmed, decisively: the cumulative-energy-counter
over-reporting already established for `idle_after` is pervasive across
active-phase readings too.** It is not legitimate short-term boost overshoot
— live telemetry never gets close to 71-78W in any of the 20 windows tested,
and the clock-throttling behavior independently corroborates that the GPU is
actually running clamped at ~60W, not above it.

**The "active 71-78W, physically plausible" baseline used since the first
September validation sprint is wrong.** The real, physically-measured active
power on this GPU/workload is ~60W (at the enforced cap), not 71-78W.
`nvmlDeviceGetTotalEnergyConsumption` has been inflating every active-window
energy reading in this project by roughly 30% — not just during the
`idle_after` anomaly, but as a baseline characteristic of the backend this
harness has used throughout.

**This needs to be resolved before any more GPU data is collected for Stage
2+.** Candidates to discuss (not decided here, per instruction not to change
defaults yet):
1. Switch the CUDA backend from `nvmlDeviceGetTotalEnergyConsumption`
   (cumulative counter) to `nvmlDeviceGetPowerUsage` (instant power,
   trapezoidal-integrated — the harness's own documented fallback path,
   already implemented in `Sensor`/`integrate()` for the `ranges=[]` case,
   currently unused because the cumulative counter reports as "supported").
2. Re-run enough of the existing dataset's conditions with the alternate
   backend to confirm it tracks live telemetry correctly before trusting it
   as the new default.
3. Decide whether `gross_j_per_image_mean` figures already computed and
   reported (e.g. the original `0.0447 J/image` pilot finding, the Stage 1
   exit test's numbers) need to be flagged as provisional/inflated by ~30%
   rather than treated as validated going forward.
