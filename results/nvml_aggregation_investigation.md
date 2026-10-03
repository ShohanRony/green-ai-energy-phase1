# NVML aggregation investigation (2026-10-03)

Question: is the NVML sampling-rate artifact (20ms→~307W idle-implied, ...,
500ms→~26W) caused by how `pilot.py` aggregates multiple per-poll samples
within a window, or is it a property of the underlying NVML counter itself?

## The actual code

`Sensor.read()` for CUDA (`pilot.py`):

```python
else:
    import pynvml as nv
    self.nv = nv; nv.nvmlInit(); self.handle = nv.nvmlDeviceGetHandleByIndex(0)
    try:
        nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self.backend = 'NVML cumulative energy'; self.ranges = [None]
    except nv.NVMLError_NotSupported:
        self.backend = 'NVML sampled power'; self.ranges = []
def read(self):
    ...
    if self.ranges: return [self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000]
    return [self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000]
```

This GPU supports `nvmlDeviceGetTotalEnergyConsumption` (confirmed in Task 0), so every
window uses the `ranges=[None]` cumulative-energy path, not the sampled-power fallback.

`integrate()` — called once per window on the full list of per-poll samples:

```python
def integrate(trace, ranges):
    energy = 0.
    for (t0, x0), (t1, x1) in zip(trace, trace[1:]):
        if not ranges: energy += (t1-t0) * (x0[0]+x1[0])/2
        else:
            for a,b,limit in zip(x0,x1,ranges):
                d = b-a
                if d < 0:
                    if limit is None: raise RuntimeError('Energy counter reset')
                    d += limit
                energy += d
    return energy
```

**Answer to the question as posed:** it's the latter — `integrate()` sums many
per-poll deltas (`d = b-a` for every consecutive pair in the trace), not a single
`(E_end - E_start)` using only the first and last reading.

## But that distinction turns out not to matter here

For a monotonically increasing counter with no resets (`ranges=[None]`, so any
`d<0` raises instead of silently correcting), summing consecutive deltas is a
**telescoping sum**: `(x2-x1)+(x3-x2)+...+(xn-xn-1) = xn-x1`, algebraically
identical to a single first/last delta. The two methods can only diverge if a
genuine counter reset/wrap happens mid-window (which would raise, not silently
distort the number) or from floating-point rounding (negligible at these
magnitudes).

Verified empirically rather than just argued: recomputed every stored window
from the 2026-10-03 interval sweep (`results/interval_sweep_{20,50,100,200,300,500}ms/`)
using a pure first/last delta, and diffed against `integrate()`'s own number from
the same stored trace:

```python
def integrate_current(trace):   # pilot.py's actual integrate(), NVML-cumulative path
    energy = 0.
    for (t0, x0), (t1, x1) in zip(trace, trace[1:]):
        d = x1[0] - x0[0]
        if d < 0: raise RuntimeError('reset')
        energy += d
    return energy

def integrate_firstlast(trace): # only the first and last sample, nothing else
    return trace[-1][1][0] - trace[0][1][0]
```

Checked all 72 windows (4 phases × 6 intervals × 3 reps): **0 mismatches.**
Every single window's `integrate_current()` result equals `integrate_firstlast()`
to floating-point precision. Sample (idle windows, 20ms interval, 85-86 polls each):

| interval | n_samples | current (J) | first/last only (J) | match |
|---|---|---|---|---|
| 20ms | 85 | 613.3630 | 613.3630 | yes |
| 20ms | 85 | 635.2690 | 635.2690 | yes |
| 20ms | 86 | 616.5940 | 616.5940 | yes |
| 500ms | 5 | 64.4090 | 64.4090 | yes |
| 500ms | 5 | 57.2620 | 57.2620 | yes |
| 500ms | 5 | 44.4950 | 44.4950 | yes |

## Part 1 finding (superseded by Part 2/3 below, kept for the record)

The sampling-rate dependence does not disappear with a first/last-only
recomputation of the *same already-recorded trace* — because it was never
caused by our aggregation method. That only rules out "it's a bug in how we
sum samples." It does **not** explain the actual mechanism. Two more specific
checks, below, were needed before accepting "real upstream behavior" as final.

## Check A: is the window duration itself interval-dependent?

If windows at different `--interval` settings ran for meaningfully different
wall-clock durations, that alone — nothing to do with the sensor or polling
rate — would explain different energy totals. Pulled `t_start`/`t_end` from
every stored window's trace across all 6 intervals:

| interval | duration_s range (4 phases × 3 reps) |
|---|---|
| 20ms | 2.0043 – 2.0131 |
| 50ms | 2.0039 – 2.0125 |
| 100ms | 2.0044 – 2.0138 |
| 200ms | 2.0044 – 2.0133 |
| 300ms | 2.0044 – 2.0099 |
| 500ms | 2.0040 – 2.0140 |

All 72 windows fall in a 10ms band (2.0039–2.0140s) with **no trend by
interval** — the 20ms rows aren't systematically longer or shorter than the
500ms rows. Window duration is not the explanation. Moves to Check B.

## Check B: does polling frequency itself change what the GPU reports (observer effect)?

Ran three variants back-to-back, today, same GPU, same idle workload, 5s each:

```python
def variant_current(interval, seconds):
    """Same style as pilot.py: full trace, telescoping sum of all deltas."""
    trace = []
    t0 = time.perf_counter()
    trace.append((t0, nv.nvmlDeviceGetTotalEnergyConsumption(h)))
    next_t = t0 + interval
    while True:
        now = time.perf_counter()
        if now - t0 >= seconds: break
        if now >= next_t:
            trace.append((now, nv.nvmlDeviceGetTotalEnergyConsumption(h)))
            next_t += interval
        else:
            time.sleep(min(0.001, next_t-now))
    trace.append((time.perf_counter(), nv.nvmlDeviceGetTotalEnergyConsumption(h)))
    energy = sum(b[1]-a[1] for a,b in zip(trace, trace[1:]))
    return dict(n_calls=len(trace), energy_mJ=energy, duration_s=trace[-1][0]-trace[0][0])

def variant_firstlast_but_same_call_rate(interval, seconds):
    """Call NVML at the SAME cadence, but only keep first and last value -- nothing summed."""
    t0 = time.perf_counter()
    first = nv.nvmlDeviceGetTotalEnergyConsumption(h)
    n_calls = 1; next_t = t0 + interval; last = first
    while True:
        now = time.perf_counter()
        if now - t0 >= seconds: break
        if now >= next_t:
            last = nv.nvmlDeviceGetTotalEnergyConsumption(h) # same call, same frequency -- discarded except kept in `last`
            n_calls += 1; next_t += interval
        else:
            time.sleep(min(0.001, next_t-now))
    last = nv.nvmlDeviceGetTotalEnergyConsumption(h); t1 = time.perf_counter(); n_calls += 1
    return dict(n_calls=n_calls, energy_mJ=last-first, duration_s=t1-t0)
```

Results (idle GPU, 5s windows, same session, back-to-back):

| variant | calls | duration_s | energy_J | implied_W |
|---|---|---|---|---|
| (1) 20ms, current (full-trace, telescoping sum) | 251 | 5.0001 | 2864.11 | **572.81** |
| (2) 20ms, same call rate, first/last only | 251 | 5.0062 | 2866.05 | **572.50** |
| (3) 500ms, current (fresh baseline, today) | 11 | 5.0001 | 139.44 | **27.89** |

**(1) ≈ (2), both ≫ (3).** Variant 2 calls `nvmlDeviceGetTotalEnergyConsumption`
at the exact same 20ms cadence as variant 1, and does *less* work per call
(no list growth, no per-sample delta, just overwriting `last`) — yet it comes
back essentially identical to variant 1 (572.50W vs 572.81W, ~0.05% apart,
consistent with ordinary run-to-run noise from two separate real executions,
not a suspiciously exact match that would suggest a bookkeeping bug). Variant
3, run fresh today at 500ms, reproduces the same ~28W plausible-idle figure
as every prior slow-polling run.

This rules out "it's about what we do with the samples" a second, stronger
way: even minimizing post-call bookkeeping to the bare minimum while keeping
the same query frequency doesn't change the result. **The effect tracks call
frequency itself**, not our aggregation style.

## Conclusion

Per the user's framing: `(1) ≈ (2)` and both are high → this is consistent with
the elevated reading being tied to *how often we query the GPU*, not to
anything in our post-processing (confirmed twice now, independently).

One honest caveat this experiment doesn't resolve: "tied to call frequency"
has two sub-explanations that this setup can't distinguish between —
(a) **genuine observer effect**: frequent NVML queries themselves keep the GPU
in a less idle power/clock state, so it is actually drawing more real power
while being polled that often, or (b) **driver/firmware reporting quirk**:
`nvmlDeviceGetTotalEnergyConsumption()`'s internal accumulator is derived from
the GPU's own power-sensor refresh (likely well under 20ms resolution) and
over-reports when queried faster than that refresh, without any real extra
watts being drawn. Telling these apart would need an independent power
reference (e.g. a wall-socket meter, or cross-checking against RAPL `psys` if
it ever covered discrete-GPU rails, which it doesn't on this laptop) — out of
scope for this investigation.

**Recommendation unchanged: keep the 0.3-0.5s floor.** Whichever sub-mechanism
it is, it isn't a bug in `pilot.py`'s integration (ruled out twice: Part 1's
telescoping-sum identity, and Check B's same-call-rate comparison) — the
floor is compensating for something real happening at or below the NVML/driver
layer, independent of our aggregation code. No harness code changed; this
remains investigation only, per the request.
