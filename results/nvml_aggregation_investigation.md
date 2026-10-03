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

## Conclusion

**The sampling-rate dependence does not disappear with a first/last-only
recomputation — because it was never caused by our aggregation method in the
first place.** The 85-sample 20ms window and the 5-sample 500ms window compute
to the same energy either way; what differs between them is the **raw value
`nvmlDeviceGetTotalEnergyConsumption()` itself reports** over a comparable
wall-clock span, depending on how often it's polled. That's a property of the
GPU driver/firmware's counter, not a bug in `pilot.py`'s integration.

**Recommendation: keep the 0.3-0.5s floor.** It isn't masking a fixable bug in
our own code — it's compensating for a real, confirmed-reproducible (Task 2,
twice now) upstream counter artifact. No code change made; this was
investigation only, per the request.
