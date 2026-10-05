# Stage 4 Pre-flight Findings — Batch Size + GPU Power-Regime Investigation

Run 2026-10-05, following the Stage 3 report's flagged batch-size-vs-speedup question. Three tasks from
the pre-flight instruction: Task 1 (batch=1 pilot), Task 2 (lock batch-size policy), Task 3 (sudoers
rule). Task 2 escalated into a more fundamental finding than originally scoped — documented in full
below, with a decision point at the end, not a unilateral policy choice.

A power loss interrupted this session partway through (machine rebooted, confirmed via `uptime`). **No
data was lost or corrupted** — every run directory from before the outage shows `pairs=9` (complete).
One determinism-check rerun was invalid for a different reason (ran under the wrong CPU governor, reset
by the reboot) and was discarded and redone cleanly; see "Power-loss recovery" below.

---

## Task 1 — Batch=1 pilot (6 states, 10 reps each)

Same ResNet-18 6-state pilot as Stage 3, rerun at `--batches 1`. All 6 states complete, `pairs=9`, in
`results_stage4_preflight/resnet18_{state}_b1/`.

| State | J/image (batch=1) | J/image (batch=16, Stage 3) |
|---|---|---|
| FP32 | 0.10965 | 0.03663 |
| FP16 | 0.05999 | 0.01221 |
| Pruned 30% | 0.08803 | 0.01791 |
| Pruned 50% | 0.02985 | 0.00659 |
| Pruned 70% | 0.02698 | 0.00540 |
| INT8 (CPU) | 0.22622 | 0.18395 |

At batch=1, pruned50/70 stop being power-cap-pinned (confirmed from per-window traces: ~46-52W instead
of the usual ~60W) — the GPU idles between launches for these very fast, heavily-reduced models. This
genuinely dilutes pruning's apparent benefit at batch=1.

## Task 2 — Batch-size policy: escalated, not simply resolved

**Three-point comparison (FP32/compressed energy ratio) raised the first flag:**

| State | batch=1 | batch=16 | batch=128 (Stage 2 latency) |
|---|---|---|---|
| FP16 | 1.83x | **2.98x** | 1.78x |
| Pruned 30% | 1.25x | **2.04x** | 1.19x |
| Pruned 50% | 3.82x | **5.54x** | 2.43x |
| Pruned 70% | 4.40x | **6.76x** | 3.87x |

Not monotonic — batch=16 is a consistent outlier spike above both neighbors, across all 4 states. Ruled
out idle-power-dilution as the explanation (net-of-idle energy gives the same pattern).

**Finer scan (batch=4,8,32,64 for FP32/FP16/pruned70) revealed the real mechanism: a two-mode GPU power
regime, not a smooth function of batch size.**

`total_J_mean` per 5s active window (implied W):

| Batch | FP32 | FP16 | Pruned70 |
|---|---|---|---|
| 1 | pinned (60W) | pinned (60W) | dip (~46W) |
| 4 | pinned | pinned | dip (~35W) |
| 8 | pinned | pinned | dip (~35W) |
| 16 | pinned | pinned | **pinned (60W)** |
| 32 | pinned | pinned | dip (~35W) |
| 64 | pinned | **dip (~38W)** | dip (~35W) |

FP32 never dips. FP16 dips only at the largest batch tested (64) — opposite of an overhead-bound
story. Pruned70 dips at every batch except 16. No monotonic or overhead-bound explanation fits this
pattern cleanly.

## Determinism check — the real finding

Re-ran 3 of the above combinations a second time, same config, to test whether the pin/dip assignment
is a fixed property of (state, batch) or can flip between runs:

| Combo | Original | Rerun(s) | Verdict |
|---|---|---|---|
| Pruned70 @ b16 | pinned (60W) | pinned, pinned (3 total) | **Deterministic** |
| Pruned70 @ b32 | dip (~35W) | **pinned (60W)** | **Flipped** |
| FP16 @ b64 | dip (~38W) | **pinned (60W)** | **Flipped** |

2 of 3 flipped between identical invocations. **Within a single run, the regime is stable across all
9-10 reps** (no mid-run switching observed in any run collected today) — it appears to lock in early
(plausibly during warmup) and hold for that invocation's duration. This means more reps within one
`pilot.py` call will NOT average out this effect; only independent repeat invocations would.

**Implication for Stage 4 and for every energy number collected so far (Stage 2, Stage 3, this
pre-flight):** a single harness invocation's energy/latency number for a given (model, state, batch)
may reflect an essentially arbitrary choice between two real, physically distinct GPU power states, not
a stable property of the configuration. Comparisons between states measured in different invocations
risk being contaminated by which regime each one happened to land in, independent of real compute
differences.

**Not yet understood:** the underlying mechanism (GPU boost-clock dwell/decay behavior most likely,
given it's a laptop GPU with aggressive power management, but unconfirmed — would need direct
driver-level clock-state telemetry at finer time resolution than this project's energy-focused harness
collects). Out of scope to chase further here without direction.

## Task 3 — Sudoers rule for CPU governor

`scripts/setup_governor_sudoers.sh` written and reviewed, not yet run as of this report (power loss
interrupted before the user got to it). Still pending.

## Power-loss recovery (mid-session interruption)

Machine rebooted during this session (confirmed via `uptime`). Audited every `results_stage4_preflight/`
and `results_stage3/` directory touched this session: **all show `pairs=9`, nothing missing or
truncated** — the 6 batch=1 runs and 12 batch-scan runs all completed and were written to disk well
before the outage.

One casualty, caught not silently absorbed: the first post-reboot determinism-check rerun
(`pruned70_b16_rerun`) ran under `powersave` (the reboot reset the governor; nothing had re-set it yet)
while the original Stage 3 measurement it was being compared against ran under `performance` — a real
confound, not a data-loss problem. Discarded and rerun cleanly under `performance` once the governor was
manually restored; the clean rerun is what's reported in the determinism-check table above. (Note: the
contaminated rerun also landed pinned, same as the clean one — suggesting CPU governor alone doesn't
explain the pin/dip split, but it's one data point and wasn't treated as evidence either way.)

---

## Decision needed before Stage 4 proceeds

This is no longer just "which batch size." The power-regime non-determinism affects *any* batch size,
and arguably affects the validity of energy comparisons already made in Stage 2/3. Options, not a
recommendation from this report alone:

1. **Investigate the mechanism further** (e.g., direct `nvidia-smi dmon`/clock-state logging during a
   run to see if the regime is visible/predictable from clock state at warmup time).
2. **Design around it empirically**: require N ≥ 2-3 independent repeat invocations per (model, state,
   batch) in Stage 4, explicitly report regime bimodality if it recurs, rather than trusting a single
   invocation's reps.
3. **Treat it as within-hardware noise** Stage 4's ≥30-rep-per-condition design already budgets for,
   accepting wider variance bands rather than resolving the mechanism.
4. Something else — this is flagged for a decision, not resolved here.
