# Stage 4 Pre-flight Findings — Batch Size + GPU Power-Regime Investigation

Run 2026-10-05, following the Stage 3 report's flagged batch-size-vs-speedup question. Three tasks from
the pre-flight instruction: Task 1 (batch=1 pilot), Task 2 (lock batch-size policy), Task 3 (sudoers
rule). Task 2 escalated into a deeper investigation than originally scoped — a GPU power-regime split
not explained by batch size, state, or CPU governor. **Mechanism investigation closed 2026-10-05**
(boot-identity hypothesis falsified by a deliberate reboot test; uptime-duration is the surviving,
unconfirmed candidate) — see "Deliberate reboot test" below. Resolved by building two harness guards
into `pilot.py` instead of waiting on the mechanism: automatic power-regime flagging and an enforced
fresh-boot check. Full detail below.

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

**Done.** `scripts/setup_governor_sudoers.sh` run successfully; hit and fixed a real bug in
`set_cpu_governor.sh` along the way (see the diagnostic-sweep section above). Verified:
`sudo -n /usr/local/sbin/set_cpu_governor.sh performance` exits 0 with no password prompt.

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

## Follow-up diagnostic sweep (2026-10-05 decision: option 1, bounded)

Scoped per the decision: Task 3's sudoers rule first (hit and fixed a real bug along the way, see
below), then launch-time telemetry (GPU temp, persistence mode, SM clock, AC state, idle-gap-since-
last-invocation) logged for N=8 independent invocations each of exactly 3 combos — Pruned70@b16
(previously deterministic), Pruned70@b32 and FP16@b64 (previously flipped). Not expanded further.
`diagnostic_power_regime_sweep.py`, full log in `results_stage4_preflight/diagnostic_sweep/launch_log.jsonl`.

**Result: 24/24 pinned. Zero dips, including both combos that flipped earlier.**

| Combo | Invocations | Pinned | Dip |
|---|---|---|---|
| Pruned70 @ b16 | 8 | 8 | 0 |
| Pruned70 @ b32 | 8 | 8 | 0 |
| FP16 @ b64 | 8 | 8 | 0 |

Temp held steady (41°C warming to 54-55°C then flat), SM clock constant at 1492MHz (boost) throughout,
persistence mode `Disabled` throughout (didn't vary — no pre-reboot baseline to compare against),
`idle_gap_s` ≈0.00 for every invocation after the first (they were launched back-to-back by design) and
AC online throughout.

**The real signal is in the timing, not the telemetry table:** every post-reboot measurement taken
today — these 24 invocations plus the 3 determinism-check reruns before them (27 total) — landed
pinned. **Every single dip observed in this entire investigation happened pre-reboot** (the original
Task 1 batch=1 pilot and the 12-run batch scan). That's a cleaner, more falsifiable candidate
explanation than per-invocation randomness: something about the reboot put the GPU into a stable
"always pinned" state for this boot session, rather than the regime being re-rolled on each invocation.

**What this sweep can't tell us, disclosed rather than glossed over:** because every invocation was
launched back-to-back, `idle_gap_s` never varied (always ≈0), and neither did `regime` (always
`pinned`) — with zero variation in either variable, this design cannot confirm or rule out idle-gap (or
persistence mode, held constant throughout) as a contributing factor. It can only speak to the
boot-session pattern, which is the strongest lead so far but itself untested directly (would require
another reboot to see if dips reappear, not done here — out of the bounded scope).

**Governor script bug caught and fixed along the way:** Task 3's `set_cpu_governor.sh` initially failed
every invocation with `Refusing unknown governor: performance}` — reproduced in the user's own terminal
(not a sandbox artifact) and confirmed the sudoers file itself was byte-clean (`cat -A`, no hidden
characters). Root cause suspected: `${1:?usage: ...{performance|powersave|...}}` embeds a second,
unescaped `{...}` pair inside the `:?` error-message word, which can confuse bash's own matching for the
outer `${...}`'s terminator. Rewrote without any brace characters in the script at all; verified working
(exit 0, no password prompt) after reinstall.

## Governor ruled out as a confound (checked, not assumed)

Before trusting the boot-session hypothesis, checked whether the CPU governor itself — not the Task 3
wrapper-script bug, the actual `/sys`-level value `pilot.py` logs into every run's `environment.json` —
could explain the pattern, since the wrapper script's bug and the reboot happened close together in time.

| Phase | Runs | Governor logged | Pinned | Dip |
|---|---|---|---|---|
| Pre-reboot finer scan | 18 | `performance`, every run | 12 | 6 |
| Determinism check, contaminated rerun | 1 | `powersave` | 1 | 0 |
| Determinism check, clean reruns | 3 | `performance` | 3 | 0 |
| Diagnostic sweep | 24 | `performance`, every run | 24 | 0 |

**Ruled out:** all 6 dips occurred with governor correctly logged as `performance` — the governor was
never wrong during a dip. The one run that genuinely ran under `powersave` still landed pinned, not
dip. Governor has no variation that correlates with regime in either direction. (The wrapper-script bug
itself was moot for this question too — it wasn't in use pre-reboot at all; governor was set manually
via `echo performance | sudo tee ...` the whole time, and that worked correctly per every logged value.)

This leaves the pre-reboot/post-reboot split (6/18 dip vs. 0/27 dip, governor held constant throughout)
as the only variable collected so far that actually co-varies with regime.

## Deliberate reboot test (closing the mechanism investigation)

Scoped to exactly the 2 combos that had flipped: Pruned70@b32, FP16@b64. Before rebooting: checked
persistence mode (`Disabled`, unchanged throughout this entire investigation — never varied, so it
can't be the explanation either) and reran both combos once more in the *same* long-running boot
session, to separate "boot identity" from "thermal/driver state accumulated over this session's
uptime" before spending an actual reboot cycle. Then rebooted for real and reran both.

| Combo | Original (long-uptime session) | After 1st reboot (accidental) | Same session, hours later | After 2nd reboot (deliberate) |
|---|---|---|---|---|
| Pruned70@b32 | **dip** (~35W) | pinned | pinned (59.9W) | **pinned** (59.9W) |
| FP16@b64 | **dip** (~38W) | pinned | pinned (60.0W) | **pinned** (60.0W) |

**Verdict: the dip did not reappear. Boot-session *identity* is falsified as the mechanism** — if each
boot got its own fixed-but-arbitrary assignment, a second fresh reboot should have had a real chance of
landing on dip again for at least one of the two combos. It landed pinned both times, extending the
pinned streak to 30+ consecutive post-reboot/post-control measurements with zero dips since the
original session.

**What survives:** dips have *only* ever been observed in that one original session, which — unlike
every session measured since — had an unknown, likely much longer uptime before it was tested. The
more precise surviving hypothesis is **uptime-duration-since-boot, not boot identity**: something
degrades over a long session and resets on any reboot, rather than each boot independently rolling a
regime. This is not confirmed (would need a session deliberately run for many hours post-reboot without
rebooting again, to see if dip reappears as uptime grows) — flagged as the leading candidate, not
asserted as fact. **Mechanism investigation stopped here per the 2026-10-05 decision** — not chased
further; the harness changes below are designed to be correct regardless of whether this hypothesis
turns out to be right.

## Harness changes landed for Stage 4 (not waiting on the mechanism)

Two additions to `pilot.py`, independent of whether the uptime hypothesis is ever confirmed:

1. **Automatic power-regime flagging.** Every active window is classified `pinned`/`dip`
   (`classify_power_regime()`: cuda-only, `implied_w >= 0.9 * power_cap_w`, threshold chosen with a
   clean margin against the two empirically observed clusters — pinned always ≥98.8% of cap, dip
   always ≤88.5%). `window()`'s return dict carries this per-window (`raw.jsonl`/`windows.csv`); summary
   rows carry a `power_regime` field that is `pinned`, `dip`, or `mixed` if a config's reps disagreed —
   `mixed` is the one that needs a human to look, since it hasn't happened in any run collected so far
   (every run's reps, even the longer 10-rep ones, have agreed with each other throughout).
2. **Fresh-boot guard, enforced not documented.** `check_fresh_boot()` reads `/proc/uptime` and refuses
   to start (same raise-or-warn-behind-an-override pattern as every other pre-flight guard in this file)
   if uptime exceeds `--max-uptime-min` (default 30, explicitly disclosed as a conservative placeholder
   — not a measured decay boundary, since that boundary isn't known yet). Override: `--allow-stale-boot`,
   logs a warning, matches the existing `--override-fast-interval` style. `uptime_s` is now logged in
   every run's `environment.json` regardless, so future Stage 4 data carries this variable even if the
   guard is overridden.

Both verified end-to-end (not just unit-tested): a real run logged `power_regime: pinned` correctly in
its summary; the fresh-boot guard raised by default at a forced-low threshold and the `--allow-stale-boot`
override correctly downgraded it to a warning. 34/34 tests pass (6 new: 3 for the regime classifier, 3
for the fresh-boot guard).

**What this buys Stage 4:** the full factorial will self-report which regime every single run landed
in (no more silently averaging across both without knowing), and will refuse to start unattended after
a long idle/uptime period without an explicit override — converting an unresolved hardware mechanism
into a controlled, disclosed variable rather than hidden noise.
