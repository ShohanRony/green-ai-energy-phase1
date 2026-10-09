# Battery check for the Stage 4b confirmatory data (read-only, no Stage 4b statistic computed)

No `windows.csv`/`environment.json` field records power source for any existing `results_stage4b/`
run (confirmed by inspection — this gap is exactly what D26's new `ac_online` field closes for
future runs). Evidence here is therefore indirect.

**Correction, 2026-10-10: absence of an ACPI AC-adapter transition event does not by itself show AC
was online — it shows no *change* occurred.** A machine that was already on battery for the entire
swept window, with no adapter plugged or unplugged during it, would produce exactly the same silence
in the kernel log as a machine that was on AC the whole time — a steady state, battery or AC, leaves
no transition event to find. The original version of this document treated that silence as
"AC online throughout," which overstates what an absence-of-transition result can show on its own.
It is still true, and still meaningful (see the 2026-10-09 contrast below), but it needed a second,
independent signal to actually distinguish "steady AC" from "steady battery." That corroboration is
added below, per session.

## Method

- **Signal 1 (original, kept): kernel/systemd transition evidence.** Kernel ACPI AC-adapter events
  and `ConditionACPower`-gated systemd service skips, swept across the full combined window all 9
  session directories fall within (2026-10-07T00:00 UTC through 2026-10-09T02:00 UTC). The journal
  was confirmed non-empty for this window (49,991 lines) — the absence of AC-related events is a
  real negative result, not a data-retention gap. As established above, this signal alone only rules
  out a *change* in power source during the window, not a steady battery state.
- **Signal 2 (new, 2026-10-10): GPU power regime, from the session's own `windows.csv`.** Every
  session's `resnet18_fp32` condition (heaviest sustained compute in the matrix, batch=1, 14 reps
  per full session) was checked for `power_regime` and mean GPU wattage
  (`gpu_energy_j / duration_s`, active phases only). This is diagnostic because this project's own
  AC-pinned power cap is well-characterised (~59-60 W, matching D25's "AC re-validation" numbers and
  the researcher's own ~60 W reference), and D25 independently found the battery-affected runs showed
  a **clearly lower** sustained power profile than the AC-pinned cap. A session whose `fp32` condition
  sits at the known AC-pinned wattage, consistently across every rep, is corroborating evidence of AC
  — not proof on its own (no second instrument was used to cross-check the GPU's own reporting), but
  independent of Signal 1 (one is a systemd/kernel log, the other is the harness's own NVML reading),
  so agreement between the two is stronger than either alone.
  - **Not used for corroboration: the lighter-compute conditions (e.g. `pruned70`).** Checked and
    found `dip` regime in every session uniformly — but `dip` at batch=1 for heavily pruned states is
    this project's own documented *expected* carve-out behaviour regardless of power source (§7 of
    `stage5_analysis_plan.md`; the sub-saturation dip is a compute-threshold effect, not an
    escalation trigger). Its uniformity says nothing about AC vs. battery, so it is not used as a
    signal either way here.
- **Signal 3 (checked, found uninformative): `power_watchdog.log` coverage.** The file's single
  line (`[power_watchdog] starting, on_ac=True, polling every 7s`) is timestamped 2026-10-09 19:57 —
  this is the watchdog's own later startup, near the D25 power-outage period, over a full day after
  `main_session6` (the last of the 9) ended (2026-10-08T19:05:42Z). It provides **zero temporal
  coverage** of any of the 9 sessions. Confirmed, not assumed: no earlier watchdog log file or
  rotated copy exists anywhere in the repository or its untracked working directories.

## Per-session table

| Directory | Start (UTC) | Last condition start (UTC) | Signal 1 (kernel/systemd) | Signal 2 (GPU regime, `fp32`, mean W) | Signal 3 (watchdog) | Grade |
|---|---|---|---|---|---|---|
| `main_session1_unpinned` | 2026-10-06T20:22:34Z | 2026-10-06T22:11:56Z | no transition event | `pinned`, 59.9 W, n=14/14 reps | no coverage | **Evidenced** |
| `main_session1` | 2026-10-07T04:35:23Z | 2026-10-07T06:24:45Z | no transition event | `pinned`, 59.9 W, n=14/14 reps | no coverage | **Evidenced** |
| `main_session2` | 2026-10-07T06:45:07Z | 2026-10-07T08:34:30Z | no transition event | `pinned`, 59.9 W, n=14/14 reps | no coverage | **Evidenced** |
| `main_session3_aborted` | 2026-10-07T10:31:47Z | 2026-10-07T10:48:19Z | no transition event | `pinned`, 59.9 W, n=4/4 reps (short, aborted session) | no coverage | **Evidenced** (smaller n) |
| `main_session4_aborted` | 2026-10-07T12:46:49Z | 2026-10-07T13:03:21Z | no transition event | `pinned`, 59.9 W, n=4/4 reps (short, aborted session) | no coverage | **Evidenced** (smaller n) |
| `main_session3` | 2026-10-07T19:22:35Z | 2026-10-07T21:11:58Z | no transition event | `pinned`, 59.9 W, n=14/14 reps | no coverage | **Evidenced** |
| `main_session4` | 2026-10-08T13:03:12Z | 2026-10-08T14:52:35Z | no transition event | `pinned`, 59.9 W, n=14/14 reps | no coverage | **Evidenced** |
| `main_session5` | 2026-10-08T14:59:16Z | 2026-10-08T16:48:39Z | no transition event | `pinned`, 59.9 W, n=14/14 reps | no coverage | **Evidenced** |
| `main_session6` | 2026-10-08T17:16:19Z | 2026-10-08T19:05:42Z | no transition event | `pinned`, 59.9 W, n=14/14 reps | no coverage | **Evidenced** |

**Grading rule, stated so it isn't read as arbitrary:** "Evidenced" = at least two independent
signals agree and neither contradicts (here: Signal 1 agrees with Signal 2 for all 9; Signal 3 is
absent, not contradictory, for all 9). "Weakly evidenced" would mean only one signal is available, or
signals disagree, or the session falls outside what any signal actually covers — not the case for any
of these 9. **No session in this table is graded "weakly evidenced"** under this corrected method; all
9 upgrade from the original single-signal "AC online throughout" claim to a two-signal "Evidenced"
grade. This is a stronger claim than the original version of this document supported, now that it has
a second, independent signal behind it.

## Remaining limitation

Neither signal is a direct, continuous, ground-truth power-source reading. Signal 1 only rules out a
*change*; Signal 2 is the GPU's own self-reported regime classification (not an external power-source
sensor) and is only diagnostic because this project happens to have a well-characterised AC-pinned
wattage to compare against — a different machine or a different GPU power-cap configuration would not
have this corroboration available. This gap is exactly what D26's `ac_online` field (direct
`/sys/class/power_supply` read, every condition start and every window) closes for all future runs;
sessions 1-6 predate that guard and will never have a first-party direct reading.

This stands in direct contrast to the 2026-10-09 evening period (D25), where the same kernel/systemd
method (Signal 1) found clear, repeated *transition* evidence of battery operation and an apparent
power-loss shutdown, and the GPU regime for those battery-affected runs was independently found to be
clearly lower than the AC-pinned cap used as Signal 2's reference here — the same two signals, in
agreement, pointing the other way. Both signals are shown to detect real events when they occur, which
is part of why their combined silence/consistency here is meaningful rather than just an absence of
data.
