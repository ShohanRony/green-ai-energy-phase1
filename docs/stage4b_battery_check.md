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

## Revision, 2026-10-10 (previous grading above is kept as-is, not edited)

**(a) When the grading rule was written relative to when the 9 sessions' power data was looked at —
disclosed plainly, because this matters for whether the rule can be trusted.** It was **not**
written blind. The sequence, honestly: the `resnet18_fp32` regime and wattage for all 9 sessions was
checked first; all 9 came back `pinned` at ~59.9 W, uniformly consistent with the known AC-pinned
cap; the two-signal "Evidenced" grading rule was written *after* seeing that result, in the same
revision. **This is a real order-of-operations problem** — a rule constructed after seeing data that
happens to support it is weaker evidence than a rule pre-registered blind, even if the rule itself is
reasonable. This revision's response is not to pretend otherwise, but to run the rule against a case
whose outcome is already independently known and different — D25's battery run — below. If the rule
cannot correctly flag a known-battery run as *not* evidenced, the rule would be shown unreliable
regardless of how it was derived; if it does correctly flag it, that is real (if limited, see (d))
evidence the rule has discriminating power, not just a restatement of what it was built to show.

**(b) How dip-regime sessions are graded, given that dip power is below the cap on AC too.** Checked
directly (`classify_power_regime`, `pilot.py`): `pinned` requires `implied_w >= 0.9 * power_cap_w`
(54.0 W at this machine's 60 W cap); anything below is `dip`. **Dip is the normal, expected AC
behaviour for lightly-loaded conditions** (heavily pruned states at batch=1 — `stage5_analysis_plan.md`
§7's own carve-out) — so a `dip` reading on a condition that is *expected* to dip on AC anyway carries
no signal either way, battery or AC. It is only diagnostic for a condition with an established,
exceptionless AC-only pattern: `resnet18_fp32` is `pinned` in **9 of 9** known-AC sessions checked
here, with zero dip readings — that track record, not the reading in isolation, is what makes `dip`
on `resnet18_fp32` specifically a meaningful departure. **Rule, stated for any future session
(including conditions this check hasn't covered): if the diagnostic condition's own regime is `dip`,
Signal 2 is uninformative for that session, Signal 1 alone is not sufficient (established in the
previous revision above), and the session is graded "weakly evidenced" — not upgraded to "Evidenced"
by assuming the dip is benign.** None of the 9 sessions below need this downgrade — all 9 show
`pinned` for `resnet18_fp32` — but the rule is stated so it's not read as never applying.

**(c) The D25 battery run, graded by the identical rule — the rule's own falsification check.**
D25's quoted validation window: `nvml_power_usage_energy_j = 174.94 J` over a `5.005 s` window,
`resnet18_fp32` — implied power `174.94 / 5.005 = 34.95 W`. Against this machine's 60 W cap,
`classify_power_regime`'s own threshold (`0.9 × 60 = 54.0 W`) classifies this as **`dip`**
(`34.95 < 54.0`), **not** `pinned`. Because `resnet18_fp32` has never once shown `dip` in any of the
9 known-AC sessions below, this dip reading is a genuine departure from the established AC-only
pattern — the rule from (b) above applies directly, and this run grades **not evidenced** (worse than
"weakly evidenced": the diagnostic condition's own regime contradicts the AC pattern, it does not
merely fail to confirm it). **The rule passes its own check: it correctly separates a run already
known, by independent kernel/systemd evidence (D25's Signal 1 transition timeline), to have been on
battery.**

**(d) D25 is one battery observation.** The 34.95 W figure comes from a single quoted example window
(`D24`'s validation entry says "e.g. ... for one window"); the full per-rep distribution (mean, min,
across the validation run's 3 reps × 2 runs) was never retained — those directories were scratch,
deleted after use, per that entry's own disclosure. **This check in (c) is therefore one data point
confirming the rule doesn't fail its one known test case, not a characterisation of what battery
power looks like in general** on this machine — no claim is made here about battery-power variance,
only that this one observed instance fails the pinned threshold clearly (34.95 W vs. a 54.0 W floor
is not a close call).

## Per-session table, with mean/min GPU power and window counts (new in this revision)

All figures from `resnet18_fp32`, the diagnostic condition per (b) above. "Windows" = active-phase
(`a1`/`a2`) reps with a valid duration, same reps the existing `summary.csv` pairing uses.

| Directory | Regime | Mean GPU power | Min GPU power | Windows | Watchdog coverage | Grade |
|---|---|---|---|---|---|---|
| `main_session1_unpinned` | `pinned` | 59.91 W | 59.86 W | 14 | none (predates watchdog's only log line) | **Evidenced** |
| `main_session1` | `pinned` | 59.92 W | 59.88 W | 14 | none | **Evidenced** |
| `main_session2` | `pinned` | 59.92 W | 59.88 W | 14 | none | **Evidenced** |
| `main_session3_aborted` | `pinned` | 59.94 W | 59.92 W | 4 (short, aborted session) | none | **Evidenced** (smaller n) |
| `main_session4_aborted` | `pinned` | 59.91 W | 59.88 W | 4 (short, aborted session) | none | **Evidenced** (smaller n) |
| `main_session3` | `pinned` | 59.91 W | 59.89 W | 14 | none | **Evidenced** |
| `main_session4` | `pinned` | 59.90 W | 59.87 W | 14 | none | **Evidenced** |
| `main_session5` | `pinned` | 59.94 W | 59.90 W | 14 | none | **Evidenced** |
| `main_session6` | `pinned` | 59.94 W | 59.92 W | 14 | none | **Evidenced** |
| *(control)* D25 battery run | `dip` | 34.95 W | *(not retained, (d))* | 1 (one quoted example) | none | **Not evidenced** |

**No session among the 9 is downgraded to "weakly evidenced" under this revision** — every one shows
`pinned` with mean/min both clustered tightly around 59.9-60.0 W (never below the 54.0 W pinned floor,
and never more than ~0.15 W off the cap, i.e. no partial-dip or borderline reading in the set). The
control row confirms the rule can and does produce a different grade when the underlying condition is
actually different, which is the main thing (a)'s disclosure asked this revision to check for.

## Note, 2026-10-10 (previous text above kept as-is, not edited)

Stated plainly, so the grade's actual evidentiary weight isn't overread: **Signal 1 (kernel/systemd
transition evidence) alone cannot distinguish steady AC from steady battery** — it only rules out a
power-source *change* during the swept window (stated already above, in the "Correction, 2026-10-10"
section). Every "Evidenced" grade in the table above therefore rests on **one discriminating signal**
(Signal 2 — GPU power sitting at the known AC-pinned cap, which a steady battery state would not
reproduce) **plus one battery contrast** (the D25 control row, the one case where this project has
independently confirmed ground truth, used to check that the rule actually grades a known-battery
run differently). It is not two independent discriminating signals in the stronger sense — Signal 1
contributes only its (real, but non-discriminating-alone) absence-of-transition finding, and Signal 3
(watchdog) contributes nothing for any of the 9 sessions. This does not change any grade already
given; it is a precision statement about what the grade is actually built on.

**Suggested neutral Methods-section sentence, for whoever writes the thesis text:** *"GPU power draw
was consistent with AC operation in all sessions; sessions 1-6 have no direct power-source record."*
Deliberately plain and free of the grading mechanics above — states the one finding that matters for
a reader (power draw pattern matches AC) and the one limitation that matters (no direct record exists
for this data), without importing the signal-counting discussion into the thesis text itself.
