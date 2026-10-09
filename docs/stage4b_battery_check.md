# Battery check for the Stage 4b confirmatory data (read-only, no Stage 4b statistic computed)

No `windows.csv`/`environment.json` field records power source for any existing `results_stage4b/`
run (confirmed by inspection — this gap is exactly what D26's new `ac_online` field closes for
future runs). Evidence here is therefore indirect: kernel ACPI AC-adapter events and
`ConditionACPower`-gated systemd service checks, swept across the full combined window all 9
session directories fall within (2026-10-07T00:00 UTC through 2026-10-09T02:00 UTC). The journal
was confirmed non-empty for this window (49,991 lines) — the absence of AC-related events below is
a real negative result, not a data-retention gap.

| Directory | Start (UTC) | Last condition start (UTC) | AC evidence |
|---|---|---|---|
| `main_session1_unpinned` | 2026-10-06T20:22:34Z | 2026-10-06T22:11:56Z | AC online throughout: evidenced by absence of any kernel ACPI AC-adapter transition event and absence of any `ConditionACPower`-skipped systemd service run in or around this window |
| `main_session1` | 2026-10-07T04:35:23Z | 2026-10-07T06:24:45Z | same |
| `main_session2` | 2026-10-07T06:45:07Z | 2026-10-07T08:34:30Z | same |
| `main_session3_aborted` | 2026-10-07T10:31:47Z | 2026-10-07T10:48:19Z | same |
| `main_session4_aborted` | 2026-10-07T12:46:49Z | 2026-10-07T13:03:21Z | same |
| `main_session3` | 2026-10-07T19:22:35Z | 2026-10-07T21:11:58Z | same |
| `main_session4` | 2026-10-08T13:03:12Z | 2026-10-08T14:52:35Z | same |
| `main_session5` | 2026-10-08T14:59:16Z | 2026-10-08T16:48:39Z | same |
| `main_session6` | 2026-10-08T17:16:19Z | 2026-10-08T19:05:42Z | same |

**All 9 directories: "AC online throughout."** One combined sweep covers all of them, since the
swept range spans the full set. This is an absence-of-contrary-evidence result, not a continuous
direct reading — no second-by-second power-source log exists for this period (that gap is what
D26 closes going forward). `power_watchdog.log` was checked and found uninformative for this
purpose (it belongs to the P8 campaign's own later watchdog instance, not a continuous monitor
running during Stage 4b).

This stands in direct contrast to the 2026-10-09 evening period (D25), where the same method
found clear, repeated evidence of battery operation and an apparent power-loss shutdown — the
method is shown to detect real events when they occur, which is part of why its silence here is
meaningful rather than just an absence of data.
