# Superseded: collected with the old backend

This data was collected via `nvmlDeviceGetTotalEnergyConsumption` (the
cumulative-energy counter), which this project's own investigation confirmed
(2026-10-03) over-reports active-phase GPU power by ~30% on this hardware --
not legitimate boost overshoot, a counter-reporting quirk. See
`results/active_power_baseline_investigation.md`.

The original *purpose* of this run (confirming a guard/flag/field logs and
behaves correctly) is unaffected -- that's about row schema and control flow,
not energy accuracy. Any energy *value* in this directory should not be
cited as a validated measurement. Kept for the evidence trail only.
