# Not superseded data to discard -- this IS the evidence that found the problem

Collected via `nvmlDeviceGetTotalEnergyConsumption` (the cumulative-energy
counter) before the 2026-10-03 correction. This directory's data is exactly
what the investigation analyzed to discover that backend's ~30% active-phase
over-reporting and its connection to the counter-telescoping/idle_after
artifacts. See `results/nvml_aggregation_investigation.md`,
`results/idle_after_anomaly_investigation.md`, and
`results/active_power_baseline_investigation.md`.

Do not cite the energy *values* here as validated GPU power measurements --
they're superseded by the corrected `nvmlDeviceGetPowerUsage` default
(commit `672ce93`). Keep this data; it's the proof, not noise to clean up.
