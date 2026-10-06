# Pilot-only (D12) AND CPU energy here is package+psys, not package-0 (D16)

Already pilot-only per `deviation_log.md` D12 -- collected before the power-regime guard
existed, excluded from confirmatory analysis.

Separately, its CPU energy (`energy_j` in `raw.jsonl`/`summary.csv`) sums two RAPL domains,
`package-0` and `psys`, not `package-0` alone -- see `deviation_log.md` D16. `psys` is a
superset platform-power domain, not an independent rail; the combined figure is inflated
relative to package-only CPU energy.

**Corrected (package-0 only) figures: `summary_package.csv`, same directory** -- recomputed
offline from this directory's own `raw.jsonl` trace (never modified) by
`scripts/recompute_cpu_package_energy.py`. `psys` is secondary, not reported in a separate
file; recoverable from the same trace if needed.
