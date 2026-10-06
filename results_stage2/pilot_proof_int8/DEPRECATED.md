# Pilot-only (D12) AND CPU energy here is package+psys, not package-0 (D16)

This directory predates the `power_regime` field entirely and has only 1 repeat -- already
pilot-only per `deviation_log.md` D12, excluded from confirmatory analysis.

Separately, its CPU energy (`energy_j` in `raw.jsonl`/`summary.csv`) sums two RAPL domains,
`package-0` and `psys`, not `package-0` alone -- see `deviation_log.md` D16. `psys` is a
superset platform-power domain, not an independent rail; the combined figure is inflated
relative to package-only CPU energy.

No corrected summary exists for this directory (`summarize()` needs >=3 pairs; this run has
1 rep). `raw.jsonl` still has the full per-domain trace if a package-only recompute is ever
needed here specifically -- see `scripts/recompute_cpu_package_energy.py`.
