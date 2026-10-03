# green-ai-energy-phase1

RAPL+NVML paired energy-measurement harness for the Masaryk Green AI proposal's
Phase 1 (see `docs/phase1-execution-plan.md`, `docs/stage1-implementation-brief.md`).
Ported from `research/energy-pilot` (2026-10-03) — that repo stays with the other,
earlier proposal (calibration/reliability angle); this repo is this proposal's own
harness going forward.

## RAPL domain in use

This machine exposes two domains:

```
intel-rapl:0 -> package-0
intel-rapl:1 -> psys
```

Both are world-readable without sudo via `/etc/udev/rules.d/51-rapl-permissions.rules`
(`chmod -R a+r /sys%p` on powercap add). The harness reads `package-0` (CPU package
energy), not `psys` (whole-system). This is a stated deviation, not a bug: package
energy excludes DRAM/VRM/peripheral rails that `psys` would include, so reported
CPU-side energy is a lower bound on true system energy for that component.

## NVML sampling interval

Confirmed-reproducible counter-telescoping artifact on this RTX 3050: polling
`nvmlDeviceGetTotalEnergyConsumption` faster than ~0.3s inflates implied power
(20ms -> ~300W against a 60W cap; converges to a plausible idle wattage only at
0.3-0.5s). Swept again 2026-10-03, same shape as the original September finding.

The floor is enforced in code, not just documented: `--interval` defaults to 0.4s,
and any `--device cuda` run with `--interval < 0.3` raises unless
`--override-fast-interval` is passed, which prints a warning and proceeds anyway.
Enforced in `pilot.py`'s `check_interval_floor()`, called from `main()`.

## RAPL sampling-rate ceiling

Checklist item 4's other half: RAPL/CPU reads via perf-events must never sample
faster than 100Hz (interval < 0.01s). Enforced the same way, same function:
`pilot.py`'s `check_interval_floor()`, called a second time from `main()` for
`--device cpu`. Default rejects; `--override-fast-rapl-interval` allows it and
logs a warning.

## What's here

- `pilot.py` — RAPL+NVML paired measurement logic, idle-baseline capture, the
  interval floor/ceiling above.
- `tests/test_pilot.py` — the 4 original unit tests (counter wrap, power
  integration, counter-reset rejection, idle-duration/detection) plus the
  interval-floor/ceiling tests added 2026-10-03.
- `docs/` — the Phase 1 execution plan and Stage 1 implementation brief this repo
  is built against.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python pilot.py --device cuda --data <cifar10-root> --out results/run1 --sizes 32 --batches 16 --windows 2 --repeats 30
```
