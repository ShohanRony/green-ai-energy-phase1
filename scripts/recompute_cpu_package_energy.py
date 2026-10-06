#!/usr/bin/env python3
"""D16 offline recompute: package-0-only CPU energy from existing raw.jsonl traces.

Read-only on raw.jsonl and summary.csv -- writes summary_package.csv alongside, using
pilot.py's own summarize()/csv_write() so aggregation logic matches the original exactly,
differing only in which RAPL domain(s) the per-window energy comes from (package-0 only,
not package-0+psys -- see deviation_log.md D16).

Per directory:
  1. Determine which trace column is package-0 vs psys from idle-phase mean power
     (package-0 < psys, confirmed empirically 2026-10-06: 12.31W vs 40.93W idle).
  2. Verify package+psys recomputed from trace reproduces the recorded (combined) energy_j.
  3. Recompute package-only energy_j per row, re-run summarize(), write summary_package.csv.

RAPL_RANGE_J is the max_energy_range_uj counter width, confirmed identical for package-0
and psys on this machine (262143328850 uj) -- a hardware register-width property that does
not change across reboots, used here for the same historical data this machine always ran.
"""
import json, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pilot import summarize, csv_write  # noqa: E402

RAPL_RANGE_J = 262143328850 / 1e6

DIRS = [
    'results_stage2/pilot_proof_int8',
    'results_stage3/resnet18_int8',
    'results_stage4_preflight/resnet18_int8_b1',
    'results_stage4/resnet18_int8',
    'results_stage4/mobilenet_v3_small_int8',
    'results_stage4/efficientnet_b0_int8',
    'results_stage4/resnet18_fp32_cpu',
    'results_stage4/mobilenet_v3_small_fp32_cpu',
    'results_stage4/efficientnet_b0_fp32_cpu',
    'results_stage4/codecarbon_check/resnet18_int8',
]


def load_raw(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def integrate_domain(trace, idx):
    energy = 0.
    for (t0, x0), (t1, x1) in zip(trace, trace[1:]):
        d = x1[idx] - x0[idx]
        if d < 0:
            d += RAPL_RANGE_J
        energy += d
    return energy


def domain_index_map(rows):
    """package-0 has the lower idle-phase mean power; psys the higher (confirmed 2026-10-06)."""
    rates = [[], []]
    for r in rows:
        if r.get('phase') not in ('idle_before', 'idle_after'):
            continue
        trace = r.get('trace')
        if not trace or len(trace) < 2:
            continue
        for (t0, v0), (t1, v1) in zip(trace, trace[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            for i in (0, 1):
                d = v1[i] - v0[i]
                if d < 0:
                    d += RAPL_RANGE_J
                rates[i].append(d / dt)
    if not rates[0] or not rates[1]:
        raise RuntimeError('no idle-phase trace data to disambiguate RAPL domains')
    mean0, mean1 = statistics.mean(rates[0]), statistics.mean(rates[1])
    return (0, 1, mean0, mean1) if mean0 < mean1 else (1, 0, mean1, mean0)


def process_dir(rel):
    d = ROOT / rel
    raw_path = d / 'raw.jsonl'
    if not raw_path.exists():
        print(f'SKIP {rel}: no raw.jsonl')
        return
    rows = load_raw(raw_path)
    cpu_rows = [r for r in rows if r.get('device') == 'cpu']
    if not cpu_rows:
        print(f'SKIP {rel}: no CPU rows')
        return

    pkg_idx, psys_idx, pkg_rate, psys_rate = domain_index_map(cpu_rows)
    print(f'{rel}')
    print(f'  domain map: trace[{pkg_idx}]=package-0 ({pkg_rate:.2f}W idle), '
          f'trace[{psys_idx}]=psys ({psys_rate:.2f}W idle)')

    mismatches = []
    pkg_rows = []
    for r in cpu_rows:
        trace = r.get('trace')
        if not trace or len(trace) < 2:
            continue
        pkg_e = integrate_domain(trace, pkg_idx)
        psys_e = integrate_domain(trace, psys_idx)
        combined = pkg_e + psys_e
        recorded = r['energy_j']
        if recorded > 0:
            rel_err = abs(combined - recorded) / recorded
            if rel_err > 1e-4:
                mismatches.append((r['phase'], r['repeat'], recorded, combined, rel_err))
        new_row = {k: v for k, v in r.items() if k != 'trace'}
        new_row['energy_j'] = pkg_e
        new_row.setdefault('power_regime', None)  # field didn't exist before D7 (pre-flight guard)
        pkg_rows.append(new_row)

    if mismatches:
        print(f'  MISMATCH: {len(mismatches)}/{len(pkg_rows)} rows, combined recompute vs recorded energy_j:')
        for m in mismatches[:5]:
            print(f'    phase={m[0]} repeat={m[1]} recorded={m[2]:.4f} recomputed={m[3]:.4f} rel_err={m[4]:.2e}')
    else:
        print(f'  OK: package+psys recompute reproduces recorded energy_j for all {len(pkg_rows)} rows (rel_err<1e-4)')

    summary = summarize(pkg_rows)
    if not summary:
        print(f'  WARNING: summarize() produced no rows for {rel} (insufficient pairs)')
        return
    out_path = d / 'summary_package.csv'
    csv_write(out_path, summary)
    for s in summary:
        print(f'  -> {out_path.relative_to(ROOT)}: pairs={s["pairs"]} '
              f'total_j_mean={s["total_j_mean"]:.2f} total_j_sd={s["total_j_sd"]:.3f} '
              f'above_idle_j_mean={s["above_idle_j_mean"]:.2f}')
    print()


def main():
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    except Exception:
        git_hash = 'unknown'
    print(f'D16 offline recompute -- harness git hash {git_hash}\n')
    for rel in DIRS:
        process_dir(rel)


if __name__ == '__main__':
    main()
