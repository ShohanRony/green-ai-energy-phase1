#!/usr/bin/env python3
"""Computes the M1 thermal guard threshold T from the calibration evidence in
this directory. Reproduces config/thermal_thresholds_m1.json's T_mhz exactly.

Reference set (included): FP32-TorchScript continuous windows from Test 0
(n=180) and Test 1 (n=180), plus ALL Test 2 duty-cycle active windows
(n=168, both INT8 and FP32-TS blocks) = 528 windows.

Excluded: INT8 continuous runs (Test 0 n=180, Test 1 n=180) -- both show a
statistically significant sustained negative frequency trend (sustained-load
throttling), so they are not used to characterize the "nominal, not
throttled" reference distribution.

Idle check (idle_check_pre_test1/) is NOT part of the reference set -- it is
background/pre-check evidence (confirms the machine started cold), not an
active-inference window, and is not combined into this computation.

Rule (pre-registered, unchanged here): T = 0.90 x pooled 5th percentile of
the reference set's per-window mean of CPU 4-7 frequency (MHz).
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


def load_p47(path):
    return [w['p47_mean_mhz'] for w in json.loads(Path(path).read_text())]


def main():
    test0_fp32 = load_p47(HERE / 'test0_resnet18_fp32_ts' / 'windows.json')
    test1_fp32 = load_p47(HERE / 'test1_resnet18_fp32_ts' / 'windows.json')
    test2_all = json.loads((HERE / 'test2_dutycycle' / 'active_windows.json').read_text())
    test2_p47 = [w['p47_mean_mhz'] for w in test2_all]

    reference = test0_fp32 + test1_fp32 + test2_p47
    print(f"test0_fp32_ts: n={len(test0_fp32)}")
    print(f"test1_fp32_ts: n={len(test1_fp32)}")
    print(f"test2_dutycycle (all active windows, INT8+FP32): n={len(test2_p47)}")
    print(f"reference set total: n={len(reference)}")

    s = sorted(reference)
    p5 = percentile(s, 5)
    T = 0.90 * p5
    print(f"P5 = {p5:.6f} MHz")
    print(f"T = 0.90 x P5 = {T:.6f} MHz")
    print(f"min={s[0]:.4f} max={s[-1]:.4f}")

    config_path = HERE.parent.parent / 'config' / 'thermal_thresholds_m1.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        print(f"\nconfig/thermal_thresholds_m1.json T_mhz = {config['T_mhz']}")
        print(f"matches computed T (rounded to 2dp): {round(T, 2) == config['T_mhz']}")


if __name__ == '__main__':
    main()
