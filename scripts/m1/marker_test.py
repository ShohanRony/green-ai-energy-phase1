#!/usr/bin/env python3
"""Marker test on Apple M1 Mac.
5 trials measuring the timing alignment between in-process load markers
and powermetrics sample intervals via gai_powermetrics.sh.
"""
import glob
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
import torch

OUTDIR = Path('/var/gai-powermetrics')
WRAPPER = '/usr/local/sbin/gai_powermetrics.sh'
HEADER = '*** Sampled system activity'

def check_guards():
    batt = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    therm = subprocess.check_output(['pmset', '-g', 'therm'], text=True)
    first_line = batt.splitlines()[0] if batt.splitlines() else ''
    if 'AC Power' not in first_line:
        raise RuntimeError(f"ABORT: Not on AC power: {batt}")
    if 'No thermal warning level' not in therm and 'Nominal' not in therm:
        raise RuntimeError(f"ABORT: Thermal status not nominal: {therm}")
    return batt.strip(), therm.strip()

def run_trial(trial_idx):
    tag = f"markertest_t{trial_idx}"
    t0_perf = time.perf_counter()
    t0_wall = time.time()
    
    # Start powermetrics via wrapper
    proc = subprocess.Popen(['sudo', '-n', WRAPPER, '400', '40', tag],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Locate output file
    outfile = None
    find_deadline = time.perf_counter() + 30
    while outfile is None and time.perf_counter() < find_deadline:
        for c in glob.glob(str(OUTDIR / f"{tag}_*.txt")):
            try:
                if os.path.getmtime(c) >= t0_wall - 2:
                    outfile = c
                    break
            except FileNotFoundError:
                continue
        if outfile is None:
            if proc.poll() is not None:
                break
            time.sleep(0.005)
            
    if outfile is None:
        raise RuntimeError(f"Trial {trial_idx}: output file not found")
        
    arrivals = {}  # sample_idx -> (arrival_perf, elapsed_ms)
    stop_event = threading.Event()
    first_header_event = threading.Event()
    
    header_pattern = re.compile(r'\*\*\* Sampled system activity .*? \(([\d.]+)ms elapsed\) \*\*\*')
    
    def poller():
        last_count = 0
        while not stop_event.is_set():
            try:
                with open(outfile, 'r', errors='ignore') as f:
                    content = f.read()
                matches = list(header_pattern.finditer(content))
                count = len(matches)
                if count > last_count:
                    now = time.perf_counter()
                    for idx in range(last_count, count):
                        sample_num = idx + 1
                        elapsed = float(matches[idx].group(1))
                        if sample_num not in arrivals:
                            arrivals[sample_num] = (now, elapsed)
                    last_count = count
                    if count >= 1 and not first_header_event.is_set():
                        first_header_event.set()
            except Exception:
                pass
            time.sleep(0.02)
            
    poller_thread = threading.Thread(target=poller, daemon=True)
    poller_thread.start()
    
    # Wait for first header
    if not first_header_event.wait(timeout=10.0):
        stop_event.set()
        proc.kill()
        raise RuntimeError(f"Trial {trial_idx}: timed out waiting for first header")
        
    # Idle 4 s
    time.sleep(4.0)
    
    # 4-thread CPU load (torch matmul) for 5 s
    torch.set_num_threads(4)
    A = torch.randn(2048, 2048, dtype=torch.float32)
    B = torch.randn(2048, 2048, dtype=torch.float32)
    C = torch.empty(2048, 2048, dtype=torch.float32)
    
    t_load_start = time.perf_counter()
    t_load_target_end = t_load_start + 5.0
    matmul_count = 0
    while time.perf_counter() < t_load_target_end:
        torch.matmul(A, B, out=C)
        matmul_count += 1
    t_load_end = time.perf_counter()
    
    # Idle 4 s
    time.sleep(4.0)
    
    # Wait for powermetrics to finish
    stdout, _ = proc.communicate()
    stop_event.set()
    poller_thread.join(timeout=5.0)
    
    # Final read to ensure all samples captured
    with open(outfile, 'r', errors='ignore') as f:
        full_content = f.read()
    matches = list(header_pattern.finditer(full_content))
    now = time.perf_counter()
    for idx, m in enumerate(matches):
        s_num = idx + 1
        elapsed = float(m.group(1))
        if s_num not in arrivals:
            arrivals[s_num] = (now, elapsed)
            
    # Parse CPU power per sample
    blocks = full_content.split('*** Sampled system activity')
    # blocks[0] is header, blocks[1..N] are samples
    cpu_powers = {}
    for idx, b in enumerate(blocks[1:], 1):
        m_cpu = re.search(r'CPU Power:\s*(\d+)\s*mW', b)
        if m_cpu:
            cpu_powers[idx] = int(m_cpu.group(1))
            
    return {
        'trial_idx': trial_idx,
        'outfile': outfile,
        't0_perf': t0_perf,
        't_load_start': t_load_start,
        't_load_end': t_load_end,
        'load_duration_s': t_load_end - t_load_start,
        'matmul_count': matmul_count,
        'arrivals': arrivals,
        'cpu_powers': cpu_powers,
    }

def analyze_trial(data):
    arrivals = data['arrivals']
    cpu_powers = data['cpu_powers']
    t_start = data['t_load_start']
    t_end = data['t_load_end']
    
    sample_nums = sorted(set(arrivals.keys()) & set(cpu_powers.keys()))
    
    samples = []
    for k in sample_nums:
        arr_t, el_ms = arrivals[k]
        dur_s = el_ms / 1000.0
        interval_start = arr_t - dur_s
        interval_end = arr_t
        power = cpu_powers[k]
        
        # Overlap with [t_start, t_end]
        overlap_s = max(0.0, min(interval_end, t_end) - max(interval_start, t_start))
        frac = overlap_s / dur_s if dur_s > 0 else 0.0
        
        samples.append({
            'k': k,
            'arrival_t': arr_t,
            'elapsed_ms': el_ms,
            'interval_start': interval_start,
            'interval_end': interval_end,
            'dur_s': dur_s,
            'power_mw': power,
            'overlap_frac': frac,
        })
        
    # Baseline idle power: samples with interval_end < t_start - 0.2 or interval_start > t_end + 0.2
    idle_powers = [s['power_mw'] for s in samples if s['interval_end'] < t_start - 0.2 or s['interval_start'] > t_end + 0.2]
    load_powers = [s['power_mw'] for s in samples if s['interval_start'] > t_start + 0.2 and s['interval_end'] < t_end - 0.2]
    
    p_idle = sum(idle_powers) / len(idle_powers) if idle_powers else 0.0
    p_load = sum(load_powers) / len(load_powers) if load_powers else 0.0
    
    # Identify straddling samples containing t_start and t_end
    straddle_start_sample = None
    straddle_end_sample = None
    for s in samples:
        if s['interval_start'] <= t_start < s['interval_end']:
            straddle_start_sample = s
        if s['interval_start'] <= t_end < s['interval_end']:
            straddle_end_sample = s
            
    # Compute predicted power for straddling samples
    for s in samples:
        s['pred_power_mw'] = (1.0 - s['overlap_frac']) * p_idle + s['overlap_frac'] * p_load
        
    # Find best-fit time shift delta in [-200ms, +200ms]
    # Shift convention: interval(delta) = [start + delta, end + delta]
    # Positive delta means intervals shift forward in time
    best_delta = 0.0
    best_rss = float('inf')
    
    # Grid search: -0.200 to +0.200 s in 0.2 ms steps
    for step in range(-1000, 1001):
        delta = step * 0.0002
        rss = 0.0
        for s in samples:
            i_start = s['interval_start'] + delta
            i_end = s['interval_end'] + delta
            ov = max(0.0, min(i_end, t_end) - max(i_start, t_start))
            f = ov / s['dur_s'] if s['dur_s'] > 0 else 0.0
            pred = (1.0 - f) * p_idle + f * p_load
            diff = s['power_mw'] - pred
            rss += diff * diff
        if rss < best_rss:
            best_rss = rss
            best_delta = delta
            
    return {
        'data': data,
        'samples': samples,
        'p_idle': p_idle,
        'p_load': p_load,
        'straddle_start': straddle_start_sample,
        'straddle_end': straddle_end_sample,
        'best_shift_s': best_delta,
        'best_shift_ms': best_delta * 1000.0,
        'best_rss': best_rss,
    }

def main():
    print("=" * 70)
    print("M1 MARKER TEST — 5 TRIALS")
    print("=" * 70)
    
    batt_info, therm_info = check_guards()
    print("AC Power Guard: PASS")
    print(batt_info)
    print("Thermal Guard: PASS")
    print(therm_info)
    print()
    
    trials = []
    for i in range(1, 6):
        print(f"--- Running Trial {i}/5 ---")
        trial_data = run_trial(i)
        analysis = analyze_trial(trial_data)
        trials.append(analysis)
        print(f"Trial {i} finished: best-fit shift = {analysis['best_shift_ms']:+.2f} ms")
        if i < 5:
            time.sleep(3.0)  # Cooldown between trials
            
    print("\n" + "=" * 70)
    print("RAW TRIAL RESULTS AND DETAIL TABLES")
    print("=" * 70)
    
    shifts_ms = [t['best_shift_ms'] for t in trials]
    abs_shifts_ms = [abs(s) for s in shifts_ms]
    
    for t in trials:
        idx = t['data']['trial_idx']
        d = t['data']
        print(f"\n### TRIAL {idx} ###")
        print(f"Outfile: {d['outfile']}")
        print(f"Load start perf_counter: {d['t_load_start']:.6f} s")
        print(f"Load end perf_counter:   {d['t_load_end']:.6f} s")
        print(f"Load duration:           {d['load_duration_s']:.6f} s ({d['matmul_count']} matmuls)")
        print(f"Estimated Idle Power:    {t['p_idle']:.2f} mW")
        print(f"Estimated Load Power:    {t['p_load']:.2f} mW")
        print(f"Best-fit time shift:     {t['best_shift_ms']:+.2f} ms (RSS: {t['best_rss']:.1f})")
        
        ss = t['straddle_start']
        se = t['straddle_end']
        print("\nStraddling sample at load start:")
        if ss:
            print(f"  Sample #{ss['k']}: interval [{ss['interval_start']:.4f}, {ss['interval_end']:.4f}] s (elapsed {ss['elapsed_ms']:.2f} ms)")
            print(f"  Load start {d['t_load_start']:.4f} s | Overlap fraction: {ss['overlap_frac']:.4f}")
            print(f"  Observed CPU Power: {ss['power_mw']} mW | Predicted: {ss['pred_power_mw']:.2f} mW")
        else:
            print("  None detected (boundary landed exactly between samples)")
            
        print("Straddling sample at load end:")
        if se:
            print(f"  Sample #{se['k']}: interval [{se['interval_start']:.4f}, {se['interval_end']:.4f}] s (elapsed {se['elapsed_ms']:.2f} ms)")
            print(f"  Load end {d['t_load_end']:.4f} s | Overlap fraction: {se['overlap_frac']:.4f}")
            print(f"  Observed CPU Power: {se['power_mw']} mW | Predicted: {se['pred_power_mw']:.2f} mW")
        else:
            print("  None detected (boundary landed exactly between samples)")
            
        print("\nAll Samples Table:")
        print("k\tarr_rel_s\tel_ms\tint_start_s\tint_end_s\tov_frac\tobs_mW\tpred_mW")
        t0 = d['t0_perf']
        for s in t['samples']:
            print(f"{s['k']}\t{s['arrival_t']-t0:.4f}\t{s['elapsed_ms']:.2f}\t{s['interval_start']-t0:.4f}\t{s['interval_end']-t0:.4f}\t{s['overlap_frac']:.4f}\t{s['power_mw']}\t{s['pred_power_mw']:.1f}")

    print("\n" + "=" * 70)
    print("CROSS-TRIAL SHIFT SUMMARY (Reviewer judgment, no PASS/FAIL assertion)")
    print("=" * 70)
    print("Trial\tShift (ms)\t|Shift| (ms)\tIdle (mW)\tLoad (mW)\tStart Straddle (Obs/Pred)\tEnd Straddle (Obs/Pred)")
    for t in trials:
        idx = t['data']['trial_idx']
        ss = t['straddle_start']
        se = t['straddle_end']
        ss_str = f"{ss['power_mw']}/{ss['pred_power_mw']:.1f}" if ss else "N/A"
        se_str = f"{se['power_mw']}/{se['pred_power_mw']:.1f}" if se else "N/A"
        print(f"{idx}\t{t['best_shift_ms']:+.2f}\t{abs(t['best_shift_ms']):.2f}\t\t{t['p_idle']:.1f}\t\t{t['p_load']:.1f}\t\t{ss_str}\t\t\t{se_str}")
        
    shifts_sorted = sorted(shifts_ms)
    abs_sorted = sorted(abs_shifts_ms)
    n = len(shifts_ms)
    median_shift = shifts_sorted[n//2]
    median_abs_shift = abs_sorted[n//2]
    mean_shift = sum(shifts_ms) / n
    variance = sum((x - mean_shift)**2 for x in shifts_ms) / (n - 1) if n > 1 else 0.0
    std_shift = variance**0.5
    min_shift = min(shifts_ms)
    max_shift = max(shifts_ms)
    spread = max_shift - min_shift
    
    print("\nStatistical Distribution of Time Shift (delta, ms):")
    print(f"Individual shifts (ms):  {[round(x, 2) for x in shifts_ms]}")
    print(f"Absolute shifts (ms):    {[round(x, 2) for x in abs_shifts_ms]}")
    print(f"Median shift:            {median_shift:+.2f} ms")
    print(f"Median |shift|:          {median_abs_shift:.2f} ms")
    print(f"Mean shift:              {mean_shift:+.2f} ms")
    print(f"Std dev of shift:        {std_shift:.2f} ms")
    print(f"Min shift:               {min_shift:+.2f} ms")
    print(f"Max shift:               {max_shift:+.2f} ms")
    print(f"Spread (Max - Min):      {spread:.2f} ms")
    print(f"Max |shift|:             {max(abs_shifts_ms):.2f} ms")
    print("=" * 70)

if __name__ == '__main__':
    main()
