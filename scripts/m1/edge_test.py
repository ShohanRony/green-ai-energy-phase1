#!/usr/bin/env python3
"""M1 Edge-Structure Test (Non-Energy), two-poller variant.
Measures rise-edge offset r and fall-edge offset f between in-process load markers
and powermetrics sample intervals across a 10-pulse train, fitted independently from
two concurrent arrival-time pollers (P20: whole-file re-read every 20ms; P2: stat +
incremental read of new bytes only, every ~2ms), so the two can be compared against
each other directly. 6 trials, all under default scheduling (no taskpolicy, no OpenMP
env var changes between trials). Numbers only, no PASS/FAIL, no interpretation.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from guards import GuardError, check_ac_power_m1, check_thermal_nominal_m1

OUTDIR = Path('/var/gai-powermetrics')
WRAPPER = '/usr/local/sbin/gai_powermetrics.sh'
HEADER_RE = re.compile(r'\*\*\* Sampled system activity .*? \(([\d.]+)ms elapsed\) \*\*\*')
CPU_POWER_RE = re.compile(r'CPU Power:\s*(\d+)\s*mW')


def check_guards():
    """Delegates to guards.py (A1's fail-closed thermal parsing included) instead
    of duplicating the parsing logic inline. Raises GuardError, not RuntimeError,
    on failure -- the caller's abort message comes from guards.py directly."""
    batt = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    therm = subprocess.check_output(['pmset', '-g', 'therm'], text=True)
    check_ac_power_m1(batt)
    check_thermal_nominal_m1(therm)
    return batt.strip(), therm.strip()


def run_worker_trial(trial_idx, out_json):
    import torch

    tag = f"edgetest_t{trial_idx}"
    sample_interval_ms = 400
    sample_count = 155  # 400ms x 155 = 62s; covers ~0.4s(first header)+4s(idle)+40s(train)+8s(margin)

    # All setup BEFORE powermetrics starts.
    torch.set_num_threads(4)
    A = torch.randn(2048, 2048, dtype=torch.float32)
    B = torch.randn(2048, 2048, dtype=torch.float32)
    C = torch.empty(2048, 2048, dtype=torch.float32)
    for _ in range(20):
        torch.matmul(A, B, out=C)  # 20 untimed warm-up matmuls, before powermetrics starts

    t0_perf = time.perf_counter()
    t0_wall = time.time()

    proc = subprocess.Popen(['sudo', '-n', WRAPPER, str(sample_interval_ms), str(sample_count), tag],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    outfile = None
    find_deadline = time.perf_counter() + 30.0
    while outfile is None and time.perf_counter() < find_deadline:
        for c in glob.glob(str(OUTDIR / f"{tag}_*.txt")):
            try:
                if os.path.getmtime(c) >= t0_wall - 2.0:
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

    arrivals_p20 = {}  # sample_idx -> (arrival_perf, elapsed_ms)
    arrivals_p2 = {}
    stop_event = threading.Event()
    first_header_event_p20 = threading.Event()
    first_header_event_p2 = threading.Event()

    def poller_p20():
        """Re-reads the whole file every 20ms (matches the flush-test method already
        validated in m1_flush_test.txt)."""
        last_count = 0
        while not stop_event.is_set():
            try:
                with open(outfile, 'r', errors='ignore') as f:
                    content = f.read()
                matches = list(HEADER_RE.finditer(content))
                count = len(matches)
                if count > last_count:
                    now = time.perf_counter()
                    for idx in range(last_count, count):
                        sample_num = idx + 1
                        elapsed = float(matches[idx].group(1))
                        if sample_num not in arrivals_p20:
                            arrivals_p20[sample_num] = (now, elapsed)
                    last_count = count
                    if count >= 1 and not first_header_event_p20.is_set():
                        first_header_event_p20.set()
            except Exception:
                pass
            time.sleep(0.020)

    def poller_p2():
        """Stats the file every ~2ms; only reads bytes after the last-seen offset.
        A header can straddle a read boundary, so unmatched trailing text from one
        read is carried over and prepended to the next."""
        last_offset = 0
        last_count = 0
        carry = ''
        while not stop_event.is_set():
            try:
                size = os.path.getsize(outfile)
                if size > last_offset:
                    with open(outfile, 'r', errors='ignore') as f:
                        f.seek(last_offset)
                        chunk = f.read()
                    last_offset = size
                    text = carry + chunk
                    matches = list(HEADER_RE.finditer(text))
                    count_total = last_count + len(matches)
                    if matches:
                        now = time.perf_counter()
                        for idx, m in enumerate(matches):
                            sample_num = last_count + idx + 1
                            elapsed = float(m.group(1))
                            if sample_num not in arrivals_p2:
                                arrivals_p2[sample_num] = (now, elapsed)
                        last_count = count_total
                        if last_count >= 1 and not first_header_event_p2.is_set():
                            first_header_event_p2.set()
                        # keep only text after the last header start, in case a new
                        # header is mid-arrival at the next poll
                        last_match_end = matches[-1].end()
                        carry = text[last_match_end:]
                    else:
                        carry = text
                    # bound carry size so a non-header-containing file can't grow it forever
                    if len(carry) > 8192:
                        carry = carry[-8192:]
            except Exception:
                pass
            time.sleep(0.002)

    t_p20 = threading.Thread(target=poller_p20, daemon=True)
    t_p2 = threading.Thread(target=poller_p2, daemon=True)
    t_p20.start()
    t_p2.start()

    if not (first_header_event_p20.wait(timeout=15.0) and first_header_event_p2.wait(timeout=15.0)):
        stop_event.set()
        proc.kill()
        raise RuntimeError(f"Trial {trial_idx}: timed out waiting for first header on one or both pollers")

    t_first_header = time.perf_counter()

    # Idle 4 s
    time.sleep(4.0)

    # Pulse train of 10 pulses: 2.0 s load, 2.0 s idle. ZERO subprocess calls during the train.
    pulses = []
    for p in range(10):
        t_on = time.perf_counter()
        t_target_off = t_on + 2.0
        count = 0
        while time.perf_counter() < t_target_off:
            torch.matmul(A, B, out=C)
            count += 1
        t_off = time.perf_counter()
        pulses.append({
            'pulse_idx': p + 1,
            't_on': t_on,
            't_off': t_off,
            'duration_s': t_off - t_on,
            'matmul_count': count
        })
        time.sleep(2.0)

    t_train_done = time.perf_counter()

    # 8 s post-train margin (2.0s already in pulse 10's trailing sleep, sleep 6.0s more)
    time.sleep(6.0)

    stdout, _ = proc.communicate(timeout=30.0)
    stop_event.set()
    t_p20.join(timeout=5.0)
    t_p2.join(timeout=5.0)

    # Final read to ensure all samples captured by both bookkeeping dicts
    with open(outfile, 'r', errors='ignore') as f:
        full_content = f.read()
    matches = list(HEADER_RE.finditer(full_content))
    now = time.perf_counter()
    for idx, m in enumerate(matches):
        s_num = idx + 1
        elapsed = float(m.group(1))
        if s_num not in arrivals_p20:
            arrivals_p20[s_num] = (now, elapsed)
        if s_num not in arrivals_p2:
            arrivals_p2[s_num] = (now, elapsed)

    blocks = full_content.split('*** Sampled system activity')
    cpu_powers = {}
    for idx, b in enumerate(blocks[1:], 1):
        m_cpu = CPU_POWER_RE.search(b)
        if m_cpu:
            cpu_powers[idx] = int(m_cpu.group(1))

    trial_data = {
        'trial_idx': trial_idx,
        'outfile': outfile,
        't0_perf': t0_perf,
        't_first_header': t_first_header,
        'pulses': pulses,
        't_train_done': t_train_done,
        'arrivals_p20': {str(k): list(v) for k, v in arrivals_p20.items()},
        'arrivals_p2': {str(k): list(v) for k, v in arrivals_p2.items()},
        'cpu_powers': {str(k): v for k, v in cpu_powers.items()}
    }

    with open(out_json, 'w') as f:
        json.dump(trial_data, f, indent=2)


def compute_overlap_single(a, b, pulses, r, f):
    total = 0.0
    for p in pulses:
        u = p['t_on'] + r
        v = p['t_off'] + f
        if b > u and a < v:
            total += max(0.0, min(b, v) - max(a, u))
    return total


def fit_rf_least_squares(sample_list, pulse_list, p_idle, p_load, r_range=(-0.200, 0.200), f_range=(-0.200, 0.400)):
    """Fit r and f independently by least squares using vectorized grid search."""
    r_grid = np.linspace(r_range[0], r_range[1], 4001)
    f_grid = np.linspace(f_range[0], f_range[1], 6001)

    rise_candidates = []
    fall_candidates = []

    for s in sample_list:
        a = s['a']
        b = s['b']
        dur = s['dur_s']
        pw = s['power_mw']

        for p in pulse_list:
            if b >= p['t_on'] - 0.25 and a <= p['t_on'] + 0.25:
                rise_candidates.append({
                    'a': a, 'b': b, 'dur': dur, 'pw': pw, 't_on': p['t_on'], 'pulse_idx': p['pulse_idx']
                })
                break

        for p in pulse_list:
            if b >= p['t_off'] - 0.25 and a <= p['t_off'] + 0.45:
                fall_candidates.append({
                    'a': a, 'b': b, 'dur': dur, 'pw': pw, 't_off': p['t_off'], 'pulse_idx': p['pulse_idx']
                })
                break

    rss_r = np.zeros_like(r_grid)
    for c in rise_candidates:
        ov = np.clip(c['b'] - (c['t_on'] + r_grid), 0.0, c['dur'])
        frac = ov / c['dur']
        pred = p_idle + (p_load - p_idle) * frac
        rss_r += (c['pw'] - pred)**2

    best_r_idx = np.argmin(rss_r)
    best_r = r_grid[best_r_idx]

    r_fine = np.linspace(best_r - 0.002, best_r + 0.002, 401)
    rss_r_fine = np.zeros_like(r_fine)
    for c in rise_candidates:
        ov = np.clip(c['b'] - (c['t_on'] + r_fine), 0.0, c['dur'])
        frac = ov / c['dur']
        pred = p_idle + (p_load - p_idle) * frac
        rss_r_fine += (c['pw'] - pred)**2
    best_r = r_fine[np.argmin(rss_r_fine)]

    rss_f = np.zeros_like(f_grid)
    for c in fall_candidates:
        ov = np.clip((c['t_off'] + f_grid) - c['a'], 0.0, c['dur'])
        frac = ov / c['dur']
        pred = p_idle + (p_load - p_idle) * frac
        rss_f += (c['pw'] - pred)**2

    best_f_idx = np.argmin(rss_f)
    best_f = f_grid[best_f_idx]

    f_fine = np.linspace(best_f - 0.002, best_f + 0.002, 401)
    rss_f_fine = np.zeros_like(f_fine)
    for c in fall_candidates:
        ov = np.clip((c['t_off'] + f_fine) - c['a'], 0.0, c['dur'])
        frac = ov / c['dur']
        pred = p_idle + (p_load - p_idle) * frac
        rss_f_fine += (c['pw'] - pred)**2
    best_f = f_fine[np.argmin(rss_f_fine)]

    total_rss = 0.0
    for s in sample_list:
        ov = compute_overlap_single(s['a'], s['b'], pulse_list, best_r, best_f)
        frac = ov / s['dur_s'] if s['dur_s'] > 0 else 0.0
        pred = (1.0 - frac) * p_idle + frac * p_load
        diff = s['power_mw'] - pred
        total_rss += diff * diff

    return float(best_r), float(best_f), float(total_rss)


def build_samples(arrivals, cpu_powers):
    sample_nums = sorted(set(arrivals.keys()) & set(cpu_powers.keys()))
    out = []
    for k in sample_nums:
        arr_t, el_ms = arrivals[k]
        dur_s = el_ms / 1000.0
        a = arr_t - dur_s
        b = arr_t
        out.append({'k': k, 'arrival_t': arr_t, 'elapsed_ms': el_ms, 'dur_s': dur_s,
                    'a': a, 'b': b, 'power_mw': cpu_powers[k]})
    return out


def fit_one_poller(all_samples, pulses, t_start_train, t_end_train, trial_idx, rng_seed):
    train_samples = [s for s in all_samples if s['a'] >= t_start_train and s['b'] <= t_end_train]

    pure_idle_samples = []
    pure_load_samples = []
    for s in train_samples:
        is_load = any(s['a'] >= p['t_on'] + 0.05 and s['b'] <= p['t_off'] - 0.05 for p in pulses)
        is_idle = all(s['b'] <= p['t_on'] - 0.05 or s['a'] >= p['t_off'] + 0.05 for p in pulses)
        if is_load:
            pure_load_samples.append(s)
        elif is_idle:
            pure_idle_samples.append(s)

    p_idle_mean = sum(s['power_mw'] for s in pure_idle_samples) / len(pure_idle_samples) if pure_idle_samples else 0.0
    p_load_mean = sum(s['power_mw'] for s in pure_load_samples) / len(pure_load_samples) if pure_load_samples else 0.0

    best_r, best_f, total_rss = fit_rf_least_squares(train_samples, pulses, p_idle_mean, p_load_mean)

    lopo_r = []
    lopo_f = []
    for j in range(len(pulses)):
        sub_pulses = [p for i, p in enumerate(pulses) if i != j]
        p_left = pulses[j]
        sub_samples = [s for s in train_samples if not (s['b'] >= p_left['t_on'] + best_r - 0.1 and s['a'] <= p_left['t_off'] + best_f + 0.1)]
        r_j, f_j, _ = fit_rf_least_squares(sub_samples, sub_pulses, p_idle_mean, p_load_mean)
        lopo_r.append(r_j)
        lopo_f.append(f_j)

    lopo_r_spread = (max(lopo_r) - min(lopo_r)) * 1000.0
    lopo_f_spread = (max(lopo_f) - min(lopo_f)) * 1000.0

    return {
        'trial_idx': trial_idx,
        'n_samples_in_train_window': len(train_samples),
        'pure_idle_count': len(pure_idle_samples),
        'pure_load_count': len(pure_load_samples),
        'p_idle_mean': p_idle_mean,
        'p_load_mean': p_load_mean,
        'best_r_ms': best_r * 1000.0,
        'best_f_ms': best_f * 1000.0,
        'total_rss': total_rss,
        'lopo_r_spread_ms': lopo_r_spread,
        'lopo_f_spread_ms': lopo_f_spread,
        'all_samples': train_samples,
    }


def analyze_trial_data(trial_dict):
    pulses = trial_dict['pulses']
    arrivals_p20 = {int(k): v for k, v in trial_dict['arrivals_p20'].items()}
    arrivals_p2 = {int(k): v for k, v in trial_dict['arrivals_p2'].items()}
    cpu_powers = {int(k): v for k, v in trial_dict['cpu_powers'].items()}

    t_start_train = pulses[0]['t_on'] - 1.0
    t_end_train = pulses[-1]['t_off'] + 7.5

    samples_p20 = build_samples(arrivals_p20, cpu_powers)
    samples_p2 = build_samples(arrivals_p2, cpu_powers)

    fit_p20 = fit_one_poller(samples_p20, pulses, t_start_train, t_end_train, trial_dict['trial_idx'], 0)
    fit_p2 = fit_one_poller(samples_p2, pulses, t_start_train, t_end_train, trial_dict['trial_idx'], 0)

    # P20-minus-P2 arrival difference, per header present in both dicts
    common_keys = sorted(set(arrivals_p20.keys()) & set(arrivals_p2.keys()))
    diffs_ms = [(arrivals_p20[k][0] - arrivals_p2[k][0]) * 1000.0 for k in common_keys]

    return {
        'trial_idx': trial_dict['trial_idx'],
        'pulses': pulses,
        'fit_p20': fit_p20,
        'fit_p2': fit_p2,
        'n_headers_p20': len(arrivals_p20),
        'n_headers_p2': len(arrivals_p2),
        'n_headers_common': len(common_keys),
        'p20_minus_p2_diffs_ms': diffs_ms,
    }


def format_output_tables(trials, output_path):
    lines = []

    def p(s=''):
        lines.append(s)

    p("=" * 80)
    p("M1 EDGE-STRUCTURE TEST, TWO-POLLER VARIANT (P20 vs P2)")
    p("=" * 80)
    p("Protocol: 10 pulses of 2.0 s load (4-thread torch.matmul 2048x2048 float32), 2.0 s idle")
    p("Setup (import torch, 4 threads, allocate A/B/C, 20 untimed warm-up matmuls) BEFORE powermetrics starts")
    p("Sampling: 400 ms powermetrics")
    p("P20 poller: re-reads whole file every 20 ms")
    p("P2 poller: os.stat size every ~2 ms, reads only new bytes since last offset")
    p("Model: power(t) = idle + (load - idle) * 1[t in (t_on + r, t_off + f)]")
    p("6 trials, all default scheduling (no taskpolicy, no OMP env var changes)")
    p("Numbers only. No PASS/FAIL stated here.")
    p()

    all_idle_p20 = [t['fit_p20']['p_idle_mean'] for t in trials]
    median_idle_p20 = float(np.median(all_idle_p20))

    p("-" * 80)
    p("SECTION 1: PER-TRIAL FIT, P20 POLLER")
    p("-" * 80)
    p("Trial\tIdle(mW)\tLoad(mW)\tr(ms)\tf(ms)\tLOPO_r_spread(ms)\tLOPO_f_spread(ms)\tRSS\tIdle>3xMedian")
    for t in trials:
        f20 = t['fit_p20']
        flag = f20['p_idle_mean'] > 3.0 * median_idle_p20
        p(f"T{t['trial_idx']}\t{f20['p_idle_mean']:.2f}\t\t{f20['p_load_mean']:.2f}\t{f20['best_r_ms']:+.2f}\t{f20['best_f_ms']:+.2f}\t{f20['lopo_r_spread_ms']:.2f}\t\t\t{f20['lopo_f_spread_ms']:.2f}\t\t\t{f20['total_rss']:.1f}\t{flag}")
    p(f"Median idle power (P20) across {len(trials)} trials: {median_idle_p20:.2f} mW")
    p()

    all_idle_p2 = [t['fit_p2']['p_idle_mean'] for t in trials]
    median_idle_p2 = float(np.median(all_idle_p2))

    p("-" * 80)
    p("SECTION 2: PER-TRIAL FIT, P2 POLLER")
    p("-" * 80)
    p("Trial\tIdle(mW)\tLoad(mW)\tr(ms)\tf(ms)\tLOPO_r_spread(ms)\tLOPO_f_spread(ms)\tRSS\tIdle>3xMedian")
    for t in trials:
        f2 = t['fit_p2']
        flag = f2['p_idle_mean'] > 3.0 * median_idle_p2
        p(f"T{t['trial_idx']}\t{f2['p_idle_mean']:.2f}\t\t{f2['p_load_mean']:.2f}\t{f2['best_r_ms']:+.2f}\t{f2['best_f_ms']:+.2f}\t{f2['lopo_r_spread_ms']:.2f}\t\t\t{f2['lopo_f_spread_ms']:.2f}\t\t\t{f2['total_rss']:.1f}\t{flag}")
    p(f"Median idle power (P2) across {len(trials)} trials: {median_idle_p2:.2f} mW")
    p()

    p("-" * 80)
    p("SECTION 3: PER-HEADER ARRIVAL DIFFERENCE, P20 MINUS P2 (ms)")
    p("-" * 80)
    p("Trial\tN_common_headers\tMean\tSD\tMin\tMax")
    all_diffs = []
    for t in trials:
        d = np.array(t['p20_minus_p2_diffs_ms'])
        all_diffs.extend(t['p20_minus_p2_diffs_ms'])
        if len(d) > 0:
            p(f"T{t['trial_idx']}\t{t['n_headers_common']}\t\t\t{d.mean():+.3f}\t{d.std(ddof=1) if len(d) > 1 else 0.0:.3f}\t{d.min():+.3f}\t{d.max():+.3f}")
        else:
            p(f"T{t['trial_idx']}\t{t['n_headers_common']}\t\t\tnot measured\tnot measured\tnot measured\tnot measured")
    if all_diffs:
        ad = np.array(all_diffs)
        p(f"POOLED (all trials)\t{len(all_diffs)}\t\t\t{ad.mean():+.3f}\t{ad.std(ddof=1):.3f}\t{ad.min():+.3f}\t{ad.max():+.3f}")
    p()

    p("-" * 80)
    p("SECTION 4: HEADER COUNTS PER TRIAL")
    p("-" * 80)
    p("Trial\tN_headers_P20\tN_headers_P2\tN_common")
    for t in trials:
        p(f"T{t['trial_idx']}\t{t['n_headers_p20']}\t\t{t['n_headers_p2']}\t\t{t['n_headers_common']}")
    p()

    p("=" * 80)
    p("END OF TWO-POLLER EDGE-STRUCTURE REPORT")
    p("=" * 80)

    text = "\n".join(lines)
    with open(output_path, 'a') as f:
        f.write('\n\n')
        f.write(text)
        f.write('\n')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true', help='Run a single worker trial')
    parser.add_argument('--trial', type=int, default=1, help='Trial index')
    parser.add_argument('--json-out', type=str, default='/tmp/trial_out.json', help='Output JSON path')
    args = parser.parse_args()

    if args.worker:
        run_worker_trial(args.trial, args.json_out)
        return

    print("Checking guards on host...")
    batt, therm = check_guards()
    print("AC Battery Guard:")
    print(batt)
    print("Thermal Guard:")
    print(therm)
    print()

    trials_data = []
    for i in range(1, 7):
        print(f"=== Starting Trial {i}/6 (default scheduling) ===")
        json_path = f"/tmp/edgetest3_t{i}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--trial', str(i), '--json-out', json_path]
        subprocess.run(cmd, check=True)
        with open(json_path, 'r') as f:
            td = json.load(f)
        analysis = analyze_trial_data(td)
        trials_data.append(analysis)
        print(f"Trial {i} completed: r(P20)={analysis['fit_p20']['best_r_ms']:+.2f}ms "
              f"r(P2)={analysis['fit_p2']['best_r_ms']:+.2f}ms")
        if i < 6:
            time.sleep(4.0)

    out_file = Path.home() / 'm1_edge_test.txt'
    report_text = format_output_tables(trials_data, out_file)
    print("\nReport appended to:", out_file)
    print()
    print(report_text)


if __name__ == '__main__':
    main()
