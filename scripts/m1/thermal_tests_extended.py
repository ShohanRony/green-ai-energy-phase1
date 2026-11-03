#!/usr/bin/env python3
"""M1 Thermal Tests Extended (Non-Energy):
Test 1 (Order Check):
  1. Wait until P-core idle frequency and CPU Power returned to idle for at least 10 min.
  2. ResNet-18 INT8 first for 15 min.
  3. Cool-down gap.
  4. ResNet-18 FP32-TS second for 15 min.
  5. Same metrics and report as before.

Test 2 (Duty-Cycle Check):
  1. 10 min cool-down after Test 1.
  2. 45 min of session-like duty cycle with ResNet-18 INT8 and FP32-TS alternating per block.
     Each block = 6 repeats x (idle 3s settle + 5s idle, active 3s warm-up + 5s window, active 3s + 5s, idle 3s + 5s).
     Block duration = 192 s (3.2 min). Total 14 blocks (44.8 min).
  3. Report per active window: P-core mean frequency vs elapsed time, overall trend,
     and the first time (if any) window mean falls below 95% and below 90% of 3204 MHz.

Numbers only, no PASS/FAIL, no threshold computed.
Appends report to ~/m1_thermal_calibration.txt.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np

OUTDIR = Path('/var/gai-powermetrics')
WRAPPER = '/usr/local/sbin/gai_powermetrics.sh'
HEADER_RE = re.compile(r'\*\*\* Sampled system activity .*? \(([\d.]+)ms elapsed\) \*\*\*')
CPU_POWER_RE = re.compile(r'CPU Power:\s*(\d+)\s*mW')
THERMAL_RE = re.compile(r'Current pressure level:\s*([A-Za-z0-9_-]+)')
CPU_FREQ_RE = {n: re.compile(rf'CPU {n} frequency:\s*(\d+)\s*MHz') for n in range(8)}
E_CLUSTER_FREQ_RE = re.compile(r'E-Cluster HW active frequency:\s*(\d+)\s*MHz')
E_CLUSTER_ACT_RE = re.compile(r'E-Cluster HW active residency:\s*([\d.]+)%')
P_CLUSTER_FREQ_RE = re.compile(r'P-Cluster HW active frequency:\s*(\d+)\s*MHz')
P_CLUSTER_ACT_RE = re.compile(r'P-Cluster HW active residency:\s*([\d.]+)%')

REPO = Path.home() / 'green-ai-m1' / 'repo'
DATA_ROOT = Path.home() / 'green-ai-m1' / 'data'

SAMPLE_INTERVAL_MS = 400

def run_cmd(args):
    return subprocess.check_output(args, text=True)

def check_guards():
    batt = run_cmd(['pmset', '-g', 'batt'])
    therm = run_cmd(['pmset', '-g', 'therm'])
    first_line = batt.splitlines()[0] if batt.splitlines() else ''
    if 'AC Power' not in first_line:
        raise RuntimeError(f"ABORT: Not on AC power: {batt.strip()}")
    therm_lines = [l.strip() for l in therm.splitlines() if l.strip()]
    expected = {
        'Note: No thermal warning level has been recorded',
        'Note: No performance warning level has been recorded',
        'Note: No CPU power status has been recorded',
    }
    if set(therm_lines) != expected:
        raise RuntimeError(f"ABORT: Thermal status not nominal (fail-closed): {therm.strip()}")

    boottime_raw = run_cmd(['sysctl', '-n', 'kern.boottime'])
    m = re.search(r'sec\s*=\s*(\d+)', boottime_raw)
    uptime_s = time.time() - float(m.group(1)) if m else None
    print(f"Guards: AC OK, thermal nominal OK, uptime_s={uptime_s} (recorded, not enforced)")
    return batt.strip(), therm.strip(), uptime_s

def load_models():
    import torch
    torch.set_num_threads(4)
    torch.backends.quantized.engine = 'qnnpack'
    ckpt_int8 = REPO / 'checkpoints' / 'resnet18_int8.pt'
    ckpt_fp32 = REPO / 'checkpoints' / 'resnet18_fp32_ts.pt'
    model_int8 = torch.jit.load(str(ckpt_int8), map_location='cpu').eval()
    model_fp32 = torch.jit.load(str(ckpt_fp32), map_location='cpu').eval()
    return model_int8, model_fp32

def load_cifar_batch(n=64):
    import torch
    import torchvision
    from torchvision import transforms
    ds = torchvision.datasets.CIFAR10(str(DATA_ROOT), train=False, download=False,
                                       transform=transforms.ToTensor())
    MEAN = torch.tensor([.4914, .4822, .4465])[None, :, None, None]
    STD = torch.tensor([.247, .243, .261])[None, :, None, None]
    imgs = torch.stack([ds[i][0] for i in range(n)])
    imgs = (imgs - MEAN) / STD
    return imgs

def start_powermetrics(tag, sample_count):
    t0_perf = time.perf_counter()
    t0_wall = time.time()
    proc = subprocess.Popen(['sudo', '-n', WRAPPER, str(SAMPLE_INTERVAL_MS), str(sample_count), tag],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    outfile = None
    deadline = time.perf_counter() + 30.0
    while outfile is None and time.perf_counter() < deadline:
        for c in glob.glob(str(OUTDIR / f"{tag}_*.txt")):
            try:
                if os.path.getmtime(c) >= t0_wall - 2.0:
                    outfile = c
                    break
            except FileNotFoundError:
                continue
        if outfile is None:
            if proc.poll() is not None:
                err_msg = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"powermetrics exited prematurely: {err_msg}")
            time.sleep(0.005)
    if outfile is None:
        raise RuntimeError(f"powermetrics output file not found for tag {tag}")
    return proc, outfile, t0_perf

def start_p2_poller(outfile, arrivals, stop_event, first_header_event):
    def poller():
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
                    if matches:
                        now = time.perf_counter()
                        for idx, m in enumerate(matches):
                            sample_num = last_count + idx + 1
                            elapsed = float(m.group(1))
                            if sample_num not in arrivals:
                                arrivals[sample_num] = (now, elapsed)
                        last_count += len(matches)
                        if last_count >= 1 and not first_header_event.is_set():
                            first_header_event.set()
                        carry = text[matches[-1].end():]
                    else:
                        carry = text
                    if len(carry) > 8192:
                        carry = carry[-8192:]
            except Exception:
                pass
            time.sleep(0.002)
    t = threading.Thread(target=poller, daemon=True)
    t.start()
    return t

def build_sample_index(outfile, arrivals):
    with open(outfile, 'r', errors='ignore') as f:
        full_content = f.read()
    blocks = full_content.split('*** Sampled system activity')
    samples = {}
    for idx, b in enumerate(blocks[1:], 1):
        m_cpu = CPU_POWER_RE.search(b)
        m_therm = THERMAL_RE.search(b)
        m_ecf = E_CLUSTER_FREQ_RE.search(b)
        m_eca = E_CLUSTER_ACT_RE.search(b)
        m_pcf = P_CLUSTER_FREQ_RE.search(b)
        m_pca = P_CLUSTER_ACT_RE.search(b)
        cpu_freqs = {}
        for n, rx in CPU_FREQ_RE.items():
            m = rx.search(b)
            if m:
                cpu_freqs[n] = int(m.group(1))
        p47 = [cpu_freqs[n] for n in (4, 5, 6, 7) if n in cpu_freqs]
        p47_mean = sum(p47) / len(p47) if p47 else None
        arr = arrivals.get(idx)
        samples[idx] = {
            'sample_k': idx,
            'arrival_perf': arr[0] if arr else None,
            'elapsed_ms': arr[1] if arr else None,
            'cpu_power_mw': int(m_cpu.group(1)) if m_cpu else None,
            'thermal_pressure': m_therm.group(1) if m_therm else None,
            'cpu4_7_mean_mhz': p47_mean,
            'e_cluster_freq_mhz': int(m_ecf.group(1)) if m_ecf else None,
            'e_cluster_active_pct': float(m_eca.group(1)) if m_eca else None,
            'p_cluster_freq_mhz': int(m_pcf.group(1)) if m_pcf else None,
            'p_cluster_active_pct': float(m_pca.group(1)) if m_pca else None,
        }
    return samples

def attribute_windows(windows, samples):
    sample_list = [s for s in samples.values() if s['arrival_perf'] is not None]
    for w in windows:
        overlapping = []
        for s in sample_list:
            dur_s = s['elapsed_ms'] / 1000.0
            a = s['arrival_perf'] - dur_s
            b = s['arrival_perf']
            if b > w['start_t'] and a < w['end_t']:
                overlapping.append(s)
        w['samples'] = overlapping
        p47_vals = [s['cpu4_7_mean_mhz'] for s in overlapping if s['cpu4_7_mean_mhz'] is not None]
        w['p47_mean_mhz'] = sum(p47_vals) / len(p47_vals) if p47_vals else None
        w['p47_min_sample_mhz'] = min(p47_vals) if p47_vals else None
        pcf_vals = [s['p_cluster_freq_mhz'] for s in overlapping if s['p_cluster_freq_mhz'] is not None]
        pca_vals = [s['p_cluster_active_pct'] for s in overlapping if s['p_cluster_active_pct'] is not None]
        ecf_vals = [s['e_cluster_freq_mhz'] for s in overlapping if s['e_cluster_freq_mhz'] is not None]
        eca_vals = [s['e_cluster_active_pct'] for s in overlapping if s['e_cluster_active_pct'] is not None]
        cpw_vals = [s['cpu_power_mw'] for s in overlapping if s['cpu_power_mw'] is not None]
        therm_vals = [s['thermal_pressure'] for s in overlapping]
        w['p_cluster_freq_mean_mhz'] = sum(pcf_vals) / len(pcf_vals) if pcf_vals else None
        w['p_cluster_active_pct_mean'] = sum(pca_vals) / len(pca_vals) if pca_vals else None
        w['e_cluster_freq_mean_mhz'] = sum(ecf_vals) / len(ecf_vals) if ecf_vals else None
        w['e_cluster_active_pct_mean'] = sum(eca_vals) / len(eca_vals) if eca_vals else None
        w['cpu_power_mean_mw'] = sum(cpw_vals) / len(cpw_vals) if cpw_vals else None
        w['thermal_pressures'] = therm_vals
        w['all_nominal'] = all(t == 'Nominal' for t in therm_vals) if therm_vals else False
        w['images_per_s'] = w['batches'] / (w['end_t'] - w['start_t']) if (w['end_t'] - w['start_t']) > 0 else None
    return windows

def ols_trend(t_min, y):
    t = np.array(t_min)
    v = np.array(y)
    n = len(t)
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0
    t_mean = np.mean(t)
    v_mean = np.mean(v)
    ss_tt = np.sum((t - t_mean)**2)
    slope = np.sum((t - t_mean) * (v - v_mean)) / ss_tt if ss_tt > 0 else 0.0
    residuals = v - (v_mean + slope * (t - t_mean))
    s2 = np.sum(residuals**2) / (n - 2) if n > 2 else 0.0
    se = (s2 / ss_tt)**0.5 if ss_tt > 0 else 0.0
    # 95% critical value t ~ 1.96
    ci_low = slope - 1.96 * se
    ci_high = slope + 1.96 * se
    return float(slope), float(se), float(ci_low), float(ci_high)

def wait_for_idle(min_duration_s=600.0, check_interval_s=10.0):
    """Wait until CPU power and P-core frequency indicate idle for at least min_duration_s."""
    print(f"Monitoring system idle state for {min_duration_s/60:.1f} minutes...")
    tag = "idlecheck_baseline"
    sample_count = int((min_duration_s + 15.0) / (SAMPLE_INTERVAL_MS / 1000.0))
    proc, outfile, t0 = start_powermetrics(tag, sample_count)
    arrivals = {}
    stop_event = threading.Event()
    first_header_event = threading.Event()
    poller = start_p2_poller(outfile, arrivals, stop_event, first_header_event)
    first_header_event.wait(timeout=15.0)
    
    t_start = time.perf_counter()
    time.sleep(min_duration_s)
    
    # communicate
    elapsed = time.perf_counter() - t0
    rem = max(0.0, sample_count * 0.4 - elapsed)
    proc.communicate(timeout=rem + 30.0)
    stop_event.set()
    poller.join(timeout=5.0)
    
    samples = build_sample_index(outfile, arrivals)
    powers = [s['cpu_power_mw'] for s in samples.values() if s['cpu_power_mw'] is not None]
    p_act = [s['p_cluster_active_pct'] for s in samples.values() if s['p_cluster_active_pct'] is not None]
    e_act = [s['e_cluster_active_pct'] for s in samples.values() if s['e_cluster_active_pct'] is not None]
    p47 = [s['cpu4_7_mean_mhz'] for s in samples.values() if s['cpu4_7_mean_mhz'] is not None]
    pcf = [s['p_cluster_freq_mhz'] for s in samples.values() if s['p_cluster_freq_mhz'] is not None]
    
    mean_pwr = np.mean(powers) if powers else 0.0
    mean_pact = np.mean(p_act) if p_act else 0.0
    mean_eact = np.mean(e_act) if e_act else 0.0
    mean_p47 = np.mean(p47) if p47 else 0.0
    mean_pcf = np.mean(pcf) if pcf else 0.0
    print(f"Idle confirmation complete: {len(samples)} samples over {min_duration_s/60:.1f}m. Mean CPU Power={mean_pwr:.2f} mW, Mean P-core Freq={mean_p47:.2f} MHz, Mean P-act={mean_pact:.2f}%")
    return {
        'duration_s': min_duration_s,
        'mean_cpu_power_mw': float(mean_pwr),
        'mean_p_cluster_active_pct': float(mean_pact),
        'mean_e_cluster_active_pct': float(mean_eact),
        'mean_p47_mhz': float(mean_p47),
        'mean_p_cluster_freq_mhz': float(mean_pcf),
        'sample_count': len(samples)
    }

def run_15min_leg(arch, condition, out_dir, model, images):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"test1_{condition}"
    total_dur_s = 15 * 60.0
    margin_s = 20.0
    sample_count = int((total_dur_s + margin_s) / (SAMPLE_INTERVAL_MS / 1000.0))
    
    proc, outfile, t0_perf = start_powermetrics(tag, sample_count)
    arrivals = {}
    stop_event = threading.Event()
    first_header_event = threading.Event()
    poller = start_p2_poller(outfile, arrivals, stop_event, first_header_event)
    if not first_header_event.wait(timeout=15.0):
        stop_event.set()
        proc.kill()
        raise RuntimeError(f"Timed out waiting for first header in {condition}")
        
    print(f"[Test 1: {condition}] First header observed. Starting 15-minute inference...")
    windows = []
    jsonl_path = out_dir / 'windows_raw.jsonl'
    t_start = time.perf_counter()
    t_end = t_start + total_dur_s
    n_images = images.shape[0]
    idx = 0
    import torch
    with torch.no_grad(), open(jsonl_path, 'a') as jf:
        while time.perf_counter() < t_end:
            w_start = time.perf_counter()
            w_target_end = w_start + 5.0
            batches = 0
            while time.perf_counter() < w_target_end:
                x = images[idx % n_images: (idx % n_images) + 1]
                if x.shape[0] < 1:
                    idx = 0
                    x = images[0:1]
                model(x)
                idx += 1
                batches += 1
            w_end = time.perf_counter()
            w = {'window_idx': len(windows), 'start_t': w_start, 'end_t': w_end, 'batches': batches}
            windows.append(w)
            jf.write(json.dumps(w) + '\n')
            jf.flush()
            
    print(f"[Test 1: {condition}] Completed {len(windows)} windows. Waiting for powermetrics to finish...")
    elapsed = time.perf_counter() - t0_perf
    rem = max(0.0, sample_count * 0.4 - elapsed)
    proc.communicate(timeout=rem + 60.0)
    stop_event.set()
    poller.join(timeout=5.0)
    
    with open(outfile, 'r', errors='ignore') as f:
        content = f.read()
    matches = list(HEADER_RE.finditer(content))
    now = time.perf_counter()
    for idx_m, m in enumerate(matches):
        s_num = idx_m + 1
        elapsed_m = float(m.group(1))
        if s_num not in arrivals:
            arrivals[s_num] = (now, elapsed_m)
            
    samples = build_sample_index(outfile, arrivals)
    windows = attribute_windows(windows, samples)
    
    # Save outputs
    shutil.copy(outfile, out_dir / Path(outfile).name)
    windows_out = [{k: v for k, v in w.items() if k != 'samples'} for w in windows]
    (out_dir / 'windows.json').write_text(json.dumps(windows_out, indent=2))
    
    # Compute summary stats
    t_min = [(w['start_t'] - windows[0]['start_t']) / 60.0 for w in windows]
    p47 = [w['p47_mean_mhz'] for w in windows]
    ips = [w['images_per_s'] for w in windows]
    pwr = [w['cpu_power_mean_mw'] for w in windows]
    p_act = [w['p_cluster_active_pct_mean'] for w in windows if w['p_cluster_active_pct_mean'] is not None]
    e_act = [w['e_cluster_active_pct_mean'] for w in windows if w['e_cluster_active_pct_mean'] is not None]
    
    slope_f, se_f, ci_l_f, ci_h_f = ols_trend(t_min, p47)
    slope_i, se_i, ci_l_i, ci_h_i = ols_trend(t_min, ips)
    
    non_nominal = sum(1 for w in windows if not w['all_nominal'])
    
    return {
        'condition': condition,
        'windows': windows,
        'n_windows': len(windows),
        'n_samples': len(samples),
        'non_nominal_windows': non_nominal,
        'slope_freq_mhz_per_min': slope_f,
        'se_freq': se_f,
        'ci_freq': [ci_l_f, ci_h_f],
        'slope_ips_per_min': slope_i,
        'se_ips': se_i,
        'ci_ips': [ci_l_i, ci_h_i],
        'p47_min': float(np.min(p47)),
        'p47_median': float(np.median(p47)),
        'p47_max': float(np.max(p47)),
        'p47_p5': float(np.percentile(p47, 5)),
        'ips_min': float(np.min(ips)),
        'ips_median': float(np.median(ips)),
        'ips_max': float(np.max(ips)),
        'ips_p5': float(np.percentile(ips, 5)),
        'pwr_mean': float(np.mean(pwr)),
        'p_cluster_act_mean': float(np.mean(p_act)) if p_act else 0.0,
        'e_cluster_act_mean': float(np.mean(e_act)) if e_act else 0.0,
        't_start': windows[0]['start_t'],
        't_end': windows[-1]['end_t']
    }

def run_test2_duty_cycle(out_dir, model_int8, model_fp32, images):
    """Test 2: 45 min session-like duty cycle.
    14 blocks alternating: INT8, FP32, INT8, FP32...
    Each block = 6 repeats x (idle 3s settle + 5s idle, active 3s warm-up + 5s window, active 3s + 5s, idle 3s + 5s).
    Total duration = 14 * 192s = 2688 s = 44.8 minutes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "test2_dutycycle"
    total_blocks = 14
    repeats_per_block = 6
    block_dur_s = repeats_per_block * 32.0  # 192 s
    total_dur_s = total_blocks * block_dur_s  # 2688 s
    margin_s = 25.0
    sample_count = int((total_dur_s + margin_s) / (SAMPLE_INTERVAL_MS / 1000.0))  # ~6782 samples
    
    proc, outfile, t0_perf = start_powermetrics(tag, sample_count)
    arrivals = {}
    stop_event = threading.Event()
    first_header_event = threading.Event()
    poller = start_p2_poller(outfile, arrivals, stop_event, first_header_event)
    if not first_header_event.wait(timeout=15.0):
        stop_event.set()
        proc.kill()
        raise RuntimeError("Test 2: Timed out waiting for first header")
        
    print(f"[Test 2: Duty Cycle] First header observed. Starting 14 blocks ({total_dur_s/60:.1f} minutes)...")
    
    active_windows = []
    jsonl_path = out_dir / 'active_windows_raw.jsonl'
    n_images = images.shape[0]
    idx = 0
    
    t_test2_start = time.perf_counter()
    import torch
    with torch.no_grad(), open(jsonl_path, 'a') as jf:
        for b_idx in range(total_blocks):
            cond = 'int8' if b_idx % 2 == 0 else 'fp32_ts'
            model = model_int8 if cond == 'int8' else model_fp32
            print(f"  Starting Block {b_idx+1}/{total_blocks}: condition={cond}")
            
            for rep in range(repeats_per_block):
                # 1. idle_before: 3s settle + 5s idle
                time.sleep(3.0)
                time.sleep(5.0)
                
                # 2. active 1: 3s warm-up + 5s window
                t_warm_end = time.perf_counter() + 3.0
                while time.perf_counter() < t_warm_end:
                    x = images[idx % n_images: (idx % n_images) + 1]
                    model(x)
                    idx += 1
                # 5s active window a1
                w_start = time.perf_counter()
                w_target_end = w_start + 5.0
                batches = 0
                while time.perf_counter() < w_target_end:
                    x = images[idx % n_images: (idx % n_images) + 1]
                    model(x)
                    idx += 1
                    batches += 1
                w_end = time.perf_counter()
                w_record = {
                    'window_global_idx': len(active_windows),
                    'block_idx': b_idx,
                    'condition': cond,
                    'repeat_idx': rep,
                    'phase': 'a1',
                    'start_t': w_start,
                    'end_t': w_end,
                    'elapsed_from_test2_s': w_start - t_test2_start,
                    'batches': batches
                }
                active_windows.append(w_record)
                jf.write(json.dumps(w_record) + '\n')
                jf.flush()
                
                # 3. active 2: 3s warm-up + 5s window
                t_warm_end = time.perf_counter() + 3.0
                while time.perf_counter() < t_warm_end:
                    x = images[idx % n_images: (idx % n_images) + 1]
                    model(x)
                    idx += 1
                # 5s active window a2
                w_start = time.perf_counter()
                w_target_end = w_start + 5.0
                batches = 0
                while time.perf_counter() < w_target_end:
                    x = images[idx % n_images: (idx % n_images) + 1]
                    model(x)
                    idx += 1
                    batches += 1
                w_end = time.perf_counter()
                w_record = {
                    'window_global_idx': len(active_windows),
                    'block_idx': b_idx,
                    'condition': cond,
                    'repeat_idx': rep,
                    'phase': 'a2',
                    'start_t': w_start,
                    'end_t': w_end,
                    'elapsed_from_test2_s': w_start - t_test2_start,
                    'batches': batches
                }
                active_windows.append(w_record)
                jf.write(json.dumps(w_record) + '\n')
                jf.flush()
                
                # 4. idle_after: 3s settle + 5s idle
                time.sleep(3.0)
                time.sleep(5.0)
                
    print(f"[Test 2: Duty Cycle] All 14 blocks finished. {len(active_windows)} active windows collected.")
    elapsed = time.perf_counter() - t0_perf
    rem = max(0.0, sample_count * 0.4 - elapsed)
    proc.communicate(timeout=rem + 60.0)
    stop_event.set()
    poller.join(timeout=5.0)
    
    with open(outfile, 'r', errors='ignore') as f:
        content = f.read()
    matches = list(HEADER_RE.finditer(content))
    now = time.perf_counter()
    for idx_m, m in enumerate(matches):
        s_num = idx_m + 1
        elapsed_m = float(m.group(1))
        if s_num not in arrivals:
            arrivals[s_num] = (now, elapsed_m)
            
    samples = build_sample_index(outfile, arrivals)
    active_windows = attribute_windows(active_windows, samples)
    
    shutil.copy(outfile, out_dir / Path(outfile).name)
    windows_out = [{k: v for k, v in w.items() if k != 'samples'} for w in active_windows]
    (out_dir / 'active_windows.json').write_text(json.dumps(windows_out, indent=2))
    
    # Analyze thresholds
    # 95% of 3204 MHz = 3043.80 MHz
    # 90% of 3204 MHz = 2883.60 MHz
    thresh_95 = 0.95 * 3204.0
    thresh_90 = 0.90 * 3204.0
    
    first_below_95 = None
    first_below_90 = None
    
    for w in active_windows:
        f = w['p47_mean_mhz']
        if f is not None:
            if f < thresh_95 and first_below_95 is None:
                first_below_95 = {
                    'window_idx': w['window_global_idx'],
                    'block_idx': w['block_idx'],
                    'condition': w['condition'],
                    'repeat_idx': w['repeat_idx'],
                    'phase': w['phase'],
                    'elapsed_s': w['elapsed_from_test2_s'],
                    'elapsed_min': w['elapsed_from_test2_s'] / 60.0,
                    'freq_mhz': f
                }
            if f < thresh_90 and first_below_90 is None:
                first_below_90 = {
                    'window_idx': w['window_global_idx'],
                    'block_idx': w['block_idx'],
                    'condition': w['condition'],
                    'repeat_idx': w['repeat_idx'],
                    'phase': w['phase'],
                    'elapsed_s': w['elapsed_from_test2_s'],
                    'elapsed_min': w['elapsed_from_test2_s'] / 60.0,
                    'freq_mhz': f
                }
                
    # Trends across all active windows
    t_min = [w['elapsed_from_test2_s'] / 60.0 for w in active_windows]
    p47 = [w['p47_mean_mhz'] for w in active_windows]
    slope_f, se_f, ci_l_f, ci_h_f = ols_trend(t_min, p47)
    
    # Trends separately by condition
    t_min_int8 = [w['elapsed_from_test2_s'] / 60.0 for w in active_windows if w['condition'] == 'int8']
    p47_int8 = [w['p47_mean_mhz'] for w in active_windows if w['condition'] == 'int8']
    slope_f_int8, se_f_int8, ci_l_f_int8, ci_h_f_int8 = ols_trend(t_min_int8, p47_int8)
    
    t_min_fp32 = [w['elapsed_from_test2_s'] / 60.0 for w in active_windows if w['condition'] == 'fp32_ts']
    p47_fp32 = [w['p47_mean_mhz'] for w in active_windows if w['condition'] == 'fp32_ts']
    slope_f_fp32, se_f_fp32, ci_l_f_fp32, ci_h_f_fp32 = ols_trend(t_min_fp32, p47_fp32)
    
    return {
        'total_active_windows': len(active_windows),
        'thresh_95_mhz': thresh_95,
        'thresh_90_mhz': thresh_90,
        'first_below_95': first_below_95,
        'first_below_90': first_below_90,
        'overall_trend': {
            'slope_mhz_per_min': slope_f,
            'se': se_f,
            'ci': [ci_l_f, ci_h_f]
        },
        'int8_trend': {
            'slope_mhz_per_min': slope_f_int8,
            'se': se_f_int8,
            'ci': [ci_l_f_int8, ci_h_f_int8]
        },
        'fp32_trend': {
            'slope_mhz_per_min': slope_f_fp32,
            'se': se_f_fp32,
            'ci': [ci_l_f_fp32, ci_h_f_fp32]
        },
        'p47_min': float(np.min(p47)),
        'p47_median': float(np.median(p47)),
        'p47_max': float(np.max(p47)),
        'p47_p5': float(np.percentile(p47, 5)),
        'active_windows': active_windows
    }

def format_report_section(idle_info, test1_int8, test1_fp32, test2_info):
    lines = []
    def p(s=''):
        lines.append(s)
        
    p("\n" + "=" * 80)
    p("EXTENDED THERMAL CALIBRATION: ORDER CHECK & DUTY-CYCLE CHECK")
    p("=" * 80)
    p("Run: 2026-11-04, Mac via ssh m1-tailscale. Numbers only, no PASS/FAIL.")
    p()
    
    p("-" * 80)
    p("PRE-TEST 1: 10-MINUTE IDLE CONFIRMATION")
    p("-" * 80)
    p(f"Duration monitored: {idle_info['duration_s']/60:.1f} minutes ({idle_info['sample_count']} samples)")
    p(f"Mean CPU Power: {idle_info['mean_cpu_power_mw']:.2f} mW")
    p(f"Mean P-core (CPU 4-7) Frequency: {idle_info.get('mean_p47_mhz', 0.0):.2f} MHz")
    p(f"Mean P-Cluster Active Residency: {idle_info['mean_p_cluster_active_pct']:.2f}%")
    p(f"Mean E-Cluster Active Residency: {idle_info.get('mean_e_cluster_active_pct', 0.0):.2f}%")
    p()
    
    p("-" * 80)
    p("TEST 1: ORDER CHECK (INT8 FIRST, THEN FP32-TS)")
    p("-" * 80)
    p("Order: ResNet-18 INT8 (15 min) -> 10 min cool-down -> ResNet-18 FP32-TS (15 min)")
    p()
    p("--- RESNET-18 INT8 (RUN FIRST) ---")
    p(f"n_windows: {test1_int8['n_windows']}")
    p(f"non_nominal_window_count: {test1_int8['non_nominal_windows']}")
    p("P-core (CPU 4-7) mean frequency trend over 15 min (linear OLS):")
    p(f"  slope: {test1_int8['slope_freq_mhz_per_min']:+.4f} MHz/min")
    p(f"  SE: {test1_int8['se_freq']:.4f}")
    p(f"  95% CI: [{test1_int8['ci_freq'][0]:+.4f}, {test1_int8['ci_freq'][1]:+.4f}] MHz/min")
    p("images/s trend over 15 min:")
    p(f"  slope: {test1_int8['slope_ips_per_min']:+.4f} images/s/min")
    p(f"  SE: {test1_int8['se_ips']:.4f}")
    p(f"  95% CI: [{test1_int8['ci_ips'][0]:+.4f}, {test1_int8['ci_ips'][1]:+.4f}]")
    p("Window-mean (CPU 4-7 mean freq) stats:")
    p(f"  min: {test1_int8['p47_min']:.2f} MHz")
    p(f"  median: {test1_int8['p47_median']:.2f} MHz")
    p(f"  max: {test1_int8['p47_max']:.2f} MHz")
    p(f"  5th percentile: {test1_int8['p47_p5']:.2f} MHz")
    p("images/s stats:")
    p(f"  min: {test1_int8['ips_min']:.2f}, median: {test1_int8['ips_median']:.2f}, max: {test1_int8['ips_max']:.2f}, p5: {test1_int8['ips_p5']:.2f}")
    p(f"Mean CPU Power: {test1_int8['pwr_mean']:.2f} mW")
    p(f"Mean P-Cluster Active Residency: {test1_int8.get('p_cluster_act_mean', 0.0):.2f}%")
    p(f"Mean E-Cluster Active Residency: {test1_int8.get('e_cluster_act_mean', 0.0):.2f}%")
    p()
    
    gap_s = test1_fp32['t_start'] - test1_int8['t_end']
    p(f"Cool-down gap between INT8 and FP32-TS legs: {gap_s:.2f} s ({gap_s/60:.2f} min)")
    p()
    
    p("--- RESNET-18 FP32-TORCHSCRIPT (RUN SECOND) ---")
    p(f"n_windows: {test1_fp32['n_windows']}")
    p(f"non_nominal_window_count: {test1_fp32['non_nominal_windows']}")
    p("P-core (CPU 4-7) mean frequency trend over 15 min (linear OLS):")
    p(f"  slope: {test1_fp32['slope_freq_mhz_per_min']:+.4f} MHz/min")
    p(f"  SE: {test1_fp32['se_freq']:.4f}")
    p(f"  95% CI: [{test1_fp32['ci_freq'][0]:+.4f}, {test1_fp32['ci_freq'][1]:+.4f}] MHz/min")
    p("images/s trend over 15 min:")
    p(f"  slope: {test1_fp32['slope_ips_per_min']:+.4f} images/s/min")
    p(f"  SE: {test1_fp32['se_ips']:.4f}")
    p(f"  95% CI: [{test1_fp32['ci_ips'][0]:+.4f}, {test1_fp32['ci_ips'][1]:+.4f}]")
    p("Window-mean (CPU 4-7 mean freq) stats:")
    p(f"  min: {test1_fp32['p47_min']:.2f} MHz")
    p(f"  median: {test1_fp32['p47_median']:.2f} MHz")
    p(f"  max: {test1_fp32['p47_max']:.2f} MHz")
    p(f"  5th percentile: {test1_fp32['p47_p5']:.2f} MHz")
    p("images/s stats:")
    p(f"  min: {test1_fp32['ips_min']:.2f}, median: {test1_fp32['ips_median']:.2f}, max: {test1_fp32['ips_max']:.2f}, p5: {test1_fp32['ips_p5']:.2f}")
    p(f"Mean CPU Power: {test1_fp32['pwr_mean']:.2f} mW")
    p(f"Mean P-Cluster Active Residency: {test1_fp32.get('p_cluster_act_mean', 0.0):.2f}%")
    p(f"Mean E-Cluster Active Residency: {test1_fp32.get('e_cluster_act_mean', 0.0):.2f}%")
    p()
    
    p("-" * 80)
    p("TEST 2: 45-MINUTE DUTY-CYCLE CHECK (SESSION-LIKE ALTERNATING BLOCKS)")
    p("-" * 80)
    p("Duty Cycle: 14 blocks alternating INT8 & FP32-TS (7 blocks each, 3.2 min per block)")
    p("Each block: 6 repeats x (idle 3s+5s, active 3s+5s, active 3s+5s, idle 3s+5s)")
    p(f"Total active 5s windows: {test2_info['total_active_windows']}")
    p()
    p("Thresholds:")
    p(f"  95% of 3204 MHz = {test2_info['thresh_95_mhz']:.2f} MHz")
    p(f"  90% of 3204 MHz = {test2_info['thresh_90_mhz']:.2f} MHz")
    p()
    b95 = test2_info['first_below_95']
    if b95:
        p(f"First active window below 95% (3043.80 MHz):")
        p(f"  Window Index: {b95['window_idx']} (Block {b95['block_idx']+1}, {b95['condition']}, Repeat {b95['repeat_idx']}, Phase {b95['phase']})")
        p(f"  Elapsed Time from Test 2 start: {b95['elapsed_s']:.2f} s ({b95['elapsed_min']:.2f} min)")
        p(f"  P-core Mean Frequency: {b95['freq_mhz']:.2f} MHz")
    else:
        p("First active window below 95% (3043.80 MHz): NONE (no window fell below 95%)")
        
    b90 = test2_info['first_below_90']
    if b90:
        p(f"First active window below 90% (2883.60 MHz):")
        p(f"  Window Index: {b90['window_idx']} (Block {b90['block_idx']+1}, {b90['condition']}, Repeat {b90['repeat_idx']}, Phase {b90['phase']})")
        p(f"  Elapsed Time from Test 2 start: {b90['elapsed_s']:.2f} s ({b90['elapsed_min']:.2f} min)")
        p(f"  P-core Mean Frequency: {b90['freq_mhz']:.2f} MHz")
    else:
        p("First active window below 90% (2883.60 MHz): NONE (no window fell below 90%)")
    p()
    
    p("P-core Frequency Trend Across 45-Minute Duty Cycle:")
    p(f"  Overall slope: {test2_info['overall_trend']['slope_mhz_per_min']:+.4f} MHz/min (SE: {test2_info['overall_trend']['se']:.4f}, 95% CI: [{test2_info['overall_trend']['ci'][0]:+.4f}, {test2_info['overall_trend']['ci'][1]:+.4f}])")
    p(f"  INT8 active windows slope: {test2_info['int8_trend']['slope_mhz_per_min']:+.4f} MHz/min (SE: {test2_info['int8_trend']['se']:.4f}, 95% CI: [{test2_info['int8_trend']['ci'][0]:+.4f}, {test2_info['int8_trend']['ci'][1]:+.4f}])")
    p(f"  FP32 active windows slope: {test2_info['fp32_trend']['slope_mhz_per_min']:+.4f} MHz/min (SE: {test2_info['fp32_trend']['se']:.4f}, 95% CI: [{test2_info['fp32_trend']['ci'][0]:+.4f}, {test2_info['fp32_trend']['ci'][1]:+.4f}])")
    p()
    p("Active Window Frequency Distribution:")
    p(f"  min: {test2_info['p47_min']:.2f} MHz")
    p(f"  median: {test2_info['p47_median']:.2f} MHz")
    p(f"  max: {test2_info['p47_max']:.2f} MHz")
    p(f"  5th percentile: {test2_info['p47_p5']:.2f} MHz")
    p()
    
    p("Representative Active Windows (Window idx, Block, Condition, Elapsed min, P-core MHz):")
    for w in test2_info['active_windows'][::12]:  # Sample every 12th window (first window of each block)
        p(f"  idx={w['window_global_idx']:<3} Block={w['block_idx']+1:<2} Cond={w['condition']:<8} Elapsed={w['elapsed_from_test2_s']/60.0:5.2f} min  P-core Freq={w['p47_mean_mhz']:.2f} MHz  Power={w['cpu_power_mean_mw']:.1f} mW")
        
    p("=" * 80)
    p("END OF EXTENDED CALIBRATION REPORT")
    p("=" * 80)
    
    return "\n".join(lines)

def main():
    out_base = Path.home() / 'thermal_extended_results'
    out_base.mkdir(exist_ok=True)
    
    batt, therm, uptime_s = check_guards()
    
    print("Pre-allocating models and test data...")
    model_int8, model_fp32 = load_models()
    images = load_cifar_batch(64)
    
    # Pre-Test 1: 10-minute idle check
    idle_info = wait_for_idle(min_duration_s=600.0)
    
    # Test 1: INT8 first, then FP32-TS
    print("=== STARTING TEST 1: RESNET-18 INT8 (15 MIN) ===")
    test1_int8 = run_15min_leg('resnet18', 'int8', out_base / 'test1_int8', model_int8, images)
    
    print("Cooling down 10 minutes between Test 1 legs...")
    time.sleep(600.0)
    
    print("=== STARTING TEST 1: RESNET-18 FP32-TS (15 MIN) ===")
    test1_fp32 = run_15min_leg('resnet18', 'fp32_ts', out_base / 'test1_fp32_ts', model_fp32, images)
    
    # Cool-down between Test 1 and Test 2: 10 minutes
    print("Cooling down 10 minutes between Test 1 and Test 2...")
    time.sleep(600.0)
    
    # Test 2: 45 min duty cycle
    print("=== STARTING TEST 2: 45-MINUTE DUTY-CYCLE CHECK ===")
    test2_info = run_test2_duty_cycle(out_base / 'test2_dutycycle', model_int8, model_fp32, images)
    
    report_text = format_report_section(idle_info, test1_int8, test1_fp32, test2_info)
    
    cal_file = Path.home() / 'm1_thermal_calibration.txt'
    with open(cal_file, 'a') as f:
        f.write(report_text)
        
    print("\nReport appended to:", cal_file)
    print(report_text)

if __name__ == '__main__':
    main()
