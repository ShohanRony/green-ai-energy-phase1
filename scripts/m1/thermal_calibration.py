#!/usr/bin/env python3
"""M1 thermal calibration (non-energy), mirroring the registered x86 A11 procedure
(docs/stage5_analysis_plan.md A11, 2026-10-12).

Pre-registered rule (not to be changed by this script):
  - per-sample metric: mean of CPU 4-7 frequency (MHz) -- the 4 P-cores on this
    M1 (confirmed from the real powermetrics sample block: "CPU 4 frequency"
    through "CPU 7 frequency" are under the P-Cluster section).
  - per-window metric: mean over samples overlapping the 5s window.
  - threshold T = 0.90 x pooled 5th percentile of window means over calibration
    windows where every sample has thermal pressure "Nominal" -- NOT computed
    or proposed here. This script reports the raw distribution only.

Two runs: ResNet-18 FP32-TorchScript, then ResNet-18 INT8 (qnnpack). 15 minutes
of continuous batch-1 32x32 inference each, no subprocess during inference.
"""
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

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

WINDOW_S = 5.0
TOTAL_DURATION_S = 15 * 60.0  # 15 minutes
SAMPLE_INTERVAL_MS = 400
# Margin: first-header wait (~1-2s) + model-load time already elapsed before
# powermetrics start + a buffer so powermetrics does not exit before the 15-
# minute inference finishes. Chosen here as 3 minutes -- stated explicitly,
# not silently assumed; this is a one-off calibration run, not the main
# per-condition loop (whose own COUNT margin remains an open item per
# MEASUREMENT_LOOP_DESIGN.md section 4).
MARGIN_S = 3 * 60.0
SAMPLE_COUNT = int((TOTAL_DURATION_S + MARGIN_S) / (SAMPLE_INTERVAL_MS / 1000.0))


def run_cmd(args):
    return subprocess.check_output(args, text=True)


def check_guards():
    """AC and thermal enforced (abort on failure); uptime recorded only, not
    enforced, per this task's explicit instruction."""
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
    uptime_s = None
    if m:
        uptime_s = time.time() - float(m.group(1))
    print(f"Guards: AC OK, thermal nominal OK, uptime_s={uptime_s} (recorded, not enforced)")
    return batt.strip(), therm.strip(), uptime_s


def load_model(arch, condition):
    import torch
    torch.set_num_threads(4)
    engine_logged = None
    if condition == 'int8':
        torch.backends.quantized.engine = 'qnnpack'
        engine_logged = str(torch.backends.quantized.engine)
        print(f"quantized.engine set to: {engine_logged}")
        ckpt = REPO / 'checkpoints' / f'{arch}_int8.pt'
    elif condition == 'fp32_ts':
        ckpt = REPO / 'checkpoints' / f'{arch}_fp32_ts.pt'
    else:
        raise ValueError(f"unknown condition {condition}")
    model = torch.jit.load(str(ckpt), map_location='cpu').eval()
    return model, engine_logged


def load_cifar_batch(n):
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


def start_powermetrics(tag):
    t0_perf = time.perf_counter()
    t0_wall = time.time()
    proc = subprocess.Popen(['sudo', '-n', WRAPPER, str(SAMPLE_INTERVAL_MS), str(SAMPLE_COUNT), tag],
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
                break
            time.sleep(0.005)
    if outfile is None:
        raise RuntimeError(f"powermetrics output file not found for tag {tag}")
    return proc, outfile, t0_perf


def start_p2_poller(outfile, arrivals, stop_event, first_header_event):
    """Thread-based P2 poller (acceptable here per instruction: this is a
    non-energy calibration run, not the real measurement loop where GIL
    contention would matter)."""
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


def run_inference_15min(model, images, out_dir, batch_size=1):
    """Continuous batch-1 32x32 inference for 15 minutes, marking consecutive
    5s windows by perf_counter. No subprocess calls inside this function.
    Each window is appended to windows_raw.jsonl on disk immediately, so a
    crash after this function returns (e.g. the proc.communicate() timeout
    bug found in the first run attempt) cannot lose window boundaries that
    otherwise exist only in this process's memory."""
    import torch
    windows = []
    jsonl_path = out_dir / 'windows_raw.jsonl'
    t_start_total = time.perf_counter()
    t_end_total = t_start_total + TOTAL_DURATION_S
    idx = 0
    n_images = images.shape[0]
    with torch.no_grad(), open(jsonl_path, 'a') as jf:
        while time.perf_counter() < t_end_total:
            w_start = time.perf_counter()
            w_target_end = w_start + WINDOW_S
            batches = 0
            while time.perf_counter() < w_target_end:
                x = images[idx % n_images: (idx % n_images) + batch_size]
                if x.shape[0] < batch_size:
                    idx = 0
                    x = images[0:batch_size]
                model(x)
                idx += batch_size
                batches += 1
            w_end = time.perf_counter()
            w = {'window_idx': len(windows), 'start_t': w_start, 'end_t': w_end,
                 'batches': batches}
            windows.append(w)
            jf.write(json.dumps(w) + '\n')
            jf.flush()
    return windows


def build_sample_index(outfile, arrivals):
    """Parse the full raw file once at the end to build per-sample records
    (power, thermal pressure, cpu freqs, cluster freq/residency), matched to
    each sample's P2 arrival time for window placement."""
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
    """For each window, collect samples whose [arrival-elapsed, arrival]
    interval overlaps [start_t, end_t], no offset correction (delta=0)."""
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


def write_arrivals_csv(path, samples):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_k', 'arrival_perf', 'elapsed_ms', 'cpu_power_mw', 'thermal_pressure'])
        for k in sorted(samples.keys()):
            s = samples[k]
            w.writerow([s['sample_k'], s['arrival_perf'], s['elapsed_ms'],
                        s['cpu_power_mw'], s['thermal_pressure']])


def run_one(arch, condition, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"thermalcal_{condition}"

    model, engine_logged = load_model(arch, condition)
    images = load_cifar_batch(64)  # cycle through 64 images repeatedly

    proc, outfile, t0_perf = start_powermetrics(tag)

    arrivals = {}
    stop_event = threading.Event()
    first_header_event = threading.Event()
    poller_thread = start_p2_poller(outfile, arrivals, stop_event, first_header_event)

    if not first_header_event.wait(timeout=15.0):
        stop_event.set()
        proc.kill()
        raise RuntimeError(f"{condition}: timed out waiting for first powermetrics header")

    print(f"[{condition}] first header received, starting 15-minute inference run")
    windows = run_inference_15min(model, images, out_dir, batch_size=1)
    print(f"[{condition}] inference run complete, {len(windows)} windows")

    # powermetrics still has (SAMPLE_COUNT * interval - TOTAL_DURATION_S) left to run
    # (the margin, §comment at MARGIN_S) -- the previous 30.0s fixed timeout here was
    # a bug: it did not account for that remaining margin and raised TimeoutExpired
    # while powermetrics was still legitimately running, losing in-memory state.
    elapsed_since_powermetrics_start = time.perf_counter() - t0_perf
    total_powermetrics_duration_s = SAMPLE_COUNT * (SAMPLE_INTERVAL_MS / 1000.0)
    remaining_s = max(0.0, total_powermetrics_duration_s - elapsed_since_powermetrics_start)
    communicate_timeout_s = remaining_s + 60.0
    print(f"[{condition}] waiting up to {communicate_timeout_s:.1f}s for powermetrics to exit "
          f"(remaining margin ~{remaining_s:.1f}s)")
    stdout, _ = proc.communicate(timeout=communicate_timeout_s)
    stop_event.set()
    poller_thread.join(timeout=5.0)

    with open(outfile, 'r', errors='ignore') as f:
        full_content = f.read()
    matches = list(HEADER_RE.finditer(full_content))
    now = time.perf_counter()
    for idx, m in enumerate(matches):
        s_num = idx + 1
        elapsed = float(m.group(1))
        if s_num not in arrivals:
            arrivals[s_num] = (now, elapsed)

    samples = build_sample_index(outfile, arrivals)
    windows = attribute_windows(windows, samples)

    write_arrivals_csv(out_dir / 'arrivals.csv', samples)
    # Copy raw powermetrics file into this run's output dir too (kept at
    # /var/gai-powermetrics as well; this is a convenience copy, not a move).
    import shutil
    shutil.copy(outfile, out_dir / Path(outfile).name)

    windows_out = []
    for w in windows:
        windows_out.append({k: v for k, v in w.items() if k != 'samples'})
    (out_dir / 'windows.json').write_text(json.dumps(windows_out, indent=2))

    run_meta = {
        'arch': arch,
        'condition': condition,
        'engine_logged': engine_logged,
        'outfile': str(outfile),
        'n_windows': len(windows),
        'n_samples': len(samples),
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))
    return windows, samples, run_meta


def main():
    out_base = Path.home() / 'thermal_calibration_results'
    out_base.mkdir(exist_ok=True)

    batt, therm, uptime_s = check_guards()

    results = {}
    for condition in ['fp32_ts', 'int8']:
        print(f"=== Starting ResNet-18 {condition} 15-minute run ===")
        out_dir = out_base / f'resnet18_{condition}'
        windows, samples, meta = run_one('resnet18', condition, out_dir)
        results[condition] = {'windows': windows, 'meta': meta}
        if condition == 'fp32_ts':
            print("Cooling gap before next run: 60s")
            time.sleep(60.0)

    (out_base / 'ambient.json').write_text(json.dumps({
        'ac_batt_output': batt,
        'ac_therm_output': therm,
        'uptime_s_recorded_not_enforced': uptime_s,
    }, indent=2))

    print("Done. Results in", out_base)


if __name__ == '__main__':
    main()
