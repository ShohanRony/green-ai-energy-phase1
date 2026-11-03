#!/usr/bin/env python3
"""Flush test for gai_powermetrics.sh at 50 ms poll interval.
Determines whether powermetrics flushes per-sample at 50 ms poll.
"""
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

OUTDIR = Path('/var/gai-powermetrics')
WRAPPER = '/usr/local/sbin/gai_powermetrics.sh'
HEADER = '*** Sampled system activity'
POLL_INTERVAL = 0.05  # 50 ms

t0_perf = time.perf_counter()
t0_wall = time.time()

proc = subprocess.Popen(['sudo', '-n', WRAPPER, '400', '20', 'flushtest50ms'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

outfile = None
find_deadline = time.perf_counter() + 30
while outfile is None and time.perf_counter() < find_deadline:
    for c in glob.glob(str(OUTDIR / 'flushtest50ms_*.txt')):
        try:
            if os.path.getmtime(c) >= t0_wall - 2:
                outfile = c
                break
        except FileNotFoundError:
            continue
    if outfile is None:
        if proc.poll() is not None:
            break
        time.sleep(0.01)

arrivals = []  # (t_rel_s, count, size_bytes)
last_count = 0
poll_count = 0
while True:
    poll_count += 1
    if outfile and os.path.exists(outfile):
        try:
            size = os.path.getsize(outfile)
            with open(outfile, 'r', errors='ignore') as f:
                data = f.read()
        except (FileNotFoundError, OSError):
            size = 0
            data = ''
        count = data.count(HEADER)
        if count > last_count:
            t_rel = time.perf_counter() - t0_perf
            arrivals.append((t_rel, count, size))
            last_count = count
    if proc.poll() is not None:
        break
    time.sleep(POLL_INTERVAL)

stdout, _ = proc.communicate()
total_time = time.perf_counter() - t0_perf

final_count = last_count
elapsed_values = []
final_size = None
if outfile and os.path.exists(outfile):
    final_size = os.path.getsize(outfile)
    with open(outfile, 'r', errors='ignore') as f:
        data = f.read()
    for line in data.splitlines():
        if HEADER in line:
            m = re.search(r'\(([\d.]+)ms elapsed\)', line)
            if m:
                elapsed_values.append(float(m.group(1)))

print('t0_perf_counter:', t0_perf)
print('t0_time:', t0_wall)
print('POLL_INTERVAL_REQUESTED:', POLL_INTERVAL)
print('OUTFILE:', outfile)
print('WRAPPER_EXIT_CODE:', proc.returncode)
print('TOTAL_RUN_TIME_S:', total_time)
print('POLL_ITERATIONS:', poll_count)
print()
print('ARRIVAL_TABLE (t_rel_s, header_count, file_size_bytes):')
for t_rel, count, size in arrivals:
    print(f'{t_rel:.4f}\t{count}\t{size}')
print()
print('FINAL_HEADER_COUNT:', final_count)
print('FINAL_FILE_SIZE_BYTES:', final_size)
print()
print(f'ELAPSED_VALUES ({len(elapsed_values)} samples, ms):')
for v in elapsed_values:
    print(v)
print()
print('WRAPPER_STDOUT:')
print(stdout)
