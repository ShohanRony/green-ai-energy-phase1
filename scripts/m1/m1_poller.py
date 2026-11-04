#!/usr/bin/env python3
"""Standalone out-of-process P2 powermetrics poller.

Runs as an independent OS process to eliminate Python GIL contention
with inference forward loops.

Polls the powermetrics output file via os.stat every ~2ms (default),
reads only bytes after the last-seen offset, and logs live sample
arrival timestamps (time.perf_counter()) and elapsed durations.
Distinguishes between live arrivals and post-mortem recoveries.

Spec: scripts/m1/MEASUREMENT_LOOP_DESIGN.md sections 1.3, 1.5; Amendment A13.
"""
import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

HEADER_RE = re.compile(r'\*\*\* Sampled system activity .*? \(([\d.]+)ms elapsed\) \*\*\*')


def run_poller(
    output_file: Path,
    ipc_file: Path,
    ready_file: Path = None,
    first_header_file: Path = None,
    poll_interval_s: float = 0.002,
):
    stop_requested = False

    def handle_signal(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Touch ready file if requested
    if ready_file:
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        ready_file.touch()

    last_offset = 0
    last_count = 0
    carry = ''
    ipc_file.parent.mkdir(parents=True, exist_ok=True)

    with open(ipc_file, 'a', encoding='utf-8') as ipc_out:
        while not stop_requested:
            try:
                if output_file.exists():
                    size = os.path.getsize(output_file)
                    if size > last_offset:
                        with open(output_file, 'r', errors='ignore') as f:
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
                                rec = {
                                    'sample_k': sample_num,
                                    'arrival_perf': now,
                                    'elapsed_ms': elapsed,
                                    'is_live': True,
                                }
                                ipc_out.write(json.dumps(rec) + '\n')
                                ipc_out.flush()
                                if sample_num == 1 and first_header_file:
                                    first_header_file.parent.mkdir(parents=True, exist_ok=True)
                                    first_header_file.touch()
                            last_count += len(matches)
                            last_match_end = matches[-1].end()
                            carry = text[last_match_end:]
                        else:
                            carry = text
                        if len(carry) > 8192:
                            carry = carry[-8192:]
            except Exception:
                pass
            time.sleep(poll_interval_s)

        # Final post-mortem read: capture any samples missed while stopping
        # Any sample discovered ONLY here gets is_live=False!
        try:
            if output_file.exists():
                with open(output_file, 'r', errors='ignore') as f:
                    full_content = f.read()
                matches = list(HEADER_RE.finditer(full_content))
                now = time.perf_counter()
                for idx, m in enumerate(matches):
                    sample_num = idx + 1
                    if sample_num > last_count:
                        elapsed = float(m.group(1))
                        rec = {
                            'sample_k': sample_num,
                            'arrival_perf': now,
                            'elapsed_ms': elapsed,
                            'is_live': False,  # Missed live!
                        }
                        ipc_out.write(json.dumps(rec) + '\n')
                        ipc_out.flush()
                        last_count = sample_num
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="P2 Powermetrics poller child process")
    parser.add_argument('--output-file', required=True, type=Path, help="Powermetrics output file to monitor")
    parser.add_argument('--ipc-file', required=True, type=Path, help="File to append arrival JSON records")
    parser.add_argument('--ready-file', type=Path, default=None, help="File to touch when poller starts")
    parser.add_argument('--first-header-file', type=Path, default=None, help="File to touch on first sample header")
    parser.add_argument('--poll-interval-s', type=float, default=0.002, help="Poll interval in seconds")
    args = parser.parse_args()

    run_poller(
        output_file=args.output_file,
        ipc_file=args.ipc_file,
        ready_file=args.ready_file,
        first_header_file=args.first_header_file,
        poll_interval_s=args.poll_interval_s,
    )


if __name__ == '__main__':
    main()
