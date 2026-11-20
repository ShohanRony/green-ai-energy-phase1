"""M1 per-condition measurement loop and condition runner.

Implements the M1-CPU per-condition measurement loop mapped from x86 pilot.py
per scripts/m1/MEASUREMENT_LOOP_DESIGN.md and Amendment A13:
- Out-of-process P2 powermetrics poller (os.stat every ~2ms, zero GIL contention)
- Powermetrics sizing: COUNT = ceil(expected_duration_s / 0.405) + 5
- Sequential powermetrics process lifecycle
- No subprocess inside measurement windows; pmset checks only at settle start
- Preflight guards: blocked processes, git tree clean, checkpoint SHA-256
- caffeinate ownership daemon per condition
- INT8 explicit backend: torch.backends.quantized.engine = 'qnnpack'
- Void rule: any sample overlapping a measurement window lacking live arrival voids condition
- Thermal rule: qualifying windows (a1/a2, repeat > 0) flagged if freq < T or pressure != 'Nominal'
- Artifacts: windows.csv, summary.csv, arrivals.csv, environment.json, summary.json, raw powermetrics
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from guards import (
    GuardError,
    check_ac_power_m1,
    check_thermal_nominal_m1,
    check_uptime_guard,
    parse_boottime,
)
from m1_pipeline import (
    WindowPhase,
    check_condition_thermal_flags,
    process_condition_energy,
    read_thermal_thresholds_m1,
    write_summary_csv,
    write_windows_csv,
)
from powermetrics_parser import parse_powermetrics_file

BLOCKED_PROCESS_NAMES = ('AnyDesk', 'OneDrive', 'remoting_host', 'screensharingd')
DEFAULT_HASHES_FILE = Path(__file__).resolve().parent.parent.parent / 'm1_preflight' / 'results' / 'm1_checkpoint_hashes.txt'
DEFAULT_WRAPPER = '/usr/local/sbin/gai_powermetrics.sh'
POLLER_SCRIPT = Path(__file__).resolve().parent / 'm1_poller.py'


def _spawn_subprocess(*args, **kwargs):
    """Subprocess spawn wrapper to enable intercepting/testing subprocess calls."""
    return subprocess.Popen(*args, **kwargs)


def _get_running_processes() -> List[str]:
    """Get list of currently running process command names via ps."""
    try:
        out = subprocess.check_output(['ps', '-A', '-o', 'comm='], text=True, errors='ignore')
        return [Path(line.strip()).name for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def compute_powermetrics_count(expected_duration_s: float) -> int:
    """Amendment A13 formula: COUNT = ceil(expected_duration_s / 0.405) + 5.
    
    0.405 is slightly below measured mean actual sample interval of 410.36 ms.
    """
    if expected_duration_s <= 0.0:
        return 5
    return math.ceil(expected_duration_s / 0.405) + 5


def verify_checkpoint_sha256(checkpoint_path: Path, hashes_file: Path) -> str:
    """Re-verify checkpoint SHA-256 against registered hash table."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint file not found: {checkpoint_path}")

    h = hashlib.sha256()
    with open(checkpoint_path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest().lower()

    if not hashes_file.is_file():
        raise RuntimeError(f"Expected hashes file not found: {hashes_file}")

    expected_hashes: Dict[str, str] = {}
    with open(hashes_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sha = parts[0].strip().lower()
                rel_path = parts[1].strip()
                fname = Path(rel_path).name
                expected_hashes[fname] = sha

    fname = checkpoint_path.name
    if fname not in expected_hashes:
        raise RuntimeError(f"No registered SHA-256 hash found for checkpoint {fname} in {hashes_file}")

    expected_sha = expected_hashes[fname]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Checkpoint SHA-256 hash mismatch for {fname}: "
            f"expected {expected_sha}, got {actual_sha}"
        )

    return actual_sha


def run_condition_preflight(
    arch: str,
    condition: str,
    checkpoints_dir: Path,
    hashes_file: Path = DEFAULT_HASHES_FILE,
    skip_git_check: bool = False,
    skip_hash_check: bool = False,
    max_uptime_s: float = 10800.0,
    check_ac_fn: Callable[[], bool] = check_ac_power_m1,
    check_therm_fn: Callable[[], bool] = check_thermal_nominal_m1,
):
    """Run all pre-condition guards before touching models or sensors."""
    # 1. Blocked background process check (presence alone blocks)
    running = _get_running_processes()
    for blocked in BLOCKED_PROCESS_NAMES:
        if blocked in running:
            raise GuardError(f"Blocked process '{blocked}' is currently running (fail-closed guard).")

    # 2. Checkpoint SHA-256 check
    if not skip_hash_check and condition != 'fp32_eager':
        from m1_session_runner import checkpoint_for
        ckpt_name = checkpoint_for(arch, condition)
        if ckpt_name:
            ckpt_path = Path(checkpoints_dir) / ckpt_name
            verify_checkpoint_sha256(ckpt_path, hashes_file)

    # 3. Git tree clean check
    if not skip_git_check:
        repo_root = Path(__file__).resolve().parent.parent.parent
        git_res = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        # Note: ignore untracked files or report if porcelain has modified tracked files
        tracked_changes = [line for line in git_res.stdout.splitlines() if not line.startswith('??')]
        if tracked_changes:
            raise RuntimeError(f"Git tree has tracked uncommitted changes: {tracked_changes}")

    # 4. AC and thermal nominal guards
    check_ac_fn()
    check_therm_fn()

    # 5. Uptime guard
    try:
        bt_out = subprocess.check_output(['sysctl', '-n', 'kern.boottime'], text=True)
        boottime_sec = parse_boottime(bt_out)
        check_uptime_guard(boottime_sec, max_uptime_seconds=max_uptime_s, context=f"condition {arch}_{condition}")
    except (FileNotFoundError, subprocess.SubprocessError):
        pass  # On non-Darwin dev workstations, sysctl may be absent in mock mode


def verify_live_arrivals_coverage(
    arrivals: List[Dict[str, Any]],
    phase_windows: List[WindowPhase],
) -> None:
    """Void rule: any sample overlapping a measurement window must have been captured live.
    
    If any sample overlapping a window has is_live=False, the condition is void.
    """
    for w in phase_windows:
        w_start = w.start_t
        w_end = w.end_t
        for a in arrivals:
            # Note: arrival_perf (poller child) and start_t/end_t (parent) use
            # perf_counter from different processes. On macOS (mach_absolute_time),
            # the epoch is shared; validated in calibration/m1_edge_20261103
            # (cross-process offset: 0.477 µs). Not portable to Linux.
            arr_t = float(a['arrival_perf'])
            dur_s = float(a['elapsed_ms']) / 1000.0
            sample_start = arr_t - dur_s
            sample_end = arr_t

            # Check overlap
            if max(sample_start, w_start) < min(sample_end, w_end):
                if not a.get('is_live', True):
                    raise RuntimeError(
                        f"Condition voided: sample {a.get('sample_k')} overlapping window "
                        f"{w.phase} (repeat {w.repeat}) lacks live arrival (captured post-mortem)."
                    )


class CaffeinateScope:
    """Context manager ensuring caffeinate runs for the condition's duration."""
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        try:
            self.proc = _spawn_subprocess(
                ['caffeinate', '-dimsu', '-w', str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            self.proc = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def measure_condition(
    arch: str,
    condition: str,
    out_dir: Union[str, Path],
    checkpoints_dir: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
    repeats: int = 6,
    settle_s: float = 3.0,
    window_s: float = 5.0,
    skip_preflight: bool = False,
    mock_forward: bool = False,
    mock_powermetrics: bool = False,
    injected_samples: Optional[List[Dict[str, Any]]] = None,
    check_ac_fn: Callable[[], bool] = check_ac_power_m1,
    check_therm_fn: Callable[[], bool] = check_thermal_nominal_m1,
    poll_interval_s: float = 0.002,
) -> Dict[str, Any]:
    """Execute complete measurement loop for one (arch, condition) pair.
    
    Returns condition summary dictionary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(checkpoints_dir)
    cond_t0 = time.time()

    # Preflight
    if not skip_preflight:
        run_condition_preflight(
            arch=arch,
            condition=condition,
            checkpoints_dir=checkpoints_dir,
            check_ac_fn=check_ac_fn,
            check_therm_fn=check_therm_fn,
        )

    with CaffeinateScope():
        # Setup PyTorch environment and model
        quantized_engine = None
        work_fn = None
        cifar_x = None
        weights_desc = None
        weights_sha256 = None
        input_source = None

        if not mock_forward:
            import torch
            import torchvision
            from torchvision import transforms

            torch.set_num_threads(4)
            if 'int8' in condition:
                torch.backends.quantized.engine = 'qnnpack'
                quantized_engine = 'qnnpack'

            from m1_session_runner import checkpoint_for
            ckpt_name = checkpoint_for(arch, condition)
            if ckpt_name:
                ckpt_path = checkpoints_dir / ckpt_name
                try:
                    model = torch.jit.load(str(ckpt_path), map_location='cpu')
                except RuntimeError:
                    from train_baseline import build_model
                    model = build_model(arch)
                    model.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=True))
                weights_desc = 'checkpoint'
            else:
                # fp32_eager mirrors x86 pilot.py exactly: seeded random weights, built the
                # same way, for the timing-only eager-vs-TorchScript comparison (A7/D2) --
                # not a trained baseline. Seed set immediately before build_model, same as
                # pilot.py's torch.manual_seed(2026); random.seed(2026) ordering.
                torch.manual_seed(2026)
                random.seed(2026)
                from train_baseline import build_model
                model = build_model(arch)
                weights_desc = 'seeded random weights (seed 2026), timing only, as x86 pilot.py'
            model.eval()
            if weights_desc != 'checkpoint':
                sd_bytes = b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())
                weights_sha256 = hashlib.sha256(sd_bytes).hexdigest()

            # Data prep: CIFAR-10 single image (32x32, normalized) -- same source, split,
            # index and normalisation as x86 pilot.py (ds[0][0], ToTensor(), same mean/std).
            # A missing/unloadable dataset is a measurement error: no synthetic fallback.
            candidate_dirs = []
            if data_dir is not None:
                candidate_dirs.append(Path(data_dir))
            else:
                resolved_ckpt_dir = Path(checkpoints_dir).resolve()
                candidate_dirs.append(resolved_ckpt_dir.parent / 'data')
                candidate_dirs.append(resolved_ckpt_dir.parent.parent / 'data')

            ds = None
            used_data_dir = None
            load_errors = []
            for cand in candidate_dirs:
                if not cand.exists():
                    load_errors.append(f"{cand}: does not exist")
                    continue
                try:
                    ds = torchvision.datasets.CIFAR10(str(cand), train=False, download=False,
                                                        transform=transforms.ToTensor())
                    used_data_dir = cand
                    break
                except Exception as e:
                    load_errors.append(f"{cand}: {e}")
            if ds is None:
                raise RuntimeError(
                    f"Could not load real CIFAR-10 test data for condition {arch}_{condition}. "
                    f"Tried: {'; '.join(load_errors)}. No synthetic fallback is used."
                )
            raw = ds[0][0].unsqueeze(0)
            input_source = {
                'dataset': 'CIFAR-10',
                'path': str(used_data_dir),
                'split': 'test',
                'index': 0,
                'transform': 'ToTensor',
            }

            norm_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            norm_std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
            cifar_x = (raw - norm_mean) / norm_std
            try:
                target_dtype = next(model.parameters()).dtype
            except StopIteration:
                target_dtype = cifar_x.dtype
            cifar_x = cifar_x.to(target_dtype)
            work_fn = lambda: model(cifar_x)
        else:
            if 'int8' in condition:
                quantized_engine = 'qnnpack'
            work_fn = lambda: None

        # Powermetrics and Poller setup
        pm_proc: Optional[subprocess.Popen] = None
        poller_proc: Optional[subprocess.Popen] = None
        raw_pm_path = None
        arrivals: List[Dict[str, Any]] = []

        if injected_samples is None and not mock_powermetrics:
            # 1. Wait for any previous powermetrics process to exit
            while True:
                running_procs = _get_running_processes()
                if 'powermetrics' not in running_procs and 'gai_powermetrics.sh' not in running_procs:
                    break
                time.sleep(0.5)

            # 2. Sizing formula
            expected_duration_s = repeats * 4 * (settle_s + window_s)
            count = compute_powermetrics_count(expected_duration_s)
            tag = f"gai_{arch}_{condition}_{int(time.time())}"

            pm_start_perf = time.perf_counter()
            pm_proc = _spawn_subprocess(
                ['sudo', '-n', DEFAULT_WRAPPER, '400', str(count), tag],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Locate output file
            out_pattern = f"/var/gai-powermetrics/{tag}_*.txt"
            raw_file_candidate = None
            for _ in range(100):
                matches = glob.glob(out_pattern)
                if matches:
                    raw_file_candidate = Path(matches[0])
                    break
                time.sleep(0.1)

            if not raw_file_candidate or not raw_file_candidate.exists():
                if pm_proc:
                    pm_proc.kill()
                raise RuntimeError(f"Timed out locating powermetrics output file matching {out_pattern}")

            raw_pm_path = raw_file_candidate

            # 3. Start P2 poller child process
            ipc_file = out_dir / 'arrivals_raw.jsonl'
            first_header_file = out_dir / 'first_header.sentinel'
            if ipc_file.exists():
                ipc_file.unlink()
            if first_header_file.exists():
                first_header_file.unlink()

            poller_proc = _spawn_subprocess([
                sys.executable,
                str(POLLER_SCRIPT),
                '--output-file', str(raw_pm_path),
                '--ipc-file', str(ipc_file),
                '--first-header-file', str(first_header_file),
                '--poll-interval-s', str(poll_interval_s),
            ])

            # Wait for first header event
            t_hdr_wait = time.perf_counter()
            while not first_header_file.exists():
                if time.perf_counter() - t_hdr_wait > 20.0:
                    poller_proc.terminate()
                    pm_proc.kill()
                    raise RuntimeError("Timed out waiting for first header from P2 poller (>20s)")
                time.sleep(0.005)

        # Execute 24-window phase loop
        phase_windows: List[WindowPhase] = []
        raw_windows_path = out_dir / 'windows_raw.jsonl'

        try:
            for rep in range(repeats):
                for phase in ['idle_before', 'a1', 'a2', 'idle_after']:
                    # 1. Settle: run pmset checks at START of settle
                    settle_start = time.perf_counter()
                    check_ac_fn()
                    check_therm_fn()

                    settle_elapsed = time.perf_counter() - settle_start
                    settle_rem = max(0.0, settle_s - settle_elapsed)

                    if phase.startswith('a'):
                        warm_until = time.perf_counter() + settle_rem
                        while time.perf_counter() < warm_until:
                            work_fn()
                    else:
                        time.sleep(settle_rem)

                    # 2. Window (5.0s): ZERO subprocess calls permitted
                    w_start = time.perf_counter()
                    batches = 0
                    if phase.startswith('a'):
                        while (time.perf_counter() - w_start) < window_s:
                            work_fn()
                            batches += 1
                    else:
                        time.sleep(window_s)
                    w_end = time.perf_counter()

                    wp = WindowPhase(
                        device='cpu',
                        arch=arch,
                        condition=condition,
                        size=32,
                        batch=1,
                        requested_s=window_s,
                        repeat=rep,
                        phase=phase,
                        start_t=w_start,
                        end_t=w_end,
                        batches=batches,
                    )
                    phase_windows.append(wp)

                    # Write boundary incrementally
                    with open(raw_windows_path, 'a', encoding='utf-8') as f_rw:
                        f_rw.write(json.dumps({
                            'repeat': rep, 'phase': phase,
                            'start_t': w_start, 'end_t': w_end,
                            'duration_s': w_end - w_start,
                            'batches': batches,
                        }) + '\n')

        finally:
            # Poller and powermetrics teardown
            if poller_proc is not None:
                try:
                    poller_proc.send_signal(signal.SIGTERM)
                    poller_proc.wait(timeout=5.0)
                except Exception:
                    try:
                        poller_proc.kill()
                    except Exception:
                        pass

            if pm_proc is not None:
                try:
                    pm_proc.wait(timeout=30.0)
                except Exception:
                    try:
                        pm_proc.kill()
                    except Exception:
                        pass

        # Collect arrivals
        if injected_samples is not None:
            arrivals = []
            if phase_windows and injected_samples and float(injected_samples[0].get('arrival_perf', 0.0)) < 100000.0:
                t_anchor = phase_windows[0].start_t - 0.2
                t_first = float(injected_samples[0].get('arrival_perf', 0.0))
                time_delta = t_anchor - t_first
                for s in injected_samples:
                    s_copy = dict(s)
                    s_copy['arrival_perf'] = float(s['arrival_perf']) + time_delta
                    s_copy['arrival_t'] = s_copy['arrival_perf']
                    arrivals.append(s_copy)
            else:
                for s in injected_samples:
                    s_copy = dict(s)
                    if 'arrival_t' not in s_copy and 'arrival_perf' in s_copy:
                        s_copy['arrival_t'] = s_copy['arrival_perf']
                    arrivals.append(s_copy)
            parsed_samples = arrivals
        elif raw_pm_path is not None:
            # Read arrivals from poller IPC file
            ipc_file = out_dir / 'arrivals_raw.jsonl'
            if ipc_file.exists():
                with open(ipc_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            arrivals.append(json.loads(line))

            # Parse full powermetrics output
            pm_run = parse_powermetrics_file(raw_pm_path)
            # Retain raw copy in condition out_dir
            shutil.copy(raw_pm_path, out_dir / 'raw_powermetrics.txt')

            # Align arrivals with powermetrics samples
            arr_by_sample = {int(a['sample_k']): a for a in arrivals}
            parsed_samples = []
            for idx, s in enumerate(pm_run, 1):
                s_dict = dict(s)
                arr = arr_by_sample.get(idx)
                if arr is not None:
                    s_dict['arrival_t'] = float(arr['arrival_perf'])
                    s_dict['is_live'] = arr.get('is_live', True)
                else:
                    # Sample not in poller record
                    s_dict['is_live'] = False
                parsed_samples.append(s_dict)
        else:
            # Mock mode with no samples injected: generate synthetic samples covering windows
            parsed_samples = []
            t_curr = phase_windows[0].start_t - 0.4 if phase_windows else 0.0
            idx = 1
            last_t = phase_windows[-1].end_t + 0.8 if phase_windows else 10.0
            while t_curr < last_t:
                parsed_samples.append({
                    'sample_k': idx,
                    'arrival_perf': t_curr + 0.4,
                    'arrival_t': t_curr + 0.4,
                    'elapsed_ms': 400.0,
                    'is_live': True,
                    'cpu_power_mw': 4500,
                    'pcore_mean_freq_mhz': 3204.0,
                    'thermal_pressure': 'Nominal',
                })
                t_curr += 0.4
                idx += 1
            arrivals = parsed_samples

        # 4. Check live arrival void rule
        verify_live_arrivals_coverage(arrivals, phase_windows)

        # 5. Overlap energy post-processing
        # Pass arrival_t as sample anchor and evaluate thermal guard
        thresh = read_thermal_thresholds_m1()
        rows, summary = process_condition_energy(
            parsed_samples=parsed_samples,
            phase_windows=phase_windows,
            time_shift_s=0.0,
            thermal_thresholds=thresh,
            require_thermal_guard=True,
        )

        # 6. Write all artifacts
        write_windows_csv(rows, out_dir / 'windows.csv')
        write_summary_csv(summary, out_dir / 'summary.csv')
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

        # Write arrivals.csv
        with open(out_dir / 'arrivals.csv', 'w', newline='', encoding='utf-8') as f_arr:
            fieldnames = ['sample_k', 'arrival_perf', 'elapsed_ms', 'cpu_power_mw', 'thermal_pressure', 'is_live']
            writer = csv.DictWriter(f_arr, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for a in arrivals:
                writer.writerow(a)

        # Write environment.json
        env_record = {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python': sys.version,
            'arch': arch,
            'condition': condition,
            'quantized_engine': quantized_engine,
            'threads': 4,
            'timestamp': cond_t0,
            'settle_s': settle_s,
            'window_s': window_s,
            'repeats': repeats,
            'weights': weights_desc,
            'weights_sha256': weights_sha256,
            'input_source': input_source,
        }
        (out_dir / 'environment.json').write_text(json.dumps(env_record, indent=2))

        return summary


def main():
    parser = argparse.ArgumentParser(description="M1 single condition runner")
    parser.add_argument('--arch', required=True, help="Model architecture")
    parser.add_argument('--condition', required=True, help="Condition name")
    parser.add_argument('--out-dir', required=True, type=Path, help="Output directory")
    parser.add_argument('--checkpoints-dir', required=True, type=Path, help="Checkpoints directory")
    parser.add_argument('--data-dir', type=Path, default=None, help="CIFAR-10 data directory")
    args = parser.parse_args()

    res = measure_condition(
        arch=args.arch,
        condition=args.condition,
        out_dir=args.out_dir,
        checkpoints_dir=args.checkpoints_dir,
        data_dir=args.data_dir,
    )
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
