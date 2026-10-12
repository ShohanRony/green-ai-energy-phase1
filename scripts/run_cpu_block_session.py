#!/usr/bin/env python3
"""x86-CPU block session runner (BLOCK I item 3). Orchestrates one session's worth of
conditions for the CPU block design registered in stage5_analysis_plan.md A7r1(e):
FP32-TorchScript (baseline), FP32-eager, INT8 (fbgemm), pruned30/50/70, each paired with
its _bnrecal counterpart where meaningful. Condition order is randomised per session using
a registered seed rule (see CONDITION_SEED_BASE below) -- unlike the original Stage 4b
main sessions, which ran conditions in a fixed sequence (a known limitation, D-logged in
stage5_analysis_plan.md's section 4 serial-reps discussion).

This script only orchestrates -- it shells out to pilot.py for the actual measurement,
exactly once per condition. It does not implement any measurement logic itself.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot import check_ac_power

# New seed family, distinct from the standing checkpoint seed (2026), the P8 campaign's
# 1001-1003, and series R's 2000+N (A8r1) -- same reasoning A8r1 gave for choosing its own
# family: never confusable with another family's provenance.
CONDITION_SEED_BASE = 3000

BNRECAL_SKIP = {
    # Known-collapsed per docs/bnrecal_cpu_equivalence.md -- recalibration produces zero
    # recovery (single-class prediction before and after). Excluded from the default
    # condition list as "not meaningful"; callers can override via --include-collapsed-bnrecal.
    ('mobilenet_v3_small', 'pruned50'),
    ('mobilenet_v3_small', 'pruned70'),
    ('efficientnet_b0', 'pruned70'),
}


def conditions_for(arch, include_collapsed_bnrecal=False):
    """The x86-CPU block's fixed condition list for one architecture, per A7r1(e)."""
    conds = ['fp32_ts', 'fp32_eager', 'int8', 'pruned30', 'pruned50', 'pruned70']
    for ratio in ('pruned30', 'pruned50', 'pruned70'):
        if include_collapsed_bnrecal or (arch, ratio) not in BNRECAL_SKIP:
            conds.append(f'{ratio}_bnrecal')
    return conds


def session_order(session_n, architectures, include_collapsed_bnrecal=False):
    """Deterministic, reproducible shuffle of (arch, condition) pairs for one session.
    Same (session_n, architectures) always produces the same order -- the point of a
    registered seed rule is that the order is fixed before the session runs, not chosen
    after seeing anything."""
    pairs = [(arch, c) for arch in architectures
             for c in conditions_for(arch, include_collapsed_bnrecal)]
    rng = random.Random(CONDITION_SEED_BASE + session_n)
    rng.shuffle(pairs)
    return pairs


def get_system_uptime():
    return float(Path('/proc/uptime').read_text().split()[0])


def checkpoint_for(arch, condition):
    """Return the checkpoint filename for an (arch, condition) pair, or None if eager."""
    if condition == 'fp32_eager':
        return None
    if condition == 'fp32_ts':
        return f'{arch}_fp32_ts.pt'
    if condition == 'int8':
        return f'{arch}_int8.pt'
    return f'{arch}_{condition}.pt'


def pilot_args_for(arch, condition, out_dir, checkpoints_dir):
    """Translate one (arch, condition) pair into the pilot.py invocation that measures it.
    Always passes --arch {arch} for every condition. fp32_eager builds fresh; every other
    condition loads a checkpoint."""
    args = ['--device', 'cpu', '--out', str(out_dir), '--cpu-affinity', 'pcores',
            '--sizes', '32', '--batches', '1',
            '--max-uptime-min', '180',
            '--arch', arch]
    ckpt_name = checkpoint_for(arch, condition)
    if ckpt_name is not None:
        args += ['--checkpoint', str(Path(checkpoints_dir) / ckpt_name)]
    return args


def preflight_checkpoints(architectures, checkpoints_dir, include_collapsed_bnrecal=False):
    """Verify that every non-eager checkpoint for the session exists and loads with torch.jit.load."""
    import torch
    checkpoints_dir = Path(checkpoints_dir)
    for arch in architectures:
        for cond in conditions_for(arch, include_collapsed_bnrecal):
            ckpt_name = checkpoint_for(arch, cond)
            if ckpt_name is None:
                continue
            ckpt_path = checkpoints_dir / ckpt_name
            if not ckpt_path.is_file():
                raise RuntimeError(f"Required checkpoint file missing: {ckpt_path}")
            try:
                torch.jit.load(str(ckpt_path), map_location='cpu')
            except Exception as e:
                raise RuntimeError(f"Checkpoint failed torch.jit.load: {ckpt_path} ({e})")


def default_runner(pilot_py, python_exe, args):
    """Production path: actually invoke pilot.py as a subprocess. Replaced with a stub in
    tests -- this function is the only place a real measurement would be launched from."""
    return subprocess.run([python_exe, str(pilot_py)] + args, check=True)


def run_session(session_n, architectures, out_base, checkpoints_dir,
                 runner=default_runner, python_exe=sys.executable,
                 pilot_py=None, skip_ac_check=False, include_collapsed_bnrecal=False,
                 governor_check=None, uptime_fn=None,
                 checkpoint_preflight=None):
    if uptime_fn is None:
        uptime_fn = get_system_uptime
    uptime_s = uptime_fn()
    if uptime_s > 30 * 60:
        raise RuntimeError(f'System uptime is {uptime_s/60:.1f} min (> 30 min) -- reboot and start within 30 min.')

    if pilot_py is None:
        pilot_py = Path(__file__).resolve().parent.parent / 'pilot.py'
    
    if governor_check is None:
        from pilot import check_governor
        governor_check = check_governor
    governor_check()

    if checkpoint_preflight is None:
        checkpoint_preflight = preflight_checkpoints
    checkpoint_preflight(architectures, checkpoints_dir, include_collapsed_bnrecal)
    
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    if not skip_ac_check and not check_ac_power():
        raise RuntimeError('AC power is not connected -- refusing to start a CPU-block '
                            'session (D26 precondition).')

    order = session_order(session_n, architectures, include_collapsed_bnrecal)
    log = {'session': session_n, 'seed': CONDITION_SEED_BASE + session_n,
           'session_start_uptime_s': uptime_s,
           'order': [f'{a}_{c}' for a, c in order], 'conditions': []}
           
    void_log = {'session': session_n, 'voids': []}
    
    reruns = []
    void_count = 0
    
    def process_condition(arch, condition, is_rerun=False):
        nonlocal void_count
        label = f'{arch}_{condition}'
        if is_rerun:
            label += '_rerun'
        cond_out = out_base / label
        args = pilot_args_for(arch, condition, cond_out, checkpoints_dir)
        
        crashed = False
        try:
            runner(pilot_py, python_exe, args)
        except subprocess.CalledProcessError as e:
            crashed = True
            
        # check AC abort
        if not skip_ac_check and not check_ac_power():
            raise RuntimeError('AC power lost mid-session -- aborting session.')
            
        # Check if flagged
        flagged = False
        if not crashed:
            import csv
            csv_path = cond_out / 'windows.csv'
            if not csv_path.exists():
                raise RuntimeError(f"windows.csv is missing in {cond_out}")
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                if 'thermal_flag' not in reader.fieldnames:
                    raise RuntimeError(f"thermal_flag column missing from {csv_path}")
                for row in reader:
                    if row.get('phase') in {"a1", "a2"} and int(row['repeat']) > 0:
                        flag_val = row.get('thermal_flag', '')
                        if not flag_val:
                            raise RuntimeError(f"Evaluation missing (empty thermal_flag) in {csv_path}")
                        if flag_val == 'True':
                            flagged = True
                            break
            
        if crashed or flagged:
            void_log['voids'].append({'condition': f'{arch}_{condition}', 'crashed': crashed, 'thermal_flagged': flagged})
            void_count += 1
            if void_count > 2:
                (out_base / 'void_log.json').write_text(json.dumps(void_log, indent=2))
                raise RuntimeError(f"Session voided: more than 2 voided conditions ({void_count}).")
            if not is_rerun:
                reruns.append((arch, condition))
        else:
            log['conditions'].append(label)

    for arch, condition in order:
        process_condition(arch, condition, is_rerun=False)
        
    for arch, condition in reruns:
        process_condition(arch, condition, is_rerun=True)

    (out_base / 'session_log.json').write_text(json.dumps(log, indent=2))
    (out_base / 'void_log.json').write_text(json.dumps(void_log, indent=2))
    return log

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=int, required=True)
    p.add_argument('--architectures', nargs='+',
                    default=['resnet18', 'mobilenet_v3_small', 'efficientnet_b0'])
    p.add_argument('--out-base', required=True)
    p.add_argument('--checkpoints-dir', default='checkpoints')
    p.add_argument('--include-collapsed-bnrecal', action='store_true')
    a = p.parse_args()
    run_session(a.session, a.architectures, a.out_base, a.checkpoints_dir,
                include_collapsed_bnrecal=a.include_collapsed_bnrecal)


if __name__ == '__main__':
    main()
