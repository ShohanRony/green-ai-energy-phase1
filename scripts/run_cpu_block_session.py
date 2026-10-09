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


def pilot_args_for(arch, condition, out_dir, checkpoints_dir):
    """Translate one (arch, condition) pair into the pilot.py invocation that measures it.
    fp32_eager uses --arch (builds the model fresh, no TorchScript checkpoint); every other
    condition loads a checkpoint, which already carries its own architecture when traced."""
    args = ['--device', 'cpu', '--out', str(out_dir), '--cpu-affinity', 'pcores']
    if condition == 'fp32_eager':
        args += ['--arch', arch]
    else:
        ckpt_name = {
            'fp32_ts': f'{arch}_fp32.pt',
            'int8': f'{arch}_int8.pt',
        }.get(condition, f'{arch}_{condition}.pt')
        args += ['--checkpoint', str(Path(checkpoints_dir) / ckpt_name)]
    return args


def default_runner(pilot_py, python_exe, args):
    """Production path: actually invoke pilot.py as a subprocess. Replaced with a stub in
    tests -- this function is the only place a real measurement would be launched from."""
    return subprocess.run([python_exe, str(pilot_py)] + args, check=True)


def run_session(session_n, architectures, out_base, checkpoints_dir,
                 runner=default_runner, python_exe=sys.executable,
                 pilot_py=None, skip_ac_check=False, include_collapsed_bnrecal=False):
    """Runs one full session: AC guard, then every (arch, condition) pair in this
    session's randomised order, each via `runner`. Returns the session log dict (also
    written to out_base/session_log.json)."""
    if pilot_py is None:
        pilot_py = Path(__file__).resolve().parent.parent / 'pilot.py'
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    if not skip_ac_check and not check_ac_power():
        raise RuntimeError('AC power is not connected -- refusing to start a CPU-block '
                            'session (D26 precondition).')

    order = session_order(session_n, architectures, include_collapsed_bnrecal)
    log = {'session': session_n, 'seed': CONDITION_SEED_BASE + session_n,
           'order': [f'{a}_{c}' for a, c in order], 'conditions': []}
    for arch, condition in order:
        label = f'{arch}_{condition}'
        cond_out = out_base / label
        args = pilot_args_for(arch, condition, cond_out, checkpoints_dir)
        runner(pilot_py, python_exe, args)
        log['conditions'].append(label)
    (out_base / 'session_log.json').write_text(json.dumps(log, indent=2))
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
