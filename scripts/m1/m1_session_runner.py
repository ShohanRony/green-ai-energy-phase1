"""M1-CPU block session runner.

Orchestrates conditions for the 6 M1 replication sessions per the frozen plan:
- 23 conditions per session (structural exclusion: mobilenet_v3_small int8 missing under qnnpack)
- Randomised condition order seeded with CONDITION_SEED_BASE + session_n (4000 + N)
- Fresh-boot guard: system uptime < 30 min at session start, < 180 min per condition
- AC guard (pre-session and per-condition)
- Thermal guard (nominal check)
- Maximum 2 voided conditions per session before voiding the entire session

Spec: scripts/m1/HARNESS_PLAN.md sections 1, 4, 6; docs/m1_harness_design.md;
frozen plan in docs/stage5_consolidated_plan.md.
"""
import argparse
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).parent))

from guards import (
    GuardError,
    check_ac_power_m1,
    check_thermal_nominal_m1,
    check_uptime_guard,
    compute_uptime,
    parse_boottime,
    parse_vm_stat,
)


def _write_session_abort(out_base: Path, session_n: int, reason: str, condition: Optional[str],
                          uptime_s: Optional[float], current_time_fn: Optional[Callable[[], float]]):
    """A3: on any GuardError, record why before re-raising -- so a session that
    aborts leaves a file explaining itself, not just a propagated exception the
    caller might not log anywhere."""
    abort_record = {
        'session': session_n,
        'reason': reason,
        'condition': condition,
        'uptime_s': uptime_s,
        'time': current_time_fn() if current_time_fn else time.time(),
    }
    (out_base / 'session_abort.json').write_text(json.dumps(abort_record, indent=2))

# Registered seed family for M1 block (distinct from checkpoint 2026, P8 1001-1003, series R 2000+N, x86-CPU 3000+N)
CONDITION_SEED_BASE = 4000

# Known-collapsed per docs/bnrecal_cpu_equivalence.md
BNRECAL_SKIP = {
    ('mobilenet_v3_small', 'pruned50'),
    ('mobilenet_v3_small', 'pruned70'),
    ('efficientnet_b0', 'pruned70'),
}

# Structurally missing on M1 under qnnpack (m1_preflight/README.md)
M1_STRUCTURAL_EXCLUSIONS = {
    ('mobilenet_v3_small', 'int8'),
}


def conditions_for(arch: str, include_collapsed_bnrecal: bool = False) -> List[str]:
    """The fixed M1 condition list for one architecture.

    A6 fix: ordered to match scripts/run_cpu_block_session.py's conditions_for
    exactly (all base states first, then all bnrecal states appended at the
    end) -- not interleaved per-ratio as before. This pre-shuffle order feeds
    directly into session_order()'s random.Random(seed).shuffle(), so it
    changes the resulting permutation even though the seed family (4000+N)
    and the 23-condition total are unchanged.
    """
    conds = ['fp32_ts', 'fp32_eager']
    if (arch, 'int8') not in M1_STRUCTURAL_EXCLUSIONS:
        conds.append('int8')
    for ratio in ('pruned30', 'pruned50', 'pruned70'):
        conds.append(ratio)
    for ratio in ('pruned30', 'pruned50', 'pruned70'):
        if include_collapsed_bnrecal or (arch, ratio) not in BNRECAL_SKIP:
            conds.append(f'{ratio}_bnrecal')
    return conds


def session_order(
    session_n: int,
    architectures: List[str],
    include_collapsed_bnrecal: bool = False,
) -> List[Tuple[str, str]]:
    """Deterministic, reproducible shuffle of (arch, condition) pairs for session_n.
    
    Seed is 4000 + session_n. For standard 3 architectures, produces exactly 23 conditions.
    """
    pairs = [
        (arch, c)
        for arch in architectures
        for c in conditions_for(arch, include_collapsed_bnrecal)
    ]
    rng = random.Random(CONDITION_SEED_BASE + session_n)
    rng.shuffle(pairs)
    return pairs


def checkpoint_for(arch: str, condition: str) -> Optional[str]:
    """Return checkpoint filename for (arch, condition) pair, or None if eager."""
    if condition == 'fp32_eager':
        return None
    if condition == 'fp32_ts':
        return f'{arch}_fp32_ts.pt'
    if condition == 'int8':
        if (arch, 'int8') in M1_STRUCTURAL_EXCLUSIONS:
            raise ValueError(f"{arch} INT8 is structurally missing on M1")
        return f'{arch}_int8.pt'
    return f'{arch}_{condition}.pt'


def preflight_m1_checkpoints(
    architectures: List[str],
    checkpoints_dir: Path,
    include_collapsed_bnrecal: bool = False,
) -> None:
    """Verify that all required checkpoints exist and can load with torch.jit.load."""
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


def run_m1_session(
    session_n: int,
    architectures: List[str],
    out_base: Union[str, Path],
    checkpoints_dir: Union[str, Path],
    condition_runner: Callable[..., Any],
    get_boottime_fn: Callable[[], float],
    check_ac_fn: Callable[[], bool] = check_ac_power_m1,
    check_therm_fn: Callable[[], bool] = check_thermal_nominal_m1,
    get_vm_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    include_collapsed_bnrecal: bool = False,
    max_session_uptime_s: float = 1800.0,
    max_condition_uptime_s: float = 10800.0,
    current_time_fn: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    """Execute one M1 CPU-block session across all randomized conditions.
    
    Enforces fresh boot (< 30 min), AC guard, thermal guard, void tracking,
    and rerun scheduling.
    """
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Pre-session Guards. A3: any GuardError here writes session_abort.json
    # before re-raising, so the abort reason is on disk, not just in the
    # propagated exception.
    try:
        check_ac_fn()
        check_therm_fn()

        boottime_sec = get_boottime_fn()
        session_start_uptime = check_uptime_guard(
            boottime_sec,
            max_uptime_seconds=max_session_uptime_s,
            current_time=current_time_fn() if current_time_fn else None,
            context="session start",
        )
    except GuardError as e:
        _write_session_abort(out_base, session_n, str(e), condition=None,
                              uptime_s=None, current_time_fn=current_time_fn)
        raise

    vm_snapshot = get_vm_fn() if get_vm_fn else None

    # 2. Preflight checkpoints
    preflight_m1_checkpoints(architectures, checkpoints_dir, include_collapsed_bnrecal)

    # 3. Determine order
    order = session_order(session_n, architectures, include_collapsed_bnrecal)

    log: Dict[str, Any] = {
        'session': session_n,
        'seed': CONDITION_SEED_BASE + session_n,
        'session_start_uptime_s': session_start_uptime,
        'total_conditions': len(order),
        'order': [f"{a}_{c}" for a, c in order],
        'vm_start_snapshot': vm_snapshot,
        'conditions': [],
    }

    void_log: Dict[str, Any] = {'session': session_n, 'voids': []}
    reruns: List[Tuple[str, str]] = []
    void_count = 0

    def process_condition(arch: str, condition: str, is_rerun: bool = False):
        nonlocal void_count
        label = f"{arch}_{condition}" + ("_rerun" if is_rerun else "")
        cond_out = out_base / label
        cond_out.mkdir(parents=True, exist_ok=True)

        # Pre-condition guards. A3: GuardError here also writes session_abort.json
        # before re-raising -- a guard failure mid-session previously propagated
        # with no file recording why.
        try:
            check_ac_fn()
            cond_uptime = check_uptime_guard(
                boottime_sec,
                max_uptime_seconds=max_condition_uptime_s,
                current_time=current_time_fn() if current_time_fn else None,
                context=f"condition {label} start",
            )
        except GuardError as e:
            _write_session_abort(out_base, session_n, str(e), condition=label,
                                  uptime_s=None, current_time_fn=current_time_fn)
            raise

        crashed = False
        flagged = False
        error_text = None

        try:
            res = condition_runner(
                arch=arch,
                condition=condition,
                out_dir=cond_out,
                checkpoints_dir=checkpoints_dir,
            )
            # Evaluate thermal or condition flags returned by runner
            if isinstance(res, dict) and res.get('thermal_flagged'):
                flagged = True
        except Exception as e:
            crashed = True
            # A3: record the exception text/traceback, not just a crashed=True flag.
            error_text = ''.join(traceback.format_exception(type(e), e, e.__traceback__))

        # Post-condition AC guard. A3: also wrapped, same as above.
        try:
            check_ac_fn()
        except GuardError as e:
            _write_session_abort(out_base, session_n, str(e), condition=label,
                                  uptime_s=None, current_time_fn=current_time_fn)
            raise

        if crashed or flagged:
            void_log['voids'].append({
                'condition': f"{arch}_{condition}",
                'is_rerun': is_rerun,
                'crashed': crashed,
                'thermal_flagged': flagged,
                'error_text': error_text,
            })
            void_count += 1
            if void_count > 2:
                (out_base / 'void_log.json').write_text(json.dumps(void_log, indent=2))
                raise RuntimeError(
                    f"Session {session_n} voided: more than 2 voided conditions ({void_count})."
                )
            if not is_rerun:
                reruns.append((arch, condition))
        else:
            log['conditions'].append(label)

    # Main pass
    for arch, condition in order:
        process_condition(arch, condition, is_rerun=False)

    # Reruns pass
    for arch, condition in reruns:
        process_condition(arch, condition, is_rerun=True)

    (out_base / 'session_log.json').write_text(json.dumps(log, indent=2))
    (out_base / 'void_log.json').write_text(json.dumps(void_log, indent=2))

    return log
