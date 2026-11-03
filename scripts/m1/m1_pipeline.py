"""M1 condition pipeline and energy post-processing.

Connects powermetrics sample streams, timing windows, overlap-splitting,
and summarization to generate windows.csv and summary.json matching x86 pilot.py.

Spec: scripts/m1/HARNESS_PLAN.md sections 1, 4, 5; pilot.py summarize() logic.
"""
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from powermetrics_parser import PowermetricsRun, split_energy_by_overlap


@dataclass
class WindowPhase:
    device: str
    arch: str
    condition: str
    size: int
    batch: int
    requested_s: float
    repeat: int
    phase: str  # 'idle_before', 'a1', 'a2', 'idle_after'
    start_t: float
    end_t: float
    batches: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_t - self.start_t)


def process_condition_energy(
    parsed_samples: List[Dict[str, Any]],
    phase_windows: List[WindowPhase],
    time_shift_s: float = 0.0,
    power_key: str = 'cpu_power_mw',
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attribute powermetrics sample energy to phase windows via overlap-splitting.
    
    Args:
        parsed_samples: list of dicts from parse_powermetrics_file, with 'arrival_t' and 'elapsed_ms'
                        or 'start' and 'end'.
        phase_windows: list of WindowPhase objects defining measurement windows.
        time_shift_s: time offset delta to apply to sample intervals (default 0.0).
        power_key: field name for power (default 'cpu_power_mw').
        
    Returns:
        (rows, summary_dict)
        rows: per-window detailed rows suitable for windows.csv
        summary_dict: aggregated condition summary matching pilot.py summary.json
    """
    # 1. Prepare sample timing with time_shift_s
    prepared_samples = []
    for s in parsed_samples:
        if 'arrival_t' in s and 'elapsed_ms' in s:
            dur = float(s['elapsed_ms']) / 1000.0
            end_t = float(s['arrival_t']) + time_shift_s
            start_t = end_t - dur
        elif 'start' in s and 'end' in s:
            start_t = float(s['start']) + time_shift_s
            end_t = float(s['end']) + time_shift_s
            dur = end_t - start_t
        else:
            dur = float(s.get('elapsed_ms', 400.0)) / 1000.0
            end_t = float(s.get('end_t', dur)) + time_shift_s
            start_t = end_t - dur

        power_val = float(s.get(power_key, 0.0))
        prepared_samples.append({
            'start': start_t,
            'end': end_t,
            'power': power_val,
        })

    # 2. Build window specs for overlap splitter
    window_specs = []
    for idx, w in enumerate(phase_windows):
        window_specs.append({
            'name': f"{w.repeat}_{w.phase}_{idx}",
            'start': w.start_t,
            'end': w.end_t,
        })

    # 3. Call overlap splitter (returns energies in mW * s = mJ)
    split_res = split_energy_by_overlap(prepared_samples, window_specs, power_key='power')
    window_energies = split_res['window_energies']

    # 4. Build per-window detailed rows
    rows = []
    for idx, w in enumerate(phase_windows):
        key = f"{w.repeat}_{w.phase}_{idx}"
        energy_mj = window_energies[key]
        energy_j = energy_mj / 1000.0
        dur_s = w.duration_s
        mean_power_w = energy_j / dur_s if dur_s > 0 else 0.0

        row = {
            'device': w.device,
            'arch': w.arch,
            'condition': w.condition,
            'size': w.size,
            'batch': w.batch,
            'requested_s': w.requested_s,
            'repeat': w.repeat,
            'phase': w.phase,
            'start_t': w.start_t,
            'end_t': w.end_t,
            'duration_s': dur_s,
            'batches': w.batches,
            'energy_j': energy_j,
            'mean_power_w': mean_power_w,
            'energy_mj': energy_mj,
        }
        rows.append(row)

    # 5. Compute condition summary
    summary = summarize_condition(rows)
    summary['overlap_diagnostics'] = {
        'total_sample_energy_j': split_res['total_sample_energy'] / 1000.0,
        'total_accounted_energy_j': split_res['total_accounted_energy'] / 1000.0,
        'outside_energy_j': split_res['outside_energy'] / 1000.0,
        'relative_discrepancy': split_res['relative_discrepancy'],
    }

    return rows, summary


def summarize_condition(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics following x86 pilot.py summarize() logic.
    
    Excludes cold first repeat from summary averages, computes paired differences
    between a1 and a2, estimated idle baseline, and gross/net J per image.
    """
    if not rows:
        return {}

    first = rows[0]
    dev = first['device']
    arch = first['arch']
    cond = first['condition']
    size = first['size']
    batch = first['batch']
    requested_s = first['requested_s']

    rep_ids = sorted({r['repeat'] for r in rows})
    pairs = []
    diffs = []
    active_energies = []
    idle_powers = []
    net_energies = []
    fractions = []
    ips_vals = []

    for rep in rep_ids:
        d = {r['phase']: r for r in rows if r['repeat'] == rep}
        if set(d.keys()) < {'idle_before', 'a1', 'a2', 'idle_after'}:
            continue

        cold = (rep == rep_ids[0] and len(rep_ids) > 1)

        # Baseline idle power across before and after idle phases
        idle_power = statistics.mean(
            d[p]['energy_j'] / d[p]['duration_s']
            for p in ['idle_before', 'idle_after']
            if d[p]['duration_s'] > 0
        )

        per_rep = []
        for p in ['a1', 'a2']:
            r = d[p]
            gross = r['energy_j']
            dur = r['duration_s']
            est_idle = idle_power * dur
            above = gross - est_idle
            r['estimated_idle_j'] = est_idle
            r['above_idle_j'] = above
            r['above_idle_fraction'] = above / gross if gross > 0 else 0.0

            imgs = r['batches'] * batch
            j_per_img = gross / imgs if imgs > 0 else float('nan')
            r['gross_j_per_image'] = j_per_img
            per_rep.append(j_per_img)

            if not cold:
                active_energies.append(gross)
                net_energies.append(above)
                if gross > 0:
                    fractions.append(above / gross)
                if dur > 0 and imgs > 0:
                    ips_vals.append(imgs / dur)

        if cold:
            continue

        idle_powers.append(idle_power)
        if len(per_rep) == 2 and not math.isnan(per_rep[0]) and not math.isnan(per_rep[1]):
            pairs.append(statistics.mean(per_rep))
            diffs.append(per_rep[1] - per_rep[0])

    n_pairs = len(pairs)
    if n_pairs < 2:
        return {
            'device': dev,
            'arch': arch,
            'condition': cond,
            'size': size,
            'batch': batch,
            'requested_s': requested_s,
            'pairs': n_pairs,
            'note': 'Insufficient pairs (< 2) for standard deviation',
        }

    mean_j_per_img = statistics.mean(pairs)
    sd_diffs = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    mde = 2.80 * sd_diffs / math.sqrt(n_pairs)
    bias = abs(statistics.mean(diffs))
    limit = bias + mde

    return {
        'device': dev,
        'arch': arch,
        'condition': cond,
        'size': size,
        'batch': batch,
        'window_s': requested_s,
        'pairs': n_pairs,
        'gross_j_per_image_mean': mean_j_per_img,
        'paired_difference_sd': sd_diffs,
        'paired_order_bias': statistics.mean(diffs),
        'approx_mde_j_per_image': mde,
        'conservative_screen_fraction': limit / mean_j_per_img if mean_j_per_img > 0 else None,
        # A4: n_pairs >= 30 is structurally unreachable on M1. Each condition runs
        # 6 repeats (HARNESS_PLAN.md / m1_session_runner.py); the first repeat is
        # always excluded as cold (see the `cold` check above), leaving at most
        # 5 pairs per condition -- nowhere close to 30. This field will always be
        # False for M1 data as a direct consequence of the fixed repeat count, not
        # a measurement result. Documented here, not changed: the field is kept
        # for schema parity with x86's summarize() (pilot.py), where 6 repeats is
        # also the actual run count in practice, so this is not M1-specific dead
        # code so much as a repeat-count fact that happens to make the threshold
        # unreachable on both platforms at the currently configured repeat count.
        'candidate_for_confirmation': bool(n_pairs >= 30 and mean_j_per_img > 0 and limit <= 0.05 * mean_j_per_img),
        'total_active_j_mean': statistics.mean(active_energies) if active_energies else None,
        'above_idle_j_mean': statistics.mean(net_energies) if net_energies else None,
        'above_idle_fraction_mean': statistics.mean(fractions) if fractions else None,
        'idle_power_w_mean': statistics.mean(idle_powers) if idle_powers else None,
        'images_per_s_mean': statistics.mean(ips_vals) if ips_vals else None,
    }
