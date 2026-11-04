"""M1 condition pipeline and energy post-processing.

Connects powermetrics sample streams, timing windows, overlap-splitting,
and summarization to generate windows.csv and summary.json matching x86 pilot.py.

Spec: scripts/m1/HARNESS_PLAN.md sections 1, 4, 5; pilot.py summarize() logic.
"""
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def read_thermal_thresholds_m1(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Read and validate M1 thermal thresholds configuration.
    
    Verifies:
    - config/thermal_thresholds_m1.json exists
    - T_mhz is a positive number
    - rule_version == 'm1-v1'
    - source is a non-empty string
    - metric == 'pcore_mean_freq_mhz'
    - flag_rule == 'window_mean_below_T'
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'thermal_thresholds_m1.json'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise RuntimeError(f"M1 runs require config/thermal_thresholds_m1.json for thermal guard (not found at {config_path}).")

    data = json.loads(config_path.read_text(encoding='utf-8'))
    t_mhz = data.get('T_mhz')
    version = data.get('rule_version')
    source = data.get('source')
    metric = data.get('metric')
    flag_rule = data.get('flag_rule')

    if not (isinstance(t_mhz, (int, float)) and t_mhz > 0):
        raise RuntimeError("Invalid or missing T_mhz in thermal_thresholds_m1.json")
    if version != "m1-v1":
        raise RuntimeError("Invalid or missing rule_version in thermal_thresholds_m1.json (must be 'm1-v1')")
    if not (isinstance(source, str) and source):
        raise RuntimeError("Invalid or missing source in thermal_thresholds_m1.json")
    if metric != "pcore_mean_freq_mhz":
        raise RuntimeError("Invalid or missing metric in thermal_thresholds_m1.json (must be 'pcore_mean_freq_mhz')")
    if flag_rule != "window_mean_below_T":
        raise RuntimeError("Invalid or missing flag_rule in thermal_thresholds_m1.json (must be 'window_mean_below_T')")

    return data


def evaluate_window_thermal_flag(
    pcore_freq_mhz: Optional[float],
    phase: str,
    repeat: int,
    threshold_t_mhz: float,
    thermal_pressures: Optional[List[str]] = None,
) -> Optional[str]:
    """Evaluate thermal flag for a single window.

    Per frozen design: only a1/a2 phases with repeat > 0 are evaluated for throttling.
    Returns 'True' if throttled (mean freq < T) OR if any sample overlapping the
    window reports a thermal pressure other than "Nominal" (independent of
    frequency -- a non-Nominal pressure reading is itself cause for exclusion,
    not just a frequency drop; added because the calibration evidence showed
    pressure staying "Nominal" through a real frequency decline, i.e. pressure
    alone under-detects, but a non-Nominal reading should still never be
    waved through just because frequency happened to stay above T).
    Returns 'False' if nominal (mean freq >= T and all overlapping samples,
    if any pressure data was passed, are "Nominal").
    Returns '' if active window with repeat > 0 is missing frequency data
    (triggers evaluation error) -- checked before the pressure check, since a
    missing primary metric is still a missing evaluation regardless of
    pressure.
    Returns None for idle phases or cold repeat 0.
    """
    if phase in {"a1", "a2"} and repeat > 0:
        if pcore_freq_mhz is None or math.isnan(pcore_freq_mhz):
            return ""  # Missing evaluation -> triggers error when checked
        if thermal_pressures and any(p != "Nominal" for p in thermal_pressures):
            return "True"
        return "True" if pcore_freq_mhz < threshold_t_mhz else "False"
    return None


def check_condition_thermal_flags(
    rows_or_csv: Union[List[Dict[str, Any]], str, Path],
    csv_path_for_err: Optional[str] = None,
) -> bool:
    """Evaluate whether any active measurement window in the condition was thermally flagged.
    
    Mirrors run_cpu_block_session.py:170-197:
    - Checks only phases a1 and a2 with repeat > 0.
    - Requires non-empty thermal_flag evaluation on all qualifying windows;
      raises RuntimeError if evaluation is missing.
    - Returns True if any window has thermal_flag == 'True' (condition voided).
    - Returns False if all qualifying windows have thermal_flag == 'False'.
    """
    if isinstance(rows_or_csv, (str, Path)):
        csv_path = Path(rows_or_csv)
        if not csv_path.exists():
            raise RuntimeError(f"windows.csv is missing in {csv_path.parent if csv_path.is_file() else csv_path}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'thermal_flag' not in reader.fieldnames:
                raise RuntimeError(f"thermal_flag column missing from {csv_path}")
            for row in reader:
                if row.get('phase') in {"a1", "a2"} and int(row.get('repeat', 0)) > 0:
                    flag_val = row.get('thermal_flag', '')
                    if not flag_val or flag_val == 'None':
                        raise RuntimeError(f"Evaluation missing (empty thermal_flag) in {csv_path}")
                    if flag_val == 'True':
                        return True
        return False

    # List of dicts
    for row in rows_or_csv:
        phase = row.get('phase')
        repeat = int(row.get('repeat', 0))
        if phase in {"a1", "a2"} and repeat > 0:
            if 'thermal_flag' not in row:
                raise RuntimeError(f"Evaluation missing (empty thermal_flag) in {csv_path_for_err or 'windows'}")
            flag_val = row.get('thermal_flag')
            if flag_val is None or str(flag_val).strip() in ('', 'None'):
                raise RuntimeError(f"Evaluation missing (empty thermal_flag) in {csv_path_for_err or 'windows'}")
            if flag_val is True or str(flag_val).strip() == 'True':
                return True
    return False


def write_windows_csv(rows: List[Dict[str, Any]], out_path: Union[str, Path]) -> None:
    """Write window rows to windows.csv matching x86 format with thermal_flag."""
    if not rows:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Collect all fieldnames from all rows preserving encounter order
    seen = set()
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    if 'thermal_flag' not in seen:
        fieldnames.append('thermal_flag')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summary_dict: Dict[str, Any], out_path: Union[str, Path]) -> None:
    """Write condition summary row to summary.csv matching x86 format."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter out nested dicts/lists from summary_dict for the flat CSV row
    flat_row = {
        k: v for k, v in summary_dict.items()
        if not isinstance(v, (dict, list, tuple))
    }
    fieldnames = list(flat_row.keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(flat_row)



def process_condition_energy(
    parsed_samples: List[Dict[str, Any]],
    phase_windows: List[WindowPhase],
    time_shift_s: float = 0.0,
    power_key: str = 'cpu_power_mw',
    thermal_thresholds: Optional[Dict[str, Any]] = None,
    require_thermal_guard: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attribute powermetrics sample energy to phase windows via overlap-splitting.
    
    Args:
        parsed_samples: list of dicts from parse_powermetrics_file, with 'arrival_t' and 'elapsed_ms'
                        or 'start' and 'end'.
        phase_windows: list of WindowPhase objects defining measurement windows.
        time_shift_s: time offset delta to apply to sample intervals (default 0.0).
        power_key: field name for power (default 'cpu_power_mw').
        thermal_thresholds: optional thresholds dict from read_thermal_thresholds_m1.
        require_thermal_guard: if True and thermal_thresholds is None, loads from config.
        
    Returns:
        (rows, summary_dict)
        rows: per-window detailed rows suitable for windows.csv
        summary_dict: aggregated condition summary matching pilot.py summary.json
    """
    # 1. Prepare sample timing with time_shift_s
    prepared_samples = []
    for s in parsed_samples:
        arr_val = s.get('arrival_t')
        if arr_val is None:
            arr_val = s.get('arrival_perf')
        if arr_val is not None and 'elapsed_ms' in s:
            dur = float(s['elapsed_ms']) / 1000.0
            end_t = float(arr_val) + time_shift_s
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
        freq_val = s.get('pcore_mean_freq_mhz')
        if freq_val is None:
            freq_val = s.get('cpu4_7_mean_mhz')
        if freq_val is None:
            freq_val = s.get('p_cluster_active_freq_mhz')
        if freq_val is not None:
            freq_val = float(freq_val)
        pressure_val = s.get('thermal_pressure')

        prepared_samples.append({
            'start': start_t,
            'end': end_t,
            'power': power_val,
            'freq': freq_val,
            'thermal_pressure': pressure_val,
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

    # Load thermal thresholds if requested
    t_thresh = thermal_thresholds
    if t_thresh is None and require_thermal_guard:
        t_thresh = read_thermal_thresholds_m1()

    # 4. Build per-window detailed rows
    rows = []
    for idx, w in enumerate(phase_windows):
        key = f"{w.repeat}_{w.phase}_{idx}"
        energy_mj = window_energies[key]
        energy_j = energy_mj / 1000.0
        dur_s = w.duration_s
        mean_power_w = energy_j / dur_s if dur_s > 0 else 0.0

        # Frequency attribution from overlapping samples
        overlapping_freqs = [
            ps['freq'] for ps in prepared_samples
            if ps.get('freq') is not None and ps['end'] > w.start_t and ps['start'] < w.end_t
        ]
        pcore_mean_freq = (
            sum(overlapping_freqs) / len(overlapping_freqs)
            if overlapping_freqs else None
        )

        # Thermal pressure of every sample overlapping the window
        overlapping_pressures = [
            ps['thermal_pressure'] for ps in prepared_samples
            if ps.get('thermal_pressure') is not None and ps['end'] > w.start_t and ps['start'] < w.end_t
        ]

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
            'pcore_mean_freq_mhz': pcore_mean_freq,
            'thermal_pcore_mean_freq_mhz': pcore_mean_freq,
        }

        if t_thresh is not None:
            t_mhz = t_thresh.get('T_mhz', 2881.11)
            row['thermal_flag'] = evaluate_window_thermal_flag(
                pcore_mean_freq, w.phase, w.repeat, t_mhz, overlapping_pressures
            )

        rows.append(row)

    # 5. Compute condition summary
    summary = summarize_condition(rows)
    summary['overlap_diagnostics'] = {
        'total_sample_energy_j': split_res['total_sample_energy'] / 1000.0,
        'total_accounted_energy_j': split_res['total_accounted_energy'] / 1000.0,
        'outside_energy_j': split_res['outside_energy'] / 1000.0,
        'relative_discrepancy': split_res['relative_discrepancy'],
    }

    if t_thresh is not None:
        is_flagged = check_condition_thermal_flags(rows)
        summary['thermal_flagged'] = is_flagged
        summary['thermal_threshold_t_mhz'] = t_thresh.get('T_mhz', 2881.11)
        summary['condition_voided'] = is_flagged

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
