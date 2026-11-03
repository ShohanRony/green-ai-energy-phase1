"""Parser for Apple Silicon powermetrics output files and energy attribution.

Spec: scripts/m1/HARNESS_PLAN.md sections 4 and 5.
Parses output from `powermetrics --samplers cpu_power,thermal`.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class MalformedPowermetricsError(ValueError):
    """Raised when powermetrics output is missing required fields or corrupted."""
    pass


class PowermetricsSample(dict):
    """Dictionary representing a single parsed powermetrics sample block."""
    pass


class PowermetricsRun(list):
    """List of PowermetricsSample dicts with file-level header metadata."""
    def __init__(self, samples: List[Dict[str, Any]], header: Dict[str, Any]):
        super().__init__(samples)
        self.header = header

    def __getitem__(self, index):
        res = super().__getitem__(index)
        if isinstance(index, slice):
            return PowermetricsRun(res, self.header)
        return res


def parse_powermetrics_header(content_or_path: Union[str, Path]) -> Dict[str, str]:
    """Parse file-level header metadata from powermetrics output."""
    if isinstance(content_or_path, Path) or (isinstance(content_or_path, str) and '\n' not in content_or_path and Path(content_or_path).exists()):
        with open(content_or_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    else:
        text = str(content_or_path)

    header: Dict[str, str] = {}
    
    # Machine model
    m = re.search(r'^Machine model:\s*(.*?)$', text, re.MULTILINE)
    if not m:
        raise MalformedPowermetricsError("Missing 'Machine model:' in powermetrics header")
    header['machine_model'] = m.group(1).strip()
    header['Machine model'] = header['machine_model']

    # OS version
    m = re.search(r'^OS version:\s*(.*?)$', text, re.MULTILINE)
    if not m:
        raise MalformedPowermetricsError("Missing 'OS version:' in powermetrics header")
    header['os_version'] = m.group(1).strip()
    header['OS version'] = header['os_version']

    # Boot arguments (can be empty)
    m = re.search(r'^Boot arguments:\s*(.*?)$', text, re.MULTILINE)
    if m:
        header['boot_arguments'] = m.group(1).strip()
        header['Boot arguments'] = header['boot_arguments']
    else:
        header['boot_arguments'] = ''
        header['Boot arguments'] = ''

    # Boot time
    m = re.search(r'^Boot time:\s*(.*?)$', text, re.MULTILINE)
    if not m:
        raise MalformedPowermetricsError("Missing 'Boot time:' in powermetrics header")
    header['boot_time'] = m.group(1).strip()
    header['Boot time'] = header['boot_time']

    return header


def _parse_sample_block(block_text: str, header_line: str, sample_index: int) -> Dict[str, Any]:
    """Parse a single sample block. Raises MalformedPowermetricsError on missing fields."""
    sample: Dict[str, Any] = {}

    # 1. Header line: *** Sampled system activity (Tue Nov  3 21:38:52 2026 +0900) (1005.14ms elapsed) ***
    m_head = re.search(
        r'\*\*\* Sampled system activity \((.*?)\)\s*\(([\d.]+)ms elapsed\) \*\*\*',
        header_line
    )
    if not m_head:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Malformed sample header line: {header_line}")

    ts_str = m_head.group(1).strip()
    elapsed_str = m_head.group(2).strip()

    try:
        sample['timestamp'] = datetime.strptime(ts_str, "%a %b %d %H:%M:%S %Y %z")
    except ValueError as e:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Invalid timestamp format '{ts_str}': {e}") from e

    try:
        sample['elapsed_ms'] = float(elapsed_str)
    except ValueError as e:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Invalid elapsed_ms '{elapsed_str}': {e}") from e

    # 2. Power values (mW)
    m_cpu = re.search(r'^CPU Power:\s*(\d+)\s*mW', block_text, re.MULTILINE)
    if not m_cpu:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'CPU Power:'")
    sample['cpu_power_mw'] = int(m_cpu.group(1))

    m_gpu = re.search(r'^GPU Power:\s*(\d+)\s*mW', block_text, re.MULTILINE)
    if not m_gpu:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'GPU Power:'")
    sample['gpu_power_mw'] = int(m_gpu.group(1))

    m_ane = re.search(r'^ANE Power:\s*(\d+)\s*mW', block_text, re.MULTILINE)
    if not m_ane:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'ANE Power:'")
    sample['ane_power_mw'] = int(m_ane.group(1))

    m_comb = re.search(r'^Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW', block_text, re.MULTILINE)
    if not m_comb:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'Combined Power:'")
    sample['combined_power_mw'] = int(m_comb.group(1))

    # 3. Thermal pressure
    m_therm = re.search(r'^\s*Current pressure level:\s*([A-Za-z0-9_-]+)', block_text, re.MULTILINE)
    if not m_therm:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'Current pressure level:'")
    sample['thermal_pressure'] = m_therm.group(1).strip()

    # 4. E-Cluster metrics
    m_efreq = re.search(r'^E-Cluster HW active frequency:\s*(\d+)\s*MHz', block_text, re.MULTILINE)
    if not m_efreq:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'E-Cluster HW active frequency:'")
    sample['e_cluster_active_freq_mhz'] = int(m_efreq.group(1))

    m_eact = re.search(r'^E-Cluster HW active residency:\s*([\d.]+)%', block_text, re.MULTILINE)
    if not m_eact:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'E-Cluster HW active residency:'")
    sample['e_cluster_active_residency_pct'] = float(m_eact.group(1))

    m_eidle = re.search(r'^E-Cluster idle residency:\s*([\d.]+)%', block_text, re.MULTILINE)
    if not m_eidle:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'E-Cluster idle residency:'")
    sample['e_cluster_idle_residency_pct'] = float(m_eidle.group(1))

    # 5. P-Cluster metrics
    m_pfreq = re.search(r'^P-Cluster HW active frequency:\s*(\d+)\s*MHz', block_text, re.MULTILINE)
    if not m_pfreq:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'P-Cluster HW active frequency:'")
    sample['p_cluster_active_freq_mhz'] = int(m_pfreq.group(1))

    m_pact = re.search(r'^P-Cluster HW active residency:\s*([\d.]+)%', block_text, re.MULTILINE)
    if not m_pact:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'P-Cluster HW active residency:'")
    sample['p_cluster_active_residency_pct'] = float(m_pact.group(1))

    m_pidle = re.search(r'^P-Cluster idle residency:\s*([\d.]+)%', block_text, re.MULTILINE)
    if not m_pidle:
        raise MalformedPowermetricsError(f"Sample {sample_index}: Missing or malformed 'P-Cluster idle residency:'")
    sample['p_cluster_idle_residency_pct'] = float(m_pidle.group(1))

    # 6. Per-CPU frequencies and P-core mean frequency
    pcore_freqs = []
    for c in range(4, 8):
        m_c = re.search(rf'^CPU {c} frequency:\s*(\d+)\s*MHz', block_text, re.MULTILINE)
        if m_c:
            freq = int(m_c.group(1))
            sample[f'cpu_{c}_freq_mhz'] = freq
            pcore_freqs.append(freq)
    if pcore_freqs:
        sample['pcore_mean_freq_mhz'] = sum(pcore_freqs) / len(pcore_freqs)
        sample['cpu4_7_mean_mhz'] = sample['pcore_mean_freq_mhz']
    elif 'p_cluster_active_freq_mhz' in sample:
        sample['pcore_mean_freq_mhz'] = float(sample['p_cluster_active_freq_mhz'])
        sample['cpu4_7_mean_mhz'] = sample['pcore_mean_freq_mhz']

    return sample


def parse_powermetrics_file(path_or_content: Union[str, Path]) -> PowermetricsRun:
    """Parse a full powermetrics output file or string.
    
    Returns a PowermetricsRun (list of sample dicts) with `.header` containing
    the parsed file header metadata.
    Raises MalformedPowermetricsError on any missing or malformed fields.
    """
    if isinstance(path_or_content, Path) or (isinstance(path_or_content, str) and '\n' not in path_or_content and Path(path_or_content).exists()):
        with open(path_or_content, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    else:
        text = str(path_or_content)

    header = parse_powermetrics_header(text)

    # Split by sample activity header
    header_regex = re.compile(r'(\*\*\* Sampled system activity .*? \([\d.]+ms elapsed\) \*\*\*)')
    parts = header_regex.split(text)
    # parts[0] is preamble / header
    # parts[1] is header line 1, parts[2] is body 1
    # parts[3] is header line 2, parts[4] is body 2, etc.

    if len(parts) < 3:
        raise MalformedPowermetricsError("No valid '*** Sampled system activity' sample blocks found in file")

    samples: List[Dict[str, Any]] = []
    sample_idx = 1
    for i in range(1, len(parts), 2):
        head_line = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ''
        sample_dict = _parse_sample_block(body, head_line, sample_idx)
        # Also copy header fields for convenience
        sample_dict['machine_model'] = header['machine_model']
        sample_dict['os_version'] = header['os_version']
        sample_dict['boot_time'] = header['boot_time']
        samples.append(PowermetricsSample(sample_dict))
        sample_idx += 1

    return PowermetricsRun(samples, header)


def split_energy_by_overlap(
    samples: List[Dict[str, Any]],
    windows: List[Dict[str, Any]],
    power_key: str = 'power',
) -> Dict[str, Any]:
    """Split per-sample energy across measurement windows by time overlap.
    
    Energy for sample k = power_k * elapsed_k (in mJ or consistent units).
    Distributed in proportion to the time overlap with each window:
    overlap = max(0, min(sample_end, win_end) - max(sample_start, win_start)).
    
    Args:
        samples: list of dicts. Each must provide:
            - 'start': float (start time, seconds)
            - 'end': float (end time, seconds)
            - power_key: float (power, e.g. mW)
            (or 'elapsed_ms' / 'duration_s')
        windows: list of dicts. Each must provide:
            - 'name': str
            - 'start': float (seconds)
            - 'end': float (seconds)
        power_key: key for power in sample dict (default 'power')
        
    Returns:
        dict with:
            - 'window_energies': dict[name -> float]
            - 'window_mean_powers': dict[name -> float]
            - 'outside_energy': float
            - 'total_sample_energy': float
            - 'total_accounted_energy': float
            - 'relative_discrepancy': float (abs(accounted - total) / total)
    """
    total_sample_energy = 0.0
    prepared_samples = []

    for idx, s in enumerate(samples):
        if 'start' in s and 'end' in s:
            s_start = float(s['start'])
            s_end = float(s['end'])
            dur = s_end - s_start
        elif 'arrival_t' in s and 'elapsed_ms' in s:
            dur = float(s['elapsed_ms']) / 1000.0
            s_end = float(s['arrival_t'])
            s_start = s_end - dur
        elif 'elapsed_ms' in s:
            dur = float(s['elapsed_ms']) / 1000.0
            s_start = 0.0
            s_end = dur
        else:
            raise ValueError(f"Sample {idx} missing timing boundaries ('start'/'end' or 'arrival_t'/'elapsed_ms')")

        if dur <= 0:
            raise ValueError(f"Sample {idx} has non-positive duration: {dur}")

        p = float(s.get(power_key, s.get('cpu_power_mw', 0.0)))
        energy = p * dur
        total_sample_energy += energy
        prepared_samples.append({
            'start': s_start,
            'end': s_end,
            'dur': dur,
            'power': p,
            'energy': energy,
        })

    window_energies = {w['name']: 0.0 for w in windows}
    window_durations = {w['name']: float(w['end']) - float(w['start']) for w in windows}

    total_window_energy = 0.0
    for ps in prepared_samples:
        sample_allocated = 0.0
        for w in windows:
            w_start = float(w['start'])
            w_end = float(w['end'])
            overlap = max(0.0, min(ps['end'], w_end) - max(ps['start'], w_start))
            if overlap > 0.0:
                frac = overlap / ps['dur']
                alloc = ps['energy'] * frac
                window_energies[w['name']] += alloc
                sample_allocated += alloc
        total_window_energy += sample_allocated

    outside_energy = total_sample_energy - total_window_energy
    total_accounted = total_window_energy + outside_energy

    rel_disc = (
        abs(total_accounted - total_sample_energy) / total_sample_energy
        if total_sample_energy > 0
        else 0.0
    )

    window_mean_powers = {}
    for w in windows:
        name = w['name']
        dur = window_durations[name]
        window_mean_powers[name] = window_energies[name] / dur if dur > 0 else 0.0

    return {
        'window_energies': window_energies,
        'window_mean_powers': window_mean_powers,
        'outside_energy': outside_energy,
        'total_sample_energy': total_sample_energy,
        'total_accounted_energy': total_accounted,
        'relative_discrepancy': rel_disc,
    }
