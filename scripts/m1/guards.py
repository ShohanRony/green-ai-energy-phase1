"""System guards and telemetry parsers for Apple M1 harness.

Implements M1-specific adaptations for:
1. AC power guard (pmset -g batt)
2. Uptime / fresh-boot guard (sysctl -n kern.boottime)
3. Thermal status guard (pmset -g therm)
4. Memory pressure telemetry (vm_stat)

Spec: scripts/m1/HARNESS_PLAN.md sections 1, 3, and 6.
"""
import re
import time
from typing import Any, Callable, Dict, Optional, Union


class GuardError(RuntimeError):
    """Raised when a pre-run or mid-run system guard condition fails."""
    pass


# -----------------------------------------------------------------------------
# 1. AC Power Guard (pmset -g batt)
# -----------------------------------------------------------------------------

def parse_pmset_batt(output: str) -> Dict[str, Any]:
    """Parse `pmset -g batt` output into structured fields.
    
    Expected patterns:
      Now drawing from 'AC Power'
       -InternalBattery-0 (id=34865251)	100%; finishing charge; 0:00 remaining present: true
      or:
      Now drawing from 'Battery Power'
       -InternalBattery-0 (id=34865251)	95%; discharging; 4:12 remaining present: true
    """
    text = output.strip()
    if not text:
        raise GuardError("Empty pmset -g batt output")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0]

    # Power source detection
    m_src = re.search(r"Now drawing from '([^']+)'", first_line)
    if not m_src:
        raise GuardError(f"Could not parse power source from: '{first_line}'")
    power_source = m_src.group(1).strip()
    is_ac = (power_source == "AC Power")

    # Battery line details
    battery_pct: Optional[float] = None
    battery_state = "unknown"
    is_present = False

    if len(lines) > 1:
        batt_line = lines[1]
        m_pct = re.search(r"(\d+)%", batt_line)
        if m_pct:
            battery_pct = float(m_pct.group(1))

        m_state = re.search(r"%;\s*([^;]+);", batt_line)
        if m_state:
            battery_state = m_state.group(1).strip().lower()

        is_present = "present: true" in batt_line.lower()

    return {
        "power_source": power_source,
        "is_ac": is_ac,
        "battery_pct": battery_pct,
        "battery_state": battery_state,
        "is_present": is_present,
        "raw": text,
    }


def check_ac_power_m1(output_or_fn: Optional[Union[str, Callable[[], str]]] = None) -> bool:
    """Verify system is operating on AC power and not discharging.
    
    Raises GuardError if AC power is not connected or if battery is discharging.
    """
    if callable(output_or_fn):
        raw_output = output_or_fn()
    elif isinstance(output_or_fn, str):
        raw_output = output_or_fn
    else:
        # Default: execute local pmset (only on macOS)
        import subprocess
        try:
            raw_output = subprocess.check_output(["pmset", "-g", "batt"], text=True)
        except Exception as e:
            raise GuardError(f"Failed to query `pmset -g batt`: {e}") from e

    parsed = parse_pmset_batt(raw_output)
    if not parsed["is_ac"]:
        raise GuardError(
            f"AC power is not connected (source: '{parsed['power_source']}'). "
            "Refusing to measure on battery power."
        )
    if parsed["battery_state"] == "discharging":
        raise GuardError(
            f"System is discharging despite AC indication (battery state: '{parsed['battery_state']}')."
        )
    return True


# -----------------------------------------------------------------------------
# 2. Uptime / Boot Guard (sysctl -n kern.boottime)
# -----------------------------------------------------------------------------

def parse_boottime(output: str) -> float:
    """Parse `sysctl -n kern.boottime` output into boot timestamp (seconds epoch).
    
    Expected format:
      `{ sec = 1756405599, usec = 852783 } Sat Aug 29 02:26:39 2026`
      or plain integer/float epoch string `1756405599`.
    """
    text = output.strip()
    if not text:
        raise GuardError("Empty kern.boottime output")

    m = re.search(r"sec\s*=\s*(\d+)", text)
    if m:
        sec = float(m.group(1))
        m_usec = re.search(r"usec\s*=\s*(\d+)", text)
        if m_usec:
            sec += float(m_usec.group(1)) / 1_000_000.0
        return sec

    # Fallback to direct numeric parse
    try:
        return float(text)
    except ValueError as e:
        raise GuardError(f"Could not parse boottime from '{text}': {e}") from e


def compute_uptime(boottime_sec: float, current_time: Optional[float] = None) -> float:
    """Compute uptime in seconds from boot epoch timestamp."""
    now = time.time() if current_time is None else current_time
    uptime = now - boottime_sec
    if uptime < 0:
        raise GuardError(f"Negative uptime computed: current_time ({now}) < boottime ({boottime_sec})")
    return uptime


def check_uptime_guard(
    boottime_sec: float,
    max_uptime_seconds: float = 1800.0,
    current_time: Optional[float] = None,
    context: str = "session start",
) -> float:
    """Verify system uptime is within the required fresh-boot or condition window.
    
    Default max_uptime_seconds = 1800.0 (30 minutes for session start).
    For per-condition limit, caller passes 10800.0 (180 minutes).
    Returns the computed uptime in seconds.
    """
    uptime_s = compute_uptime(boottime_sec, current_time=current_time)
    if uptime_s > max_uptime_seconds:
        raise GuardError(
            f"System uptime is {uptime_s/60:.1f} min (> {max_uptime_seconds/60:.1f} min) "
            f"at {context}. Reboot required."
        )
    return uptime_s


# -----------------------------------------------------------------------------
# 3. Thermal Guard (pmset -g therm)
# -----------------------------------------------------------------------------

_THERM_NOMINAL_LINES = {
    "Note: No thermal warning level has been recorded",
    "Note: No performance warning level has been recorded",
    "Note: No CPU power status has been recorded",
}


def parse_pmset_therm(output: str) -> Dict[str, Any]:
    """Parse `pmset -g therm` output for throttling or warning states.

    FAIL CLOSED (A1): nominal is returned ONLY when the output consists of
    exactly the three expected "No ... has been recorded" notes, in any order,
    and nothing else (no extra thermal/limit/pressure line of any kind,
    no "Current pressure level:" line, no unrecognized text). Any other
    content -- including a genuinely nominal-looking but differently-worded
    line, an unknown format, or empty output -- raises GuardError rather than
    defaulting to nominal. The previous version defaulted to nominal whenever
    none of its known "bad" substrings matched, which is fail-open: unknown
    output was silently treated as safe. This is the opposite.
    """
    text = output.strip()
    if not text:
        raise GuardError("Empty pmset -g therm output (fail closed)")

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if set(lines) == _THERM_NOMINAL_LINES and len(lines) == len(_THERM_NOMINAL_LINES):
        return {
            "is_nominal": True,
            "has_thermal_warning": False,
            "has_perf_warning": False,
            "has_power_status": False,
            "pressure_level": "Nominal",
            "raw": text,
        }

    raise GuardError(
        f"Thermal guard: unrecognized or non-nominal pmset -g therm output "
        f"(fail closed, not defaulting to nominal): {text!r}"
    )


def check_thermal_nominal_m1(output_or_fn: Optional[Union[str, Callable[[], str]]] = None) -> bool:
    """Verify thermal state is nominal. Raises GuardError if throttled or warned."""
    if callable(output_or_fn):
        raw_output = output_or_fn()
    elif isinstance(output_or_fn, str):
        raw_output = output_or_fn
    else:
        import subprocess
        try:
            raw_output = subprocess.check_output(["pmset", "-g", "therm"], text=True)
        except Exception as e:
            raise GuardError(f"Failed to query `pmset -g therm`: {e}") from e

    parsed = parse_pmset_therm(raw_output)
    if not parsed["is_nominal"]:
        raise GuardError(
            f"Thermal guard failed: not nominal. Details: {parsed['raw']}"
        )
    return True


# -----------------------------------------------------------------------------
# 4. Memory Telemetry (vm_stat)
# -----------------------------------------------------------------------------

def parse_vm_stat(output: str) -> Dict[str, Any]:
    """Parse macOS `vm_stat` page counts and compute memory metrics.
    
    Apple Silicon M1 page size is 16,384 bytes (16 KB).
    """
    text = output.strip()
    if not text:
        raise GuardError("Empty vm_stat output")

    # Detect page size
    m_page = re.search(r"page size of (\d+) bytes", text)
    page_size = int(m_page.group(1)) if m_page else 16384

    metrics = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        k_clean = key.strip().strip('"').lower().replace(" ", "_").replace("-", "_")
        m_num = re.search(r"(\d+)", val)
        if m_num:
            metrics[k_clean] = int(m_num.group(1))

    pages_free = metrics.get("pages_free", 0)
    pages_active = metrics.get("pages_active", 0)
    pages_inactive = metrics.get("pages_inactive", 0)
    pages_speculative = metrics.get("pages_speculative", 0)
    pages_wired = metrics.get("pages_wired_down", 0)
    pages_occupied_compressor = metrics.get("pages_occupied_by_compressor", 0)

    # A2: swapins/swapouts/compressions/decompressions are cumulative event
    # counters since boot (not page counts), present in vm_stat output on
    # this Mac (confirmed against a real capture) but previously dropped.
    swapins = metrics.get("swapins", 0)
    swapouts = metrics.get("swapouts", 0)
    compressions = metrics.get("compressions", 0)
    decompressions = metrics.get("decompressions", 0)

    # Compute MB (1024 * 1024 bytes)
    bytes_to_mb = page_size / (1024.0 * 1024.0)
    free_mb = (pages_free + pages_speculative) * bytes_to_mb
    active_mb = pages_active * bytes_to_mb
    inactive_mb = pages_inactive * bytes_to_mb
    wired_mb = pages_wired * bytes_to_mb
    compressed_mb = pages_occupied_compressor * bytes_to_mb

    return {
        "page_size_bytes": page_size,
        "pages_free": pages_free,
        "pages_speculative": pages_speculative,
        "pages_active": pages_active,
        "pages_inactive": pages_inactive,
        "pages_wired": pages_wired,
        "pages_occupied_by_compressor": pages_occupied_compressor,
        "swapins": swapins,
        "swapouts": swapouts,
        "compressions": compressions,
        "decompressions": decompressions,
        "available_mb": free_mb + inactive_mb,
        "free_mb": free_mb,
        "active_mb": active_mb,
        "inactive_mb": inactive_mb,
        "wired_mb": wired_mb,
        "compressed_mb": compressed_mb,
    }
