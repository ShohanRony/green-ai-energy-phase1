"""Background power-state watchdog for loadshedding resilience.

Run with: nohup python3 power_watchdog.py > power_watchdog.log 2>&1 &

Polls the AC online file every POLL_S seconds. On AC->battery it writes the
stop marker that train_baseline.py / prune_full_grid.py / pilot.py check
between safe points (epoch end, ratio end, rep end). On battery->AC it
clears the marker and prints whatever resume hint the interrupted script
left behind -- it does NOT relaunch anything itself (see
stage3-implementation-brief-adjacent instructions: resume is a manual,
confirmed step, not automatic, since power can flicker before it's stable
and a measurement run specifically needs the CPU governor settled back to
`performance`, which pilot.py's own startup check already enforces).

AC path confirmed on this system (not assumed): /sys/class/power_supply/ACAD/online.
If this script is ever run on different hardware, re-confirm with
`ls /sys/class/power_supply/` and `cat .../*/type` first -- the name isn't portable.
"""
import subprocess, sys, time
from pathlib import Path

import power_state

AC_ONLINE = Path('/sys/class/power_supply/ACAD/online')
POLL_S = 7
GOVERNOR_SCRIPT = '/usr/local/sbin/set_cpu_governor.sh'  # installed by scripts/setup_governor_sudoers.sh


def on_ac(path: Path = AC_ONLINE) -> bool:
    return path.read_text().strip() == '1'


def restore_governor() -> bool:
    """Best-effort, non-blocking: `sudo -n` fails immediately (never prompts for a
    password) if scripts/setup_governor_sudoers.sh hasn't been run yet, so this is
    safe to call unconditionally -- it just logs and falls back to a manual instruction."""
    try:
        subprocess.run(['sudo', '-n', GOVERNOR_SCRIPT, 'performance'],
                        check=True, capture_output=True, timeout=10, text=True)
        print('[power_watchdog] CPU governor restored to performance', flush=True)
        return True
    except Exception as e:
        print(f'[power_watchdog] could not auto-restore the governor ({e}); run manually: '
              f'sudo {GOVERNOR_SCRIPT} performance (or scripts/setup_governor_sudoers.sh '
              f"hasn't been run yet)", flush=True)
        return False


def classify_transition(was_on_ac: bool, now_on_ac: bool) -> str | None:
    """Pure decision logic, separated from I/O so it's unit-testable without touching
    real hardware state. Returns 'lost', 'restored', or None (no transition)."""
    if was_on_ac == now_on_ac:
        return None
    return 'restored' if now_on_ac else 'lost'


def handle_transition(kind: str, ts: str) -> None:
    if kind == 'lost':
        power_state.STOP_MARKER.write_text(ts)
        power_state.RESTORED_MARKER.unlink(missing_ok=True)
        print(f'[power_watchdog] {ts} AC lost -- stop requested', flush=True)
    else:
        power_state.STOP_MARKER.unlink(missing_ok=True)
        power_state.RESTORED_MARKER.write_text(ts)
        print(f'[power_watchdog] {ts} AC restored', flush=True)
        restore_governor()
        if power_state.RESUME_HINT.exists():
            print(f'[power_watchdog] ready to resume: {power_state.RESUME_HINT.read_text().strip()}', flush=True)
        else:
            print('[power_watchdog] no interrupted job found (no resume hint on file)', flush=True)


def main():
    if not AC_ONLINE.exists():
        sys.exit(f'{AC_ONLINE} not found -- confirm the AC power-supply path on this system '
                  f"first (`ls /sys/class/power_supply/`), don't assume ACAD.")
    power_state.STATE_DIR.mkdir(exist_ok=True)
    was_on_ac = on_ac()
    print(f'[power_watchdog] starting, on_ac={was_on_ac}, polling every {POLL_S}s', flush=True)
    while True:
        time.sleep(POLL_S)
        now_on_ac = on_ac()
        kind = classify_transition(was_on_ac, now_on_ac)
        if kind:
            handle_transition(kind, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        was_on_ac = now_on_ac


if __name__ == '__main__':
    main()
