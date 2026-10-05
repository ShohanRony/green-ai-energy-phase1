"""Shared power-interruption marker protocol used by power_watchdog.py and the
long-running scripts it protects (train_baseline.py, prune_full_grid.py, pilot.py).

Lifecycle owned by the watchdog, not by the scripts:
- AC lost: watchdog writes STOP_MARKER, removes RESTORED_MARKER.
- AC restored: watchdog removes STOP_MARKER, writes RESTORED_MARKER, prints RESUME_HINT if present.
Scripts only ever read stop_requested() and write a resume hint when they checkpoint-and-exit.
"""
from pathlib import Path

STATE_DIR = Path('.power_state')
STOP_MARKER = STATE_DIR / 'stop'
RESTORED_MARKER = STATE_DIR / 'restored'
RESUME_HINT = STATE_DIR / 'resume_hint.txt'


def stop_requested() -> bool:
    return STOP_MARKER.exists()


def write_resume_hint(command: str) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    RESUME_HINT.write_text(command.strip() + '\n')


def clear_resume_hint() -> None:
    RESUME_HINT.unlink(missing_ok=True)
