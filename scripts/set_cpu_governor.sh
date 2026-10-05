#!/bin/bash
# Installed to /usr/local/sbin/set_cpu_governor.sh, root-owned, mode 0755.
# Lock all CPU cores' scaling_governor to a fixed value. This is the scoped sudoers
# target for the power-resilience watchdog (needs to restore performance unattended
# after AC loss, without an interactive password prompt) -- see
# scripts/setup_governor_sudoers.sh for how it's installed and why a fixed-path
# wrapper script is used instead of granting sudo on tee/echo directly (sudoers
# can't safely match a shell-glob-expanded argument list like cpu star .../scaling_governor).
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: set_cpu_governor.sh GOVERNOR" >&2
  exit 1
fi
GOV="$1"
case "$GOV" in
  performance) ;;
  powersave) ;;
  ondemand) ;;
  conservative) ;;
  schedutil) ;;
  *) echo "Refusing unknown governor: $GOV" >&2; exit 1 ;;
esac
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo "$GOV" > "$f"
done
