#!/bin/bash
# Reverses scripts/m1/setup_powermetrics_sudoers.sh. Run with:
#   sudo bash scripts/m1/remove_powermetrics_sudoers.sh
#
# Removes the sudoers drop-in and the installed wrapper script. Deliberately
# leaves /var/gai-powermetrics and any data already collected there in
# place -- this script only removes the privilege grant and the executable,
# not prior measurement output.
set -euo pipefail
if [ "$EUID" -ne 0 ]; then
  echo "Run with sudo: sudo bash $0" >&2
  exit 1
fi

rm -f /etc/sudoers.d/gai-powermetrics
rm -f /usr/local/sbin/gai_powermetrics.sh

echo "Removed /etc/sudoers.d/gai-powermetrics and /usr/local/sbin/gai_powermetrics.sh."
echo "/var/gai-powermetrics and its contents were left in place."
