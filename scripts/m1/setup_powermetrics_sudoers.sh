#!/bin/bash
# One-time setup on the M1 Mac. Run with: sudo bash scripts/m1/setup_powermetrics_sudoers.sh
#
# Installs scripts/m1/gai_powermetrics.sh as a root-owned wrapper at
# /usr/local/sbin/gai_powermetrics.sh, creates its fixed output directory
# /var/gai-powermetrics, then grants the real (non-root) user NOPASSWD sudo
# on exactly that one wrapper path via a validated
# /etc/sudoers.d/gai-powermetrics drop-in.
#
# Narrowly scoped, same pattern as scripts/setup_governor_sudoers.sh on the
# x86 side -- not a blanket NOPASSWD, and the wrapper itself (not sudoers)
# enforces argument shape/range, so sudoers only needs to match the fixed
# script path with no arguments pinned.
set -euo pipefail
if [ "$EUID" -ne 0 ]; then
  echo "Run with sudo: sudo bash $0" >&2
  exit 1
fi
REAL_USER="${SUDO_USER:?run via sudo, not as root directly, so the real username is known}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUDOERS_LINE="$REAL_USER ALL=(root) NOPASSWD: /usr/local/sbin/gai_powermetrics.sh"
TMP_SUDOERS="$(mktemp)"
echo "$SUDOERS_LINE" > "$TMP_SUDOERS"
if ! visudo -c -f "$TMP_SUDOERS"; then
  echo "Generated sudoers line failed validation, aborting (nothing installed):" >&2
  cat "$TMP_SUDOERS" >&2
  rm -f "$TMP_SUDOERS"
  exit 1
fi

install -d -o root -g wheel -m 0755 /usr/local/sbin
install -o root -g wheel -m 0755 "$HERE/gai_powermetrics.sh" /usr/local/sbin/gai_powermetrics.sh

mkdir -p /var/gai-powermetrics
chown root:wheel /var/gai-powermetrics
chmod 0755 /var/gai-powermetrics

install -o root -g wheel -m 0440 "$TMP_SUDOERS" /etc/sudoers.d/gai-powermetrics
rm -f "$TMP_SUDOERS"

echo "Installed. Test with (no password prompt expected):"
echo "  sudo -n /usr/local/sbin/gai_powermetrics.sh 1000 5 smoketest"
