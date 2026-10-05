#!/bin/bash
# One-time setup. Run with: sudo bash scripts/setup_governor_sudoers.sh
#
# Installs scripts/set_cpu_governor.sh as a root-owned wrapper at
# /usr/local/sbin/set_cpu_governor.sh, then grants the real (non-root) user NOPASSWD
# sudo on exactly one invocation of it (`... performance`) via a validated
# /etc/sudoers.d/green-ai-governor drop-in.
#
# Narrowly scoped, same pattern as the M1 sudoers rule for `powermetrics` in
# docs/stage1-implementation-brief.md -- not a blanket NOPASSWD, and a shell-glob
# target like scaling_governor can't be matched safely in sudoers directly (the glob
# expands in the user's shell before sudo ever sees it), hence the fixed-path wrapper
# script instead of granting sudo on `tee`/`echo`.
set -euo pipefail
if [ "$EUID" -ne 0 ]; then
  echo "Run with sudo: sudo bash $0" >&2
  exit 1
fi
REAL_USER="${SUDO_USER:?run via sudo, not as root directly, so the real username is known}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install -o root -g root -m 0755 "$HERE/set_cpu_governor.sh" /usr/local/sbin/set_cpu_governor.sh

SUDOERS_LINE="$REAL_USER ALL=(root) NOPASSWD: /usr/local/sbin/set_cpu_governor.sh performance"
TMP_SUDOERS="$(mktemp)"
echo "$SUDOERS_LINE" > "$TMP_SUDOERS"
if ! visudo -c -f "$TMP_SUDOERS"; then
  echo "Generated sudoers line failed validation, aborting (nothing installed):" >&2
  cat "$TMP_SUDOERS" >&2
  rm -f "$TMP_SUDOERS"
  exit 1
fi
install -o root -g root -m 0440 "$TMP_SUDOERS" /etc/sudoers.d/green-ai-governor
rm -f "$TMP_SUDOERS"

echo "Installed. Test with (no password prompt expected):"
echo "  sudo -n /usr/local/sbin/set_cpu_governor.sh performance"
