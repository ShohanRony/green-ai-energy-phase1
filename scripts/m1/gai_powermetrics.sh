#!/bin/bash
# Installed to /usr/local/sbin/gai_powermetrics.sh, root:wheel, mode 0755.
# Fixed-path, fixed-argument-shape wrapper so sudoers can grant NOPASSWD on
# exactly this script without matching a shell-glob or free-form argument
# list (same reasoning as scripts/setup_governor_sudoers.sh on the x86 side).
#
# Samplers are hard-coded, not passed in: cpu_power,thermal. Confirmed on
# this Mac via `man powermetrics` (section 27-340 lists "cpu_power" and
# "thermal" samplers by name; "gpu_power" is NOT a real sampler name here --
# GPU power is reported as part of the cpu_power sampler on this platform).
# thermal was also confirmed by a manual run on this Mac on 2026-11-03
# ("Thermal pressure / Current pressure level: Nominal"); an earlier run
# with cpu_power alone printed CPU, GPU, ANE and Combined Power.
set -euo pipefail
umask 022
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export LC_ALL=C

SAMPLERS="cpu_power,thermal"
OUTDIR="/var/gai-powermetrics"
POWERMETRICS="/usr/bin/powermetrics"

usage() {
  echo "usage: gai_powermetrics.sh INTERVAL_MS COUNT TAG" >&2
  echo "  INTERVAL_MS: integer, 100-5000" >&2
  echo "  COUNT:       integer, 1-20000" >&2
  echo "  TAG:         matches ^[A-Za-z0-9_-]{1,64}\$" >&2
}

if [ $# -ne 3 ]; then
  usage
  exit 2
fi

INTERVAL_MS="$1"
COUNT="$2"
TAG="$3"

[[ "$INTERVAL_MS" =~ ^[1-9][0-9]{0,3}$ ]] || { echo "Refusing: INTERVAL_MS must be an integer 100-5000 without leading zeros: $INTERVAL_MS" >&2; exit 2; }
if [ "$INTERVAL_MS" -lt 100 ] || [ "$INTERVAL_MS" -gt 5000 ]; then
  echo "Refusing: INTERVAL_MS out of range [100,5000]: $INTERVAL_MS" >&2
  exit 2
fi

[[ "$COUNT" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "Refusing: COUNT must be an integer 1-20000 without leading zeros: $COUNT" >&2; exit 2; }
if [ "$COUNT" -lt 1 ] || [ "$COUNT" -gt 20000 ]; then
  echo "Refusing: COUNT out of range [1,20000]: $COUNT" >&2
  exit 2
fi

[[ "$TAG" =~ ^[A-Za-z0-9_-]{1,64}$ ]] || { echo "Refusing: invalid TAG: $TAG" >&2; exit 2; }

TOTAL_MS=$((INTERVAL_MS * COUNT))
FOUR_HOURS_MS=$((4 * 60 * 60 * 1000))
if [ "$TOTAL_MS" -gt "$FOUR_HOURS_MS" ]; then
  echo "Refusing: INTERVAL_MS * COUNT ($TOTAL_MS ms) exceeds 4 hours ($FOUR_HOURS_MS ms)" >&2
  exit 2
fi

if [ ! -d "$OUTDIR" ]; then
  echo "Refusing: output dir does not exist: $OUTDIR" >&2
  exit 2
fi
if [ -L "$OUTDIR" ]; then
  echo "Refusing: output dir is a symlink: $OUTDIR" >&2
  exit 2
fi
OUTDIR_OWNER="$(stat -f '%Su' "$OUTDIR")"
OUTDIR_MODE="$(stat -f '%Lp' "$OUTDIR")"
if [ "$OUTDIR_OWNER" != "root" ]; then
  echo "Refusing: output dir not owned by root: $OUTDIR (owner=$OUTDIR_OWNER)" >&2
  exit 2
fi
if [ "$OUTDIR_MODE" != "755" ]; then
  echo "Refusing: output dir mode is not 0755: $OUTDIR (mode=$OUTDIR_MODE)" >&2
  exit 2
fi

OUTFILE="$OUTDIR/${TAG}_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
set -o noclobber
: > "$OUTFILE"
set +o noclobber

"$POWERMETRICS" --samplers "$SAMPLERS" -i "$INTERVAL_MS" -n "$COUNT" -o "$OUTFILE"

chmod 0644 "$OUTFILE"
echo "$OUTFILE"
