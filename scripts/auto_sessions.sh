#!/usr/bin/env bash
# Stage 4b unattended session orchestrator (A4 sessions 3-6), run by stage4b-auto.service at boot.
# One session per fresh boot. Halts (no further reboots) on any failure. State lives outside the
# repo so the runner's clean-tree preflight is not tripped.
set -uo pipefail

REPO=/home/shohan/green-ai-research/green-ai-energy-phase1
STATE=/home/shohan/.stage4b_auto
QUEUE="$STATE/queue"        # lines: "<session> <not_before_utc_epoch>"
LOG="$STATE/orchestrator.log"
HALT="$STATE/HALTED"
AC=/sys/class/power_supply/ACAD/online
mkdir -p "$STATE"
exec >>"$LOG" 2>&1
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

halt() {
  say "HALT: $*"
  echo "$(date -u '+%FT%TZ') $*" > "$HALT"
  sudo -n /usr/bin/systemctl disable stage4b-auto.service || say "could not disable service"
  exit 1
}
on_ac() { [[ -f $AC && $(cat $AC) == 1 ]]; }
reboot_now() { say "Rebooting: $*"; sync; sudo -n /usr/bin/systemctl reboot || halt "reboot failed"; exit 0; }

say "=== boot detected, uptime $(awk '{print $1}' /proc/uptime)s ==="
[[ -f $HALT ]] && { say "HALTED marker present, doing nothing"; exit 0; }
[[ -s $QUEUE ]] || { say "Queue empty: all sessions done. Disabling service."; sudo -n /usr/bin/systemctl disable stage4b-auto.service; exit 0; }

read -r SESSION NOT_BEFORE < "$QUEUE"
say "Next: session $SESSION (not before $(date -u -d @"$NOT_BEFORE" '+%FT%TZ'))"

# Calendar-spread gate: if too early, idle, then reboot so the session still gets a fresh boot.
now=$(date +%s)
if (( now < NOT_BEFORE )); then
  say "Too early; sleeping $((NOT_BEFORE-now))s then rebooting for a fresh boot."
  sleep $((NOT_BEFORE-now)); reboot_now "not_before reached for session $SESSION"
fi

# Power gate: never start on battery. Wait for AC, require 10 min stable, then reboot fresh.
if ! on_ac; then
  say "On battery at boot; waiting for AC."
  while ! on_ac; do sleep 60; done
  say "AC back; waiting 600s for stability."; sleep 600
  on_ac && reboot_now "AC restored before session $SESSION" || halt "AC unstable after restore"
fi

say "Waiting 120s post-boot settle."; sleep 120
on_ac || halt "lost AC during settle"
rm -f "$REPO/.power_state/stop" 2>/dev/null && true   # never inherit a marker across a clean AC boot (D20)

cd "$REPO" || halt "repo missing"
say "Starting runner for main session $SESSION (uptime $(awk '{print $1}' /proc/uptime)s)"
bash scripts/run_stage4b_session.sh --kind main --session "$SESSION"
rc=$?
say "Runner exit code $rc"
(( rc == 0 )) || halt "runner exit $rc for session $SESSION"

# Independent completeness check: 30 conditions x 29 window rows (7 reps).
D="$REPO/results_stage4b/main_session$SESSION"
n=0; bad=0
while read -r c; do
  [[ -z $c ]] && continue; n=$((n+1))
  r=$(wc -l < "$D/$c/windows.csv" 2>/dev/null || echo 0)
  [[ $r -eq 29 && -s $D/$c/summary.csv ]] || { bad=$((bad+1)); say "INCOMPLETE $c rows=$r"; }
done < "$D/order.txt"
(( n == 30 && bad == 0 )) || halt "session $SESSION incomplete: $n conditions, $bad bad"
say "Session $SESSION verified complete (30/30 x 29 rows)."

sed -i '1d' "$QUEUE"
if [[ -s $QUEUE ]]; then
  read -r NEXT NB < "$QUEUE"; now=$(date +%s)
  if (( now < NB )); then say "Session $NEXT not allowed until $(date -u -d @"$NB" '+%FT%TZ'); idling."; sleep $((NB-now)); fi
  reboot_now "fresh boot for session $NEXT"
fi
say "All queued sessions complete."
sudo -n /usr/bin/systemctl disable stage4b-auto.service
