#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# Stage 4b Session Runner Script
# Usage: scripts/run_stage4b_session.sh --kind main|rq1|b16 --session N
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="/home/shohan/green-ai-research/measurement_env/bin/python"
WRAPPER="/usr/local/sbin/set_cpu_governor.sh"
DATA_DIR="/home/shohan/green-ai-research/data"

KIND=""
SESSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)
      KIND="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$KIND" || -z "$SESSION" ]]; then
  echo "Usage: $0 --kind main|rq1|b16 --session N" >&2
  exit 1
fi

if [[ "$KIND" != "main" && "$KIND" != "rq1" && "$KIND" != "b16" ]]; then
  echo "Error: --kind must be main, rq1, or b16" >&2
  exit 1
fi

SESSION_DIR_NAME="${KIND}_session${SESSION}"
OUT_BASE="$REPO_ROOT/results_stage4b/$SESSION_DIR_NAME"
mkdir -p "$OUT_BASE"
RUNNER_LOG="$OUT_BASE/runner.log"

exec > >(tee -a "$RUNNER_LOG") 2>&1

echo "========================================================================"
echo "Starting Stage 4b Runner: kind=$KIND session=$SESSION at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Output directory: $OUT_BASE"
echo "========================================================================"

# --- Preflight Checks ---
echo "[Preflight] Checking tracked files git status..."
if ! git diff --quiet HEAD; then
  echo "Preflight Error: Uncommitted changes in tracked files!" >&2
  exit 1
fi

echo "[Preflight] Ensuring CPU governor is performance..."
if [[ -x "$WRAPPER" ]]; then
  sudo "$WRAPPER" performance || true
fi
GOVS=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u)
if [[ "$GOVS" != "performance" ]]; then
  echo "Preflight Error: Scaling governor is '$GOVS', not 'performance'!" >&2
  exit 1
fi

echo "[Preflight] Checking platform profile..."
if [[ -f /sys/firmware/acpi/platform_profile ]]; then
  PROF=$(cat /sys/firmware/acpi/platform_profile)
  if [[ "$PROF" != "performance" ]]; then
    echo "Preflight Error: Platform profile is '$PROF', not 'performance'!" >&2
    exit 1
  fi
fi

echo "[Preflight] Checking AC online..."
if [[ -f /sys/class/power_supply/ACAD/online ]]; then
  AC=$(cat /sys/class/power_supply/ACAD/online)
  if [[ "$AC" != "1" ]]; then
    echo "Preflight Error: AC adapter is not online (value: $AC)!" >&2
    exit 1
  fi
fi

echo "[Preflight] Checking foreign GPU processes..."
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | wc -l)
if [[ "$GPU_PROCS" -gt 0 ]]; then
  echo "Preflight Error: Active GPU compute processes found!" >&2
  nvidia-smi
  exit 1
fi

echo "[Preflight] Checking power_watchdog..."
if ! pgrep -fa power_watchdog >/dev/null; then
  echo "Starting power_watchdog in background..."
  nohup python3 power_watchdog.py > power_watchdog.log 2>&1 &
  sleep 1
fi

echo "[Preflight] Checking for stale power stop marker..."
if [[ -f ".power_state/stop" ]]; then
  echo "Preflight Error: Stale .power_state/stop marker found! Did you restart on AC without clearing it?" >&2
  exit 1
fi

BOOT_TIME=$(uptime -s)
UPTIME=$(cat /proc/uptime | awk '{print $1}')
echo "[Preflight] System boot time: $BOOT_TIME, uptime: ${UPTIME}s"

# --- Generate Order ---
ORDER_FILE="$OUT_BASE/order.txt"
if [[ ! -f "$ORDER_FILE" ]]; then
  echo "Generating presentation order for $KIND session $SESSION..."
  "$PYTHON" - <<EOF
import random

models = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']
states = [
    'fp32', 'fp16',
    'pruned30', 'pruned50', 'pruned70',
    'pruned30_bnrecal', 'pruned50_bnrecal', 'pruned70_bnrecal',
    'int8', 'fp32_cpu'
]
conditions = [f"{m}_{s}" for m in models for s in states]

seed_base = 1000 if "$KIND" == "main" else (2000 if "$KIND" == "rq1" else 3000)
seed = seed_base + int("$SESSION")
rng = random.Random(seed)
rng.shuffle(conditions)

with open("$ORDER_FILE", "w") as f:
    for c in conditions:
        f.write(c + "\n")
print(f"Generated order with seed {seed}: {len(conditions)} conditions.")
EOF
fi

# --- Helper function to determine condition args ---
get_condition_args() {
  local cond="$1"
  local arch=""
  local state=""

  if [[ "$cond" =~ ^resnet18_(.*)$ ]]; then
    arch="resnet18"
    state="${BASH_REMATCH[1]}"
  elif [[ "$cond" =~ ^mobilenet_v3_small_(.*)$ ]]; then
    arch="mobilenet_v3_small"
    state="${BASH_REMATCH[1]}"
  elif [[ "$cond" =~ ^efficientnet_b0_(.*)$ ]]; then
    arch="efficientnet_b0"
    state="${BASH_REMATCH[1]}"
  else
    echo "Unknown condition: $cond" >&2
    return 1
  fi

  local extra_flags=()
  case "$state" in
    fp32)
      extra_flags+=(--device cuda --arch "$arch" --checkpoint "checkpoints/${arch}_fp32.pt" --concurrent-cpu-package)
      ;;
    fp16)
      extra_flags+=(--device cuda --checkpoint "checkpoints/${arch}_fp16.pt" --concurrent-cpu-package)
      ;;
    pruned30|pruned50|pruned70)
      extra_flags+=(--device cuda --checkpoint "checkpoints/${arch}_${state}.pt" --concurrent-cpu-package)
      ;;
    pruned30_bnrecal|pruned50_bnrecal|pruned70_bnrecal)
      extra_flags+=(--device cuda --checkpoint "checkpoints/${arch}_${state}.pt" --concurrent-cpu-package)
      ;;
    int8)
      extra_flags+=(--device cpu --checkpoint "checkpoints/${arch}_int8.pt")
      ;;
    fp32_cpu)
      extra_flags+=(--device cpu --arch "$arch" --checkpoint "checkpoints/${arch}_fp32.pt")
      ;;
    *)
      echo "Unknown state: $state" >&2
      return 1
      ;;
  esac

  echo "${extra_flags[@]}"
}

# --- Execution Parameters ---
REPEATS=7
BATCH_ARG=(--batches 1)
EXTRA_KIND_FLAGS=()

if [[ "$KIND" == "rq1" ]]; then
  EXTRA_KIND_FLAGS+=(--codecarbon)
elif [[ "$KIND" == "b16" ]]; then
  BATCH_ARG=(--batches 16)
fi

if [[ -n "$STAGE4B_DRYRUN" ]]; then
  REPEATS=4
  echo "[DRY RUN MODE] Overriding repeats=$REPEATS"
fi

START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
declare -A COND_UPTIMES
INTERRUPTIONS=()
FAILED_CONDITIONS=()

run_condition() {
  local cond="$1"
  local cond_dir="$OUT_BASE/$cond"
  mkdir -p "$cond_dir"

  # Resume-safe check
  if [[ -f "$cond_dir/summary.csv" && -s "$cond_dir/summary.csv" ]]; then
    echo "[$cond] Already complete. Skipping."
    return 0
  fi

  local cond_uptime
  cond_uptime=$(cat /proc/uptime | awk '{print $1}')
  COND_UPTIMES["$cond"]="$cond_uptime"

  echo "------------------------------------------------------------------------"
  echo "[$cond] Starting at uptime ${cond_uptime}s..."
  echo "------------------------------------------------------------------------"

  local cond_args
  cond_args=$(get_condition_args "$cond")

  local cmd=("$PYTHON" pilot.py
    --data "$DATA_DIR"
    --sizes 32
    "${BATCH_ARG[@]}"
    --windows 5
    --interval 0.4
    --warmup 3
    --threads 4
    --repeats "$REPEATS"
    --allow-stale-boot
    --session-id "$SESSION_DIR_NAME"
    --condition-label "$cond"
    --cpu-affinity pcores
    --out "$cond_dir"
    "${EXTRA_KIND_FLAGS[@]}"
    $cond_args
  )

  # Check for resume
  if [[ -f "$cond_dir/resume_state.json" && -f "$cond_dir/raw.jsonl" ]]; then
    echo "[$cond] Incomplete run found, resuming..."
    cmd=("$PYTHON" pilot.py --resume "$cond_dir")
  fi

  local status=0
  "${cmd[@]}" || status=$?

  # Power interruption recovery loop
  # pilot.py returns 0 when it stops safely on a power marker, so we check the marker directly.
  while [[ -f ".power_state/stop" ]]; do
    local int_start
    int_start=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    echo "[$cond] Interrupted by power loss marker at $int_start. Waiting for AC power recovery..."
    INTERRUPTIONS+=("Power loss during $cond at $int_start")

    local wait_elapsed=0
    local max_wait=3600
    while true; do
      sleep 30
      wait_elapsed=$((wait_elapsed + 30))
      
      if [[ -f /sys/class/power_supply/ACAD/online ]] && [[ $(cat /sys/class/power_supply/ACAD/online) == "1" ]]; then
        echo "AC online detected. Waiting 10 minutes (600s) for power stabilization..."
        sleep 600
        wait_elapsed=$((wait_elapsed + 600))
        if [[ $(cat /sys/class/power_supply/ACAD/online) == "1" ]]; then
          echo "AC power stable. Restoring governor..."
          sudo "$WRAPPER" performance || true
          break
        fi
      fi
      
      if [[ $wait_elapsed -ge $max_wait ]]; then
        echo "[$cond] Power interruption exceeded wait cap of ${max_wait}s. Aborting." >&2
        return 1
      fi
    done

    echo "[$cond] Resuming after power interruption..."
    cmd=("$PYTHON" pilot.py --resume "$cond_dir")
    status=0
    "${cmd[@]}" || status=$?
  done

  local expected_rows=$((REPEATS * 4 + 1))
  local actual_rows=0
  if [[ -f "$cond_dir/windows.csv" ]]; then
    actual_rows=$(wc -l < "$cond_dir/windows.csv")
  fi

  if [[ $status -ne 0 || ! -s "$cond_dir/summary.csv" || $actual_rows -ne $expected_rows ]]; then
    echo "[$cond] Execution failed (exit code $status, expected $expected_rows rows but got $actual_rows)!"
    return 1
  fi

  echo "[$cond] Successfully finished."
  return 0
}

# Read conditions from order.txt
mapfile -t CONDITIONS < "$ORDER_FILE"

# If dry run, limit to first condition
if [[ -n "$STAGE4B_DRYRUN" ]]; then
  CONDITIONS=("${CONDITIONS[0]}")
  echo "[DRY RUN MODE] Running 1 condition: ${CONDITIONS[0]}"
fi

for cond in "${CONDITIONS[@]}"; do
  [[ -z "$cond" ]] && continue
  if ! run_condition "$cond"; then
    FAILED_CONDITIONS+=("$cond")
  fi
done

# Re-run failed conditions once
if [[ ${#FAILED_CONDITIONS[@]} -gt 0 ]]; then
  echo "========================================================================"
  echo "Re-running ${#FAILED_CONDITIONS[@]} failed condition(s) once..."
  echo "========================================================================"
  RETRY_FAILED=()
  for cond in "${FAILED_CONDITIONS[@]}"; do
    echo "Retrying $cond..."
    if ! run_condition "$cond"; then
      RETRY_FAILED+=("$cond")
    fi
  done
  FAILED_CONDITIONS=("${RETRY_FAILED[@]}")
fi

END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# --- Write session_log.md ---
LOG_MD="$OUT_BASE/session_log.md"
echo "Writing session log to $LOG_MD..."

"$PYTHON" - <<EOF
import csv, json
from pathlib import Path

out_base = Path("$OUT_BASE")
log_path = Path("$LOG_MD")

conditions = []
with open("$ORDER_FILE") as f:
    for line in f:
        line = line.strip()
        if line:
            conditions.append(line)

lines = []
lines.append(f"# Stage 4b Session Log — $SESSION_DIR_NAME\n")
lines.append(f"- **Start Time:** $START_TIME")
lines.append(f"- **End Time:** $END_TIME")
lines.append(f"- **Boot Time:** $BOOT_TIME")
lines.append(f"- **Uptime Start:** ${UPTIME}s")
lines.append(f"- **Total Planned Conditions:** {len(conditions)}")

lines.append("\n## Interruptions")
interruptions = "$INTERRUPTIONS"
if interruptions.strip():
    lines.append(interruptions)
else:
    lines.append("- None")

lines.append("\n## Failures")
failures = "$FAILED_CONDITIONS"
if failures.strip():
    lines.append(f"- {failures}")
else:
    lines.append("- None")

lines.append("\n## Descriptive Results Summary")
lines.append("| Condition | Regime | images/s | GPU J/img | Package J/img | System J/img |")
lines.append("|---|---|---|---|---|---|")

for cond in conditions:
    sum_csv = out_base / cond / "summary.csv"
    if sum_csv.exists() and sum_csv.stat().st_size > 0:
        with open(sum_csv) as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row:
                regime = row.get("power_regime", "N/A")
                ips = f"{float(row['images_per_s_mean']):.2f}" if row.get("images_per_s_mean") else "N/A"
                gpu_e = f"{float(row['gpu_j_per_image_mean']):.5f}" if row.get("gpu_j_per_image_mean") else "N/A"
                pkg_e = f"{float(row['cpu_package_j_per_image_mean']):.5f}" if row.get("cpu_package_j_per_image_mean") else "N/A"
                sys_e = f"{float(row['system_j_per_image_mean']):.5f}" if row.get("system_j_per_image_mean") else "N/A"
                lines.append(f"| {cond} | {regime} | {ips} | {gpu_e} | {pkg_e} | {sys_e} |")
            else:
                lines.append(f"| {cond} | EMPTY | N/A | N/A | N/A | N/A |")
    else:
        lines.append(f"| {cond} | MISSING | N/A | N/A | N/A | N/A |")

with open(log_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("session_log.md created.")
EOF

# --- Commit Session Results ---
if [[ -z "$STAGE4B_DRYRUN" ]]; then
  echo "Committing session results to git..."
  git add "$OUT_BASE"
  git commit -m "Results: Stage 4b $KIND session $SESSION" || true
  git push origin master || echo "WARNING: Git push failed, commit left locally."
fi

echo "========================================================================"
echo "Stage 4b Runner Completed: $SESSION_DIR_NAME at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================================"
