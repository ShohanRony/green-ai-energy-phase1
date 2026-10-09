#!/bin/bash
# Autonomous execution of Audit Plan P8: Accuracy Completion.
# Runs FP32 baselines, prunes them, and fine-tunes them, surviving reboots and power cuts.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# Wait for post-boot network/mounts if freshly booted
sleep 10

# Ensure power_watchdog is running
if ! pgrep -f "../measurement_env/bin/python power_watchdog.py" > /dev/null; then
    echo "Starting power_watchdog..."
    nohup ../measurement_env/bin/python power_watchdog.py > power_watchdog.log 2>&1 &
fi

function wait_for_ac() {
    while [ "$(cat /sys/class/power_supply/ACAD/online)" != "1" ]; do
        echo "Waiting for AC power..."
        sleep 60
    done
}

# The list of jobs to run.
# Format: JOB_ID|COMMAND
# We store completion in .p8_state
mkdir -p .p8_state
touch .p8_state/completed_jobs.txt

declare -a JOBS=(
    # 1. Train FP32 baselines with 45k/5k split for 3 seeds
    "fp32_resnet18_1001|../measurement_env/bin/python train_p8.py --arch resnet18 --epochs 30 --seed 1001 --out checkpoints_p8/resnet18_fp32_1001.pt"
    "fp32_resnet18_1002|../measurement_env/bin/python train_p8.py --arch resnet18 --epochs 30 --seed 1002 --out checkpoints_p8/resnet18_fp32_1002.pt"
    "fp32_resnet18_1003|../measurement_env/bin/python train_p8.py --arch resnet18 --epochs 30 --seed 1003 --out checkpoints_p8/resnet18_fp32_1003.pt"
    
    "fp32_mobilenet_v3_small_1001|../measurement_env/bin/python train_p8.py --arch mobilenet_v3_small --epochs 30 --seed 1001 --out checkpoints_p8/mobilenet_v3_small_fp32_1001.pt"
    "fp32_mobilenet_v3_small_1002|../measurement_env/bin/python train_p8.py --arch mobilenet_v3_small --epochs 30 --seed 1002 --out checkpoints_p8/mobilenet_v3_small_fp32_1002.pt"
    "fp32_mobilenet_v3_small_1003|../measurement_env/bin/python train_p8.py --arch mobilenet_v3_small --epochs 30 --seed 1003 --out checkpoints_p8/mobilenet_v3_small_fp32_1003.pt"

    "fp32_efficientnet_b0_1001|../measurement_env/bin/python train_p8.py --arch efficientnet_b0 --epochs 30 --seed 1001 --out checkpoints_p8/efficientnet_b0_fp32_1001.pt"
    "fp32_efficientnet_b0_1002|../measurement_env/bin/python train_p8.py --arch efficientnet_b0 --epochs 30 --seed 1002 --out checkpoints_p8/efficientnet_b0_fp32_1002.pt"
    "fp32_efficientnet_b0_1003|../measurement_env/bin/python train_p8.py --arch efficientnet_b0 --epochs 30 --seed 1003 --out checkpoints_p8/efficientnet_b0_fp32_1003.pt"
)

# Generate pruning and finetuning jobs dynamically
for seed in 1001 1002 1003; do
    for arch in resnet18 mobilenet_v3_small efficientnet_b0; do
        for ratio in 30 50 70; do
            ratio_f="0.${ratio}"
            JOBS+=("prune_${arch}_${ratio}_${seed}|../measurement_env/bin/python prune_models_p8.py --arch ${arch} --ratio ${ratio_f} --seed ${seed}")
            JOBS+=("ft_${arch}_${ratio}_${seed}|../measurement_env/bin/python train_p8.py --arch ${arch} --epochs 25 --lr 0.01 --seed ${seed} --finetune-from checkpoints_p8/${arch}_pruned${ratio}_${seed}.pt --out checkpoints_p8/${arch}_p${ratio}_ft_${seed}.pt")
        done
    done
done

for JOB in "${JOBS[@]}"; do
    JOB_ID="${JOB%%|*}"
    CMD="${JOB#*|}"

    if grep -q "^${JOB_ID}$" .p8_state/completed_jobs.txt; then
        continue
    fi

    echo "Starting job: ${JOB_ID}"
    
    # Check for a resume hint from an interrupted version of this job
    if [ -f "checkpoints_p8/${JOB_ID}_interrupted.pt" ] && [[ "$CMD" == *"train_p8.py"* ]]; then
        OUT_FILE=$(echo "$CMD" | grep -oP '(?<=--out ).*')
        CMD="../measurement_env/bin/python train_p8.py --resume checkpoints_p8/$(basename ${OUT_FILE} .pt)_interrupted.pt"
        echo "Resuming from interrupted state: $CMD"
    fi

    # Inner loop for retry/resume logic
    while true; do
        wait_for_ac

        # Disable exit-on-error temporarily for the command
        set +e
        $CMD
        EXIT_CODE=$?
        set -e

        if [ $EXIT_CODE -eq 0 ]; then
            echo "${JOB_ID}" >> .p8_state/completed_jobs.txt
            echo "Job ${JOB_ID} completed successfully."
            break
        elif [ $EXIT_CODE -eq 99 ]; then
            echo "Job ${JOB_ID} interrupted by power loss (Exit 99). Will retry when AC restores."
            # Sleep a bit before the outer loop checks AC again
            sleep 10
            
            # Reconstruct the resume command for the next iteration
            if [[ "$CMD" != *"--resume"* ]]; then
                OUT_FILE=$(echo "$CMD" | grep -oP '(?<=--out ).*')
                CMD="../measurement_env/bin/python train_p8.py --resume checkpoints_p8/$(basename ${OUT_FILE} .pt)_interrupted.pt"
            fi
        else
            echo "Job ${JOB_ID} failed with exit code $EXIT_CODE. Aborting P8 auto execution."
            exit 1
        fi
    done
done

echo "All P8 Accuracy Completion jobs finished!"
