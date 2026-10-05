"""Stage 4 pre-flight diagnostic (2026-10-05 decision, option 1, bounded scope): is the
GPU pin(~60W)/dip(~35-46W) power-regime split correlated with GPU temp, persistence mode,
SM clock, AC state, or idle-time-since-last-use at launch? Scoped to exactly 3 combos --
Pruned70@b16 (previously deterministic), Pruned70@b32 and FP16@b64 (previously flipped) --
N independent invocations each. Not expanded to more combos/models per the decision.

Reduced --repeats (4, not the usual 10): this sweep only needs to classify each invocation
as pinned/dip from its total_j_mean, not full-precision energy stats -- disclosed here, not
silent. The regime has been observed stable across all reps within every run collected so
far (locks in once per invocation), so 4 reps (3 pairs post-cold-discard, the summarize()
minimum) is enough for a reliable classification without the full ~4min/run cost.

Usage: python3 diagnostic_power_regime_sweep.py
"""
import csv, json, subprocess, sys, time
from pathlib import Path

COMBOS = [
    ('pruned70_b16', 'checkpoints/resnet18_pruned70.pt', 16),
    ('pruned70_b32', 'checkpoints/resnet18_pruned70.pt', 32),
    ('fp16_b64', 'checkpoints/resnet18_fp16.pt', 64),
]
N_INVOCATIONS = 8
REPEATS_PER_INVOCATION = 4
PIN_THRESHOLD_W = 55  # clean margin: observed pinned cluster >=59.5W, dip cluster <=53.1W
OUT_ROOT = Path('results_stage4_preflight/diagnostic_sweep')
LOG_PATH = OUT_ROOT / 'launch_log.jsonl'


def gpu_telemetry() -> dict:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=temperature.gpu,persistence_mode,clocks.sm,power.draw',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    temp, persistence, clock_sm, power = [x.strip() for x in out.stdout.strip().split(',')]
    return dict(temp_c=float(temp), persistence_mode=persistence,
                clock_sm_mhz=float(clock_sm), power_draw_w_at_launch=float(power))


def ac_online() -> bool:
    return Path('/sys/class/power_supply/ACAD/online').read_text().strip() == '1'


def classify(out_dir: Path):
    with open(out_dir / 'summary.csv') as f:
        row = list(csv.DictReader(f))[0]
    total_j = float(row['total_j_mean'])
    implied_w = total_j / 5.0
    regime = 'pinned' if implied_w > PIN_THRESHOLD_W else 'dip'
    return regime, implied_w, int(row['pairs'])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    last_run_end = None  # global: all GPU use in this sweep is sequential pilot.py calls

    for combo_name, ckpt, batch in COMBOS:
        for i in range(N_INVOCATIONS):
            now = time.time()
            idle_gap_s = (now - last_run_end) if last_run_end is not None else None
            telem = gpu_telemetry()
            ac = ac_online()
            out_dir = OUT_ROOT / f'{combo_name}_inv{i}'
            print(f'=== {combo_name} invocation {i + 1}/{N_INVOCATIONS} '
                  f'(idle_gap={idle_gap_s}, temp={telem["temp_c"]}C) ===', flush=True)
            subprocess.run(
                ['python3', 'pilot.py', '--device', 'cuda', '--arch', 'resnet18',
                 '--checkpoint', ckpt, '--data', '/home/shohan/green-ai-research/data',
                 '--out', str(out_dir), '--sizes', '32', '--batches', str(batch),
                 '--windows', '5', '--repeats', str(REPEATS_PER_INVOCATION),
                 '--interval', '0.4', '--warmup', '3'], check=True)
            regime, implied_w, pairs = classify(out_dir)
            record = dict(combo=combo_name, invocation=i, idle_gap_s=idle_gap_s, ac_online=ac,
                          regime=regime, implied_w=implied_w, pairs=pairs, **telem,
                          timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            with LOG_PATH.open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(f'  -> {regime} ({implied_w:.1f}W)', flush=True)
            last_run_end = time.time()

    print('DIAGNOSTIC_SWEEP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
