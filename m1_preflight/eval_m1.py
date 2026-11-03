#!/usr/bin/env python3
"""M1 CPU full-test-set accuracy check. Non-energy equivalence check only --
NOT a benchmark, no sensor/energy measurement.

Citations to code copied from the Phase 1 repo clone (~/green-ai-m1/repo),
scripts/eval_artifacts.py:
  - Normalisation constants: lines 38-39
  - Model loading logic (load_model): lines 49-55
  - CIFAR-10 test dataset loading & transform: lines 69-72
  - Batch handling, dtype casting, inference loop (batch_size=200): lines 83-95
  - Wilson 95% CI: lines 58-65

Deviation from eval_artifacts.py, stated explicitly: this script evaluates
2 fp32 variants per model (fp32_ts.pt and fp32.pt eager) instead of 1, adds
9 *_bnrecal.pt checkpoints (all that exist in checkpoints/, 3 models x 3
pruning levels -- not 6), and forces qnnpack for INT8 (fbgemm is x86-only
and absent from torch.backends.quantized.supported_engines on this Mac).

On failure (e.g. mobilenet_v3_small_int8.pt: QNNPACK rejects this model's
residual-add scale ratio -- a genuine numerical incompatibility, not a bug
here), the job is recorded as status=FAILED in summary.csv with the full
verbatim exception written to results_m1_accuracy/{model}_{state}_error.txt.
No workaround is attempted: no re-quantization, no engine switch, no editing
checkpoints, no retry. Already-completed jobs (existing *_m1cpu_preds.npy)
are not rerun; their accuracy is recomputed from the saved predictions.
"""
import csv
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms

REPO = Path.home() / 'green-ai-m1' / 'repo'
sys.path.insert(0, str(REPO))
from train_baseline import build_model  # noqa: E402

# Copied from scripts/eval_artifacts.py lines 38-39
MEAN = torch.tensor([.4914, .4822, .4465])[None, :, None, None]
STD = torch.tensor([.247, .243, .261])[None, :, None, None]

DATA_ROOT = Path.home() / 'green-ai-m1' / 'data'
OUT = Path.home() / 'green-ai-m1' / 'results_m1_accuracy'
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']

JOBS = []  # (arch, state_label, checkpoint_filename, needs_qnnpack)
for arch in MODELS:
    JOBS.append((arch, 'fp32_ts', f'{arch}_fp32_ts.pt', False))
    JOBS.append((arch, 'fp32', f'{arch}_fp32.pt', False))
    JOBS.append((arch, 'int8', f'{arch}_int8.pt', True))
    for lvl in ('30', '50', '70'):
        JOBS.append((arch, f'pruned{lvl}', f'{arch}_pruned{lvl}.pt', False))
        JOBS.append((arch, f'pruned{lvl}_bnrecal', f'{arch}_pruned{lvl}_bnrecal.pt', False))


# Copied from scripts/eval_artifacts.py lines 49-55
def load_model(ckpt_path, arch):
    try:
        model = torch.jit.load(str(ckpt_path), map_location='cpu')
    except RuntimeError:
        model = build_model(arch)
        model.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=True))
    return model.eval()


# Copied from scripts/eval_artifacts.py lines 58-65
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def main():
    print('python version:', sys.version)
    print('torch version:', torch.__version__)
    print('torch threads:', torch.get_num_threads())
    print('platform:', platform.platform())

    # Copied from scripts/eval_artifacts.py lines 69-72
    ds = torchvision.datasets.CIFAR10(str(DATA_ROOT), train=False, download=False,
                                       transform=transforms.ToTensor())
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    np.save(OUT / 'labels.npy', labels)
    n = len(ds)
    print(f'CIFAR-10 test set: n={n}')

    fieldnames = ['model', 'state', 'status', 'device', 'engine', 'n', 'correct',
                  'accuracy', 'wilson_lo', 'wilson_hi', 'wall_s']
    rows = []
    for arch, state, fname, needs_qnnpack in JOBS:
        ckpt = REPO / 'checkpoints' / fname
        if not ckpt.exists():
            print(f'MISSING: {ckpt}')
            continue

        preds_path = OUT / f'{arch}_{state}_m1cpu_preds.npy'
        engine = 'qnnpack' if needs_qnnpack else 'n/a'

        if preds_path.exists():
            preds = np.load(preds_path)
            correct = int((preds == labels).sum())
            acc = correct / n
            lo, hi = wilson_ci(correct, n)
            rows.append(dict(model=arch, state=state, status='OK', device='cpu',
                              engine=engine, n=n, correct=correct, accuracy=acc,
                              wilson_lo=lo, wilson_hi=hi, wall_s=''))
            print(f'{arch:20s} {state:18s} engine={engine:8s} acc={acc*100:.2f}% '
                  f'[{lo*100:.2f}, {hi*100:.2f}] (already done, skipped rerun)')
            continue

        if needs_qnnpack:
            torch.backends.quantized.engine = 'qnnpack'
            engine = str(torch.backends.quantized.engine)

        t0 = time.perf_counter()
        try:
            model = load_model(ckpt, arch).to('cpu')

            # Copied from scripts/eval_artifacts.py lines 83-95
            try:
                target_dtype = next(model.parameters()).dtype
            except StopIteration:
                target_dtype = images.dtype
            preds = np.empty(n, dtype=np.int64)
            batch_size = 200
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    x = images[i:i + batch_size]
                    x = (x - MEAN) / STD
                    x = x.to('cpu').to(target_dtype)
                    out = model(x)
                    preds[i:i + batch_size] = out.float().argmax(1).cpu().numpy()
            wall_s = time.perf_counter() - t0
        except Exception as e:
            wall_s = time.perf_counter() - t0
            err_text = (f'Exception type: {type(e).__name__}\n'
                        f'Exception message: {e}\n\n'
                        f'Full traceback:\n{traceback.format_exc()}')
            err_path = OUT / f'{arch}_{state}_error.txt'
            err_path.write_text(err_text)
            rows.append(dict(model=arch, state=state, status='FAILED', device='cpu',
                              engine=engine, n=n, correct='', accuracy='',
                              wilson_lo='', wilson_hi='', wall_s=round(wall_s, 2)))
            print(f'{arch:20s} {state:18s} engine={engine:8s} FAILED: '
                  f'{type(e).__name__}: {e}  (full traceback: {err_path})')
            if needs_qnnpack:
                torch.backends.quantized.engine = 'none'
            continue

        np.save(preds_path, preds)
        correct = int((preds == labels).sum())
        acc = correct / n
        lo, hi = wilson_ci(correct, n)
        rows.append(dict(model=arch, state=state, status='OK', device='cpu', engine=engine,
                          n=n, correct=correct, accuracy=acc, wilson_lo=lo, wilson_hi=hi,
                          wall_s=round(wall_s, 2)))
        print(f'{arch:20s} {state:18s} engine={engine:8s} acc={acc*100:.2f}% '
              f'[{lo*100:.2f}, {hi*100:.2f}] wall={wall_s:.1f}s')
        del model
        if needs_qnnpack:
            torch.backends.quantized.engine = 'none'

    with (OUT / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote {OUT / "summary.csv"}')


if __name__ == '__main__':
    main()
