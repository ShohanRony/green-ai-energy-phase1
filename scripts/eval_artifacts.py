#!/usr/bin/env python3
"""Full-test-set accuracy for all 18 checkpoints, loaded/preprocessed exactly as pilot.py
does (same try/except TorchScript-vs-state_dict loading, same normalization constants,
same dtype-casting rule, same per-state device). Supervisor-requested accuracy completion
-- NOT an energy measurement, no sensor involved.

Per checkpoint:
  - load via torch.jit.load; on RuntimeError, build_model(arch) + load_state_dict (mirrors
    pilot.py's main() exactly).
  - device: cuda for fp32/fp16/pruned30/50/70 (matches Stage 4's actual --device for those
    states); cpu for int8 (D2: INT8 segfaults on this GPU).
  - normalize with pilot.py's own constants (.4914,.4822,.4465)/(.247,.243,.261) -- NOT
    train_baseline.py's CIFAR_STD (0.2470,0.2435,0.2616), which differ in the 3rd decimal.
    This mismatch is deliberate to replicate here, not fixed: the question is whether
    pilot.py's own inference pipeline reproduces Stage 2's reported accuracy.
  - dtype cast: target_dtype = next(model.parameters()).dtype if it has plain parameters
    (FP16 ends up float16), else input dtype unchanged (quantized INT8 models hold no plain
    nn.Parameter) -- identical to pilot.py's StopIteration fallback.

Outputs: results_accuracy/labels.npy (shared, full test-set order) and
results_accuracy/{model}_{state}_preds.npy per checkpoint, plus summary.csv with accuracy
and a 95% Wilson CI (n=10000).
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_baseline import build_model  # noqa: E402

MEAN = torch.tensor([.4914, .4822, .4465])[None, :, None, None]
STD = torch.tensor([.247, .243, .261])[None, :, None, None]

MODELS = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']
STATES = ['fp32', 'int8', 'fp16', 'pruned30', 'pruned50', 'pruned70']
GPU_STATES = {'fp32', 'fp16', 'pruned30', 'pruned50', 'pruned70'}  # int8 is CPU-only (D2)

OUT = ROOT / 'results_accuracy'
OUT.mkdir(exist_ok=True)


def load_model(ckpt_path, arch):
    try:
        model = torch.jit.load(str(ckpt_path), map_location='cpu')
    except RuntimeError:
        model = build_model(arch)
        model.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=True))
    return model.eval()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def main():
    ds = torchvision.datasets.CIFAR10(str(ROOT.parent / 'data'), train=False, download=False,
                                       transform=transforms.ToTensor())
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    np.save(OUT / 'labels.npy', labels)
    n = len(ds)
    print(f'CIFAR-10 test set: n={n}')

    rows = []
    for arch in MODELS:
        for state in STATES:
            device = 'cuda' if state in GPU_STATES else 'cpu'
            ckpt = ROOT / 'checkpoints' / f'{arch}_{state}.pt'
            model = load_model(ckpt, arch).to(device)
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
                    x = x.to(device).to(target_dtype)
                    out = model(x)
                    preds[i:i + batch_size] = out.float().argmax(1).cpu().numpy()
            np.save(OUT / f'{arch}_{state}_preds.npy', preds)
            correct = int((preds == labels).sum())
            acc = correct / n
            lo, hi = wilson_ci(correct, n)
            rows.append(dict(model=arch, state=state, device=device, n=n, correct=correct,
                              accuracy=acc, wilson_lo=lo, wilson_hi=hi))
            print(f'{arch:20s} {state:10s} device={device:4s} acc={acc*100:.2f}% '
                  f'[{lo*100:.2f}, {hi*100:.2f}] (n={n})')
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

    with (OUT / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote {OUT / "summary.csv"}')


if __name__ == '__main__':
    main()
