#!/usr/bin/env python3
"""Supervisor-requested pruned-state controls. NOT an energy measurement.

(a) Predicted-class histogram per pruned checkpoint, from eval_artifacts.py's saved
    predictions (results_accuracy/*_preds.npy) -- no new inference needed.

(b) BN-statistics-only recalibration: forward passes on ~2000 CIFAR-10 TRAIN images in
    train() mode, no gradients, then re-evaluate test accuracy. The saved pruned checkpoints
    are torch.jit.trace()'d (materialize_checkpoints.py) -- confirmed empirically that a
    traced module's BatchNorm does NOT respond to .train() (the trace bakes training=False
    into the graph; running_mean/running_var buffers don't move after train-mode forward
    passes). So recalibration reproduces the pruned model in EAGER mode instead, using the
    exact same procedure materialize_checkpoints.py used (same seed, same
    torch_pruning.pruner.MagnitudePruner call, same ignored final layer, deepcopy from the
    same FP32 base) -- verified bit-for-bit identical to the stored traced checkpoint
    (max abs output diff 0.0 on a 16-image batch) before trusting it for recalibration.

Writes to results_pruned_controls/ -- a separate directory, nothing in results_accuracy or
checkpoints/ is overwritten.
"""
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_baseline import build_model  # noqa: E402

MEAN = torch.tensor([.4914, .4822, .4465])[None, :, None, None]
STD = torch.tensor([.247, .243, .261])[None, :, None, None]

ARCHES = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']
RATIOS = [0.3, 0.5, 0.7]
FINAL_LAYER = {
    'resnet18': lambda m: m.fc,
    'mobilenet_v3_small': lambda m: m.classifier[3],
    'efficientnet_b0': lambda m: m.classifier[1],
}

ACC_DIR = ROOT / 'results_accuracy'
OUT = ROOT / 'results_pruned_controls'
OUT.mkdir(exist_ok=True)
RECAL_N = 2000
BATCH = 200


def rebuild_pruned(arch, ratio, device):
    torch.manual_seed(2026)
    example_inputs = torch.randn(1, 3, 32, 32, device=device)
    base = build_model(arch)
    base.load_state_dict(torch.load(str(ROOT / 'checkpoints' / f'{arch}_fp32.pt'),
                                     map_location='cpu', weights_only=True))
    base = base.to(device).eval()
    pruned = copy.deepcopy(base)
    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(pruned, example_inputs, importance=imp,
                                        pruning_ratio=ratio, ignored_layers=[FINAL_LAYER[arch](pruned)])
    pruner.step()
    return pruned.eval()


def verify_matches_traced(eager, arch, ratio, device):
    traced = torch.jit.load(str(ROOT / 'checkpoints' / f'{arch}_pruned{int(ratio*100)}.pt'),
                             map_location=device).eval()
    x = torch.randn(16, 3, 32, 32, device=device)
    with torch.no_grad():
        a = eager(x)
        b = traced(x)
    max_diff = (a - b).abs().max().item()
    if max_diff > 1e-5:
        raise RuntimeError(f'{arch}_pruned{int(ratio*100)}: eager rebuild diverges from traced '
                            f'checkpoint (max abs diff {max_diff}) -- not safe to recalibrate from this.')
    return max_diff


def class_histogram(arch, state):
    preds = np.load(ACC_DIR / f'{arch}_{state}_preds.npy')
    hist = np.bincount(preds, minlength=10)
    return hist


def evaluate(model, device, images, labels, target_dtype):
    n = len(labels)
    preds = np.empty(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, BATCH):
            x = images[i:i + BATCH]
            x = (x - MEAN) / STD
            x = x.to(device).to(target_dtype)
            out = model(x)
            preds[i:i + BATCH] = out.float().argmax(1).cpu().numpy()
    return int((preds == labels).sum()) / n


def main():
    device = 'cuda'
    test_ds = torchvision.datasets.CIFAR10(str(ROOT.parent / 'data'), train=False, download=False,
                                            transform=transforms.ToTensor())
    test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_labels = np.array([test_ds[i][1] for i in range(len(test_ds))], dtype=np.int64)

    train_ds = torchvision.datasets.CIFAR10(str(ROOT.parent / 'data'), train=True, download=False,
                                             transform=transforms.ToTensor())
    torch.manual_seed(2026)
    idx = torch.randperm(len(train_ds))[:RECAL_N]
    recal_images = torch.stack([train_ds[i][0] for i in idx])

    hist_rows = []
    recal_rows = []
    for arch in ARCHES:
        for ratio in RATIOS:
            state = f'pruned{int(ratio*100)}'
            hist = class_histogram(arch, state)
            hist_rows.append(dict(model=arch, state=state, **{f'class_{i}': int(c) for i, c in enumerate(hist)}))
            print(f'{arch} {state} class histogram: {hist.tolist()}')

            eager = rebuild_pruned(arch, ratio, device)
            max_diff = verify_matches_traced(eager, arch, ratio, device)

            try:
                target_dtype = next(eager.parameters()).dtype
            except StopIteration:
                target_dtype = test_images.dtype
            acc_before = evaluate(eager, device, test_images, test_labels, target_dtype)

            eager.train()
            with torch.no_grad():
                for i in range(0, RECAL_N, BATCH):
                    x = recal_images[i:i + BATCH]
                    x = (x - MEAN) / STD
                    x = x.to(device).to(target_dtype)
                    eager(x)
            eager.eval()
            acc_after = evaluate(eager, device, test_images, test_labels, target_dtype)

            recal_rows.append(dict(model=arch, state=state, verify_max_diff=max_diff,
                                    acc_before_recal=acc_before, acc_after_bn_recal=acc_after,
                                    delta_pt=(acc_after - acc_before) * 100))
            print(f'  acc_before={acc_before*100:.2f}%  acc_after_BN_recal={acc_after*100:.2f}%  '
                  f'delta={100*(acc_after-acc_before):+.2f}pt')
            del eager
            torch.cuda.empty_cache()

    with (OUT / 'class_histograms.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
        w.writeheader(); w.writerows(hist_rows)
    with (OUT / 'bn_recalibration.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recal_rows[0].keys()))
        w.writeheader(); w.writerows(recal_rows)
    print(f'\nWrote {OUT / "class_histograms.csv"} and {OUT / "bn_recalibration.csv"}')


if __name__ == '__main__':
    main()
