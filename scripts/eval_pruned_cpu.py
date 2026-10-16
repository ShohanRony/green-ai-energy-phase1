#!/usr/bin/env python3
"""CPU accuracy evaluation for zero-finetune pruned artifacts.

Evaluates the 9 zero-finetune pruned checkpoints on CPU to test functional
equivalence between CPU and CUDA inference under Plan v1 section 9.

Citations to code copied from scripts/eval_artifacts.py:
  - Normalisation constants: lines 38-39
  - Model loading logic: lines 49-55
  - CIFAR-10 test dataset loading & transform: lines 69-72
  - Batch handling, dtype casting, and inference loop: lines 83-95
"""
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_baseline import build_model  # noqa: E402

# Copied from scripts/eval_artifacts.py lines 38-39
MEAN = torch.tensor([.4914, .4822, .4465])[None, :, None, None]
STD = torch.tensor([.247, .243, .261])[None, :, None, None]

MODELS = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']
STATES = ['pruned30', 'pruned50', 'pruned70']

OUT_ACC = ROOT / 'results_accuracy'
OUT_RES = ROOT / 'analysis' / 'results'
OUT_RES.mkdir(parents=True, exist_ok=True)


# Copied from scripts/eval_artifacts.py lines 49-55
def load_model(ckpt_path, arch):
    try:
        model = torch.jit.load(str(ckpt_path), map_location='cpu')
    except RuntimeError:
        model = build_model(arch)
        model.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=True))
    return model.eval()


def exact_mcnemar(cpu_correct: np.ndarray, cuda_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test based on discordant pairs b and c."""
    b = int((cpu_correct & ~cuda_correct).sum())
    c = int((~cpu_correct & cuda_correct).sum())
    if b + c == 0:
        p_val = 1.0
    else:
        p_val = float(stats.binomtest(min(b, c), b + c, p=0.5, alternative='two-sided').pvalue)
    return b, c, p_val


def main():
    device = 'cpu'

    # Copied from scripts/eval_artifacts.py lines 69-72
    ds = torchvision.datasets.CIFAR10(
        str(ROOT.parent / 'data'), train=False, download=False,
        transform=transforms.ToTensor()
    )
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    n = len(ds)

    stored_labels = np.load(OUT_ACC / 'labels.npy')
    assert np.array_equal(labels, stored_labels), "CIFAR-10 test labels do not match results_accuracy/labels.npy"

    results = []

    for arch in MODELS:
        for state in STATES:
            ckpt = ROOT / 'checkpoints' / f'{arch}_{state}.pt'
            model = load_model(ckpt, arch).to(device)

            # Copied from scripts/eval_artifacts.py lines 83-95
            try:
                target_dtype = next(model.parameters()).dtype
            except StopIteration:
                target_dtype = images.dtype

            preds_cpu = np.empty(n, dtype=np.int64)
            batch_size = 200
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    x = images[i:i + batch_size]
                    x = (x - MEAN) / STD
                    x = x.to(device).to(target_dtype)
                    out = model(x)
                    preds_cpu[i:i + batch_size] = out.float().argmax(1).cpu().numpy()

            out_file = OUT_ACC / f'{arch}_{state}_cpu_preds.npy'
            np.save(out_file, preds_cpu)

            # Load stored CUDA predictions
            cuda_preds = np.load(OUT_ACC / f'{arch}_{state}_preds.npy')

            cpu_correct_arr = (preds_cpu == labels)
            cuda_correct_arr = (cuda_preds == labels)

            cpu_correct = int(cpu_correct_arr.sum())
            cuda_correct = int(cuda_correct_arr.sum())

            acc_diff_pp = (cpu_correct / n - cuda_correct / n) * 100.0
            abs_acc_diff_pp = abs(acc_diff_pp)

            b, c, mcnemar_p = exact_mcnemar(cpu_correct_arr, cuda_correct_arr)
            class_diffs = int((preds_cpu != cuda_preds).sum())

            # Equivalence rule per Plan v1 section 9:
            # McNemar p >= 0.05 AND |accuracy difference| <= 0.05 pp
            equivalent = (mcnemar_p >= 0.05) and (abs_acc_diff_pp <= 0.05)

            results.append({
                'model': arch,
                'state': state,
                'cpu_correct': cpu_correct,
                'cuda_correct': cuda_correct,
                'acc_diff_pp': acc_diff_pp,
                'abs_acc_diff_pp': abs_acc_diff_pp,
                'b': b,
                'c': c,
                'mcnemar_p': mcnemar_p,
                'class_diffs': class_diffs,
                'equivalent': equivalent,
            })
            del model

    # Format text output
    lines = []
    lines.append("CPU vs CUDA Functional Equivalence Check for Zero-Finetune Pruned Artifacts")
    lines.append("Plan v1 section 9 rule: equivalent if McNemar p >= 0.05 AND |accuracy difference| <= 0.05 pp")
    lines.append("")
    header = (
        f"{'Model':20s} {'State':10s} {'CPU Correct':12s} {'CUDA Correct':13s} "
        f"{'Diff (pp)':10s} {'b':4s} {'c':4s} {'McNemar p':11s} {'Class Diffs':12s} {'Equivalent':10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    n_pass = 0
    for r in results:
        eq_str = "yes" if r['equivalent'] else "no"
        if r['equivalent']:
            n_pass += 1
        diff_str = f"{r['acc_diff_pp']:+.2f}"
        p_str = f"{r['mcnemar_p']:.4f}"
        lines.append(
            f"{r['model']:20s} {r['state']:10s} {r['cpu_correct']:12d} {r['cuda_correct']:13d} "
            f"{diff_str:10s} {r['b']:4d} {r['c']:4d} {p_str:11s} {r['class_diffs']:12d} {eq_str:10s}"
        )

    lines.append("-" * len(header))
    lines.append(f"Summary: {n_pass} of {len(results)} equivalent under Plan v1 section 9 rule.")
    lines.append("")
    lines.append("Citation of code copied from scripts/eval_artifacts.py:")
    lines.append("  - Loading (load_model via torch.jit.load with build_model fallback): lines 49-55")
    lines.append("  - Normalisation constants (MEAN, STD): lines 38-39")
    lines.append("  - CIFAR-10 test transform and dataset loading: lines 69-72")
    lines.append("  - Batch handling, dtype casting, and inference loop (batch_size=200): lines 83-95")

    text_output = "\n".join(lines) + "\n"
    print(text_output)

    out_txt = OUT_RES / 'pruned_cpu_equivalence.txt'
    with open(out_txt, 'w') as f:
        f.write(text_output)
    print(f"Wrote {out_txt}")


if __name__ == '__main__':
    main()
