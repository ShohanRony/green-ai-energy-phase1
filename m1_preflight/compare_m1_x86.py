#!/usr/bin/env python3
"""Compare M1 CPU predictions (results_m1_accuracy/ copied to ~/m1_compare/)
against existing x86 predictions (green-ai-energy-phase1/results_accuracy/).

McNemar logic and equivalence rule copied from scripts/eval_pruned_cpu.py
lines 48-56, 109-117 (exact two-sided McNemar on discordant pairs b, c;
rule: equivalent if p >= 0.05 AND |diff| <= 0.05 pp). That rule was
REGISTERED for CPU-vs-CUDA (x86 internal) comparison, NOT for M1-vs-x86.
Applying it here is an extension beyond its original scope -- stated,
not silently assumed.
"""
import csv
import numpy as np
from pathlib import Path
from scipy import stats

M1 = Path.home() / 'm1_compare'
X86 = Path('/home/shohan/green-ai-research/green-ai-energy-phase1/results_accuracy')

labels_m1 = np.load(M1 / 'labels.npy')
labels_x86 = np.load(X86 / 'labels.npy')
assert np.array_equal(labels_m1, labels_x86), "label order mismatch between M1 and x86!"
labels = labels_m1
n = len(labels)

MODELS = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']
STATES = ['fp32_ts', 'fp32', 'int8', 'pruned30', 'pruned30_bnrecal',
          'pruned50', 'pruned50_bnrecal', 'pruned70', 'pruned70_bnrecal']

# (m1_state) -> (x86_file_suffix, x86_device_label)
X86_MAP = {
    'fp32_ts': ('fp32_cpu_preds.npy', 'CPU (x86 eager fp32 -- x86 has no TorchScript-specific preds file)'),
    'fp32': ('fp32_cpu_preds.npy', 'CPU'),
    'int8': ('int8_preds.npy', 'CPU (no _cpu suffix, but device=cpu per results_accuracy/summary.csv)'),
    'pruned30': ('pruned30_cpu_preds.npy', 'CPU'),
    'pruned50': ('pruned50_cpu_preds.npy', 'CPU'),
    'pruned70': ('pruned70_cpu_preds.npy', 'CPU'),
    'pruned30_bnrecal': ('pruned30_bnrecal_preds.npy', 'CUDA (no x86 CPU variant exists for bnrecal states)'),
    'pruned50_bnrecal': ('pruned50_bnrecal_preds.npy', 'CUDA (no x86 CPU variant exists for bnrecal states)'),
    'pruned70_bnrecal': ('pruned70_bnrecal_preds.npy', 'CUDA (no x86 CPU variant exists for bnrecal states)'),
}


def exact_mcnemar(a_correct, b_correct):
    b_ = int((a_correct & ~b_correct).sum())
    c_ = int((~a_correct & b_correct).sum())
    if b_ + c_ == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(min(b_, c_), b_ + c_, p=0.5, alternative='two-sided').pvalue)
    return b_, c_, p


rows = []
for arch in MODELS:
    m1_fp32_file = M1 / f'{arch}_fp32_m1cpu_preds.npy'
    m1_fp32_base_acc = None
    if m1_fp32_file.exists():
        p = np.load(m1_fp32_file)
        m1_fp32_base_acc = (p == labels).sum() / n

    x86_fp32_file = X86 / f'{arch}_fp32_cpu_preds.npy'
    x86_fp32_base_acc = None
    if x86_fp32_file.exists():
        p = np.load(x86_fp32_file)
        x86_fp32_base_acc = (p == labels).sum() / n

    for state in STATES:
        m1_file = M1 / f'{arch}_{state}_m1cpu_preds.npy'
        x86_suffix, x86_note = X86_MAP[state]
        x86_file = X86 / f'{arch}_{x86_suffix}'

        if not m1_file.exists():
            rows.append(dict(model=arch, state=state, status='M1_FAILED',
                              x86_file=str(x86_file.name), x86_note=x86_note))
            continue

        m1_preds = np.load(m1_file)
        x86_preds = np.load(x86_file)

        m1_correct_arr = (m1_preds == labels)
        x86_correct_arr = (x86_preds == labels)
        m1_correct = int(m1_correct_arr.sum())
        x86_correct = int(x86_correct_arr.sum())
        diff_pp = (m1_correct - x86_correct) / n * 100.0
        b_, c_, p_ = exact_mcnemar(m1_correct_arr, x86_correct_arr)
        class_diffs = int((m1_preds != x86_preds).sum())
        rule_met = (p_ >= 0.05) and (abs(diff_pp) <= 0.05)

        m1_pct_of_fp32 = (m1_correct / n) / m1_fp32_base_acc * 100.0 if m1_fp32_base_acc else None
        x86_pct_of_fp32 = (x86_correct / n) / x86_fp32_base_acc * 100.0 if x86_fp32_base_acc else None

        rows.append(dict(model=arch, state=state, status='OK',
                          m1_correct=m1_correct, x86_correct=x86_correct,
                          diff_pp=diff_pp, b=b_, c=c_, mcnemar_p=p_,
                          class_diffs=class_diffs, rule_met=rule_met,
                          m1_pct_of_m1fp32=m1_pct_of_fp32, x86_pct_of_x86fp32=x86_pct_of_fp32,
                          x86_file=str(x86_file.name), x86_note=x86_note))

# Print table
hdr = (f"{'model':20s} {'state':18s} {'M1 ok':6s} {'x86 ok':7s} {'diff(pp)':9s} "
       f"{'b':4s} {'c':4s} {'McNp':8s} {'cdiff':6s} {'rule':5s} {'M1%FP32':8s} {'x86%FP32':9s}  x86_file (device)")
print(hdr)
print('-' * len(hdr))
for r in rows:
    if r['status'] != 'OK':
        print(f"{r['model']:20s} {r['state']:18s} {'--':6s} {'--':7s} {'--':9s} "
              f"{'--':4s} {'--':4s} {'--':8s} {'--':6s} {'--':5s} {'--':8s} {'--':9s}  "
              f"{r['x86_file']} ({r['x86_note']})  <- M1 side FAILED (QNNPACK, see error.txt)")
        continue
    print(f"{r['model']:20s} {r['state']:18s} {r['m1_correct']:6d} {r['x86_correct']:7d} "
          f"{r['diff_pp']:+9.3f} {r['b']:4d} {r['c']:4d} {r['mcnemar_p']:8.4f} {r['class_diffs']:6d} "
          f"{'yes' if r['rule_met'] else 'no':5s} "
          f"{r['m1_pct_of_m1fp32']:7.2f}% {r['x86_pct_of_x86fp32']:8.2f}%  "
          f"{r['x86_file']} ({r['x86_note']})")

# CSV output (same data, machine-readable)
csv_fields = ['model', 'state', 'status', 'm1_correct', 'x86_correct', 'diff_pp', 'b', 'c',
              'mcnemar_p', 'class_diffs', 'rule_met', 'm1_pct_of_m1fp32', 'x86_pct_of_x86fp32',
              'x86_file', 'x86_note']
csv_path = Path.home() / 'm1_compare' / 'comparison_table.csv'
with csv_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    for r in rows:
        row = {k: r.get(k, '') for k in csv_fields}
        w.writerow(row)
print(f'\nWrote {csv_path}')
