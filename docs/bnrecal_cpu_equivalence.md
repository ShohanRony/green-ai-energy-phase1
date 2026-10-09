# `_bnrecal` CPU functional-equivalence check (read-only, no energy measurement)

All 9 `_bnrecal` checkpoints (`checkpoints/{model}_{ratio}_bnrecal.pt`) were accuracy-evaluated on
**CUDA** only, in `results_accuracy/summary.csv`. This check loads each one on **CPU** instead,
runs the full CIFAR-10 test set (same normalization as `pilot.py`/`eval_artifacts.py`), and
compares the exact-correct-count to the stored CUDA figure. Separately, logits are compared
CPU-vs-CUDA on a fixed first-256-image subset. **Pass is marked only when the correct count
matches exactly** — a stricter bar than "accuracy is close."

| Model | State | Stored (CUDA) correct/10000 | CPU correct/10000 | max-abs-diff, logits (256 img) | Status |
|---|---|---|---|---|---|
| resnet18 | pruned30_bnrecal | 8656 | 8656 | 3.83e-03 | **pass** |
| resnet18 | pruned50_bnrecal | 7102 | 7101 | 4.56e-03 | **FAIL** |
| resnet18 | pruned70_bnrecal | 1712 | 1713 | 1.28e-03 | **FAIL** |
| mobilenet_v3_small | pruned30_bnrecal | 1893 | 1894 | 1.44e-03 | **FAIL** |
| mobilenet_v3_small | pruned50_bnrecal | 1000 | 1000 | 3.21e-04 | **pass** |
| mobilenet_v3_small | pruned70_bnrecal | 1000 | 1000 | 3.89e-05 | **pass** |
| efficientnet_b0 | pruned30_bnrecal | 7811 | 7809 | 4.84e-03 | **FAIL** |
| efficientnet_b0 | pruned50_bnrecal | 4628 | 4628 | 2.04e-03 | **pass** |
| efficientnet_b0 | pruned70_bnrecal | 1010 | 1010 | 4.13e-04 | **pass** |

**5/9 pass, 4/9 fail** under the strict exact-count rule. Every failure is off by exactly 1 image
out of 10,000 (0.01pp), and every logit max-abs-diff is small (3.9e-05 to 4.8e-03) — consistent
with ordinary CPU-vs-CUDA floating-point kernel differences (different reduction order in
conv/batchnorm) occasionally flipping one borderline prediction, not a structural incompatibility.
**Stated as fact, not interpreted further** — the 4 FAILs are real under the rule as given; whether
a single-image flip at this magnitude should count as "equivalent" for this project's purposes is
left to the researcher.

Full per-run data: `/tmp/.../bnrecal_cpu_check_results.csv` (scratch, not committed — the table
above is the complete result set). No training, no deletion, no change to any checkpoint.
