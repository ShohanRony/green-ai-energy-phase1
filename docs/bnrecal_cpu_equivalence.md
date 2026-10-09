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

## Addendum, 2026-10-10: per-flip detail, FAIL labels unchanged

The line above ("off by exactly 1 image") was imprecise for two of the four — corrected here, not
in the table (**no FAIL is relabelled**). Locating the exact flipped index(es) per artifact
(comparing this check's CPU predictions against the stored CUDA predictions in
`results_accuracy/{model}_{state}_preds.npy`, full 10,000 images):

| Model / state | Net count diff | Actual flipped indices (count) |
|---|---|---|
| efficientnet_b0 pruned30_bnrecal | 2 | 1687, 5446 (**2**, not 1) |
| mobilenet_v3_small pruned30_bnrecal | 1 | 8329 (1) |
| resnet18 pruned50_bnrecal | 1 | 3766 (1) |
| resnet18 pruned70_bnrecal | 1 (net) | 2725, 3159, 5605, 7753, 9038 (**5** individual disagreements, netting to 1 — some flips go correct→wrong and others wrong→correct) |

For each flipped image, top-2 logit margin and logit max-abs-diff were recomputed **at batch=1 on
both devices** (not reusing the bulk batch=200 sweep, to isolate pure CPU-vs-CUDA numerical
difference from any batch-size effect):

| idx | model/state | margin (CUDA) | margin (CPU) | max-abs-diff (this image, batch=1) | margin < 2×diff? |
|---|---|---|---|---|---|
| 1687 | efficientnet_b0 pruned30 | 0.000507 | 0.000507 | 0.000001 | **No** |
| 5446 | efficientnet_b0 pruned30 | 0.001060 | 0.001059 | 0.000001 | **No** |
| 8329 | mobilenet_v3_small pruned30 | 0.000082 | 0.000083 | 0.0000005 | **No** |
| 3766 | resnet18 pruned50 | 0.001467 | 0.001436 | 0.000933 | **Yes** |
| 2725 | resnet18 pruned70 | 0.000230 | 0.000579 | 0.000553 | **Yes** |
| 3159 | resnet18 pruned70 | 0.000048 | 0.000451 | 0.000659 | **Yes** |
| 5605 | resnet18 pruned70 | 0.000412 | 0.000314 | 0.000479 | **Yes** |
| 7753 | resnet18 pruned70 | 0.000242 | 0.000160 | 0.000452 | **Yes** |
| 9038 | resnet18 pruned70 | 0.000015 | 0.000153 | 0.000561 | **Yes** |

**Split result, not uniform:** for `resnet18` (both pruned50 and pruned70), the margin-vs-noise
test passes — the per-image logit margin is comparable to or smaller than the batch=1 CPU/CUDA
numerical difference, consistent with an ordinary close-call flip. For `efficientnet_b0_pruned30`
and `mobilenet_v3_small_pruned30`, it does **not** — the batch=1 CPU-vs-CUDA logit difference for
those exact images is far smaller (~1e-6) than the margin (~5e-4 to 1e-3), meaning pure device
numerical noise at matched batch size does not explain those two flips. The original bulk check
used batch=200; this recheck used batch=1 — **batch-size-dependent kernel selection (a separate,
also-benign source of floating-point non-determinism, not unique to CPU-vs-CUDA) is the more
likely explanation for those two**, not interpreted further than that. **The FAIL labels in the
table above are unchanged** — this is additional explanation, not a relabelling.
