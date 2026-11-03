# M1 Preflight — 2026-11-03

NON-ENERGY preflight on the Apple M1. No powermetrics, no energy/benchmark
measurement of any kind. Facts only.

## Machine

MacBook Air, Apple M1, 8 GB RAM, macOS 26.6.2.

Repository commit checked out on both machines:
0e426d1b772b790bc2879586048f58c809715f38.

## Environment (M1)

- Python 3.12.16 (installed via `uv`)
- torch 2.7.1
- torchvision 0.22.1
- numpy 2.5.2
- `torch.backends.quantized.supported_engines`: `['qnnpack', 'none']`

## Laptop environment (for comparison)

- Python 3.12.3
- torch 2.7.1+cu118
- torch-pruning installed: 1.6.0
- torch-pruning pinned in requirements.txt: 1.6.1
- Provenance of the version used to build the existing pruned checkpoints: unknown

## Checkpoint integrity

sha256 of all 36 checkpoint files (`*_fp32.pt`, `*_fp32_ts.pt`, `*_pruned*.pt`,
`*_int8.pt`): 36/36 match between the laptop and the M1 clone of this repo.

## CIFAR-10 test set integrity

sha256 of `cifar-10-batches-py/test_batch`, laptop and M1 — both equal:
`f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831`

## taskpolicy

`man taskpolicy` on the M1 shows the `-c` clamp flag accepts only
`utility`, `background`, or `maintenance`. `docs/m1_check_commands.md` in
this repo shows an example using `-c default`, which is not a valid value
for this flag.

## Comparison method

M1 CPU predictions (`eval_m1.py`, run on the M1, full 10,000-image CIFAR-10
test set, batch size 200, no shuffle) were compared against existing x86
prediction files in `results_accuracy/` using exact two-sided McNemar on
discordant pairs plus the accuracy-difference threshold, the equivalence
rule registered in `docs/stage5_plan_v1.md` section 9 (as applied in
`scripts/eval_pruned_cpu.py` and reported in
`docs/pruned_cpu_equivalence.md`) (equivalent if McNemar p >= 0.05 AND
|accuracy difference| <= 0.05 pp).

This rule was registered for CPU-vs-CUDA comparison on x86 only. It is
applied here, to M1-vs-x86, for reference only — not as a validated
pass/fail criterion for this cross-platform case.

`_bnrecal` state rows are compared against x86 CUDA prediction files, since
no x86 CPU prediction file exists for any `_bnrecal` state.

## FP32 and pruned findings

All numbers below are images with a differing predicted class, from
`results/comparison_table.csv` (`class_diffs` column).

- FP32: for all three models, FP32 TorchScript and FP32 eager on the M1
  gave the same predicted class as the x86 CPU FP32 file on all 10,000
  images (0 differing images in both cases, for all three models).

- Zero-finetune pruned states (vs. x86 CPU file):
  - resnet18: pruned30 0, pruned50 0, pruned70 0
  - mobilenet_v3_small: pruned30 0, pruned50 0, pruned70 0
  - efficientnet_b0: pruned30 0, pruned50 0, pruned70 0

- `_bnrecal` states (vs. x86 CUDA file):
  - resnet18: pruned30_bnrecal 0, pruned50_bnrecal 1, pruned70_bnrecal 5
  - mobilenet_v3_small: pruned30_bnrecal 1, pruned50_bnrecal 0,
    pruned70_bnrecal 0
  - efficientnet_b0: pruned30_bnrecal 2, pruned50_bnrecal 4,
    pruned70_bnrecal 0

- INT8 models that ran, correct-count out of 10,000 (M1 vs. x86 CPU file):
  ResNet-18 9320 vs. 9312; EfficientNet-B0 8711 vs. 8718.

- Several states are at or near chance accuracy for this 10-class task
  (1000/10000, or close: e.g. mobilenet_v3_small pruned50/pruned70/
  pruned50_bnrecal/pruned70_bnrecal all 1000/10000; efficientnet_b0
  pruned70 1000/10000, pruned70_bnrecal 1010/10000), where M1-vs-x86
  prediction agreement is uninformative about equivalence.

## INT8 findings

- ResNet-18 INT8 and EfficientNet-B0 INT8 both ran successfully on the M1
  under the `qnnpack` quantized engine. Compared against the x86 fbgemm
  INT8 predictions: ResNet-18 had 40 images with a different predicted
  class; EfficientNet-B0 had 312 images with a different predicted class.
- MobileNetV3-Small INT8 fails to run on the M1 under `qnnpack`. Diagnostic
  detail in `results/mbv3_int8_diag.txt`. Verbatim error line from that
  file (one occurrence):

  > Error in QNNPACK: failed to create add operator with 2.169492e-05
  > A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range

  Verdict derived from that diagnostic in this session (not a line written
  inside the file itself):

  > Reproducible failure (9/9 fresh-process attempts): the
  > fbgemm-calibrated mobilenet_v3_small_int8.pt cannot run under qnnpack
  > in torch 2.7.1 on this Mac; it loads, but forward() fails at the
  > second quantized::add (add_9) because an input-to-output scale ratio
  > of 2.169492e-05 is below the qnnpack minimum of 2^-14. The QNNPACK
  > message names input A (the first operand of add_9); the scale of
  > that input tensor was not read directly. This does not test a
  > checkpoint re-quantized with qnnpack.

## Contents

- `eval_m1.py` — evaluation script run on the M1 (27 checkpoint jobs: 2 fp32
  variants + int8 + 3 zero-finetune pruned + 3 bnrecal, per model x3 models)
- `compare_m1_x86.py` — comparison script run on the laptop
- `results/summary.csv` — per-job M1 accuracy/Wilson-CI summary from the M1
- `results/comparison_table.csv` — M1-vs-x86 comparison, McNemar, %FP32
- `results/mobilenet_v3_small_int8_error.txt` — first captured error
- `results/mbv3_int8_diag.txt` — full diagnostic (environment, integrity,
  load test, control models, graph inspection, 9 fresh-process forward
  attempts)
- `results/m1_checkpoint_hashes.txt` — sha256 of all 36 checkpoints (laptop)
- `results/m1_pin_versions.txt` — exact dependency versions, laptop vs
  requirements.txt vs pip freeze
- `preds/*_m1cpu_preds.npy` — 26 of the planned 27 prediction files. The
  27th (`mobilenet_v3_small_int8`) does not exist because that job failed
  (see INT8 findings above); it is not substituted or fabricated here.
