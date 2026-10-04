# Stage 2 Task Report — Compression Artifacts

Energy-Aware Efficiency of Lightweight Vision Models Under Post-Training Compression.
Run 2026-10-04, x86 only (Lenovo LOQ 15IRX9, RTX 3050 6GB). One session, Task 0 through Task 7,
all complete. This report walks task-by-task; `stage2_deliverable.md` in this same directory holds
the final tables in deliverable form (accuracy table, realized-vs-nominal table, disclosure sentence,
flag report) without the task narrative.

---

## Task 0 — Orient, confirm inherited state

- Confirmed `pilot.py` defaults to `nvmlDeviceGetPowerUsage` (the Stage 1-corrected backend);
  `--legacy-cumulative-counter` is `action='store_true'` — opt-in only, prints a warning when used.
- Confirmed CIFAR-10 (`cifar-10-batches-py/`) and CIFAR-10-C (20 corruption `.npy` files) present
  under `/home/shohan/green-ai-research/data`, MD5-verified per `docs/phase1-execution-plan.md` §1.5.
- **ARM/M1 explicitly logged as out of scope** for this stage — no M1 work started.
- One discrepancy flagged: the implementation brief referenced `legacy/phase1-execution-plan.md` and
  `legacy/stage1-findings-report.md`; actual repo paths are `docs/phase1-execution-plan.md` and no file
  named `stage1-findings-report.md` (closest equivalents: `results/stage1_exit_test.md` plus the
  individual Stage 1 investigation reports). Used those as the Stage 1 record.

**Verdict:** confirmed, no blockers.

---

## Task 1 — FP32 baselines, all three architectures

No training script existed in the repo before this session — `pilot.py` only *measures* energy for a
given checkpoint, it doesn't train. Built `train_baseline.py` from scratch.

**Architecture decision:** `pilot.py` already applied the standard "CIFAR-ResNet" stem fix to
ResNet-18 (3x3 stride-1 `conv1`, `maxpool` replaced with `Identity`) to stop a 32x32 input from being
over-downsampled before the first residual stage. Extended the same fix to MobileNetV3-Small and
EfficientNet-B0 (their first conv's stride dropped from 2 to 1) for consistency — otherwise their
ImageNet-derived stems would collapse CIFAR-10's spatial resolution too early as well. Disclosed, not
silently assumed equivalent to the un-adapted ImageNet architectures.

**Training recipe** (user-confirmed before spending GPU time): SGD + cosine annealing, 30 epochs,
lr=0.1, momentum=0.9, weight_decay=5e-4, batch_size=128, standard CIFAR-10 augmentation
(RandomCrop(32, pad=4) + RandomHorizontalFlip), seed 2026. Eval protocol: CIFAR-10 test set (10,000
images), augmentation off (ToTensor + normalize only).

Smoke-tested on 1 epoch before committing to the full run, then launched all three sequentially in
the background (~1.5–2.5h estimated).

| Model | Test Acc | Wall-clock | Provenance |
|---|---|---|---|
| ResNet-18 | 93.09% | 22.4 min | trained-here-fresh |
| MobileNetV3-Small | 84.99% | 3.7 min | trained-here-fresh |
| EfficientNet-B0 | 83.59% | 17.3 min | trained-here-fresh |

ResNet-18 leading by ~8-9 points over the other two is expected (residual connections, most mature
CIFAR recipe of the three). MobileNetV3-Small/EfficientNet-B0 in the low-to-mid 80s at only 30 epochs
from scratch is normal, not a collapse.

Checkpoints: `checkpoints/{arch}_fp32.pt` + sibling `.json` (accuracy, per-epoch log, provenance, seed).

**Verdict:** pass, all three.

---

## Task 2 — INT8 quantization, model by model

| Model | Path | FP32 → INT8 | Verdict |
|---|---|---|---|
| ResNet-18 | official `torchvision.models.quantization.resnet18` | 93.09% → 93.14% | **pass** |
| MobileNetV3-Small | manual pipeline (new, built this session) | 84.99% → 84.20% | **pass** |
| EfficientNet-B0 | same manual pipeline (no official wrapper exists for this arch either) | 83.61% → 78.37% | **pass**, real 5.2pt cost |

**ResNet-18**: re-confirmed on the fresh checkpoint via `quantize_int8.py`. Fuse → per-channel fbgemm
qconfig → calibrate (20 batches) → convert. Real int8 kernel dispatch confirmed by walking the
converted model's modules for `torch.ao.nn.quantized.*` types — not just "no exception raised."

**MobileNetV3-Small**: torchvision ships no `models.quantization.mobilenet_v3_small` (only
`mobilenet_v3_large` has an official quantizable wrapper — confirmed by listing
`torchvision.models.quantization`'s public names). Built `quantize_mobilenet_ptq.py` using **FX
graph-mode quantization** (`torch.ao.quantization.quantize_fx`, built into PyTorch — chosen over
hand-writing a `QuantizableMobileNetV3` because FX mode auto-detects fusable conv/bn/relu patterns on
any `nn.Module`) for the fuse/observe/convert mechanics, with the **Nagel et al. (2021)** procedure
layered on top of FX's default round-to-nearest:

> Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., & Blankevoort, T. (2021).
> *A White Paper on Neural Network Quantization.* arXiv:2106.08295.

- **Cross-layer equalization: deliberately skipped, disclosed in the code docstring and metadata.**
  CLE exists to fix per-channel weight-range imbalance for backends that quantize weights per-tensor.
  fbgemm's default qconfig already uses a **per-channel** weight observer, so the problem CLE targets
  is already handled by the backend — same "tooling already covers what the plan assumed needed
  building" pattern as Stage 1's INT4 Conv2d-kernel finding.
- **Bias correction + AdaRound: implemented.** AdaRound learns a soft per-weight round-up/down
  decision (sigmoid relaxation, β-annealed regularizer, 200 iterations/layer, Adam) to minimize each
  layer's own output-reconstruction error, instead of naive round-to-nearest. Bias correction then
  shifts each layer's bias by the measured mean output error on calibration data. 54 layers covered
  for MobileNetV3-Small, 82 for EfficientNet-B0.
- **Disclosed simplification:** AdaRound here is layer-wise (each layer calibrated against the FP32
  model's own activations), not the paper's full sequential/network-wise calibration (which would
  calibrate layer *N* against the already-quantized output of layer *N-1*). Simpler, slightly weaker,
  but the core mechanism — learned rounding instead of round-to-nearest — is faithfully implemented.

**Bug caught and fixed during build, before any numbers were trusted:** the first draft of
`adaround_weight()` computed the reconstruction target for `nn.Linear` layers as `module(x_calib)`
(includes the layer's original bias) but compared it against a forward pass that excluded bias
(`bias=None`) — a systematic bias-shaped error that would have thrown off every Linear layer's AdaRound
optimization. Fixed to exclude bias on both sides before the first real run (bias is corrected
separately, afterward, by design).

**EfficientNet-B0**: confirmed no official quantized variant exists for this architecture either
(absent from `torchvision.models.quantization`'s full name list, same as MobileNetV3-Small). Reused
the same pipeline, generalized via an `--arch` flag rather than duplicating the script. The **5.2-point
accuracy cost is a genuine finding**, notably larger than MobileNetV3-Small's 0.79-point cost — plausibly
because EfficientNet-B0's SiLU activations and squeeze-excite blocks are known to be less
quantization-robust than the ReLU-based nets. Not a bug; all dtype checks confirmed real int8 kernels,
no silent FP32 fallback in any of the three models.

**Verdict:** pass, all three (ResNet-18, MobileNetV3-Small, EfficientNet-B0).

---

## Task 3 — FP16 conversion, all three

Built `fp16_speedup_check.py`: GPU-only comparison (asserted — FP16's benefit is CUDA-specific on this
hardware), warmup + `torch.cuda.synchronize()`-bracketed timing, batch_size=128, 50 reps, dtype
asserted on the output tensor.

| Model | Speedup | Dtype confirmed |
|---|---|---|
| ResNet-18 | 1.78x | yes |
| MobileNetV3-Small | 1.96x | yes |
| EfficientNet-B0 (new) | 2.01x | yes |

EfficientNet-B0 (previously unbenchmarked) behaves consistently with the other two — genuine speedup,
correct `float16` output, no silent fallback.

**Verdict:** pass, all three.

---

## Task 4 — Structured pruning, decision checkpoint then full grid

**Decision checkpoint (ResNet-18 only, 70% sparsity, zero-finetune vs. 3-epoch brief-recovery), via
`torch-pruning`'s `MagnitudePruner` (L2 importance), before touching the other 8 model/ratio
combinations:**

| | Accuracy | MACs reduction | Speedup |
|---|---|---|---|
| Zero-finetune | **10.35%** | 91.0% | 3.94x |
| Brief recovery (3 epochs) | **84.24%** | 91.0% | 3.84x |

Zero-finetune at 70% collapsed to exact chance-level on the architecture expected to be *most*
pruning-robust of the three. Per the brief, this was a stop-and-decide point, not something to resolve
silently.

**Decision, confirmed by the user (2026-10-04): zero-finetune policy stands, uniformly, across all 3
models and all 3 sparsity levels.** Matches the Stage 2 plan's own pre-decided, disclosed policy
("explicitly a post-training compression study... introducing fine-tuning after pruning would blur
pruning into pruning+retraining"). Collapse is reported as a finding about the limits of
post-training-only compression, not patched over.

**Full 3-model x 3-ratio grid**, all zero-finetune, via `prune_full_grid.py`:

| Model | 30% | 50% | 70% |
|---|---|---|---|
| ResNet-18 | 64.84% | 17.58% | **10.35% — collapsed** |
| MobileNetV3-Small | **11.17% — collapsed** | **11.16% — collapsed** | **10.00% — collapsed** |
| EfficientNet-B0 | 61.14% | 20.46% | **10.00% — collapsed** |

**MobileNetV3-Small collapsing at every tested sparsity level, including 30%, is a stronger result
than the pre-stage risk flag anticipated** (which named 70% as the likely collapse point). This
architecture has essentially no redundancy to prune away with zero recovery.

Nominal sparsity undersells realized compression everywhere — e.g. ResNet-18 "70% pruned" actually
removes 91.0% of MACs and 91.1% of params, because pruning a layer's output channels cascades through
every dependent downstream layer via `torch-pruning`'s dependency graph. Full 18-row nominal/realized
table in `stage2_deliverable.md` §4.

**Verdict:** decision resolved, full grid complete, 9/9 runs produced real (not nominal-only)
compression evidence.

---

## Task 5 — EfficientNet-B0 per-epoch timing

Not covered by the prior validation sprint (only timed VGG-19-BN, ResNet-18, MobileNetV3-Small — see
`research/mitra-reproduction/run_experiment.py`, the other proposal's repo, read-only reference for
matching conditions). Replicated its batch_size=64, num_workers=0 convention in
`efficientnet_timing_check.py`, 2 epochs (first absorbs cuDNN autotune, second is steady-state).

**Result: 37.6s/epoch** (batch_size=64, RTX 3050), warmup epoch 37.7s — negligible autotune cost for
this architecture.

**Verdict:** done. Feeds Stage 3/4 time-budget planning.

---

## Task 6 — Realized vs. nominal compression table

18-row table assembled from Tasks 2-4's saved JSON outputs (`checkpoints/*_int8.json`,
`results_stage2/*_fp16.json`, `results_stage2/task4_full_grid.json`). Full table in
`stage2_deliverable.md` §4 — nominal label + realized MACs/params reduction or dtype-check evidence,
side by side, for every one of the 18 states. No nominal label stands unaccompanied by its realized
evidence anywhere in the table, per the pre-flight checklist requirement carried over from Stage 1.

**Verdict:** done.

---

## Task 7 — Deliverable assembly

`stage2_deliverable.md` assembled with: the 18-artifact inventory (and an explicit note on which of
the 18 states have standalone checkpoint files on disk vs. are evaluated in-process and are
deterministically regenerable — FP16 and the 9 pruned variants fall in the latter category, flagged
rather than silently assumed sufficient), the accuracy table, the realized-vs-nominal table, the §1.6
disclosure sentence verbatim, and the flag report (pruning collapse extent, which INT8 path
EfficientNet-B0 needed, FP16 consistency confirmation).

**Task 4's decision is recorded as resolved, not left as "TBD"**, with the date and the actual
comparison numbers that drove it.

**Verdict:** done. Stage 2 complete pending your review of the two open flags below.

---

## Open items carried out of this stage (not blockers, flagged for visibility)

1. **FP16 and pruned-state checkpoints aren't materialized as 18 separate files** — evaluated
   in-process instead. Deterministically regenerable from the 3 FP32 checkpoints (seed 2026) if Stage
   3's pilot needs them as standalone files for `pilot.py --checkpoint`. Not yet done either way —
   your call before calling Stage 2 fully closed.
2. **Nothing from this session has been committed to git.** All code (`train_baseline.py`,
   `quantize_int8.py`, `quantize_mobilenet_ptq.py`, `fp16_speedup_check.py`,
   `prune_decision_checkpoint.py`, `prune_full_grid.py`, `efficientnet_timing_check.py`) and all
   results/checkpoints are sitting in the working tree only.

## Files produced this session

```
train_baseline.py                          # Task 1: FP32 training, all 3 architectures
quantize_int8.py                           # Task 2: ResNet-18 INT8 (official torchvision path)
quantize_mobilenet_ptq.py                  # Task 2: manual PTQ (MobileNetV3-Small + EfficientNet-B0)
fp16_speedup_check.py                      # Task 3: FP16 dtype + speedup check, all 3
prune_decision_checkpoint.py               # Task 4: zero-finetune vs brief-recovery, ResNet-18 @ 70%
prune_full_grid.py                         # Task 4: full 3x3 zero-finetune pruning grid
efficientnet_timing_check.py               # Task 5: EfficientNet-B0 per-epoch timing

checkpoints/{arch}_fp32.{pt,json}          # Task 1 artifacts (3)
checkpoints/{arch}_int8.{pt,json}          # Task 2 artifacts (3)
results_stage2/{arch}_fp16.json            # Task 3 results (3)
results_stage2/task4_decision_checkpoint.json  # Task 4 decision-point data
results_stage2/task4_full_grid.json        # Task 4 full-grid data
results_stage2/efficientnet_b0_timing.json # Task 5 result
results_stage2/stage2_deliverable.md       # Task 7 deliverable (tables + disclosure + flags)
results_stage2/stage2_task_report.md       # this file
```
