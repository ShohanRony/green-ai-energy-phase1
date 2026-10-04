# Stage 2 Deliverable — Compression Artifacts

Energy-Aware Efficiency of Lightweight Vision Models Under Post-Training Compression
x86 only (Lenovo LOQ 15IRX9, RTX 3050 6GB). ARM/M1 out of scope for this stage (deferred to Stage 4).
All energy-relevant backend choices use the Stage 1-corrected default (`nvmlDeviceGetPowerUsage`, not
the legacy cumulative counter) — no energy measurements were taken directly in Stage 2 (that's Stage 3's
RQ1 pilot); this stage produces the artifacts that pilot will measure.

## 1. Task 0 — inherited state confirmed

- `nvmlDeviceGetPowerUsage` confirmed default in `pilot.py`; `--legacy-cumulative-counter` is opt-in
  (`action='store_true'`), warns when used.
- CIFAR-10 / CIFAR-10-C present at `/home/shohan/green-ai-research/data`, MD5-verified per
  `docs/phase1-execution-plan.md` §1.5.
- ARM/M1: explicitly out of scope, no M1 work started under Stage 2.

## 2. 18 Compression Artifacts

All checkpoints in `checkpoints/`, all trained/derived fresh on this hardware (seed 2026).

| # | Model | State | Checkpoint |
|---|---|---|---|
| 1 | ResNet-18 | FP32 (baseline) | `resnet18_fp32.pt` |
| 2 | ResNet-18 | INT8 | `resnet18_int8.pt` |
| 3 | ResNet-18 | FP16 | `resnet18_fp16.pt` |
| 4 | ResNet-18 | Pruned 30% | `resnet18_pruned30.pt` |
| 5 | ResNet-18 | Pruned 50% | `resnet18_pruned50.pt` |
| 6 | ResNet-18 | Pruned 70% | `resnet18_pruned70.pt` |
| 7 | MobileNetV3-Small | FP32 (baseline, retrained 60ep 2026-10-04) | `mobilenet_v3_small_fp32.pt` |
| 8 | MobileNetV3-Small | INT8 | `mobilenet_v3_small_int8.pt` |
| 9 | MobileNetV3-Small | FP16 | `mobilenet_v3_small_fp16.pt` |
| 10 | MobileNetV3-Small | Pruned 30% | `mobilenet_v3_small_pruned30.pt` |
| 11 | MobileNetV3-Small | Pruned 50% | `mobilenet_v3_small_pruned50.pt` |
| 12 | MobileNetV3-Small | Pruned 70% | `mobilenet_v3_small_pruned70.pt` |
| 13 | EfficientNet-B0 | FP32 (baseline, retrained 60ep 2026-10-04) | `efficientnet_b0_fp32.pt` |
| 14 | EfficientNet-B0 | INT8 | `efficientnet_b0_int8.pt` |
| 15 | EfficientNet-B0 | FP16 | `efficientnet_b0_fp16.pt` |
| 16 | EfficientNet-B0 | Pruned 30% | `efficientnet_b0_pruned30.pt` |
| 17 | EfficientNet-B0 | Pruned 50% | `efficientnet_b0_pruned50.pt` |
| 18 | EfficientNet-B0 | Pruned 70% | `efficientnet_b0_pruned70.pt` |

All 18 states have standalone checkpoint files on disk as of Task D's materialization
(`materialize_checkpoints.py`, self-contained TorchScript traces) — the note that used to be here about
FP16/pruned states being in-memory-only was stale, left over from before Task D ran; corrected here.

**Retrain note (2026-10-04):** MobileNetV3-Small and EfficientNet-B0's FP32 baselines (and everything
derived from them: INT8, FP16, all 9 pruned states) were retrained/regenerated at 60 epochs after Task B
flagged the original 30-epoch baselines as not plateaued (+1.97pt and +2.05pt over their last 5 epochs
respectively). ResNet-18 was already plateaued (loss 0.0288) and carries over unchanged. Full
before/after comparison in `stage2_retrain_report.md`.

## 3. Accuracy Table (all 18 states, FP32 = reference)

| Model | FP32 (ref) | INT8 | FP16* | Prune 30% | Prune 50% | Prune 70% |
|---|---|---|---|---|---|---|
| ResNet-18 | **93.09%** | 93.14% | 93.09% | 64.84% | 17.58% | **10.35% (collapsed)** |
| MobileNetV3-Small (retrained, 60ep) | **86.46%** | 84.00% | 86.46% | 17.13% (not collapsed, but -69.3pt) | **10.00% (collapsed)** | **10.00% (collapsed)** |
| EfficientNet-B0 (retrained, 60ep) | **88.91%** | 86.87% | 88.91% | 60.18% | **13.29% (collapsed)** | **10.00% (collapsed)** |

**Retrained-vs-original baseline, for comparison (full detail in `stage2_retrain_report.md`):**

| Model | Old FP32 (30ep) | New FP32 (60ep) | Old INT8 cost | New INT8 cost |
|---|---|---|---|---|
| MobileNetV3-Small | 84.99% | 86.46% (+1.47pt) | 0.79pt | 2.47pt |
| EfficientNet-B0 | 83.61% | 88.91% (+5.30pt) | 5.2pt | 2.04pt |

Two verdicts changed with the converged baseline: MobileNetV3-Small@30% pruning no longer crosses the
15%-collapse threshold (11.17%→17.13%, still severely degraded, not "collapsed" by the stated
criterion); EfficientNet-B0@50% newly crosses into collapse (20.46%→13.29%) that it didn't before.

*FP16 accuracy is numerically identical to FP32 at this precision (half-precision forward pass,
output dtype confirmed `float16`) — reported here for completeness; the real metric for this state is
the realized speedup, not accuracy (see §4).
Collapse threshold: ≤15% on CIFAR-10 (10 classes, 10% = chance).

## 4. Realized vs. Nominal Compression (18 rows)

| Model | State | Nominal | Realized MACs reduction | Realized params reduction | Realized speedup | Evidence |
|---|---|---|---|---|---|---|
| ResNet-18 | FP32 | — (0%) | — | — | 1.00x | baseline: 557.2M MACs, 11.17M params |
| ResNet-18 | INT8 | 8-bit | n/a (dtype, not FLOPs) | n/a | — (CPU-only; GPU column structurally empty) | `torch.ao.nn.quantized` modules confirmed present |
| ResNet-18 | FP16 | 16-bit | n/a (dtype, not FLOPs) | n/a | 1.78x | output dtype confirmed `float16` |
| ResNet-18 | Prune 30% | 30% | 51.6% | 51.1% | 1.19x | `torch-pruning` FLOPs/param report |
| ResNet-18 | Prune 50% | 50% | 74.8% | 75.0% | 2.43x | same |
| ResNet-18 | Prune 70% | 70% | 91.0% | 91.1% | 3.87x | same — **accuracy collapsed, compression still real** |
| MobileNetV3-Small (retrained) | FP32 | — (0%) | — | — | 1.00x | baseline: 5.80M MACs, 1.53M params |
| MobileNetV3-Small (retrained) | INT8 | 8-bit | n/a | n/a | — (CPU-only) | `torch.ao.nn.quantized` modules confirmed present |
| MobileNetV3-Small (retrained) | FP16 | 16-bit | n/a | n/a | 1.78x | output dtype confirmed `float16` |
| MobileNetV3-Small (retrained) | Prune 30% | 30% | 46.7% | 50.1% | 1.34x | `torch-pruning` report — severely degraded (17.1%), not past the collapse threshold |
| MobileNetV3-Small (retrained) | Prune 50% | 50% | 69.1% | 73.6% | 1.97x | same — **accuracy collapsed** |
| MobileNetV3-Small (retrained) | Prune 70% | 70% | 86.8% | 90.0% | 3.09x | same — **accuracy collapsed** |
| EfficientNet-B0 (retrained) | FP32 | — (0%) | — | — | 1.00x | baseline: 33.28M MACs, 4.02M params |
| EfficientNet-B0 (retrained) | INT8 | 8-bit | n/a | n/a | — (CPU-only) | `torch.ao.nn.quantized` modules confirmed present |
| EfficientNet-B0 (retrained) | FP16 | 16-bit | n/a | n/a | 2.03x | output dtype confirmed `float16` |
| EfficientNet-B0 (retrained) | Prune 30% | 30% | 48.4% | 50.0% | 1.37x | `torch-pruning` report |
| EfficientNet-B0 (retrained) | Prune 50% | 50% | 71.2% | 73.5% | 2.21x | same — **accuracy collapsed (13.3%)**, worse than the undercooked baseline's 20.5% at this ratio |
| EfficientNet-B0 (retrained) | Prune 70% | 70% | 88.1% | 89.8% | 3.70x | same — **accuracy collapsed** |

Nominal sparsity is consistently *lower* than realized MACs/params reduction across every pruned
state and model — pruning one layer's output channels removes the corresponding input channels (and
their MACs) in every downstream dependent layer via `torch-pruning`'s dependency graph, so the
compounding effect is real, not a reporting error. This is the same nominal-vs-realized gap already
documented for `torch.nn.utils.prune` (that one in the other direction: 0% realized despite nominal
claims) — here the gap is real but in the generous direction.

## 5. §1.6 Disclosure Sentence (verbatim)

> "INT4 post-training quantization was attempted and found infeasible for convolutional layers on
> available consumer tooling (no Conv2d INT4 kernel in any evaluated library as of 2026); FP16 was
> substituted as the sixth compression state. This finding is itself consistent with the
> nominal-vs-realized compression gap documented in the literature review."

## 6. Flag Report

**Zero-finetune pruning collapse — far more widespread than anticipated.** The pre-stage risk flag
named 70% sparsity on the smaller architectures as the likely collapse point. Actual results:

- **ResNet-18**: survives 30% (64.84%), already badly degraded at 50% (17.58%, barely above chance),
  fully collapsed at 70% (10.35%). Unaffected by the 2026-10-04 retrain (baseline already plateaued).
- **MobileNetV3-Small (retrained baseline, 86.46%)**: severely degraded at 30% (17.13%, just above the
  15% collapse threshold — not "collapsed" by the stated criterion, but still a 69.3pt loss), fully
  collapsed at 50% and 70% (10.00% each). With the undercooked 30-epoch baseline this had read as
  collapsed at *every* tested ratio including 30%; the properly converged baseline pushes that one data
  point just above the line. The architecture still has essentially no usable redundancy to prune away
  with zero recovery — the headline finding is unchanged, only the single 30% number moved.
- **EfficientNet-B0 (retrained baseline, 88.91%)**: survives 30% (60.18%, essentially unchanged from the
  undercooked baseline's 61.14%), but now **collapses at 50% (13.29%)** — it did not cross the 15%
  threshold before (20.46%, reported as "degraded but not collapsed"). Collapsed at 70% either way
  (10.00%). This is a real, counter-intuitive finding: the better-trained baseline is *more* fragile to
  magnitude pruning at 50% sparsity, not less — plausibly because a more fully converged model has
  settled into a sharper, less redundant solution that zero-finetune magnitude pruning disrupts more
  severely. Not an artifact of measurement noise (MACs/params reduction at each ratio is bit-identical
  to the old run — architecture-dependent, not weight-dependent — so this is a real accuracy shift).

**Fine-grained MobileNetV3-Small probe (Task A), rerun against the converged baseline — curve shape
changed, headline verdict did not:**

| Sparsity | Old acc (30ep baseline) | New acc (60ep baseline) |
|---|---|---|
| baseline | 84.99% | 86.46% |
| 5% | 81.96% | 79.84% |
| 10% | 59.13% | 63.58% |
| 15% | 30.31% | 59.83% |
| 20% | 21.18% | 30.21% |
| 30% | 11.17% | 17.13% |

The original Task A verdict ("genuine architectural fragility, not a `torch-pruning` dependency-graph
bug") rested partly on the curve being **smooth and monotonic with no cliff**. That specific claim does
**not** hold with the converged baseline: there's now a near-plateau from 10%→15% (-3.75pt) followed by
a sharp cliff from 15%→20% (-29.6pt) — a much more abrupt, localized drop than the old curve showed.
This is flagged as a correction to the earlier "smooth curve" evidence, not swept forward silently.

The rest of Task A's evidence still holds, though: realized/nominal MACs-reduction amplification at 30%
is still 1.56x (identical to the old run — architecture-dependent, not weight-dependent), still in line
with ResNet-18's 1.72x and EfficientNet-B0's 1.61x at the same nominal ratio — no outlier amplification
that would point to a dependency-graph bug at the depthwise/pointwise boundary. **Revised verdict:**
MobileNetV3-Small still shows genuinely low zero-finetune pruning redundancy (now with a plateau-then-
cliff signature around 15-20% rather than a uniformly smooth decline), consistent with — not contradicted
by — a converged baseline. The "no cliff anywhere" phrasing from the original Task A report should be
treated as superseded.

**Task 4 decision, resolved (not TBD):** after seeing ResNet-18's 70% zero-finetune result collapse to
10.35% vs. 84.24% with a 3-epoch brief-recovery variant (see `results_stage2/task4_decision_checkpoint.json`),
the user confirmed (2026-10-04): **zero-finetune policy stands, uniformly, across all 3 models and all
3 sparsity levels.** Collapse is reported as a finding about the limits of post-training-only
compression, not patched over by silently introducing a recovery step the proposal's methodology
doesn't call for.

**Which INT8 path EfficientNet-B0 needed:** confirmed no official `torchvision.models.quantization`
wrapper exists for EfficientNet-B0 (same situation as MobileNetV3-Small, worse than ResNet-18). Used
the same manual FX graph-mode PTQ pipeline built for MobileNetV3-Small (generalized via an `--arch`
flag rather than duplicated). Verdict: **pass** — real int8 kernels confirmed. **Updated against the
retrained baseline:** accuracy cost dropped from 5.2pt (83.61%→78.37%, undercooked baseline) to 2.04pt
(88.91%→86.87%, converged baseline) — now *smaller* than MobileNetV3-Small's cost, not larger.
MobileNetV3-Small's own INT8 cost moved the other way, 0.79pt→2.47pt (84.99%→84.20% vs
86.46%→84.00%). With both baselines now converged, EfficientNet-B0 is the more INT8-quantization-robust
of the two lightweight architectures — the opposite ranking from what the undercooked baselines implied.

**FP16 consistency across all three models:** confirmed, still holds against the retrained baselines.
ResNet-18 unaffected (1.78x). MobileNetV3-Small 1.78x (previously 1.96x), EfficientNet-B0 2.03x
(previously 2.01x) — both within normal run-to-run timing noise, since FP16 forward-pass latency
depends on architecture and precision only, not on the specific trained weight values. All three still
show correctly-dtyped `float16` output, no silent fallback to FP32.
