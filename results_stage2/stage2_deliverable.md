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
| 3 | ResNet-18 | FP16 | *(converted in-memory at eval/measurement time — no separate on-disk artifact; `.half()` of the FP32 checkpoint)* |
| 4 | ResNet-18 | Pruned 30% | *(produced by `prune_full_grid.py`; not persisted to disk as a separate file this run — regenerable deterministically from seed 2026)* |
| 5 | ResNet-18 | Pruned 50% | *(same as above)* |
| 6 | ResNet-18 | Pruned 70% | *(same as above)* |
| 7 | MobileNetV3-Small | FP32 (baseline) | `mobilenet_v3_small_fp32.pt` |
| 8 | MobileNetV3-Small | INT8 | `mobilenet_v3_small_int8.pt` |
| 9 | MobileNetV3-Small | FP16 | *(in-memory, as above)* |
| 10 | MobileNetV3-Small | Pruned 30% | *(regenerable, as above)* |
| 11 | MobileNetV3-Small | Pruned 50% | *(regenerable, as above)* |
| 12 | MobileNetV3-Small | Pruned 70% | *(regenerable, as above)* |
| 13 | EfficientNet-B0 | FP32 (baseline) | `efficientnet_b0_fp32.pt` |
| 14 | EfficientNet-B0 | INT8 | `efficientnet_b0_int8.pt` |
| 15 | EfficientNet-B0 | FP16 | *(in-memory, as above)* |
| 16 | EfficientNet-B0 | Pruned 30% | *(regenerable, as above)* |
| 17 | EfficientNet-B0 | Pruned 50% | *(regenerable, as above)* |
| 18 | EfficientNet-B0 | Pruned 70% | *(regenerable, as above)* |

**Note on non-persisted states:** FP16 and the 9 pruned variants were evaluated in-process
(`fp16_speedup_check.py`, `prune_full_grid.py`) rather than saved as 9 extra checkpoint files, since
Stage 2's acceptance criteria call for accuracy + realized-compression *evidence*, not necessarily 18
files on disk. If Stage 3's RQ1 pilot needs the pruned/FP16 weights as standalone files (e.g., to load
via `pilot.py --checkpoint`), they're trivially regenerable — `prune_full_grid.py` and
`fp16_speedup_check.py` are deterministic (seed 2026) given the FP32 checkpoints already saved. Flagging
this now rather than assuming it's fine.

## 3. Accuracy Table (all 18 states, FP32 = reference)

| Model | FP32 (ref) | INT8 | FP16* | Prune 30% | Prune 50% | Prune 70% |
|---|---|---|---|---|---|---|
| ResNet-18 | **93.09%** | 93.14% | 93.09% | 64.84% | 17.58% | **10.35% (collapsed)** |
| MobileNetV3-Small | **84.99%** | 84.20% | 84.99% | **11.17% (collapsed)** | **11.16% (collapsed)** | **10.00% (collapsed)** |
| EfficientNet-B0 | **83.59%** | 78.37% | 83.59% | 61.14% | 20.46% | **10.00% (collapsed)** |

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
| MobileNetV3-Small | FP32 | — (0%) | — | — | 1.00x | baseline: 5.80M MACs, 1.53M params |
| MobileNetV3-Small | INT8 | 8-bit | n/a | n/a | — (CPU-only) | `torch.ao.nn.quantized` modules confirmed present |
| MobileNetV3-Small | FP16 | 16-bit | n/a | n/a | 1.96x | output dtype confirmed `float16` |
| MobileNetV3-Small | Prune 30% | 30% | 46.7% | 50.1% | 1.33x | `torch-pruning` report — **accuracy collapsed** |
| MobileNetV3-Small | Prune 50% | 50% | 69.1% | 73.6% | 2.06x | same — **accuracy collapsed** |
| MobileNetV3-Small | Prune 70% | 70% | 86.8% | 90.0% | 3.28x | same — **accuracy collapsed** |
| EfficientNet-B0 | FP32 | — (0%) | — | — | 1.00x | baseline: 33.28M MACs, 4.02M params |
| EfficientNet-B0 | INT8 | 8-bit | n/a | n/a | — (CPU-only) | `torch.ao.nn.quantized` modules confirmed present |
| EfficientNet-B0 | FP16 | 16-bit | n/a | n/a | 2.01x | output dtype confirmed `float16` |
| EfficientNet-B0 | Prune 30% | 30% | 48.4% | 50.0% | 1.38x | `torch-pruning` report |
| EfficientNet-B0 | Prune 50% | 50% | 71.2% | 73.5% | 2.25x | same — accuracy badly degraded (20.5%), not fully collapsed |
| EfficientNet-B0 | Prune 70% | 70% | 88.1% | 89.8% | 3.60x | same — **accuracy collapsed** |

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
  fully collapsed at 70% (10.35%).
- **MobileNetV3-Small**: collapses at **every tested sparsity level, including 30%** (11.17% / 11.16% /
  10.00%). This architecture has essentially no redundancy to prune away with zero recovery — a result
  worth stating plainly in the paper, not softened.
- **EfficientNet-B0**: survives 30% (61.14%), severely degraded at 50% (20.46%, above chance but not
  usable), collapsed at 70% (10.00%).

**Task 4 decision, resolved (not TBD):** after seeing ResNet-18's 70% zero-finetune result collapse to
10.35% vs. 84.24% with a 3-epoch brief-recovery variant (see `results_stage2/task4_decision_checkpoint.json`),
the user confirmed (2026-10-04): **zero-finetune policy stands, uniformly, across all 3 models and all
3 sparsity levels.** Collapse is reported as a finding about the limits of post-training-only
compression, not patched over by silently introducing a recovery step the proposal's methodology
doesn't call for.

**Which INT8 path EfficientNet-B0 needed:** confirmed no official `torchvision.models.quantization`
wrapper exists for EfficientNet-B0 (same situation as MobileNetV3-Small, worse than ResNet-18). Used
the same manual FX graph-mode PTQ pipeline built for MobileNetV3-Small (generalized via an `--arch`
flag rather than duplicated). Verdict: **pass** — real int8 kernels confirmed, but with a real 5.2-point
accuracy cost (83.61% → 78.37%) notably larger than MobileNetV3-Small's 0.79-point cost, consistent
with SiLU/squeeze-excite architectures being less quantization-robust than ReLU-based ones.

**FP16 consistency across all three models:** confirmed. All three show genuine GPU speedup (1.78x /
1.96x / 2.01x) with correctly-dtyped `float16` output, no silent fallback to FP32. EfficientNet-B0
(new this stage) behaves consistently with the other two, already-verified architectures.
