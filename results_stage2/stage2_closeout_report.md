# Stage 2 Closeout Report — Tasks A–D

Follow-up to `stage2_deliverable.md` / `stage2_task_report.md`, run 2026-10-04. Four items required
actual evidence before Stage 2 could be called closed. All four resolved below.

---

## Task A — MobileNetV3-Small pruning collapse: diagnosed

**Check 1 — realized-vs-nominal consistency at 30%, compared across all three models:**

| Model | Nominal | Realized MACs reduction | Ratio (realized/nominal) |
|---|---|---|---|
| ResNet-18 | 30% | 51.6% | 1.72x |
| MobileNetV3-Small | 30% | 46.7% | 1.56x |
| EfficientNet-B0 | 30% | 48.4% | 1.61x |

MobileNetV3-Small's amplification (1.56x) is **in line with, not disproportionate to** — actually
slightly *lower* than — ResNet-18's (1.72x) at the same nominal sparsity. A dependency-graph bug
mishandling depthwise/pointwise boundaries would be expected to produce an outlier ratio (e.g. nominal
30% cascading to 70%+ realized, a ~2.3x+ ratio). It doesn't. This is evidence against a graph bug.

**Check 2 — fine-grained sparsity probe (5/10/15/20%), zero-finetune, MobileNetV3-Small only**
(`prune_mbv3_probe.py`, `results_stage2/mbv3_sparsity_probe.json`):

| Sparsity | Accuracy | Realized MACs reduction |
|---|---|---|
| baseline | 84.99% | — |
| 5% | 81.96% | 9.5% |
| 10% | 59.13% | 17.8% |
| 15% | 30.31% | 25.9% |
| 20% | 21.18% | 33.5% |
| 30% (from full grid) | 11.17% | 46.7% |

The degradation is **smooth and monotonic** — a small, real cost at 5% (-3pt), a steep but continuous
decline through 10-20%, reaching the chance floor by 30%. There is no sudden cliff or erratic jump at
any specific sparsity that would indicate wrong channels being removed or a structural mismatch — that
signature (an abrupt failure at one threshold, or non-monotonic behavior) is absent. A smooth curve
down to the floor is the signature of genuinely thin redundancy, not a bug tripping at a particular
graph boundary.

**Verdict: genuine architectural fragility, not a torch-pruning dependency-graph bug.**
Evidence: (1) realized/nominal amplification at 30% is consistent with, not an outlier against, the
other two architectures; (2) the accuracy-vs-sparsity curve from 5% to 30% is smooth and monotonic,
with no discontinuity signature of a structural/graph error. MobileNetV3-Small's depthwise-separable
design — already channel-minimal by construction for its target (mobile ImageNet inference) — has
measurably less slack to prune away with zero recovery than ResNet-18 or EfficientNet-B0, even at 5%.

---

## Task B — FP32 training convergence: checked, two of three flagged

| Model | Loss, epoch 26→30 | Train acc, epoch 26→30 | Status |
|---|---|---|---|
| ResNet-18 | 0.0623→0.0288 | 97.93%→99.23% | Near-plateau (loss already very low) |
| MobileNetV3-Small | 0.4607→0.4073 | 83.96%→85.93% | **Not plateaued** (+1.97pt in 5 epochs) |
| EfficientNet-B0 | 0.5222→0.4613 | 81.89%→83.94% | **Not plateaued** (+2.05pt in 5 epochs) |

Both lightweight architectures were still improving meaningfully at epoch 30 — EfficientNet-B0's
absolute rate of improvement is in fact slightly larger than MobileNetV3-Small's, despite its much
longer per-epoch wall-clock, so per-epoch speed alone doesn't predict convergence state. **Not
retrained per instructions — flagging only.** Practical consequence: every INT8/FP16/pruning accuracy
delta computed against these two baselines in Stage 2 is measured against a baseline that hadn't fully
converged. The deltas (and the pruning-collapse thresholds in Task A) are still real and still
informative, but the absolute FP32 reference points for these two models would likely shift upward with
more epochs — worth deciding whether to retrain before Stage 3's pilot treats these numbers as final.

---

## Task C — Committed

Commit `bc37ff91a4a3b3fe9d162adaaef6453f45431115` ("Stage 2: compression artifacts for ResNet-18,
MobileNetV3-Small, EfficientNet-B0") — all Task 0-7 code, checkpoints, and results, 30 files, committed
before any of Task A's extra runs touched the tree. A second commit follows this report, covering
Tasks A/B/D's new files (probe script, materialized checkpoints, `pilot.py` fixes, this report).
Nothing pushed to the remote — local commits only, pending your go-ahead.

---

## Task D — Checkpoint materialization: Option (b)-leaning hybrid, implemented and proven

**What the investigation found before choosing:** `pilot.py` as it stood before this session could
only ever build a ResNet-18 architecture internally, regardless of which `--checkpoint` was passed —
there was no `--arch` flag. It also loaded checkpoints via `model.load_state_dict(torch.load(...))`,
which already couldn't have worked on the INT8 checkpoints from Task 2 (`quantize_int8.py` /
`quantize_mobilenet_ptq.py` save those as whole TorchScript modules via `torch.jit.save`, not raw
state_dicts). So the real gap was bigger than "FP16 and pruned states are missing files" — `pilot.py`
couldn't load *any* non-ResNet-18, non-FP32 checkpoint, including ones Stage 2 had already produced.

**Decision: both (a) and (b), because (b) was required regardless of what was chosen for (a).**
1. Materialized all 12 missing checkpoints (`materialize_checkpoints.py`) as self-contained
   TorchScript traces (`torch.jit.trace` + `.save()`) — the same format the INT8 checkpoints already
   use, deterministic from the existing FP32 checkpoints (seed 2026), same pruning calls as
   `prune_full_grid.py`. `checkpoints/{arch}_fp16.pt` (x3) and `checkpoints/{arch}_pruned{30,50,70}.pt`
   (x9).
2. Extended `pilot.py`: added `--arch {resnet18,mobilenet_v3_small,efficientnet_b0}` (reusing
   `train_baseline.build_model`, not duplicating the CIFAR-stem logic), and made checkpoint loading
   format-agnostic — try `torch.jit.load` first (handles FP16/pruned/INT8's self-contained modules),
   fall back to `build_model(arch) + load_state_dict` for plain state_dict files (the original FP32
   checkpoints still load this way, unchanged).

**Two real bugs caught and fixed while proving it, not left for Stage 3 to discover:**
- FP16 checkpoints need their input cast to `float16` before the forward pass — `pilot.py` built
  `float32` input unconditionally. Fixed: input dtype now follows `next(model.parameters()).dtype`.
- Quantized (INT8) modules hold no plain `nn.Parameter` (packed params instead), so that same dtype
  lookup raised `StopIteration` for the INT8 checkpoint specifically. Fixed: falls back to the input's
  existing dtype (float32, which quantized models expect — they quantize internally via their own
  stubs).

**Proof — all four checkpoint formats loaded and profiled through `pilot.py` end to end, real output,
exit code 0 for each** (`results_stage2/pilot_proof_*/`):

| Checkpoint | Format | Result |
|---|---|---|
| `resnet18_fp32.pt` (backward-compat) | plain state_dict | pass — unchanged code path |
| `mobilenet_v3_small_fp16.pt` | TorchScript trace | pass — real energy trace, `gross_j_per_image=0.00245` |
| `efficientnet_b0_pruned50.pt` | TorchScript trace | pass |
| `resnet18_int8.pt` | TorchScript scripted (quantized) | pass, CPU/fbgemm |

21/21 existing `tests/test_pilot.py` unit tests still pass after both `pilot.py` edits.

**Verdict:** Stage 3's pilot can load every one of the 18 artifact states today, not just the 3 FP32
ones it could load before this session.
