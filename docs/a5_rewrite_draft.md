# A5 rewrite draft — accuracy arm (draft only, not registered, nothing decided)

**Status:** a draft for discussion. **Nothing in this document is decided** — the researcher and
supervisor decide whether, and how, to adopt any of it. `stage5_analysis_plan.md`'s existing A5
(DRAFT, pending supervisor review) is unchanged by this document; this is a separate draft that
may inform a future dated revision of A5, not a replacement for it. No training is run as part of
writing this document.

## (a) Variance arm

- **Recipe, exactly the registered one** (`checkpoints/*_fp32.json`, confirmed): 50,000-image
  training set (no validation split — the registered recipe's own standing limitation,
  `stage5_analysis_plan.md` §6(f)), ResNet-18 **30 epochs**, MobileNetV3-Small **60 epochs**,
  EfficientNet-B0 **60 epochs**, SGD with Nesterov momentum, `lr=0.1`, `weight_decay=5e-4`,
  cosine LR schedule, batch size 128.
- **Change proposed: extra seeds, same recipe otherwise.** Train each architecture's FP32 baseline
  multiple times, varying only the seed — everything else (epochs, LR schedule, optimizer, split)
  identical to what's already registered. **Purpose:** this project's accuracy rule (§6) currently
  has no estimate of training-run-to-run variance at all — every existing checkpoint comes from one
  fixed seed (2026), disclosed as a limitation in §6(f) ("no multi-seed replication... the Wilson CI
  and McNemar test... say nothing about training-run-to-run variability"). A variance arm would be
  the first data in this project that actually measures that gap, rather than just disclosing it.
- **No training run as part of this draft** — this section is a proposal, not an execution.

## (b) Fine-tune arm

- **Change proposed:** a dedicated **45,000/5,000 train/validation split** of the CIFAR-10 training
  set, used specifically for fine-tuning pruned models — distinct purpose from (a)'s variance arm,
  though it borrows the same split *mechanism* P8 already implements (`docs/p8_spec.md`, confirmed
  `torch.randperm(50000, generator=seeded)`, first 45k train / last 5k held-out validation, no
  augmentation on the validation subset).
  - Held-out validation lets early stopping or best-checkpoint selection use a signal that isn't
    the test set itself — something no training script in this project has ever had (§6(f)'s
    "structural absence of a validation split" critique applies to every existing checkpoint, not
    just the FP32 baselines).
- **Scope, matching the existing A5 draft's own shape** (`stage5_analysis_plan.md` A5): fine-tune
  from already-pruned weights (not from scratch), pruned30 and pruned50 as the primary targets,
  pruned70 optional, labelled "prune + fine-tune (not post-training)" wherever it appears, same as
  A5 already specifies. This draft does not change that framing — only adds the validation-split
  detail A5's original text left unspecified.

## Comparison with `docs/p8_spec.md` — every divergence, checked directly

| | Registered recipe (`*_fp32.json`) | This draft's variance/fine-tune arm | P8 (`docs/p8_spec.md`, as actually run) |
|---|---|---|---|
| Train/val split | None (50k train, test used once) | **45k/5k**, new for (b) | **45k/5k** — already implemented, same split mechanism |
| Seeds | 2026 (fixed, single) | Multiple (proposed, count below) | **1001, 1002, 1003** — already run, a separate seed family |
| Epochs — ResNet-18 | 30 | 30 (unchanged, per (a)) | **30** — matches |
| Epochs — MobileNetV3-Small | 60 | 60 (unchanged, per (a)) | **30** — **does not match**; re-introduces the undercooked 30-epoch baseline this project already found inadequate and retrained away from (`stage2_retrain_report.md`) |
| Epochs — EfficientNet-B0 | 60 | 60 (unchanged, per (a)) | **30** — same divergence as MobileNetV3-Small |
| Optimizer/LR/batch/augmentation | SGD nesterov, lr 0.1, wd 5e-4, cosine, batch 128, `RandomCrop`+`RandomHorizontalFlip` | unchanged | **identical** — confirmed matching, `docs/p8_spec.md`'s own side-by-side table |
| Fine-tune arm | Not part of the registered recipe | Pruned30/50(/70 optional), 45k/5k split, labelled "not post-training" | P8 has its own fine-tune jobs (`ft_{arch}_{ratio}_{seed}`, 25 epochs, `lr=0.01`) — **same general shape** (fine-tune from pruned weights) but on P8's own 30-epoch-baseline-derived pruned checkpoints, not the registered 60-epoch ones for MobileNetV3-Small/EfficientNet-B0 |
| Checkpoint format | TorchScript trace (`torch.jit.trace`) | Not yet decided — flagged below | P8 saves plain pickled `nn.Module` objects, **not** TorchScript traces (`docs/p8_spec.md`'s own "Format note") — a third runtime category if ever measured for energy, per that document's own warning |

**Divergences, stated plainly, as the order requires:**
1. **P8's MobileNetV3-Small and EfficientNet-B0 baselines used 30 epochs, not the registered 60** —
   the single most consequential divergence; it means P8's non-ResNet-18 checkpoints rest on the
   same undercooked convergence depth this project already identified and fixed once.
2. **P8's seeds (1001-1003) are a different family from both the registered single seed (2026) and
   whatever seed count this draft's variance arm (a) would use** — not wrong, just not
   interchangeable; P8's checkpoints cannot silently stand in for either a registered-recipe
   checkpoint or a dedicated variance-arm checkpoint.
3. **P8's checkpoint format (plain pickled module) is not TorchScript**, unlike every other
   checkpoint this project measures energy against — this draft's fine-tune arm, if adopted, should
   specify its output format explicitly rather than inherit P8's by default, given P8's own document
   already flags this as a problem waiting to compound C1 (the phase-1 audit's runtime-confound
   finding) if ever measured as-is.
4. **P8's fine-tune jobs are 25 epochs at `lr=0.01`** — within A5's original "20-30 epoch" range, so
   not a divergence from A5's own framing, but worth noting P8's specific numbers already exist as a
   reference point if this draft's fine-tune arm reuses them rather than defining new ones.

## Proposed seed count and compute-time estimate

- **Proposed seed count for the variance arm (a): 5 seeds per architecture** (in addition to the
  standing seed 2026, so 2026 plus 4 new, or 5 entirely new — not decided here). **Assumption,
  unverified:** 5 is a common minimum in the ML-variance literature for a usable (if still wide)
  standard-error estimate, not a figure derived from this project's own data — no pilot variance
  measurement exists yet to size this properly. A smaller pilot (e.g. 3 seeds) before committing to
  the full count is one option, not decided here either.
- **Compute-time estimate — assumption, unverified, not measured:** scaling from this project's own
  already-logged training wall-times (not independently re-measured for this document) —
  ResNet-18's 30-epoch baseline and MobileNetV3-Small/EfficientNet-B0's 60-epoch baselines are the
  only real reference points available. Treating one architecture's one-seed training time as a
  unit, 5 extra seeds × 3 architectures = 15 additional full training runs for the variance arm
  alone, before the fine-tune arm's own runs are added on top. **No actual per-run wall-clock number
  is asserted here** — this project's existing logs would need to be read for real per-epoch
  timings before any concrete hour/day estimate could be defended; stating one without that check
  would be exactly the kind of unverified-number problem this project's own disclosure discipline
  exists to avoid.

## Nothing decided

This document proposes; it does not adopt. Whether any of (a), (b), the seed count, or the
compute-time tradeoff is worth running is the researcher's and supervisor's decision — consistent
with `stage5_analysis_plan.md` A5's own "DRAFT, pending supervisor review" status, which this
document does not change.
