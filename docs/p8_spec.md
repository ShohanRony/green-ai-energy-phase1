# P8 training campaign — specification reconstructed from code and logs only

**Purpose of this document:** describe what the P8 campaign (`train_p8.py`, `prune_models_p8.py`,
`scripts/auto_p8.sh`, `scripts/install_p8_service.sh`) actually does and did, from reading the
code and the logs it produced — not from any design intent that may have existed elsewhere.
Every claim below is marked **[from code]** (verified by reading the script) or **[unknown]**
(not determinable from code/logs alone). See `deviation_log.md` D22 for the process context
(ran as an untracked root systemd service before amendment A5 was adopted) and
`docs/incidents/2026-10-09/` for the full containment evidence (service status, journal,
`p8_auto.log`). Training is **not** resumed or validated as part of writing this document.

---

## Start time

**[from code/logs]** First `Starting job` log line: `2026-10-09 03:41:12` (local time,
`journalctl`/`p8_auto.log`, matching the service's first `Started p8-auto.service` event). The
first two attempts failed immediately with `ModuleNotFoundError: No module named 'torch'`
(environment issue, self-resolved by 03:42); real training began shortly after.

## Seeds

**[from code]** `scripts/auto_p8.sh` hardcodes three seeds: **1001, 1002, 1003** — looped for
every architecture and every pruning ratio. `train_p8.py --seed` sets both `torch.manual_seed`
and `random.seed`, and separately seeds the 45k/5k train/val split generator (`torch.Generator().
manual_seed(a.seed)`) with the same value. This is a different seed convention from the rest of
this project, which uses the single fixed seed **2026** throughout (`train_baseline.py`,
`materialize_checkpoints.py`) — P8's checkpoints are additional, not a replacement for the
seed-2026 checkpoints already used in every Stage 4/4b energy measurement to date. They live in
a separate directory (`checkpoints_p8/`, not `checkpoints/`).

## Epochs

**[from code]**, `scripts/auto_p8.sh`'s job definitions:
- FP32 baselines: `--epochs 30` (9 jobs: 3 architectures × 3 seeds).
- Prune + fine-tune jobs: `--epochs 25` (up to 27 jobs: 3 architectures × 3 ratios × 3 seeds).

## LR schedule

**[from code]**, `train_p8.py`:
- Optimizer: `SGD(lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)`.
- Schedule: `CosineAnnealingLR(opt, T_max=a.epochs)` — anneals to 0 over the job's own epoch count
  (30 for FP32 baselines, 25 for fine-tune jobs; the schedule resets per job, not shared across
  the prune→fine-tune transition).
- LR value: FP32 baseline jobs do not pass `--lr` in `auto_p8.sh`, so they use `train_p8.py`'s
  default, **0.1**. Fine-tune jobs explicitly pass `--lr 0.01`.

## Train/val split

**[from code]**, `train_p8.py`: a **45,000 / 5,000** split of the CIFAR-10 *training* set
(`torch.randperm(50000, generator=seeded-by-the-run's-own---seed)`, first 45k = train, last 5k =
held-out validation with no augmentation — plain `ToTensor`+`Normalize`, not the `RandomCrop`+
`RandomHorizontalFlip` used for the actual training subset). The CIFAR-10 *test* set (10,000
images, `train=False`) is untouched during training — used only once, at the very end, to compute
`test_acc` on the best-validation-accuracy checkpoint. This is the first held-out validation split
anywhere in this project; every other training script (`train_baseline.py`) uses train/test only,
no validation set (`deviation_log.md`/`stage5_analysis_plan.md` §6(f)'s "no validation split"
disclosure).

## Prune levels

**[from code]**, `prune_models_p8.py`: **30%, 50%, 70%** channel ratio (`ch_sparsity`), via
`torch_pruning.pruner.MagnitudePruner` with `MagnitudeImportance(p=2)`, ignoring each
architecture's final classification layer — the same procedure `materialize_checkpoints.py` uses
for the project's existing pruned checkpoints (not independently verified bit-identical to those;
unlike the D16-era BN-recalibration work, no cross-check against a traced reference was run here
— **[unknown]** whether P8's pruned checkpoints are numerically identical to the existing
seed-2026 pruned checkpoints at the same ratio).

## What each job type saves

**[from code]**:
- **FP32 baseline** (`train_p8.py`, no `--finetune-from`): saves `model.state_dict()` (plain
  state dict, not a full module) to `checkpoints_p8/{arch}_fp32_{seed}.pt`, plus a sibling `.json`
  with `test_acc`, `best_val_acc`, and the full per-epoch train/val log.
- **Prune** (`prune_models_p8.py`): saves the **full pruned `nn.Module` object**
  (`torch.save(model, out_path)`, not a state dict — requires `weights_only=False` to reload, and
  is not a TorchScript trace) to `checkpoints_p8/{arch}_pruned{ratio}_{seed}.pt`.
- **Fine-tune** (`train_p8.py`, `--finetune-from` set): loads the full pruned module
  (`torch.load(..., weights_only=False)`), fine-tunes it, and — because `a.finetune_from` is set
  — saves the **full fine-tuned module** (same `torch.save(model, out_path)` branch as the prune
  step, not a state dict) to `checkpoints_p8/{arch}_p{ratio}_ft_{seed}.pt`, plus a `.json` log.
- **Format note, relevant to the phase-1 audit's C1 finding:** none of P8's outputs are
  TorchScript traces. The existing pruned/FP16/INT8 checkpoints this project measures energy
  against are all `torch.jit.trace`'d (`materialize_checkpoints.py`); P8's pruned and fine-tuned
  checkpoints are plain pickled `nn.Module` objects. If these are ever used for energy
  measurement, they would need an explicit trace-and-save step first to be loaded the way
  `pilot.py` loads every other compressed state — otherwise they'd introduce a *third* runtime
  category (plain eager module, distinct from both the FP32 baselines' `state_dict`-rebuild path
  and every other state's traced path), compounding rather than resolving C1.

## Which checkpoints would be used for accuracy vs. energy

**[unknown] — not specified anywhere in the P8 code itself.** `train_p8.py`/`prune_models_p8.py`/
`auto_p8.sh` only produce and save checkpoints; nothing in them references `pilot.py`,
`results_stage4b/`, or any accuracy-evaluation script (`eval_artifacts.py`). The only place a
stated intent exists is `stage5_analysis_plan.md` amendment **A5, still DRAFT, not adopted**: it
proposes the new fine-tuned checkpoints be (a) evaluated for accuracy under §6's existing rule,
and (b) measured for energy directly in one dedicated extra session, reported as its own arm, not
pooled into the six-session confirmatory matrix. That is A5's *proposal*, not a fact established
by the P8 code — it is marked here as draft intent, not as what will happen.

## Current state (confirmed from `checkpoints_p8/`, `.p8_state/completed_jobs.txt`, logs)

**[from code/logs]**:
- 9/9 FP32 baselines complete (all three architectures × seeds 1001/1002/1003).
- 1/27 prune jobs complete (`resnet18_pruned30_1001.pt`).
- 0/27 fine-tune jobs complete. The first one (`ft_resnet18_30_1001`) failed ~318 times on a
  `torch.load` bug (missing `weights_only=False` at the time; `train_p8.py` line 81 now has it,
  edited 2026-10-09 10:50, **unverified** — see below) before reaching 4/25 epochs on its most
  recent attempt (ended 2026-10-09 20:00:18, service stopped). **No interrupted-checkpoint file
  exists for this job** — the power-loss-resilience save path only triggers on a
  `.power_state/stop` marker, not a generic process stop/SIGTERM, so those 4 epochs' progress is
  not recoverable; a future resume would restart this job from epoch 0.
- The current `train_p8.py`/`prune_models_p8.py` in this commit are the versions last run
  (edited 10:50, after the original bug was found) — **this review does not claim they are bug-free
  or validated**, only that they are what currently exists and were committed unchanged from
  what last ran, per the researcher's order.
