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

## P8 recipe vs. the registered recipe, side by side (2026-10-10)

**[from code]**, `train_p8.py` vs. `train_baseline.py` and `checkpoints/*_fp32.json`:

| | Registered (`train_baseline.py` / `*_fp32.json`) | P8 (`train_p8.py` / `auto_p8.sh`) |
|---|---|---|
| Optimizer | SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True | **identical** |
| LR schedule | `CosineAnnealingLR(T_max=epochs)` | **identical** (`T_max=a.epochs`, resets per job) |
| Batch size | 128 | 128 (`train_p8.py` default, not overridden in `auto_p8.sh`) |
| Augmentation | `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` | **identical** |
| Seed | **2026**, one fixed seed throughout | **1001 / 1002 / 1003**, three seeds — this is P8's actual new contribution (multi-seed) |
| Train/val split | none (train on full 50k, test set used once at the end) | **45,000 / 5,000**, new — P8's other actual new contribution |
| Epochs — ResNet-18 | **30** | **30** — matches |
| Epochs — MobileNetV3-Small | **60** (retrained after the original 30-epoch run was found not plateaued, `stage2_retrain_report.md`) | **30** — confirmed directly from every `fp32_mobilenet_v3_small_*` job line in `auto_p8.sh` (`--epochs 30`, all three seeds) — **does NOT match; re-introduces the exact 30-epoch undercooked baseline this project already found inadequate and fixed** |
| Epochs — EfficientNet-B0 | **60** (same reason) | **30** — same divergence, confirmed from `auto_p8.sh`'s `fp32_efficientnet_b0_*` job lines, all three seeds |

**Answering the question directly: P8's MobileNetV3-Small and EfficientNet-B0 FP32 baselines used
30 epochs, not 60.** This is a real recipe mismatch against the registered convergence depth for
those two architectures specifically, not a difference the campaign disclosed or seems aware of —
nothing in `train_p8.py`/`auto_p8.sh` references the 60-epoch retrain decision or
`stage2_retrain_report.md` at all.

## The fine-tune job that reached epoch 4

**[from code/logs]** `ft_resnet18_30_1001` — `train_p8.py --arch resnet18 --epochs 25 --lr 0.01
--seed 1001 --finetune-from checkpoints_p8/resnet18_pruned30_1001.pt --out
checkpoints_p8/resnet18_p30_ft_1001.pt` (`auto_p8.sh`'s fine-tune job template). Logged accuracy
at each epoch reached (from `p8_auto.log`, final attempt): epoch 1 train/val 0.9514/0.9080, epoch
2 0.9613/0.9104, epoch 3 0.9626/0.9022, epoch 4 0.9653/0.9006.

**Why it stopped: a clean, explicit service stop, not a crash.** `journalctl -u p8-auto.service`
for this exact boot (`27625af78f4a4efcb33001f10565a355`, 19:57:41-22:15:20) shows exactly one
start/stop pair: `Started` 19:57:44, then `Stopping... / Deactivated successfully / Stopped` at
**20:00:18** — systemd's signature for an externally-requested stop (e.g. `systemctl stop`), not
a failure exit (no "Main process exited, code=...FAILURE" line, which appears elsewhere in this
same log for the earlier `weights_only` crash loop). No OOM-killer or segfault entry exists in the
kernel log for this window either (checked directly). The service never restarted for the rest of
that boot — consistent with a deliberate stop (`Restart=on-failure` doesn't trigger after a clean
stop). `p8_auto.log`'s last write (`20:00:17`) matches this timestamp. This is a separate, earlier
event from the researcher's later ~21:38-21:40 quarantine/disable action (D23) — by 21:38 the
service was already inactive (stopped at 20:00:18), so quarantining it then didn't need to stop
anything already-stopped, consistent with no further `Stopping`/`Stopped` journal lines appearing
later in the same boot.
