# Overnight Work Order: Stage 4 Rework (x86 LOQ), 2026-10-06/07

**Who reads this:** the local coding-assistant session on the LOQ, working unattended overnight.
**Authority:** Shohan has approved every decision in this file (2026-10-06, 23:26 local). This file is the specification. Supervisor review: `docs/stage4-supervisor-review.md`.
**Goal tonight:**
- Fix everything in the review that does not need new energy measurements.
- Prepare the confirmatory **Stage 4b** design, register it, and tag it.
- Launch **Stage 4b Session 1** as a detached background job.
- Leave a morning report.

**Not tonight:** any Stage 5 statistics, any M1 work, the RQ1 CodeCarbon pass, the batch=16 sensitivity run, and Stage 4b Sessions 2–4. Shohan launches those later with the runner script you build in Step 7.

---

## 0. Hard rules (apply to every step)

1. **Never modify or delete existing data.** This covers `results*/**/raw.jsonl`, every `summary*.csv`, existing `environment.json` files, and the 18 existing checkpoints. Write only to new files or new directories.
2. **No Stage 5 statistics.** Do not run any hypothesis test or compute a confirmatory ratio on any dataset. Computing accuracy and Wilson CIs in Step 2 is allowed; that is measurement, not inference.
3. **No concurrent GPU work.** Training, evaluation and measurement must never overlap. Before every GPU step, confirm with `nvidia-smi` that no other compute process is running. While Session 1 runs (Step 8), do **nothing else** on this machine: no GPU, no heavy CPU, no pip installs, no tests.
4. **No sudo** except the existing `/usr/local/sbin/set_cpu_governor.sh` wrapper. Do not install system packages. Do not change system settings. Do not reboot.
5. **Use git safely.** Commit after every step with a clear message and push after every commit, which is the backup if the machine dies. Never use `--force`, never rewrite history, never touch other repositories.
6. **Never fabricate.** Every number in any document must come from a file you just read or a command you just ran. If something cannot be found, write `TODO — not found` and move on.
7. **Do not wait for a human, and do not improvise science.**
   - If a step fails or produces something unexpected, record it in the morning report under "Blocked/unexpected", do **not** invent a workaround that changes the design, and continue with the next independent step.
   - Never change the design in §6 after it is tagged.
8. **Save tokens.** Write long outputs to files and print only short summaries. Do not paste raw logs into the conversation.
9. **Priority if you run out of time or tokens:** Steps 1–3, then 5–8, then 4. Because every step is committed, stopping anywhere leaves a consistent state.

## Preflight (do first, about 5 minutes)

- `git status` must be clean. Run `git pull`, and record the HEAD hash.
- Add this file as `docs/overnight-instructions-2026-10-07.md` and the review as `docs/stage4-supervisor-review.md`. Shohan will have placed both in the repo root or `docs/`. Commit them.
- Record in the morning report:
  - uptime
  - governor (must be `performance`; fix it with the wrapper if not)
  - `/sys/firmware/acpi/platform_profile` (must be `performance`)
  - AC online (`/sys/class/power_supply/ACAD/online`)
  - `nvidia-smi` process list
  - free disk space
- If `power_watchdog.py` is not running, start it: `nohup python3 power_watchdog.py > power_watchdog.log 2>&1 &`.

---

## Step 1: Documentation fixes (no compute)

Make one commit per logical group. Do not rewrite registered text in `docs/stage5_analysis_plan.md`. Plan changes go in as **dated amendments** in its Amendments section.

1. **Plan amendment A2.** List every in-place edit made to the plan after A1, using `git diff 0a358c6 HEAD -- docs/stage5_analysis_plan.md` and the commits `abed585` and `6d361b5`. Give each one a date and reason. Correct A1's stale "D16 not yet remediated". From now on, no in-place edits to registered text.
2. **Stale status lines:**
   - D8 and plan §7: the stability reruns *were* executed in `e2df9c1`. Say so and summarise them.
   - D15: D16 *was* remediated (`8367662`, `70f5d1d`).
   - `docs/stage4-implementation-brief.md`:
     - §2: M1 access exists. Point to D5.
     - §5(c): "~3 hours" was really 3:57 / 4:19. Point to D7.
     - §9: the ResNet-18 pruned50/70 Stage 4 range is 44.1–53.1 W. The 35 W figure is pre-flight b4/b8.
   - `docs/phase1-execution-plan.md`: mark CodeCarbon wrapping as done, and "psys vs package check" as **missed, see D16**.
   - `results_stage3/stage3_deliverable.md`: add a note at the top that its INT8 energy and power figures are package+psys (D16), not "RAPL package".
   - Add a short `D16-NOTE.md` to each of the 6 Stage 4 CPU directories and to `codecarbon_check/resnet18_int8`, pointing to `summary_package.csv`.
3. **Mechanism wording (D7, brief §5.3/§9, preflight findings Task 1).**
   - Replace "the GPU idles between launches" with what the evidence supports.
   - `resnet18_pruned70_dmon.log` shows SM active 89–96% during active windows at about 1965 MHz, which is consistent with low per-kernel occupancy, not idle gaps. `mbv3_fp32_dmon.log` shows about 58% active SM, so gaps are plausible there. Only 2 of 12 conditions had dmon.
   - Pre-flight dips in heavy configs (FP16@b64, pruned70@b32) remain unexplained.
   - Remove "not a mystery" / "not an unresolved mystery".
4. **Plan §6 (amendment A2).**
   - Record that **Shohan confirmed the 99% primary / 99.9% strict rule on 2026-10-06**.
   - Replace the "expected to fail" sentence with the list computed from measured accuracies after Step 2. Fill it in at the end of Step 2.
5. **New deviation entries** (next free numbers, with the same fields as existing entries):
   - **D17, between-session drift and order confound.**
     - Compare each `results_stage4/dip_stability_check/*` run with its primary run: J/image, power and images/s. The range is about −12.1% to +3.7% in J/image; 9 of 12 are lower in the fresh session.
     - Note that within each model the compressed states were measured after FP32, at rising uptime.
     - Note that INT8 and FP32-CPU were measured in different boots, about 7 hours apart.
     - Consequence: the original Stage 4 matrix is reclassified as **exploratory** (one ordered session per condition). The confirmatory dataset is Stage 4b (§6).
   - **D18, brief-recovery pruning arm added** (Step 3): secondary arm; zero-finetune remains primary.
   - **D19, harness changes for Stage 4b** (Step 5): dual sensor, CPU affinity, images/s, session ID.
   - **D20, the supervisor's own design errors.** Batch=1 one-session-per-condition sequential design; a stability check specified by regime label only, not energy. Both are attributed to the supervisor's recommendations.
6. **Disclosures** (add to the deviation log or the plan limitations, wherever they fit):
   - **INT8 pipelines differ.** ResNet-18 uses eager PTQ, 1,280 calibration images, no AdaRound/BC, unseeded. MobileNetV3-Small and EfficientNet-B0 use FX PTQ with layer-wise AdaRound and bias correction on 512 images.
   - **Training:** 30 epochs for ResNet-18 vs 60 for the others; single seed 2026; cuDNN not deterministic; no validation split, and decisions were made after seeing test accuracy.
   - **Pruning labels:** "30/50/70%" is a per-layer **channel ratio**. Give realised MAC reduction from `results_stage2/task4_full_grid.json`: ResNet-18 52/75/91%, MobileNetV3-Small 47/69/87%, EfficientNet-B0 48/71/88%. Use "channel ratio 30%" etc. in all new tables.
   - **Input:** every batch=1 window repeats the same test image.
   - **Summary fields:** `approx_mde_j_per_image` and `candidate_for_confirmation` use within-rep A/A differences and understate uncertainty. Do not cite them.
   - **Plausibility flags:** plan §8 says "zero plausibility flags Stage 1–4", but the Stage 1 exit test has one documented flag. Correct the scope.
   - **Cold rep:** quoted 736.02 / 266.21 J include the cold rep. Summary values are 736.30 / 266.24.
7. **README and repo hygiene:**
   - RAPL is read via **sysfs powercap**, not perf-events.
   - Each run uses one backend. This changes in Step 5; update after it.
   - Tests: either move `test_pilot.py` into `tests/` or fix the README command so the D16 tests actually run.
   - `requirements.txt`: record the versions that actually produced the checkpoints (torch 2.7.1+cu118, torchvision 0.22.1, from `checkpoints/*.json`) and add `torch_pruning` at its installed version (`pip show torch_pruning`).
   - Note the hard-coded data path `/home/shohan/green-ai-research/data` in the README. Do not refactor scripts tonight.

## Step 2: Accuracy completion (GPU and CPU, about 20–40 minutes)

Write `scripts/eval_artifacts.py` and run it.

- **What to evaluate:** the **exact artifacts `pilot.py` loads** (`torch.jit.load` on `checkpoints/{model}_{state}.pt`; FP32 via the same fallback `pilot.py` uses).
- **Data:** full CIFAR-10 test set (n=10,000), `shuffle=False`. Use the same normalisation as `pilot.py`: mean (.4914,.4822,.4465), std (.247,.243,.261). Use the same test transform as Stage 2 (ToTensor + Normalize, no augmentation).
- **Device rule:**
  - GPU states (FP32, FP16 with `.half()` inputs, pruned) run on CUDA.
  - INT8 runs on CPU.
  - Also evaluate **FP32 on CPU**, so that INT8 has a same-device McNemar partner.
- **Outputs:** to new `results_accuracy/`:
  - per artifact, `{model}_{state}_{device}_preds.npy` (int8 predicted class per image) plus one shared `labels.npy`
  - `accuracy_summary.csv` with accuracy, a 95% Wilson CI (n=10,000), and the difference from the Stage 2 reported number
- **Checks to report, not fix:**
  - Any traced artifact whose accuracy differs from Stage 2's live-module number by more than 0.05 percentage points (5 images).
  - FP16 accuracy, now measured for the first time. If it differs from FP32, say so plainly. Update `stage2_deliverable.md` with a note at the top: "FP16 accuracy was not measured in Stage 2; measured values in results_accuracy/".
- Then compute, for every compressed state, accuracy as a % of its same-model FP32 (same device). List which states pass 99% and which pass 99.9%. Fill this list into the §6 amendment (Step 1.4). This is a descriptive tally, not a statistic.
- Commit.

## Step 3: Pruned-state controls and the recovery arm (GPU, about 30–60 minutes)

1. **Regenerate the live pruned modules** using the same code path and seed as `prune_full_grid.py` and `materialize_checkpoints.py`, without overwriting anything.
   - For each of the 9, verify it is the **same model** as the stored artifact: equal MACs and params, and output max-abs-diff below 1e-4 against `checkpoints/{model}_pruned{N}.pt` on a fixed batch of 64 test images.
   - If any does not match, record it as blocked and skip that model's Step 3 items.
2. **Predicted-class histogram** for each zero-finetune pruned state, from the Step 2 predictions. Save it to `results_pruning_controls/`.
3. **BN recalibration control.** On a deep copy of each live pruned module:
   - Reset BN running stats and set `momentum=None` (cumulative average).
   - Run forward passes in `train()` mode under `torch.no_grad()` on 2,000 **training** images (fixed seed 2026, no augmentation, batch 128).
   - Then `eval()` and test accuracy.
   - Save to `results_pruning_controls/bn_recal.csv`. Do not save these models as new states; this is a diagnostic only.
4. **Brief-recovery arm.**
   - Fine-tune each of the 9 live pruned modules with **exactly the recipe in `prune_decision_checkpoint.py`**: read `RECOVERY_EPOCHS`, the optimiser (SGD lr 0.01, momentum 0.9, wd 5e-4), batch size, scheduler if any, and the training augmentation from `train_baseline.py`.
   - Use the **training set only**, seed 2026, and take the final epoch. There is no selection on the test set.
   - Trace and save with the same `trace_and_save` method as `checkpoints/{model}_pruned{N}_ft.pt`. These are new files; never overwrite existing ones.
   - Save training logs and a JSON with the recipe, MACs/params (which must equal the zero-finetune version) and the final train loss.
   - Then run `scripts/eval_artifacts.py` on these 9 artifacts too (with preds).
   - The power-loss resilience pattern (stop marker between epochs) must be respected, the same way `train_baseline.py` does it.
5. Commit, plus an update to D18 with results (accuracies, and the 99% tally for the recovery arm).

## Step 4: INT8 pipeline ablation (CPU, optional, lowest priority)

Only if time and tokens remain, and **before** Step 8 starts, never during it.
- For MobileNetV3-Small and EfficientNet-B0, run the FX PTQ path with AdaRound and bias correction **disabled** (plain PTQ, same 512 calibration images, same seed) to a scratch location.
- Evaluate accuracy and compare it with the shipped INT8 artifacts.
- Report it in the morning report and D-log as evidence on whether AdaRound/BC changed anything.
- Do **not** replace the shipped INT8 checkpoints.

## Step 5: Harness patch for Stage 4b (one commit, unit-tested)

In `pilot.py`, keep all existing behaviour and fields (backward compatible), and add:

1. **Dual sensor in CUDA runs.**
   - Instantiate a second sensor for RAPL (all top-level domains, named as in the D16 code).
   - Read it in the **same sampler call** as NVML, so both share timestamps. Store it in the trace as an extra element, or as a parallel trace.
   - Per window, log:
     - `gpu_energy_j` (NVML; equals the existing `energy_j` for CUDA)
     - `cpu_package_energy_j`
     - `cpu_psys_energy_j` (secondary)
     - `system_energy_j = gpu_energy_j + cpu_package_energy_j`
   - `energy_j` keeps its current meaning, so old data stays comparable.
   - `power_regime` and `plausibility_flag` stay computed on GPU power only.
2. **Per-boundary summary.** `summarize()` adds per-image means for each boundary: `gpu_j_per_image_mean`, `cpu_package_j_per_image_mean`, `system_j_per_image_mean`, and `images_per_s_mean`.
3. **Images per second.** Add `images_per_s = batches*batch/duration_s` to every window row.
4. **CPU affinity.** Add `--cpu-affinity pcores|none` (default `none`, so old commands behave the same).
   - With `pcores`, read the P-core logical CPU list from `/sys/devices/cpu_core/cpus` (hybrid kernel interface). If that file is missing, raise; do not guess.
   - Apply `os.sched_setaffinity(0, cpus)` before loading torch.
   - Log the list and the applied affinity in `environment.json`.
5. **Labels.** Add `--session-id` and `--condition-label` (free text), logged in `environment.json`.
6. **Tests.** Add unit tests for:
   - dual-sensor integration on a synthetic trace
   - the `system_energy_j` sum
   - parsing of the affinity CPU list
   - the `images_per_s` field

   Run all tests.
7. **Validation.** Make two 3-rep runs into a scratch directory **outside** `results*`, and delete it afterwards:
   - `resnet18_fp32` on CUDA with `--cpu-affinity pcores`
   - `resnet18_int8` on CPU with `--cpu-affinity pcores`

   Verify that:
   - the fields exist
   - `system = gpu + package` holds within 1e-9
   - `cpu_package_energy_j` is positive and plausible (it should exceed idle package of about 12 W × window)
   - affinity is logged

   Report the numbers in the morning report.
8. Update the README "What's here" section, plus D19. Commit and push.

## Step 6: Register the Stage 4b design (before any Stage 4b data exists)

Create `docs/stage4b-design.md` with exactly this content (you may format it, but do not change the substance), and add plan amendment **A3** pointing to it.

**Stage 4b: confirmatory x86 dataset**

- **Conditions (30).** Per model (ResNet-18, MobileNetV3-Small, EfficientNet-B0):
  - FP32 (cuda)
  - FP16 (cuda)
  - pruned30/50/70 zero-finetune (cuda)
  - pruned30/50/70 `_ft` recovery arm (cuda)
  - INT8 (cpu)
  - FP32 (cpu)

  If a Step 3 artifact is blocked, drop only that condition and list it.
- **Fixed settings:** `--sizes 32 --batches 1 --windows 5 --interval 0.4 --warmup 3 --threads 4 --cpu-affinity pcores`, governor and platform profile `performance`, CodeCarbon off.
- **Reps:** 7 per condition per session (the first is cold and discarded, leaving 6 pairs).
- **Sessions:** **4**, each a **separate boot**, spread over at least 2 calendar days.
  - Session 1 runs tonight in the current boot after preparation, with its uptime logged.
  - Sessions 2–4 each start within 15 minutes of a fresh reboot.
  - The number of sessions is fixed at 4. No sessions are added or removed based on results.
- **Order:** each session runs all 30 conditions in a **random order** generated with `random.Random(1000+session_id)`. The order is written to `results_stage4b/session{N}/order.txt` before the first run.
- **Unit of replication:** the session. Within-session reps describe within-session repeatability only.
- **Primary energy boundary:**
  - GPU states: **system = NVML GPU + RAPL package-0**, per the MLPerf Power full-system principle (Tschand et al., HPCA 2025, §III-C).
  - CPU states: RAPL package-0.
  - Secondary for GPU states: GPU-only (NVML) and CPU-package-only.
  - Never compare GPU-state and CPU-state energy directly.
- **Primary outcome:** energy per image, J (gross).
- **Primary comparisons:** each compressed state vs its same-model, same-device FP32 **within the same session**. The ratio is r = J/img(state) / J/img(FP32).
- **Primary estimate:**
  - per comparison, the geometric mean of r across the 4 sessions
  - a 95% CI from the t-distribution on log r (df = 3)
  - Also fit a linear mixed model `log(J/img) ~ condition + (1 | session)` as a robustness check.
- **Smallest effect of interest:** ±5% (about the observed between-session drift).
  - A difference whose CI lies entirely within ±5% is reported as "practically equivalent" (TOST on log r, margin log(1.05)).
  - A CI that crosses ±5% but does not exclude 1 is reported as "unresolved", not as "no effect".
  - The pruning-level comparisons (30 vs 50 vs 70) are analysed the same way, within session.
- **FLOPs vs energy (RQ2):**
  - realised MAC ratio vs energy ratio for the pruned states
  - for FP16/INT8, MACs are unchanged (ratio 1.0), with bit-width-adjusted BOPs as a labelled secondary
  - Spearman/Kendall are descriptive only
- **Accuracy:** from `results_accuracy/`. Wilson CIs; McNemar vs same-device FP32 using the per-image predictions; the 99% primary / 99.9% strict rule (confirmed 2026-10-06); Pareto plots of accuracy vs energy with **all** states shown, labelled, never removed.
- **Covariates logged, not adjusted in the primary analysis:** uptime at condition start, position in the session order, `power_regime`, images/s. A secondary analysis adds uptime and position to the mixed model.
- **Exclusions:** as plan §8, plus:
  - A condition that fails or is interrupted is re-run once at the end of the same session.
  - If it fails again, it is excluded from that session and listed.
  - No other exclusions.
- **Existing Stage 4 matrix:** exploratory, reported as such (D17). It is never pooled with Stage 4b.
- **Later sessions, on the same harness version:**
  - RQ1 CodeCarbon pass: 1 session, all 30 conditions, `--codecarbon`, separate result set. It is analysed only as a paired within-window comparison, component-wise primary per plan §3.
  - Batch=16 sensitivity: 1 session, all 30 conditions.
  - Neither is pooled with the main sessions.

Commit, then create the git tag **`stage4b-registered`** and push the tag. After the tag, do not edit `docs/stage4b-design.md`; any later change goes in as a dated amendment.

## Step 7: Runner script

Write `scripts/run_stage4b_session.sh` (bash, plus a small Python helper if needed).

- **Usage:** `scripts/run_stage4b_session.sh --kind main|rq1|b16 --session N`.
- **Preflight** (abort with a clear message if any check fails):
  - no uncommitted changes to tracked files (`git diff --quiet HEAD`; untracked log files are fine)
  - governor `performance` (set it via the wrapper, then re-check)
  - platform profile `performance`
  - AC online
  - no foreign GPU compute processes
  - `power_watchdog.py` running (start it if not)
  - log uptime
- Generates the order (Step 6 rule; for `rq1` and `b16` use seeds 2000+N and 3000+N), or reuses an existing `order.txt`.
- **Per condition:** run `pilot.py` with the Step 6 settings into `results_stage4b/{kind}_session{N}/{condition}/`.
  - Use `--allow-stale-boot` and `--session-id`.
  - Add `--codecarbon` for `rq1`, and `--batches 16` for `b16`.
  - Skip a condition whose directory already has a complete summary (resume-safe).
  - Use `--resume` for a partially complete one.
- **Power loss:** if `pilot.py` exits because of the power-loss stop marker, wait until AC has been continuously online for 10 minutes, re-set the governor via the wrapper, then resume the same condition with `--resume`. Log the interruption with timestamps.
- At the end, re-run failed conditions once. Then write `results_stage4b/{kind}_session{N}/session_log.md`, containing:
  - start and end times
  - uptime at the start of each condition
  - interruptions
  - failures
  - a per-condition table of `power_regime`, `images_per_s_mean`, and the three J/image boundaries

  This table is descriptive only. **No ratios, no tests.**
- Logs go to `results_stage4b/{kind}_session{N}/runner.log`. Commit results at the end with `git add` restricted to that session directory, then push. If push fails, leave the commit local and log it.
- Do a short dry run with `--kind main --session 0`, limited to 1 condition and 2 reps, using an environment variable such as `STAGE4B_DRYRUN=1` that also reduces reps. Then delete `session0` entirely. Do not commit it.

## Step 8: Launch Session 1 and stop working

- Write the morning report first (§9), marking Session 1 as "launched, running".
- Commit and push.
- Launch it detached so it survives this assistant session ending:

  `mkdir -p ~/stage4b_logs && nohup scripts/run_stage4b_session.sh --kind main --session 1 > ~/stage4b_logs/session1_nohup.log 2>&1 &`

  The nohup log lives outside the repo so it never makes the working tree dirty.

  Record the PID in the morning report.
- **Then stop.** Do not run anything else on this machine. Do not poll frequently. If you are still active, at most check `runner.log` once an hour with a single `tail`. When the session finishes, append its end time and the "Blocked/unexpected" items to the morning report and commit.

## 9. Morning report: `docs/overnight-report-2026-10-07.md`

Keep it short and factual. Every number must have its file path.

1. **Done, with evidence:** per step, the commit hash and the key numbers. These are:
   - FP16 accuracies
   - traced-vs-Stage-2 accuracy differences
   - the 99% / 99.9% tallies for both pruning arms
   - BN-recalibration accuracies
   - recovery-arm accuracies
   - the harness validation numbers
2. **Blocked or unexpected**, including anything that contradicted an earlier finding. Do not smooth anything over.
3. **Deviation entries added** (numbers and one line each).
4. **Status in three groups:**
   - independently verified tonight
   - still open
   - deferred by Shohan's choice: M1 after x86; RQ1 pass and b16 after the main sessions
5. **For Shohan:**
   - the exact commands for Sessions 2–4: reboot, wait about 2 minutes, then `scripts/run_stage4b_session.sh --kind main --session N`
   - the RQ1 and b16 commands, to run only after Sessions 2–4
   - the still-missing items: the proposal statistics paragraph (D14) and the date M1 access began (D5)
