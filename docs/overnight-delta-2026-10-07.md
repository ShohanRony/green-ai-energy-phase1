# Overnight Delta: read AFTER `docs/overnight-instructions-2026-10-07.md`; this file overrides it where they differ

Written by the supervisor at 2026-10-07 00:40 local, after the coding assistant's commits `b990db1`, `3a68727`, `b66349d`, `c5579d6`, `37bd397` and `85d966c`. Those commits already cover much of Steps 1, 2, 3, 5 and 6, so **do not redo them. Verify them and fill the gaps.** All hard rules in §0 of the main file still apply.

## D0. First
- `git push` all local commits. They are not on GitHub yet, and the supervisor cannot verify them until they are.
- Then add both instruction files to `docs/`, commit, and push.

## D1. Verify, don't redo (write the results into the morning report)
1. **Step 2 (accuracy).**
   - List every artifact whose traced accuracy differs from Stage 2 by more than 0.05 pt, with both numbers. One is reported to differ by up to 0.31 pt; name it and give the likely reason (for example a device difference). Do not "fix" anything.
   - Report all three FP16 accuracies.
   - Confirm that per-image prediction files exist for all 18 artifacts **plus FP32 on CPU**. If FP32-on-CPU predictions are missing, add them.
2. **Step 3.1.** If the regenerated live pruned modules were not identity-checked against the stored artifacts (MACs, params, and output max-abs-diff below 1e-4), do it now.
3. **Step 5 (harness).** Compare `b66349d` against Step 5 items 1–6 of the main file, and implement only what is missing:
   - `gpu_energy_j`, `cpu_package_energy_j`, `cpu_psys_energy_j` and `system_energy_j` per window
   - the per-boundary means in `summarize()`
   - `images_per_s`
   - `--session-id` and `--condition-label`
   - affinity read from `/sys/devices/cpu_core/cpus` and logged

   Add tests for anything new, then re-run the two short validation runs (scratch directory, deleted afterwards).
4. **Step 1.** Tick off each sub-item already covered by `c5579d6` and `37bd397`. Do only the uncovered ones. Make sure §6 records that **Shohan confirmed the 99%/99.9% rule on 2026-10-06**.

## D2. Design change: BN recalibration becomes a pruning arm (new deviation entry)
Commit `3a68727` showed that gradient-free BN-statistics recalibration recovers ResNet-18 pruned50 from 17.6% to 71.0% and EfficientNet-B0 pruned50 from 13.4% to 46.2%. BN recalibration uses training data without weight updates, the same class of step as INT8 calibration, so it fits a *post-training* study. The pruning arms are therefore:

| Arm | Suffix | What it is | Energy-measured in Stage 4b? |
|---|---|---|---|
| Zero-finetune | (none) | existing checkpoints | yes |
| BN-recalibrated | `_bnrecal` | 2,000 train images, seed 2026, momentum=None, no gradients (the `3a68727` procedure) | **yes** |
| Brief-recovery fine-tune | `_ft` | Step 3.4 recipe from `prune_decision_checkpoint.py` | **no, accuracy only** |

- Save the 9 `_bnrecal` models as traced artifacts `checkpoints/{model}_pruned{N}_bnrecal.pt`, using the same `trace_and_save` method. These are new files; never overwrite existing ones.
- Do Step 3.4 (the `_ft` arm) as written. It is accuracy-only.
- Run `scripts/eval_artifacts.py` on all 18 new artifacts and save their predictions.
- **Rationale for measuring energy on `_bnrecal` but not `_ft`:** both arms have the same architecture as zero-finetune. Measuring zero-finetune and `_bnrecal` in every session directly tests whether energy depends on weights at fixed architecture. If it does not (within ±5%), the `_ft` arm's energy is taken as its architecture's measured energy, and that assumption is disclosed. If it does, that is a finding. Report it, and the `_ft` arm then needs its own energy runs later.
- Add a deviation entry (next free number) with these results, the 99%/99.9% tally for each arm, and the rationale above.

## D3. `docs/stage4b-design.md`
`85d966c`'s draft is not registered yet, so it may be replaced.
- Replace its substance with **Step 6 of the main file**, with one change: the 9 `_ft` energy conditions become the 9 **`_bnrecal`** conditions. That still gives 30 conditions: per model, FP32-cuda, FP16, pruned30/50/70, pruned30/50/70_bnrecal, INT8-cpu, FP32-cpu.
- Keep **exactly 4 sessions, 7 reps**, the random order with seed 1000+N, the session as the replicate unit, the primary boundary rules, the ±5% smallest effect of interest, and the rest of Step 6.
- State that the `_ft` arm is accuracy-only, with energy assumed from architecture, conditional on the D2 test.
- Note in plan amendment A3 that the earlier draft (`85d966c`) is superseded and remains in git history.
- Then commit, tag **`stage4b-registered`**, and push the tag.

## D4. Then Steps 7, 8 and 9 of the main file, unchanged
Runner script, a dry run with session 0 (deleted afterwards), launching Session 1 detached with nohup, then stop using the machine and write the morning report.

## Priority if time or tokens run out
D0 → D1 → D2 → D3 → Step 7 → Step 8 → report.

Each step is committed, so another agent can continue from the last commit, with `git log` and this file as the handover.
