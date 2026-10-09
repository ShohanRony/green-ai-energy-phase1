# Handoff — 2026-10-10

Written at the end of a documentation/design-only review (no training, no measurement, no
`results_*` data, no push). Six local commits made this session, none pushed. This file is for
whoever picks this project up next — human or otherwise — to get oriented quickly.

## State of `origin/master`

**Nothing described in this file is pushed.** Check `git log origin/master..HEAD --oneline` for
the current unpushed count and the current commit hashes before assuming anything below is public
or citing a specific hash — local `master` is ahead of `origin/master`, and hashes in this file can
go stale if the local history is ever rewritten (as it was once already, within this same local
range — see `docs/history_rewrite_map.md` if a hash mentioned elsewhere looks unfamiliar).

## What this session added (all local, all committed, none pushed)

1. `docs/p1_wall_meter_protocol.md` — draft wall-meter protocol (D1's eventual data source).
   Requirements only, no meter chosen, nothing executed.
2. `docs/m1_harness_design.md` — M1-CPU arm design. Lists open questions that block a real M1
   session (AC-detection guard, fresh-boot guard, `powermetrics` format, `taskpolicy` syntax, INT8
   checkpoint portability — none of these exist yet).
3. `scripts/run_cpu_block_session.py` + `tests/test_run_cpu_block_session.py` — the x86-CPU block
   session runner, orchestration only (shells out to `pilot.py`, which already has the real
   measurement logic and the D26 AC guard). Tested with a stub instrument and a dummy architecture
   list; never ran a real model. 16 new tests, 65/65 total passing as of this session.
4. `docs/a5_rewrite_draft.md` — accuracy-arm draft (variance arm + fine-tune arm), compared against
   `docs/p8_spec.md`'s actual recipe. Nothing decided.
5. `docs/a7_freeze_checklist.md` — the exact steps for freezing the A7 analysis framework and
   depositing it, written but not run. A7/A8 (and their revisions) are still DRAFT — this is the
   main blocker to actually freezing anything.
6. Reference check (not a file — see the session's own report): "Santos et al. 2026" resolves to a
   real, verified paper (Santos, Ottoni, Borgo, Ferreira, Nepomuceno, *Artificial Intelligence
   Review* 59(5), article 132, DOI `10.1007/s10462-026-11515-8`). **"Iyer et al. 2026" was not
   found** after genuine searching — whatever document cites it should be checked for a possible
   mis-citation before anyone relies on that reference.

## What is still open (the real blockers, not busywork)

- **A7/A7r1/A7r2 and A8/A8r1/A8r2 are all still DRAFT.** Nothing in `stage5_analysis_plan.md`'s
  confirmatory framework is adopted yet. This blocks the freeze checklist's first and main
  prerequisite.
- **No wall meter has been chosen.** D1 (instrument-axis decision rule) cannot be evaluated until
  one is, and the protocol's charger-efficiency figure is unmeasured.
- **No M1 hardware has ever been touched by this project.** Every M1-related document in this
  repository (`docs/m1_check_commands.md`, now also `docs/m1_harness_design.md`) is written
  blind, from documentation and general platform knowledge, not from a real machine. The researcher
  needs to run `docs/m1_check_commands.md`'s checks (prioritised in `m1_harness_design.md` §7)
  before any M1-CPU session design can move from "design" to "implementation."
- **Series R (A8) has never been run.** D2 cannot be evaluated, and the GPU block's baseline stays
  FP32-eager (runtime-confounded) until it is.
- **A5's fine-tune arm (both the original draft and this session's rewrite draft) is undecided.**
  `checkpoints_p8/` has some real fine-tuned weights from an unrelated, out-of-process campaign
  (`deviation_log.md` D22) — those are not a substitute for a deliberately-designed A5 run, and the
  divergence list in `docs/a5_rewrite_draft.md` explains exactly why not.
- **A local-only git tag** (`backup-pre-history-rewrite-20261010`) still exists from the earlier
  history-cleanup work — harmless, local, never pushed, but worth knowing it's there.

## Standing rules that apply to any future session on this repo

- Neutral wording only — no AI tool names, anywhere, in commits or docs (`deviation_log.md` D27 and
  the full commit-history rewrite exist specifically to enforce this retroactively; don't
  reintroduce what was removed).
- `stage5_analysis_plan.md` text above `## 12. Changes after registration` is never edited in place
  — append a new dated revision instead (A1/A2 are disclosed exceptions, not a precedent to repeat).
- `deviation_log.md` is append-only — corrections go in a new dated entry, not an edit to the
  original.
- D26's AC guard (`pilot.py`'s `check_ac_power()`) is a hard precondition for any new measurement
  session — no override flag exists, by design.
- Nothing gets pushed without the researcher's explicit go-ahead.
