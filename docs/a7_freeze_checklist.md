# A7 freeze checklist (written, not executed)

**Purpose:** the exact ordered steps for freezing `stage5_analysis_plan.md`'s A7 framework (A7,
A7r1, A7r2, and A8/A8r1/A8r2) as the pre-registered analysis plan, before any Stage 4b confirmatory
statistic is computed — the same "register before you compute" discipline this project has followed
throughout (`stage5_analysis_plan.md`'s own opening status note, and A1's existing but
never-executed tagging precedent for §6). **Nothing below has been run.** Every command is given
exactly as it would be typed; this document does not execute any of them.

## 1. What must be true before freezing (prerequisites)

Checked against this project's actual current state as of this review, not assumed:

- [ ] **A7 and its revisions (A7r1, A7r2) formally confirmed by the researcher** — currently
  **DRAFT, pending supervisor review** (`stage5_analysis_plan.md`, A7's own header). Freezing a
  still-DRAFT amendment would misrepresent it as settled; confirmation must come first.
- [ ] **A8 and its revisions (A8r1, A8r2) formally confirmed** — same status, same requirement.
- [ ] **The "looked at" statement (A7r1(g)) re-verified true at the moment of freezing, not just
  when it was last written** — re-run the same check this project has run at every prior
  confirmation point: no ratio, interval, or hypothesis test computed on any Stage 4b energy data.
  This is a repeat of an existing check, not a new one — stated here so it isn't skipped because
  "it was already checked once."
- [ ] **Decision rules D1r1-D5r1 (and P-CF, P-MAC) do not need to be *resolved*** — pre-registering
  them unresolved, before the data that would resolve them exists, is the entire point (same logic
  as this project's existing power-regime, exclusion, and accuracy-rule pre-registrations). They do
  need to be in **final, non-draft wording** — satisfied once A7r2 itself is confirmed (first bullet
  above), since D1r1-D5r1 live inside it.
- [ ] **P1 wall-meter protocol (`docs/p1_wall_meter_protocol.md`) exists in at least draft form** —
  satisfied as of this review (BLOCK I item 1). Not required to be executed or have a meter chosen;
  D1r1 only needs a real document to point to, not completed data collection.
- [ ] **M1 harness design (`docs/m1_harness_design.md`) exists in at least draft form** — satisfied
  as of this review (BLOCK I item 2). M1-CPU data itself does **not** need to exist before freezing
  — the plan can be frozen with M1-CPU data collection still pending, as long as no M1 statistic is
  computed before the freeze (same rule as the "looked at" check above, extended to the block that
  doesn't exist yet).
- [ ] **No uncommitted changes** to `stage5_analysis_plan.md` or `deviation_log.md` at freeze time
  (`git status` clean for both files) — the tag must point to a committed, pushed state, not a
  working-tree snapshot.
- [ ] **Pushed to `origin`** before tagging — tagging a commit that only exists locally defeats the
  purpose of an externally-verifiable freeze point.

## 2. Tag name proposal

- **Proposed: `stage5-a7-frozen-<YYYY-MM-DD>`**, date of the actual freeze (not proposed here as a
  fixed date, since the prerequisites above are not yet satisfied). Distinct from A1's existing,
  never-executed `stage5-plan-v1` proposal (which was tied specifically to §6(a)'s threshold
  confirmation, a narrower and already-superseded framing) — this tag is for the A7 framework
  specifically, named accordingly so the two are never confused if both are ever created.
- **Command (not run):**
  ```
  git tag -a stage5-a7-frozen-<YYYY-MM-DD> -m "A7 analysis framework frozen before any Stage 4b confirmatory statistic."
  git push origin stage5-a7-frozen-<YYYY-MM-DD>
  ```
  **Annotated tag (`-a`), not lightweight** — carries its own message, date, and tagger identity,
  which a bare lightweight tag does not; appropriate for a citable registration point.

## 3. Zenodo deposit contents

- **Included:**
  - `docs/stage5_analysis_plan.md` (the frozen plan itself, including A7/A7r1/A7r2/A8/A8r1/A8r2).
  - `docs/deviation_log.md` (full deviation history to date — the plan's own stated discipline
    treats deviations as part of the registered record, not a separate appendix to omit).
  - `docs/p1_wall_meter_protocol.md`, `docs/m1_harness_design.md` (referenced prerequisites,
    included so the deposit is self-contained rather than pointing to documents that could change
    after the freeze).
  - `docs/history_rewrite_map.md` (provenance record for this repository's commit history —
    included for transparency about the history cleanup, consistent with this project's general
    disclosure discipline).
  - A short manifest/README specific to the deposit: what is and isn't included, the exact git
    commit hash and tag the deposit corresponds to, and a one-paragraph plain statement that this
    is a pre-registration (no results data), not a results release.
- **Explicitly excluded:**
  - **Any `results_*` directory** — the entire point of freezing before computing is that no result
    exists yet to deposit; including one would contradict the deposit's own purpose.
  - **`checkpoints/` and `checkpoints_p8/`** — large binaries, not needed to register an analysis
    plan, and (for `checkpoints_p8/`) not yet reviewed for the same disclosure standard as the rest
    of the repository.
  - **`docs/incidents/`** — untracked in git for a reason (sensitive containment-review evidence,
    `deviation_log.md` D22/D23); must stay excluded from any public deposit, not just from git.
  - **Any file not yet committed/pushed** — the deposit should be reproducible from the tagged
    commit alone, not from working-tree state that could differ.

## 4. Open prerequisites, consolidated (not yet satisfied, as of this review)

- A7/A7r1/A7r2 and A8/A8r1/A8r2 are all still DRAFT — the primary blocker.
- No meter has been chosen for the P1 wall-meter protocol (not required for freeze per §1, listed
  here only so it isn't mistaken for a freeze blocker it isn't).
- M1 harness design has multiple open items (`docs/m1_harness_design.md` §8) — none block the
  freeze itself (§1), but all block any actual M1-CPU session, which the frozen plan already
  anticipates running after the freeze, not before.
- BLOCK H's outcomes (corrected D1-D5 wording, the battery-check revision, the tag rename) are
  already committed as of this review and do not block the freeze on their own — listed here as
  context, since the freeze checklist above assumes their content is already folded into
  `stage5_analysis_plan.md`/`deviation_log.md` by the time any freeze actually happens.
