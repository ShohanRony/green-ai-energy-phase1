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

## 5. Additions, 2026-10-10 (previous sections above kept as-is, not edited)

- **Pin the harness and session-runner commit hash in the tag.** §2's tag command is extended: the
  annotated tag message should name the exact commit hash of `pilot.py` and
  `scripts/run_cpu_block_session.py` as they exist at tag time (normally identical to the tag's own
  commit, but stated explicitly so the tag is self-describing even if read outside its own git
  history) — e.g. `git tag -a stage5-a7-frozen-<date> -m "A7 analysis framework frozen... harness:
  pilot.py @ <hash>; session runner: scripts/run_cpu_block_session.py @ <hash>"`. Not run here —
  written for when the freeze actually happens.
- **A7r3 is now also a prerequisite, alongside A7/A7r1/A7r2 and A8/A8r1/A8r2** (§1/§4's existing
  list) — all still DRAFT as of this review; A7r3 adds no new prerequisite category, it is simply
  one more revision that must be confirmed before freezing, same as the others.
- **D22, D23, D25 reviewed for host-security detail before any public deposit — findings, not a
  redaction (nothing edited here):**
  - **D23 carries the most sensitive detail of the three:** it names the real local username
    (`shohan`) directly in "`stage4b-auto.service`, which runs as `User=shohan`," and describes a
    (now-remediated) passwordless-sudo misconfiguration (`systemctl reboot` with no argument
    restriction, service enable/disable) with specific sudoers rule filenames
    (`green-ai-governor`, `stage4b-auto`). This is real operational security detail about a real
    person's real machine — even though the exposure is already closed, publishing the exact prior
    misconfiguration is a disclosure decision for the researcher, not an automatic "include" by
    default.
  - **D22 carries moderate detail:** systemd unit names, `User=root`, an absolute system path
    (`/etc/systemd/system/p8-auto.service.quarantined-20261009`) — less sensitive than D23 (no
    username, no sudoers specifics) but still real-machine configuration detail.
  - **D25 carries the least security-sensitive detail of the three**, but a different kind of
    disclosure risk: ACPI/battery timestamps across an evening reveal when the researcher was and
    wasn't near AC power at home — a personal-routine inference risk, not a security
    vulnerability, worth the same "researcher decides before public deposit" treatment.
  - **Recommendation, not a decision:** if `deviation_log.md` is included in the deposit (§3 above
    currently includes it), the researcher should explicitly decide whether D22/D23/D25 go in
    as-is, get a redacted/summarized public version prepared separately, or get excluded from the
    public deposit while remaining in the private git history. Not resolved here.
- **Old-commit-hash citation check, read-only, as requested:** no document has been "submitted or
  deposited" yet (no freeze has happened) — checked instead against every tracked document in this
  repository, as a prospective check for what a deposit would currently contain. Every hash-like
  string in every tracked `.md` file was checked against `git cat-file -e` for the current repo.
  **Result: clean.** The only hash-like strings that don't resolve as current commits are: (a)
  `docs/history_rewrite_map.md`'s own old-hash column — intentional, that is the document's entire
  purpose, not a citation error; (b) one systemd boot ID in `docs/p8_spec.md`
  (`27625af78f4a4efcb33001f10565a355`) — a 32-character boot identifier, not a git hash, correctly
  labelled as a boot in its own sentence. **No document cites a stale commit hash.** Nothing
  edited, per this item's instruction.

## 6. Dated section, 2026-10-10 (previous sections above kept as-is, not edited)

**(i) Correction: "no result exists" was inaccurate — §3's text is left as-is, corrected here.**
§3 above states (twice) that no result exists to deposit ("this is a pre-registration (no results
data)," "no result exists yet to deposit"). **This is wrong** — `results_stage4b/`'s raw GPU-block
data (sessions 1-6 and the unpinned/aborted directories) already exists in the repository and is
tracked in git, collected before A7 even existed (A7r3(j)). The accurate statement: **no
confirmatory statistic has been computed; raw GPU-block data exist in the repository.** §3's
exclusion of `results_*` from the deposit (line 73) is unaffected by this correction — raw data is
still excluded from the *deposit* for the same reason as before (the deposit is the plan, not a
results release), the correction is only to the claim that no such data exists at all anywhere.

**(ii) Upload method: manual, not the GitHub-release integration.** Zenodo's GitHub integration
archives a release as **the entire repository snapshot**, including every `results_*` directory,
`checkpoints/`, and anything else tracked at that commit — directly contradicting §3's carefully
curated inclusion/exclusion list. **Use a manual upload** through Zenodo's own web upload (or API)
interface instead, uploading **exactly** the files §3 lists — not a repository archive, not a
GitHub release artifact. This is a correction to how the deposit must be performed, not previously
stated in §2/§3.

**(iii) Push scope: master only; backup branches and tags stay local; verify with `git ls-remote`.**
§1's existing "pushed to origin before tagging" bullet is imprecise about scope. Precisely: **`git
push origin master`** is the one push command for the freeze itself. **Backup branches** (e.g.
`backup-pre-blockj-foldfix-20261010`) **and housekeeping tags** (e.g. the renamed
`backup-pre-history-rewrite-20261010`) **stay local, never pushed.** **Verify with `git ls-remote
origin` before and after the push** — compare the ref list both times to confirm exactly what
changed and that nothing besides `master` (and, per §2, the freeze tag) moved.
- **Flagged, not resolved: a real tension with §2.** §2's own tag command includes
  `git push origin stage5-a7-frozen-<date>` — pushing the freeze tag itself. Whether "tags stay
  local" in this instruction means *all* tags (which would contradict §2's command, and undercut
  the freeze tag's own stated purpose — "tagging a commit that only exists locally defeats the
  purpose of an externally-verifiable freeze point," §1) or only the *housekeeping/backup* tags
  (leaving the freeze tag's push in §2 intact) is **not resolved here.** §2's text is left as-is;
  this item is stated as read, with the apparent conflict surfaced rather than silently picked one
  way — the researcher's call.

**(iv) First-CPU-session prerequisites, distinct from the freeze prerequisites in §1.** The freeze
itself (§1) does not require any CPU-block data to exist (A7r1(j)/§1's own M1 bullet already
established this). Collecting the **first actual CPU-block session**, once the plan is frozen, has
its **own**, separate gate, not previously listed in §1: **the thermal-throttle guard code must
exist, have unit tests, and the calibrated thresholds must be registered in a dated amendment**
(`stage5_analysis_plan.md` A7r4(b)) — none of which exists yet. Listed here so it isn't conflated
with the freeze's own prerequisite list; a frozen plan can sit un-executed while this gate is still
open.

**(v) Redaction plan for the Zenodo copy of D22/D23/D25 — a concrete proposal, not yet a decision.**
Building on §5's earlier sensitivity ranking (D23 highest, D22 moderate, D25 a privacy-not-security
risk), a possible redaction, **for the Zenodo deposit copy only**:
- **Username:** every literal `shohan` → a generic placeholder (e.g. `[local user]`).
- **Sudoers rule filenames:** `green-ai-governor`, `stage4b-auto` → generic descriptions (e.g.
  `[CPU-governor sudoers rule]`, `[reboot/service-toggle sudoers rule]`).
- **Systemd absolute paths:** e.g.
  `/etc/systemd/system/p8-auto.service.quarantined-20261009` → a relative/generic description
  (e.g. `[quarantined unit file, system service directory]`).
- **D25's evening timestamps:** the exact clock times (19:57:41 through 00:07:02) enable inferring
  when the researcher was away from AC power at home — proposed treatment: round to the nearest
  hour, or replace with relative offsets from the first event, removing the precise-clock-time
  inference risk while keeping the forensic sequence (AC confirmed → battery confirmed twice →
  apparent shutdown → reboot) fully legible.
- **Explicitly: the git repository's `deviation_log.md` is unchanged by any of this** — D22/D23/D25
  stay exactly as written, append-only, in the project's own history. A redacted version, if the
  researcher chooses one, would exist **only** as a separate file prepared specifically for the
  Zenodo copy, never substituted into the repo itself. **This is a proposal; the researcher decides**
  whether to redact at all, and exactly how — nothing above is applied to any file by this entry.

## 7. Dated section, 2026-10-10: pin the analysis scripts' commit hash too

§5(i)'s existing instruction to pin the harness and session-runner commit hash in the freeze tag
is extended: the tag message should also name the exact commit hash of the `analysis/` directory
(`analysis/rq2_model.R`, `analysis/helpers.R`, `analysis/simulate_rq2.R`, `analysis/tests/`) as it
exists at tag time — e.g. `git tag -a stage5-a7-frozen-<date> -m "... harness: pilot.py @ <hash>;
session runner: scripts/run_cpu_block_session.py @ <hash>; analysis: analysis/ @ <hash>"`. Same
reasoning as §5(i): the frozen tag should be self-describing about exactly which version of the
code that will eventually run against real data was registered, not just the prose plan. Not run
here — written for when the freeze actually happens. `docs/analysis_environment.md` records the
analysis scripts' own pre-freeze/synthetic-only status and the requirement that any post-freeze
change needs a dated amendment; this section is the freeze-checklist-side counterpart of that.

## 8. Dated section, 2026-10-10: resolving §6(iii)'s flagged tag-push tension

§6(iii) flagged, but did not resolve, an apparent conflict between "backup branches and tags stay
local" and §2's own command to push the freeze tag. **Resolved here, explicitly:**

- **The freeze tag IS pushed.** It is not a "housekeeping/backup" tag — it is the citable
  registration artifact the entire freeze process exists to produce, and an unpushed tag cannot
  serve that purpose (§1's own stated reasoning). "Stay local, never pushed" in §6(iii) refers
  only to genuinely local housekeeping refs (backup branches made during unrelated git-history
  work, and the renamed local backup tag) — never to the freeze tag itself.
- **Exact command sequence, named push only — no wildcard ref-pushing command is ever used:**
  ```
  git push origin master
  git push origin stage5-a7-frozen-<YYYY-MM-DD>
  ```
  Two separate, explicitly-named pushes — `master`, then the one named tag. **Never**
  `git push origin --all`, `git push origin --mirror`, or `git push origin --tags` — all three
  would push every local branch/tag indiscriminately, including the backup branch and the local
  backup tag this checklist explicitly keeps local. A wildcard push is exactly the mistake this
  two-step, explicitly-named sequence is designed to make structurally impossible, not just
  procedurally discouraged.
- **Backup branch and backup tag stay local, unchanged from §6(iii):** e.g.
  `backup-pre-blockj-foldfix-20261010` (branch) and `backup-pre-history-rewrite-20261010` (tag) —
  neither is named in either push command above, so neither is pushed by this sequence.
  `git ls-remote origin` before and after (§6(iii)'s existing instruction) remains the verification
  step: the ref list should show exactly `master` moving and exactly one new tag appearing, nothing
  else.

## 9. Dated section, 2026-10-10: `analysis/` commit hash to pin is the one after this block

§7's instruction to pin `analysis/`'s commit hash in the freeze tag message still applies
unchanged — restated here only to say **which** commit that now means. This block added
`analysis/rq2_diagnostics.R`, `analysis/simulate_rq2_diagnostics.R`, and
`analysis/tests/test_rq2_diagnostics.R`, and extended `docs/analysis_environment.md`'s results
table — the commit hash to name in the freeze tag's `analysis: analysis/ @ <hash>` field (§7) is
whichever of this block's commits is the last one touching `analysis/`, not an earlier commit
from Block O/P/Q. Not run here — written for when the freeze actually happens, same as §7.
