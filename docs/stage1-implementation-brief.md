# Stage 1 Implementation Brief — hand this to the local coding-assistant CLI on the LOQ

Context: full plan is in `phase1-execution-plan.md` (same folder as this file — put both in your repo, e.g. under `docs/`). This brief is just Stage 1, broken into concrete, checkable tasks. Read the full plan's §1.5 (what's already built), §1.6 (the INT4→FP16 decision, already made), and §2 (the pre-flight checklist) before starting — don't rebuild what §1.5 says already exists.

## Task 0 — Orient, don't rebuild

Before writing anything new, locate and inspect these existing local directories/repos from the prior validation sprint:
- `measurement_env` (the venv + RAPL udev rule + installed packages)
- `research/energy-pilot` (the RAPL+NVML harness, currently at 3 A/A pairs)
- `research/thesis_arch_results` (training-time benchmarks)

Confirm they still work (`source` the venv, run the energy-pilot's existing test suite) before porting anything. If something that the status report says "passed" no longer passes, that's a regression to flag back, not to silently fix and move on.

## Task 1 — RAPL domain check

```bash
ls /sys/class/powercap/intel-rapl/intel-rapl:*/name
for f in /sys/class/powercap/intel-rapl/intel-rapl:*/name; do echo "$f: $(cat $f)"; done
```

Record which domains exist (`package-0`, `psys`, `core`, `uncore`, etc.) and which the existing udev rule actually makes readable without sudo. If `psys` is absent, note that explicitly — the harness will report `package` domain energy, and that's a stated deviation (whole-package ≠ whole-system), not a bug.

**Acceptance:** a one-line note in the harness README stating which RAPL domain is used and why.

## Task 2 — NVML sampling interval, confirm the existing finding

The prior sprint already swept this (20ms→291W through 500ms→16W, see plan §2 item 4). Re-run the sweep once as a sanity check that the finding reproduces on this exact machine today (driver/library versions may have changed since September), then lock the harness's NVML polling interval at **0.3–0.5s**, not configurable to go lower without a deliberate override flag that logs a warning.

**Acceptance:** sweep result matches the prior finding within reason; harness code has the interval floor enforced, not just documented.

## Task 3 — Port `research/energy-pilot` into a new, clean repo for this proposal

Don't keep developing inside the old repo (per plan §1.7, that repo's identity belongs to the other proposal). Create a new repo — e.g. `green-ai-energy-phase1` — and port:
- The RAPL+NVML paired measurement logic
- The 4 existing unit tests
- The idle-baseline capture logic

Leave `research/energy-pilot` as-is (don't delete or keep modifying it — it's the evidence trail for the other proposal).

**Acceptance:** new repo, `git log` shows a clean history starting from the port, old repo untouched since the port commit.

## Task 4 — CodeCarbon paired integration (not done in the prior sprint)

Wrap CodeCarbon so it runs **concurrently** with the RAPL+NVML reads on the same inference run — same process, same time window, not two separate executions compared after the fact. This is what makes the eventual RQ1 comparison valid (Proposal §3's instrumentation plan requires this).

**Acceptance:** one run produces a single log row with both the hardware-counter energy and the CodeCarbon-estimated energy for the same inference batch, same timestamp window.

## Task 5 — Phase separation, confirm against Bartoli et al.'s design

The existing pilot already separates idle/active watts in its smoke run. Confirm this is a genuine dual-trigger boundary (pre-inference / inference / post-inference marked explicitly in the code, e.g. via explicit start/stop calls bracketing only the model's forward pass) rather than a simpler "measure around the whole script" split.

**Acceptance:** code has three explicit, separately-timestamped phases; data-loading and post-processing energy is excluded from the "inference" number by construction, not by subtraction after the fact.

## Task 6 — M1 side: `powermetrics` setup (not started)

On the MacBook Air M1:
1. Confirm `powermetrics` runs and reports plausible power figures at idle.
2. Set up a narrowly-scoped sudoers rule for just `powermetrics` (not a blanket NOPASSWD), so it can run unattended in a script:
   ```
   <username> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
   ```
3. Port the same measurement-harness structure from Task 3's new repo to run on the M1 (same log format, different instrument).
4. Check memory pressure monitoring — 8GB RAM is tight; the harness should record `vm_stat` or similar alongside the energy reading, so a run that swapped can be flagged and excluded, not silently contaminating the dataset.

**Acceptance:** same harness, same log schema, producing a plausible idle-power reading on the M1, with a memory-pressure flag column.

## Task 7 — Exit test

Run the ported, CodeCarbon-integrated, phase-separated harness on an **uncompressed ResNet-18**, ×10 repetitions, on **both platforms**. Check every item in the plan's §2 checklist against the actual logged output — not "should satisfy," actually inspect the log and confirm each one.

**Acceptance:** a short markdown report (`stage1_exit_test.md`) listing all 13 checklist items with a ✅/❌ and the actual value observed for each (e.g. "item 4, NVML: sampled at 0.4s intervals, confirmed from log timestamps" not "item 4: done").

---

**When Task 7's report is clean, Stage 1 is done.** Report back (or just push the repo and note it) rather than continuing straight into Stage 2 — the §1.6 INT4→FP16 decision is already made, but Stage 2's own first task is to build the MobileNetV3-Small PTQ pipeline, which is new work, not a continuation of Stage 1's harness work.
