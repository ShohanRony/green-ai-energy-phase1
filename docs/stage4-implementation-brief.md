# Stage 4 Implementation Brief — Full Measurement Matrix

**Project:** Energy-Aware Efficiency of Lightweight Vision Models Under Post-Training Compression
**Depends on:** `phase1-execution-plan.md` §3 Stage 4; Stage 3's GO decision
(`results_stage3/stage3_deliverable.md`); Stage 4 pre-flight's closed investigation
(`results_stage4_preflight/stage4_preflight_findings.md`).

---

## 1. Objective

Collect the full energy/latency dataset: 3 models × 6 compression states × ≥30 reps/condition, x86
only. This is data collection, not a go/no-go gate — Stage 3 already cleared the harness. The 18
checkpoints this needs already exist (Stage 2); no new training or compression work.

## 2. Scope lock

| In scope | Out of scope (explicitly, not an oversight) |
|---|---|
| ResNet-18, MobileNetV3-Small, EfficientNet-B0 — all 6 states each (18 conditions) | ARM/M1 — carried-forward gap since Stage 1, still no access to the MacBook Air. Stage 4 runs x86-only; flag this as a real gap in the deliverable, don't call the matrix "complete" without noting it. |
| Batch size = 1, locked (see §3) | Sweeping batch size as a factor — considered during pre-flight, not adopted. |
| ≥30 reps/condition (`--repeats 31`, giving 30 pairs after the single cold-discard) | Statistical analysis (Stage 5's job). |

## 3. Batch size: locked to 1 — deployment realism, not the old speedup argument

**Decision: batch=1 is Stage 4's headline and only batch size.**

The original Stage 3 justification for a headline batch size leaned on comparing energy ratios across
batch sizes against Stage 2's batch=128 latency speedups. Pre-flight found that comparison is
**regime-confounded and unreliable**: the same (model, state, batch) can land in one of two distinct
GPU power regimes (pinned at the enforced cap, or an unsaturated ~55-90% of it) on different
invocations, and this swamps whatever batch-size effect was being measured. That old reasoning is
**retired, not reused** — do not cite Stage 2/3's batch-16-vs-128 comparison as justification for
anything in Stage 4 or the eventual manuscript.

**The actual justification: deployment realism.** Batch=1 is genuine single-image edge inference — the
deployment scenario the proposal's lightweight-model framing targets in the first place (a model
serving one request at a time on constrained hardware), not an artifact of convenience. This holds
regardless of the power-regime question, which is why it survives as the locked choice even though the
investigation that originally raised the batch-size question didn't resolve cleanly.

**Disclosure for the manuscript (same discipline as the §1.6 INT4→FP16 substitution):** any INT8/FP16/
pruning "speedup" or energy-ratio figure quoted anywhere must be a batch=1 number collected under this
stage's standing operating procedure (§5), not Stage 2's batch=128 latency speedups — those two numbers
are not interchangeable and must never be presented as if they were.

## 4. Preconditions — carried in, not re-derived

- All 18 checkpoints exist and proof-load through `pilot.py` (Stage 2 Task D, re-confirmed Stage 3).
- Corrected energy backend (`nvmlDeviceGetPowerUsage`), NVML floor, RAPL ceiling, concurrent-GPU guard,
  platform-profile guard: all enforced in code since Stage 1/3.
- **New since Stage 3, enforced in code, not just documented:**
  - `power_regime` field (`pinned`/`dip`/`mixed`) on every summary row (`classify_power_regime()`).
  - `check_fresh_boot()` — refuses to start past `--max-uptime-min` (default 30 min) unless
    `--allow-stale-boot` is passed. See §5 for how this is actually used in practice — the default
    30-minute threshold is a conservative placeholder from pre-flight, not the real safety net for a
    multi-hour session.
- Power-resilience scaffolding (`power_watchdog.py`, `--resume` on `pilot.py`) exists from the same-day
  session that built it, but has not been run as a live background process during any stage yet — if
  used for real during Stage 4, that's the first live deployment, not a re-tested mechanism.

## 5. Standing operating procedure — dip/mixed triggers reboot-and-rerun

This is the actual safety net for Stage 4, not the proactive uptime timer. Pre-flight found: (a) every
observed power-regime dip happened in a long-uptime session, (b) a fresh reboot reliably produces
`pinned` (confirmed 30+ consecutive times across two separate reboots), and (c) staying pinned for
several hours *into* a session without rebooting again is also already confirmed safe (the thermal-soak
control ran ~3 hours into a boot session and was still pinned) — so the 30-minute default guard is
**overly conservative for actual multi-hour Stage 4 sessions**, and the real governing rule is reactive,
not a timer:

1. **Reboot at the start of each work session**, before the first run of that session. Confirm with
   `uptime` that it's genuinely fresh (don't trust a claim — this project has been burned by a
   not-actually-rebooted session before).
2. **The first run of a session passes `check_fresh_boot()` naturally.** For subsequent runs in the
   same session (uptime will exceed 30 min quickly), pass `--allow-stale-boot` — this is a deliberate,
   disclosed choice given (c) above, not a workaround for an inconvenient check.
3. **After every run, check its summary row's `power_regime` field.** If it's `pinned`: accept and move
   to the next condition. If it's `dip` or `mixed`: **stop, reboot, and rerun that exact condition**
   before doing anything else — do not accept a dip/mixed result as valid Stage 4 data, and do not
   continue to the next condition on top of an unresolved flag. **Unless the carve-out in point 3a
   applies.**
3a. **Carve-out: configs already known to be genuinely sub-saturating at batch=1 are exempt from the
   reboot-and-rerun trigger.** Stage 4 pre-flight's own Task 1 (`stage4_preflight_findings.md`) already
   measured and physically explained this: at batch=1, pruned50/70 are fast enough that the GPU idles
   between launches instead of staying boosted — confirmed directly from per-window traces, not a
   guess. This is a *different* phenomenon from the pre-flight pin/dip mystery (which was about
   FP32/FP16/pruned70 at larger batches landing in different regimes *unexpectedly*). A `dip` on a
   config already known from Task 1 to sub-saturate — pruned50 and pruned70 at batch=1, confirmed
   again below for ResNet-18 — is **expected, not an escalation trigger**: log it as `dip (expected —
   sub-saturation at batch=1)` in the deliverable and move on. The reboot-and-rerun rule in point 3
   still applies in full to any config expected to saturate the GPU: FP32, FP16, pruned30 (confirmed
   pinned at batch=1 for ResNet-18 in Task 1's original data — not a borderline case), and anything at
   batch≥4. **For MobileNetV3-Small and EfficientNet-B0's pruned states, Task 1 never collected
   batch=1 data** (ResNet-18 only) — treat pruned30/50/70 for those two models as *unknown* regime
   expectation, not automatically exempt, until their own data says otherwise.
4. **Log every reboot-and-rerun event, and every carve-out invocation** (condition, regime observed,
   timestamp, which rule applied) in the eventual deliverable's provenance notes — neither a rerun nor
   an exemption is a hidden detail.
5. If dip/mixed recurs on a config that is *not* covered by the 3a carve-out — i.e., a config expected
   to saturate the GPU still lands dip/mixed after a reboot — that is new evidence against the
   uptime-duration hypothesis and should come back for a fresh look, not be pattern-matched into "just
   reboot again" forever.

## 6. Tasks

### Task 0 — Session start
Reboot. Confirm fresh via `uptime`. Confirm governor `performance` (passwordless now:
`sudo /usr/local/sbin/set_cpu_governor.sh performance`), platform profile `performance`, GPU idle
(`nvidia-smi --query-compute-apps`).

### Task 1 — Run all 18 conditions
Per model × state: `python3 pilot.py --device {cuda|cpu} --arch {model} --checkpoint checkpoints/{model}_{state}.pt --data /home/shohan/green-ai-research/data --out results_stage4/{model}_{state} --sizes 32 --batches 1 --windows 5 --repeats 31 --interval 0.4 --warmup 3` (`--device cpu` for the 3 INT8 conditions, `--allow-stale-boot` for every run after the session's first). Apply §5's rule after every single run.

Estimated wall-clock: ~12-13 min/condition for the 15 GPU conditions (3 models × 5 non-INT8 states),
~20-30 min/condition for the 3 CPU/INT8 conditions — roughly 4.5-5 hours of actual measurement time
total, not the plan's original 3-4-week estimate (that estimate likely assumed building compression
artifacts from scratch alongside M1 work; Stage 2 already built the artifacts, and M1 is out of scope
here). Spans multiple sessions realistically — §5 governs how to resume cleanly across them.

### Task 2 — Plausibility pass, all 18 conditions
Mean/SD energy-per-image and latency per condition, same discipline as Stage 3 Task 2. Sanity-check
ordering (compressed states ≤ FP32 baseline for their own model) and flag anything that doesn't fit
rather than explaining it away.

### Task 3 — Checklist compliance report
Same 14-item checklist as Stage 1/3, with real observed values, plus the new power-regime field's
value for every one of the 18 conditions (expect `pinned` everywhere per §5 — any `dip`/`mixed` that
made it into the final dataset despite §5 is itself a checklist finding, not a silent pass).

### Task 4 — Deliverable assembly
`stage4_deliverable.md`: full 18-condition table (mean/SD energy, latency, power_regime, reps), the
ARM/M1 gap stated plainly (not omitted), the batch=1 justification from §3 restated for anyone who
wasn't in this conversation, reboot-and-rerun log from §5 if any occurred.

## 7. Deliverables

- `stage4_task_report.md` — task-by-task narrative.
- `stage4_deliverable.md` — the 18-condition table + checklist + batch-size/ARM disclosures, no narrative.
- Raw per-condition data retained under `results_stage4/` (same format as Stage 3).

## 8. Timeline

Plan's original estimate: 3-4 weeks of evenings. Revised estimate given Stage 2's artifacts already
exist and ARM is out of scope here: ~4.5-5 hours of actual measurement time, likely spread across
several sessions for practical reasons (laptop availability, not needing to run unattended overnight).
Don't compress §5's checklist discipline to hit a shorter timeline — the whole point of pre-flight was
making sure Stage 4's data is trustworthy, not fast.
