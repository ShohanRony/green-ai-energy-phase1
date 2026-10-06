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
| ResNet-18, MobileNetV3-Small, EfficientNet-B0 — all 6 states each (18 conditions) | ARM/M1 — Stage 4 itself still ran x86-only. **Updated 2026-10-06 (`deviation_log.md` D5): M1 access is available** (the "no access" framing used when this brief was first written is corrected there, not asserted as having been true the whole time); x86-only was this stage's sequencing choice, not a hardware blocker. Flag the matrix as x86-only, not as blocked-pending-hardware. |
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
control ran ~4 hours into a boot session — corrected 2026-10-06 from the original "~3 hours," checked
against real `timestamp_utc` values: 3:57-4:19, see `deviation_log.md` D7 — and was still pinned) — so
the 30-minute default guard is
**overly conservative for actual multi-hour Stage 4 sessions**, and the real governing rule is reactive,
not a timer:

1. **Reboot at the start of each work session**, before the first run of that session. Confirm with
   `uptime` that it's genuinely fresh (don't trust a claim — this project has been burned by a
   not-actually-rebooted session before).
2. **The first run of a session passes `check_fresh_boot()` naturally.** For subsequent runs in the
   same session (uptime will exceed 30 min quickly), pass `--allow-stale-boot` — this is a deliberate,
   disclosed choice given (c) above, not a workaround for an inconvenient check.
3. **Resolved 2026-10-06, during real Stage 4 execution: `dip`/`mixed` is no longer a stop-and-ask
   event. Log the regime and keep running.** What changed: this was originally written as "stop,
   reboot, rerun" because pre-flight's dip/pinned split looked like an unexplained, possibly-stochastic
   GPU quirk. Confirmed during Stage 4 itself across three different models' own FP32 baselines
   (ResNet-18 557M MACs: pinned; EfficientNet-B0 33.28M MACs: dip at 38.6-40.5W; MobileNetV3-Small
   5.8M MACs: dip at 25-26.5W) that **whether a config dips or pins at batch=1 is a direct function of
   its compute cost relative to this GPU's boost-sustaining threshold, not a per-model exception or an
   unresolved mystery.** A model/state light enough simply never reaches the power cap during
   single-image inference — that's real, physically-explained hardware behavior, not a measurement
   glitch to chase with reboots. See §9 for why this is a citable finding, not just an operational note.
   **The real backstop is Task 2's plausibility pass, not the regime field**: for every condition,
   confirm energy-per-image orders sensibly against that model's own FP32 baseline and tracks the
   expected compute reduction, regardless of which regime it landed in. A correctly-classified `dip`
   confirms the GPU wasn't power-capped — it does not by itself confirm the energy number is a real
   measurement rather than an unrelated problem that happens to also look like low power. Only escalate
   a specific run if its numbers look wrong on those plausibility grounds, never because it dipped.
4. **Log every run's regime** (condition, `power_regime`, implied W) in the eventual deliverable —
   this is now expected data to report, not an exception needing a provenance note.
5. (Superseded by point 3 above — kept for history.) The original trigger was: if dip/mixed recurs on a
   config expected to saturate the GPU, that would be new evidence against the
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
value for every one of the 18 conditions — `dip` is expected and valid for light-enough models/states
at batch=1 (see §5, §9), report it as data, not as a failure.

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

## 9. Citable finding, not just a measurement-hygiene note

At batch=1, a model light enough — confirmed directly: MobileNetV3-Small (FP32 baseline, 25-26.5W),
EfficientNet-B0 (FP32 baseline, 38.6-40.5W), and ResNet-18's pruned50/70 states (35-52W) — **never
reaches this GPU's power cap at all** during single-image inference. ResNet-18's own FP32/FP16/pruned30
stay pinned at the ~60W cap in the same conditions. This is a direct, concrete illustration of this
proposal's actual thesis: **compression's real energy effect must be measured, not inferred from
compute-reduction proxies.** Any energy estimate built from "FLOPs × assumed constant power draw" would
be wrong by a large, non-uniform factor for exactly these cases — the assumed-constant power draw simply
isn't true once a model (or a sufficiently compressed state of one) stops saturating the GPU. The actual
wattage numbers above are the evidence; carry them into Results/Discussion with the real figures, not a
methods-appendix footnote — this belongs with the rest of the paper's hypothesis-relevant findings,
documented in §5 as it accumulated in real time.

**Mechanism precision, added 2026-10-06 from `nvidia-smi dmon` data (`deviation_log.md` D8's
correction):** "never reaches the power cap" is correct and is the citable claim; "idles between
launches" (an earlier, uncorroborated gloss on *why*) is not — the dmon log shows continuous, elevated
SM occupancy throughout every active window (89-96% for `resnet18_pruned70`, 57-58% for
`mobilenet_v3_small_fp32`), clock boosted to near-max the whole time, no idling between individual
batch=1 dispatches. The correct mechanism for the citable finding: a lighter model/state sustains
lower SM occupancy while continuously active, not zero occupancy — lower occupancy at full clock draws
less power. State it this way in the manuscript, not as a dispatch-idling story.
