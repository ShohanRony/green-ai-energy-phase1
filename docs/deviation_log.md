# Deviation and Decision Log — Phase 1

**Purpose:** every change from `phase1-execution-plan.md` and the proposal ("Energy-Aware Efficiency of
Lightweight Vision Models Under Post-Training Compression: Proxy Metrics, Measurement Reliability, and
Cross-Platform Behaviour", `Proposal_GreenAI_taught.NEW.docx`), recorded openly so readers and reviewers
can judge whether any decision could have shaped the results. Entries pulled from commit history, repo
reports, and the proposal document itself — not from memory. Where a specific fact could not be located
in these records, it is marked `TODO — not found in records` rather than guessed.

---

## D1. INT4 replaced by FP16 as the sixth compression state

- **Planned:** proposal §3 and Abstract both specify the six states as "FP32 baseline, INT8, INT4, and
  structured pruning at 30%, 50%, and 70%" — INT4 explicitly, not FP16.
- **Actual:** FP16 substituted for INT4, all three models (`phase1-execution-plan.md` §1.6; commit
  history predates this repo's tracked git log, carried in as an already-made decision per
  `docs/stage1-implementation-brief.md`: "the §1.6 INT4→FP16 decision is already made").
- **Trigger:** validated finding — INT4 has no `Conv2d` kernel in any installed library as of 2026
  (`bitsandbytes`-style libraries cover only `Linear4bit`; all three target models are
  convolution-dominated). Missing-tooling problem, not a measurement problem.
- **Alternatives considered:** none recorded beyond the dropped INT4 path — no evidence of a considered
  middle option (e.g., INT4-for-Linear-layers-only) in the records.
- **Rationale:** FP16 already verified working (1.7-2x genuine speedup, dtype-confirmed) on this
  hardware; disclosed substitution rather than silently dropping the state.
- **Timing:** BEFORE — decided during Stage 1 harness-build, before any compression-state energy data
  existed for any of the three models.
- **Could this change the conclusions?** Possibly. FP16 and INT4 have different expected energy/accuracy
  profiles; RQ2's FLOPs-vs-energy divergence question is tested on FP16 data, not the INT4 data the
  proposal specified. The §1.6 disclosure sentence (quoted verbatim in `results_stage2/stage2_deliverable.md`)
  states this is "consistent with the nominal-vs-realized compression gap documented in the literature
  review" — i.e., treated as itself informative rather than a gap to minimize.
- **Mitigation:** disclosed in Stage 2's deliverable notes and in this log; the exact disclosure sentence
  is reproduced verbatim in `stage2_deliverable.md` §5 for citation.

## D2. INT8 runs on CPU/fbgemm only; GPU energy column structurally empty

- **Planned:** proposal §3 doesn't specify INT8's execution device explicitly; implicitly assumes
  energy is measured consistently across states on the same instrument path.
- **Actual:** PyTorch INT8 segfaults on GPU dispatch for this hardware (confirmed, not assumed —
  `phase1-execution-plan.md` §1.6: "PyTorch INT8 on this GPU does not run; it segfaults"). INT8 for all
  three models runs CPU/fbgemm only; `pilot.py`'s GPU energy column is structurally absent (`gpu: null`)
  for INT8 runs, not silently zero'd — confirmed in Stage 3's checklist item 13
  (`results_stage3/stage3_deliverable.md`).
- **Trigger:** direct segfault reproduction on this GPU, logged in the execution plan before Stage 1
  began.
- **Alternatives considered:** ONNX Runtime's CUDA/TensorRT providers were checked and confirmed to
  silently fall back to CPU despite claiming GPU support (`phase1-execution-plan.md` §1.6) — ruled out,
  not merely unconsidered.
- **Rationale:** tooling constraint, not a methodology choice.
- **Timing:** BEFORE — confirmed during Stage 1 hardware feasibility audit, before any INT8 measurement
  was attempted.
- **Could this change the conclusions?** Yes, directly relevant to RQ1 (estimator agreement) and RQ3
  (cross-architecture rank ordering): INT8 is measured on a different power domain (CPU/RAPL) than the
  other five states (GPU/NVML) for the same models. Any energy comparison involving INT8 crosses
  instruments — flagged explicitly in `stage4-implementation-brief.md` §3 ("Never compare energy across
  different instruments directly" in the analysis plan, see File 2 §3).
- **Mitigation:** D13 (FP32-on-CPU baseline) added specifically so INT8 has a same-instrument comparison
  point rather than only being compared against GPU-measured FP32.

## D3. MobileNetV3-Small and EfficientNet-B0 INT8 via manual PTQ; cross-layer equalisation skipped

- **Planned:** proposal §2 ("Quantization method") specifies "post-training quantization follows the
  procedures documented by Nagel et al. (2021), including cross-layer equalisation, bias correction, and
  AdaRound."
- **Actual:** torchvision ships no official quantized variant for MobileNetV3-Small or EfficientNet-B0
  (only `mobilenet_v3_large` has one). Built a manual pipeline (`quantize_mobilenet_ptq.py`) using FX
  graph-mode quantization for fuse/observe/convert, with AdaRound and bias correction implemented, but
  **cross-layer equalisation deliberately skipped** — disclosed in the code docstring and
  `results_stage2/stage2_task_report.md`: "fbgemm's default qconfig already uses a per-channel weight
  observer, so the problem CLE targets is already handled by the backend." AdaRound itself is also a
  disclosed simplification: layer-wise, not the paper's full sequential/network-wise calibration.
  ResNet-18's INT8 used the official `torchvision.models.quantization.resnet18` path, no deviation.
- **Trigger:** confirmed absence of official quantized variants for these two architectures (checked by
  listing `torchvision.models.quantization`'s public names, not assumed).
- **Alternatives considered:** implementing a hand-written `QuantizableMobileNetV3`/EfficientNet wrapper
  class — rejected in favor of FX graph-mode quantization, which handles fuse/observe/convert generically
  without architecture-specific code.
- **Rationale:** CLE's purpose (fixing per-channel weight-range imbalance for per-tensor quantization
  backends) is already handled by fbgemm's per-channel weight observer — implementing it would be
  redundant with what the backend already does, not a methodology shortcut.
- **Timing:** BEFORE — decided and built during Stage 2, before INT8 accuracy numbers were measured for
  either model.
- **Could this change the conclusions?** Possibly, for RQ1/RQ2 on these two models specifically: a
  non-standard PTQ pipeline could produce different accuracy/energy characteristics than the full Nagel
  et al. procedure would. The EfficientNet-B0 INT8 accuracy cost was measurably larger than
  MobileNetV3-Small's in the original (now-superseded, see D-entries in
  `project_green_ai_stage2_plan.md`) 30-epoch data — attributed in the record to SiLU/squeeze-excite
  architectures being less quantization-robust, not to the CLE omission, but the two are not
  independently distinguished.
- **Mitigation:** disclosed in code docstring and task report; not hidden as if the full procedure were
  used.

## D4. Pruning fine-tuning policy: zero-finetune, uniform across all models and sparsities

- **Planned:** proposal §3 specifies structured pruning at 30/50/70% as three of the six states; does
  not explicitly mandate zero-finetune vs. brief-recovery — `phase1-execution-plan.md` §5 lists this as
  an "Open Technical Item... decide in Stage 2."
- **Actual:** zero-finetune policy, decided and confirmed by the user 2026-10-04, applied uniformly
  across all 3 models × 3 sparsity levels (`results_stage2/stage2_deliverable.md`, `stage2_task_report.md`).
- **Trigger:** a single-model decision checkpoint (ResNet-18 @ 70%, zero-finetune vs. 3-epoch brief
  recovery) showed zero-finetune collapsing to near-chance accuracy (10.35%) vs. 84.24% with brief
  recovery — a stop-and-decide point per the Stage 2 brief, not silently resolved.
- **Alternatives considered:** brief-recovery training after pruning (directly tested as the comparison
  arm in the decision checkpoint).
- **Rationale:** the proposal is explicitly a post-training compression study (title, Abstract); the
  methodology never specifies a retraining step, and introducing fine-tuning after pruning would blur
  pruning into pruning+retraining. Collapse is reported as a finding about the limits of post-training-only
  compression, not patched over.
- **Timing:** AFTER — the decision was made having already seen the ResNet-18 @ 70% decision-checkpoint
  result (10.35% vs. 84.24%). This is explicitly a results-informed decision, not a pre-registered one.
- **Accuracy collapse at 70%, confirmed:** ResNet-18 10.35%, MobileNetV3-Small 10.00%, EfficientNet-B0
  10.00% (all nominal-70% states, near CIFAR-10's 10% chance floor) — `stage2_deliverable.md` §3. 50% also
  collapsed for MobileNetV3-Small (10.00-11.16% across the original and retrained baselines) and later
  showed EfficientNet-B0 50% collapsing too under the retrained (60-epoch) baseline (13.29%, see D-entries
  in `project_green_ai_stage2_plan.md`).
- **Could this change the conclusions?** Yes, directly: RQ2 (FLOPs-vs-energy proportionality) cannot be
  meaningfully assessed for collapsed states on accuracy grounds — energy is still measured and real, but
  the accompanying accuracy is near-chance. This is exactly the scenario Part A File 2 §6's
  not-yet-set accuracy threshold is meant to handle.
- **Mitigation:** policy applied uniformly (not cherry-picked per model/ratio), collapse reported
  honestly rather than excluded, and flagged for the accuracy-threshold rule in the Stage 5 plan.

## D5. ARM/M1 — sequencing and prior documentation

- **Actual status, confirmed directly by the researcher 2026-10-06:** M1 hardware access IS available.
- **What the records say, verified across every stage's documentation (Stage 1 through today, 2026-10-06):**
  every prior brief and memory file states M1 as "not yet set up" / "no access" / "not started" —
  `phase1-execution-plan.md` §1 ("Not yet set up — §1.5's validation sprint was entirely on the LOQ;
  nothing has been run on the M1 yet"), `stage1-implementation-brief.md` Task 6 ("not started"),
  `stage3_pilot_report.md` and `stage4-implementation-brief.md` (both state ARM/M1 "out of scope,"
  "no access to the MacBook Air in any session"). No record anywhere in this repo's commit history,
  briefs, or reports states M1 was available before 2026-10-06, and no record flags the "no access"
  wording as inaccurate prior to this entry.
- **This entry is itself the correction**, made 2026-10-06 at the researcher's direct instruction: M1
  work was sequenced after x86 by choice, not blocked by unavailable hardware, and earlier wording
  stating "no access" should be read as imprecise going forward. **Flagged, not asserted as having been
  true all along:** this log cannot independently verify *when* M1 access actually became available
  versus when the prior "no access" wording was written, because no record establishes that date. If
  this distinction matters for the manuscript (e.g., whether M1 testing could have started earlier),
  that requires the researcher's own recollection, not something inferable from this repo's records —
  `TODO — exact date M1 access became available not found in records; researcher to confirm if needed`.
- **Could this change the conclusions?** Not on its own — this is a sequencing/documentation question,
  not a methodological one. It matters for D11 (timeline) and for whether RQ3 (cross-platform rank
  ordering) can be answered within Phase 1 as originally scoped, since RQ3 requires M1 data and none has
  been collected as of this log's writing.
- **Mitigation:** Stage 1-M1, 3-M1, and 4-M1 now explicitly planned (`stage5_analysis_plan.md` §10),
  sequenced after x86 Stage 4, before any cross-platform (RQ3) analysis.

## D6. Batch size

- **Planned:** `phase1-execution-plan.md` §2 checklist item 5: "Batch size recorded for every figure;
  vary 2–3 values as a declared control variable." The proposal itself (§3, Protocol controls): "Batch
  size is varied across a small set of values as a control variable, not as a headline question."
- **Actual, full history:**
  - Stage 2's `fp16_speedup_check.py`/`prune_full_grid.py` speedups measured at **batch=128**
    (latency only, no energy).
  - Stage 3's 6-state pilot used **batch=16** (chosen for direct comparability with Stage 1's exit-test
    convention, `--sizes 32 --batches 16`).
  - Stage 4 pre-flight explored **batch=1, 4, 8, 16, 32, 64** (diagnostic sweep investigating the
    power-regime question, not a declared-control-variable sweep for the main results).
  - Stage 4's actual matrix **locked to batch=1 only** — a single value, not 2-3 as the plan and
    proposal both specify.
- **Trigger:** Stage 3's report flagged that compressed-state energy ratios at batch=16 looked larger
  than Stage 2's batch=128 speedups would predict, prompting the pre-flight batch investigation
  (`results_stage4_preflight/stage4_preflight_findings.md`).
- **Alternatives considered:** sweeping batch size as an explicit Stage 4 factor — considered during
  pre-flight (`stage4-implementation-brief.md` §2: "Sweeping batch size as a factor — considered during
  pre-flight, not adopted"), not adopted for the main matrix.
- **Rationale given in the brief:** batch=1 justified by deployment realism (genuine single-image edge
  inference). **An inconsistency worth recording plainly:** the instruction that first set Stage 4's
  headline batch size (given before the pre-flight investigation, recorded in this conversation's own
  history, not a separate file) said: "if the trend is monotonic, adopt batch=16 as Stage 4's headline
  condition and log the decision with reasoning (edge-deployment realism + continuity with Stage 1/3)."
  Batch=16 and batch=1 were each justified by "edge-deployment realism" at different points — the same
  framing was used for two different answers, six days apart. This is recorded as an inconsistency in the
  decision trail, not resolved further here.
- **Timing:** AFTER for both the original batch=16 framing (chosen to match Stage 1/3's existing
  convention, aware of existing pilot numbers) and the final batch=1 lock (chosen after the pre-flight
  investigation's dip/pinned findings were already in hand, specifically to route around the
  power-regime confound that batch=16's comparison to batch=128 had surfaced).
- **Cost framing:** the original Stage 4 plan estimate (3-4 weeks of evenings) was cited as a reason not
  to sweep batch size as a full matrix factor; this estimate was later revised to ~4.5-5 hours once
  Stage 2's artifacts were confirmed already built and ARM was confirmed out of scope for that estimate
  (`stage4-implementation-brief.md` §8) — i.e., part of the cost basis for not sweeping batch size was
  itself revised after the sweep decision was made.
- **Could this change the conclusions?** Yes, directly and substantially — this is the subject of D7/D8
  below. Energy-per-image at batch=1 is not simply a scaled version of batch=16/128 results; the
  GPU's power regime itself depends on batch size and model compute cost in a way that was not
  understood when batch=16 was first chosen.
- **Mitigation:** batch=16 added as a secondary condition in the Stage 5 analysis plan (`stage5_analysis_plan.md`
  §9) — planned as Part B work, not yet executed as of this log's writing.

## D7. GPU power-regime investigation

- **Discovery:** Stage 4 pre-flight found this GPU lands in one of two distinct power regimes per active
  window at batch=1 — pinned at the ~60W enforced cap, or an unsaturated ~55-90% of it — in a pattern
  that didn't initially track batch size, state, or model in any simple way.
- **Ruled out, in order:** (1) CPU governor — checked via logged `cpu_governors` field across all 45
  runs collected at the time; all 6 observed dips had governor correctly logged `performance`, one run
  under `powersave` still landed pinned — zero correlation either direction
  (`results_stage4_preflight/stage4_preflight_findings.md`). (2) Thermal soak alone — reran two combos
  in the same long-running boot session (not a fresh reboot) and both stayed pinned, including one
  measured ~3 hours into that session (**checked against real `timestamp_utc` values 2026-10-06: the
  actual gaps were 3:57:15 for `pruned70_b32` and 4:19:00 for `fp16_b64`, i.e. closer to 4 hours than
  3 — "~3 hours" undercounted by roughly an hour; corrected here, original wording left visible rather
  than silently fixed**). (3) Boot-identity hypothesis — falsified directly: two separate
  deliberate reboots, same two combos, both landed pinned both times (not a fresh dip on either reboot).
- **Unconfirmed, stopped for cost reasons:** cumulative-load/uptime-duration hypothesis (something
  degrading over a long session, resetting on any reboot) remained the leading but untested candidate
  after the boot-identity falsification. Testing it would require a deliberate multi-hour session without
  rebooting to see if a dip reappears — not run; investigation explicitly closed at the researcher's
  direction rather than pursued further.
- **Resolution during real Stage 4 execution (2026-10-06):** confirmed directly, not inferred, across
  three different models' own FP32 baselines that whether a config dips or pins at batch=1 tracks
  compute cost relative to the GPU's boost-sustaining threshold: ResNet-18 (557M MACs) pinned at FP32,
  dipped only when heavily pruned (50/70%); EfficientNet-B0 (33.28M MACs) dipped at FP32 (38.6-40.5W);
  MobileNetV3-Small (5.8M MACs) dipped at FP32 (25-26.5W). This is a **real, physically-explained
  finding about this hardware**, not an unresolved mystery — it supersedes the uptime hypothesis as the
  operative understanding, though the uptime hypothesis itself was never directly disproven, only
  superseded by a better-supported explanation.
- **Harness guards added, both enforced in code:**
  - `power_regime` field (`pilot.py`'s `classify_power_regime()`): threshold `implied_w >= 0.9 *
    power_cap_w`, calibrated on the two observed clusters (pinned always ≥98.8% of cap, dip always
    ≤88.5% — verified exact values from `pilot.py`'s own docstring and `stage4_preflight_findings.md`).
  - Fresh-boot guard (`check_fresh_boot()`): refuses to start past `--max-uptime-min` (default 30,
    overridable via `--allow-stale-boot`). **This threshold is an unvalidated placeholder**, stated as
    such in the code and brief — chosen conservatively from the pre-flight sessions' own timing, not
    from a measured decay boundary (no experiment established where the actual boundary, if any, lies).
- **Timing:** the mechanism understanding evolved entirely AFTER seeing outcome data — first the
  pre-flight dip/pinned pattern, then the real Stage 4 execution data that resolved it. No part of this
  was pre-registered.
- **Could this change the conclusions?** Yes, significantly, for any energy number in this project
  collected before the `power_regime` field existed (Stage 1, 2, 3, and pre-flight's own energy
  numbers) — those runs have no recorded regime classification and cannot retroactively be checked. See
  D12.

## D8. Stage 4 §5 standing rule evolution

- **Original rule** (`stage4-implementation-brief.md`, as first written): any `dip` or `mixed`
  `power_regime` result triggers stop-reboot-rerun before continuing to the next condition — treated as
  an escalation requiring investigation.
- **First real-world trigger:** `resnet18_pruned50` at batch=1 returned `dip` on its first Stage 4
  measurement. Per the rule, the researcher rebooted and the condition was rerun — **this is the only
  condition in the full 18-condition matrix that was actually reproduced across a deliberate reboot
  during real Stage 4 execution**; it dipped both times (pre-reboot and post-reboot), both consistently
  across all reps within each run (no mixing).
- **Carve-out introduced, then generalized:** after `resnet18_pruned50`'s reboot-reproduced dip was
  recognized as matching Task 1's own pre-flight finding (pruned50/70 sub-saturate at batch=1 for
  ResNet-18, already measured and explained before Stage 4 started), the rule was amended to exempt
  configs "already known" to sub-saturate from the reboot-and-rerun trigger. When MobileNetV3-Small's
  FP32 baseline (not a pruned state) also showed a consistent, tight dip (25-26.5W across all 30 reps),
  and then EfficientNet-B0's FP32 baseline did too (38.6-40.5W), the carve-out was generalized again:
  dip/mixed is no longer a stop-and-ask trigger at all for the remainder of Stage 4; Task 2's
  plausibility pass (checking energy-per-image ordering and magnitude against each model's own FP32
  baseline) is the real backstop instead.
- **Mechanism basis:** inferred from the wattage pattern across three models' baselines (a consistent,
  monotonic-with-compute-cost relationship), not independently verified against a direct measure of GPU
  saturation (e.g., `nvidia-smi dmon` utilization/clock logging during the actual dip windows). The
  physical story (model fast enough that the GPU idles between launches) is plausible and consistent
  with the wattage data, but was not confirmed by a second, independent signal.
- **Only one condition (`resnet18_pruned50`) was reproduced across a reboot** under the full rule before
  it was relaxed — the MobileNetV3-Small and EfficientNet-B0 baseline dips that drove the later
  generalizations were each observed once, not reboot-tested.
- **Timing:** AFTER — every step of this evolution happened in direct response to Stage 4's own
  incoming data, in real time, across a single measurement session.
- **Could this change the conclusions?** Possibly: if the uptime/session-state hypothesis (not fully
  ruled out, only superseded) turns out to be a real contributing factor alongside the compute-threshold
  explanation, some fraction of the dip conditions accepted without a reboot-reproduction check could be
  measuring a different, unaccounted-for state. The compute-threshold explanation is well-supported by
  the wattage pattern but not independently confirmed via clock-state telemetry.
- **Mitigation:** stability reruns (one repeat per dip condition in a separate fresh session) plus an
  `nvidia-smi dmon` utilization/clock check during at least 2 of those reruns — planned as Part B work,
  not yet executed as of this log's writing.

## D9. Excluded, discarded, or duplicated runs

Full accounting, checked directly against logs rather than recalled:

- **Governor-contaminated determinism-check rerun** (pre-flight, not Stage 4 proper):
  `pruned70_b16_rerun` ran under `powersave` (the reboot that caused it reset the governor; nothing had
  restored it yet) while being compared against a `performance`-governor baseline. Discarded as a
  confound and rerun cleanly under `performance`. One run discarded.
- **Stage 4 `resnet18_pruned50` duplicate:** the condition was measured twice (original, pre-reboot;
  rerun, immediately post-reboot), both landing `dip` consistently. The rerun directory
  (`resnet18_pruned50_rerun1`) was deleted after confirming consistency; the original is the canonical
  dataset entry. This is a duplicate-removal for cleanliness, not an exclusion for invalidity — both
  runs were equally valid dip measurements.
- **Plausibility-guard trips** (`plausibility_flag=True`, the physically-implausible-power check):
  checked directly across every row in `results_stage3/`, `results_stage4/`, and
  `results_stage4_preflight/`'s raw logs — **zero rows flagged**, across all data collected in this
  project to date.
- **Cold-start reps:** one per condition, discarded from summary statistics by design (not an anomaly) —
  18 cold reps across Stage 4's 18 conditions (one per condition, `pairs=30` reported from 31 collected
  reps each), consistent with the same discard rule used in every stage since Stage 1
  (`phase1-execution-plan.md` §2 checklist item 3).
- **CUDA OOM skips:** none. **Closed 2026-10-06** — `find results_stage4 -iname skipped.jsonl` (the
  file `pilot.py` creates only on an actual CUDA OOM) returns zero matches, checked directly across
  every condition directory, not just noted in passing during execution.

## D10. Power outage mid pre-flight

- **What happened:** the laptop rebooted unexpectedly partway through Stage 4 pre-flight work
  (confirmed via `uptime`, not assumed — the system showed "up N min" immediately after, consistent with
  a fresh boot, not a resume-from-suspend).
- **Audit result:** every run directory from before the outage was checked directly — all showed
  `pairs=9` or `pairs=30` as expected for their stage, nothing missing or truncated. No data was lost.
- **Side effect, not data loss:** one determinism-check rerun (see D9) was invalidated by the governor
  reset this same outage caused, and was redone — a confound caught and corrected, not a gap in the
  record.
- **Consequence noted:** this outage turned out to be the event that first exposed the pin/dip
  regime's reboot-sensitivity (D7) — the investigation that followed was prompted by this accidental
  reboot, not a planned experiment.

## D11. Timeline revised from 3-4 weeks to ~5 hours of x86 measurement

- **Planned:** `phase1-execution-plan.md` §3, Stage 4: "Est. 3-4 weeks (evenings)."
- **Actual:** `stage4-implementation-brief.md` §8 revises this to "~4.5-5 hours of actual measurement
  time" for the 18-condition x86 matrix, citing two reasons: Stage 2's compression artifacts already
  existed (no new training needed) and ARM was out of scope for that specific estimate at the time it
  was written.
- **Timing:** the revision was made when the Stage 4 brief was drafted, after Stage 2's artifacts were
  already confirmed built — not a blind estimate.
- **Could this change the conclusions?** Not directly — a faster timeline doesn't change what was
  measured. Relevant context: D6 notes this original 3-4-week estimate was part of the stated reason for
  not sweeping batch size as a full Stage 4 factor; the estimate was revised downward after that
  decision was made, which is itself worth having on record given D6's documented inconsistency.

## D12. Stage 1-3 and pre-flight energy numbers designated pilot-only

- **Decision:** energy numbers collected in Stage 1, Stage 2, Stage 3, and Stage 4 pre-flight are
  designated pilot-only and excluded from confirmatory analysis — they were collected before the
  power-regime issue (D7) was characterised, and none of them carry a `power_regime` classification
  (the field didn't exist in `pilot.py` until the guard was added during Stage 4 pre-flight's closing
  commit).
- **Trigger:** D7's finding that energy-per-image depends on which of two power regimes a run landed in,
  and that this wasn't controlled for or even known during Stages 1-3.
- **Timing:** AFTER — this designation is being made now, after Stage 4's own data resolved the
  mechanism, not pre-registered.
- **Could this change the conclusions?** By design, this *prevents* those earlier numbers from
  influencing confirmatory conclusions — they're relegated to pilot status specifically because they
  can't be trusted to the same standard as Stage 4's regime-aware data.
- **Mitigation:** `stage5_analysis_plan.md` §2 formalizes this exclusion explicitly.

## D13. FP32-on-CPU baseline added after Stage 4

- **Planned:** not in the original proposal or execution plan — a new addition.
- **Actual:** 3 conditions (one per model), same settings as the INT8 CPU runs (`--device cpu`,
  batch=1, 31 repeats), added so INT8 has a same-instrument (CPU/RAPL) comparison point rather than only
  being measurable against GPU-measured FP32.
- **Trigger:** D2's observation that INT8 and the other five states are measured on different power
  domains/instruments, making any direct INT8-vs-FP32(GPU) energy comparison instrument-confounded.
- **Timing:** AFTER — added once the full x86 Stage 4 matrix and D2's instrument-mismatch implication
  were both already in hand.
- **Could this change the conclusions?** Improves RQ1/RQ2 answerability for INT8 specifically, by
  providing a same-instrument baseline; does not change any already-collected GPU-measured data.
- **Status: executed, 2026-10-06.** All 3 conditions run and committed (`7c5c6d7`): `resnet18_fp32_cpu`
  (started 2026-10-06T13:25:53Z), `mobilenet_v3_small_fp32_cpu` (13:42:27Z), `efficientnet_b0_fp32_cpu`
  (13:59:02Z) — same session (post-reboot, uptime 1840-3829s), 31 reps each, `pairs=30` after the
  cold-start discard, in `results_stage4/{arch}_fp32_cpu/`.

## D14. Stage 5 statistics: proposal's own tests restored as primary, Mann-Whitney U substituted for signed-rank

- **Planned:** proposal §3 specifies Wilcoxon signed-rank tests for paired compression-state
  comparisons, Spearman rank correlation for RQ2, Kendall's tau for RQ3.
- **Intermediate deviation (2026-10-06, same day):** `stage5_analysis_plan.md` was first registered
  with Welch's t-test, bootstrap 95% CIs, and Holm-Bonferroni correction as the primary methods
  instead — at the researcher's explicit instruction at the time of drafting — with the proposal's
  tests demoted to a status note.
- **Actual, resolved same day (2026-10-06), at the researcher's explicit instruction:** reversed back.
  The proposal's tests are primary: **Mann-Whitney U (Wilcoxon rank-sum), not the signed-rank test**,
  for compressed-vs-baseline comparisons — substituted because Stage 4's reps are independent repeated
  measurements of a static configuration, not paired observations; the signed-rank test requires a
  real pairing between the two samples (matched units or the same unit under two conditions), which
  does not exist between a compressed-state rep and a baseline rep. Applying the signed-rank test here
  would misrepresent the data's structure, not just be a weaker choice. Spearman (RQ2) and Kendall's
  tau (RQ3) are retained as specified, with an explicit caveat attached: both pool across a model's 6
  compression states, which are not independent draws (same checkpoint, related transformations), and
  n=6 per model gives low power regardless — both are reported as descriptive signals, not
  confirmatory tests. Welch's t-test/bootstrap CI/Holm-Bonferroni are kept as supplementary,
  triangulating rather than replacing the primary tests.
- **Trigger:** the researcher's direct review of the divergence after it was first logged, resolving
  which method set should be read as confirmatory.
- **Timing:** AFTER — both the original Welch's-primary registration and this correction back to the
  proposal's tests happened after `stage5_analysis_plan.md` was first drafted, same day, before any
  actual Stage 5 statistics were run on real data (none have been, pending §6's accuracy threshold).
- **Could this change the conclusions?** Not on data collected so far — no Stage 5 statistics have
  been run under either method set yet. It determines which test family will be read as confirmatory
  once they are.
- **Mitigation:** `stage5_analysis_plan.md` §4 and its top status note both updated to reflect this
  resolution; the original Welch's-primary text is not silently erased from this log's history, only
  from the live plan (which per its own §12 rule amends rather than edits in place — this D14 entry
  and the status-note update together serve as that amendment record).
- **OPEN, 2026-10-06:** the proposal's own §3 text (which specific comparisons it meant the signed-rank
  test for, and whether its design assumed paired reps) is not independently re-verifiable right now —
  the external drive hosting `Proposal_GreenAI_taught.NEW.docx` is not mounted on this machine as of
  this entry. Everything above is based on the paraphrase already committed in this log/plan from when
  the drive was last mounted, not a fresh re-read. **This entry stays open** until the drive is
  remounted and the exact §3 wording is re-quoted verbatim; if that wording turns out to describe a
  genuinely paired design for the compressed-vs-baseline comparison (not assumed here), this
  resolution would need revisiting.
- **Signed-rank may still be the right primary test elsewhere — not ruled out generally.** The
  compressed-vs-baseline energy comparison (where Mann-Whitney U was substituted above) is the
  *unpaired* case. The CodeCarbon-vs-hardware instrument-agreement comparison (D15, RQ1) is different:
  both readings come from the same run, the same active window, at the same time — a real pairing
  exists there. Wilcoxon signed-rank (or a paired bootstrap on the per-window differences) is plausibly
  the *correct* primary test for that specific comparison, not a misapplication — flagged here so the
  Mann-Whitney substitution above isn't read as a blanket "signed-rank is wrong for this project"
  conclusion.

## D15. CodeCarbon never enabled in any Stage 4 run

- **Planned:** proposal §3 requires CodeCarbon to run concurrently with hardware measurement on
  *every* run, stated as what makes the RQ1 instrument-agreement comparison a within-run paired
  comparison.
- **Actual:** confirmed directly against logged `arguments.codecarbon` in every one of the 21 primary
  Stage 4 `environment.json` files (18 compression-state conditions + 3 D13 FP32-CPU baselines):
  `False` in all 21, zero exceptions. `pilot.py`'s `--codecarbon` flag (`action='store_true'`, default
  `False`) was implemented back in Stage 1 (commit `ea89ad1`, "Wrap CodeCarbon to run concurrently with
  RAPL/NVML reads") but the flag was never passed in any Stage 4 command — `stage4-implementation-brief.md`'s
  Task 1 command template (§Task 1) has no `--codecarbon` in it, and no Stage 4 commit message
  (`d0bfb70` through the matrix) mentions enabling it.
- **Consequence:** no paired CodeCarbon readings exist anywhere in the Stage 4 dataset. RQ1 (software
  estimator vs. hardware agreement) cannot be answered from the x86 data collected so far.
- **Trigger:** direct audit requested by the researcher (2026-10-06, Part B item 3 of the 7-item
  audit), checked against the brief, the harness default, the command template, and git history
  rather than assumed.
- **Feasibility check run same day:** 3 conditions (`resnet18_fp32`, `mobilenet_v3_small_fp32` on GPU;
  `resnet18_int8` on CPU), batch=1, 6 reps each, `--codecarbon` enabled, in
  `results_stage4/codecarbon_check/`. Findings:
  - CodeCarbon's own `codecarbon_energy_j` diverges sharply from the hardware counter and not in a
    fixed direction: ResNet-18 FP32 (GPU) ≈1.95x hardware; MobileNetV3-Small FP32 (GPU) ≈3.22x
    hardware; ResNet-18 INT8 (CPU) ≈0.83x hardware. Consistent with CodeCarbon estimating whole-system
    (CPU+GPU+RAM) energy by its own internal model rather than reading the same single-device counter
    `pilot.py` reads (NVML GPU-only for the GPU runs; RAPL package for the CPU run) — not a simple
    calibration offset.
  - CodeCarbon's own reading is also far noisier than the hardware counter: CV 3.0-15.3% across the
    three conditions' `codecarbon_energy_j`, vs. 0.6-4.7% for the paired `energy_j` hardware readings
    over the same windows.
- **Rerun with per-component logging (2026-10-06, same day, `pilot.py` extended to log
  `codecarbon_cpu_energy_j`/`_gpu_energy_j`/`_ram_energy_j`, not just the total):**

  | Condition | hardware `energy_j` | CC `cpu_energy_j` | CC `gpu_energy_j` | CC `ram_energy_j` | CC total |
  |---|---|---|---|---|---|
  | resnet18_fp32 (GPU) | 296.63 | 127.31 | 346.60 | 116.09 | 590.00 |
  | mobilenet_v3_small_fp32 (GPU) | 137.99 | 131.87 | 193.56 | 116.39 | 441.82 |
  | resnet18_int8 (CPU) | 736.02 | 266.40 | 98.10 | 107.45 | 471.95 |

  This resolves the mechanism precisely: for the two GPU conditions, CodeCarbon's `gpu_energy_j` alone
  is already 1.17x (resnet18) and 1.40x (mobilenet) the hardware NVML reading — consistent with
  CodeCarbon's GPU path calling `pynvml.nvmlDeviceGetTotalEnergyConsumption` (confirmed by reading
  `codecarbon/core/gpu_nvidia.py` in the installed package, v3.3.1), the same cumulative NVML counter
  this project's own Stage 1 validation work (commits `92fd4c1`/`672ce93`/`975b7fd`) found over-reports
  active-phase power by ~30% on this GPU versus the validated `nvmlDeviceGetPowerUsage` default
  `pilot.py` uses. On top of that, CodeCarbon adds `cpu_energy_j`≈127-132 J and `ram_energy_j`≈107-116 J
  to every condition regardless of what the workload actually touches — including a nonzero
  `gpu_energy_j`=98.10 J on the **CPU-only** `resnet18_int8` run, where no CUDA code executes at all.
  CodeCarbon tracks all detected hardware on the machine as whole-system overhead, not just the
  component the measured workload uses; `pilot.py`'s hardware figure, by contrast, is always
  single-device by design (§3 of `stage5_analysis_plan.md`).
- **A second, independent finding surfaced while checking the CPU path (item 4c): `pilot.py`'s own
  RAPL reading sums two domains, not one.** `Sensor.__init__`'s glob
  (`/sys/class/powercap/intel-rapl:*/energy_uj`, filtered to one colon) matches **both**
  `intel-rapl:0` (`package-0`) **and** `intel-rapl:1` (`psys`) on this machine — confirmed directly by
  listing `/sys/class/powercap/intel-rapl:*/name`. `integrate()` sums every matched domain's delta into
  one scalar. `psys` ("platform", typically package + chipset + voltage regulators) is a superset of
  `package-0`, not an independent additive domain — summing both very likely substantially inflates
  every CPU/RAPL energy number this project has ever recorded. CodeCarbon's own RAPL path
  (`core/resource_tracker.py`'s `_setup_rapl()` → `IntelRAPL`) defaults to `rapl_prefer_psys=False`,
  i.e. **package-only** — which is almost certainly why its `cpu_energy_j` (266.40 J) is so much lower
  than `pilot.py`'s combined package+psys reading (736.02 J) for the same `resnet18_int8` run, rather
  than the two being expected to agree. **Logged separately as D16 — this affects every CPU-measured
  condition in the project (all 3 INT8 states + all 3 D13 FP32-CPU baselines, 6 conditions total), not
  just this feasibility check, and is not yet remediated.**
- **4(a) — CodeCarbon's internal sampling cadence, confirmed from source:** `pilot.py` sets
  `measure_power_secs=max(a.interval,1)` = 1 second (since `a.interval=0.4`). Confirmed from
  `emissions_tracker.py`: `.start()` takes one immediate sample, then a `PeriodicScheduler` fires the
  same sampling function every `measure_power_secs` until `.stop()`. Each active window is a fixed
  5-second box (`window_s=5.0`), so each window gets ~5-6 internal CodeCarbon samples — comfortably
  ≥2, confirmed by reading the scheduling code, not inferred from the setting alone.
- **4(d) — the wall-clock "no overhead" argument is withdrawn, not just caveated.** Each active window
  is time-boxed (`window()` loops `while time.perf_counter()-start < seconds`, a fixed 5.0s regardless
  of how much work completes) — so wall-clock duration structurally cannot reveal throughput-level
  overhead; it only shows whether *outer* per-rep bookkeeping grew, which it didn't (32.7 s/rep in two
  independent reruns, matching the primary matrix's 32.1-33.9 s/rep). The correct probe is `batches`
  completed per fixed window: with CodeCarbon on, resnet18_fp32 completed **5.8% fewer** batches/window
  than its primary (no-CodeCarbon) run (2549 vs. 2706, with-CC SD only 9.8 — not noise); mobilenet and
  resnet18_int8 showed smaller reductions (−1.5%, −0.8%). CodeCarbon's background sampling thread does
  measurably compete for CPU/GIL time during active dispatch for at least one model; this would show up
  as slightly higher **gross J/image** (this project's primary energy unit, energy ÷ images processed)
  under CodeCarbon, not in the window's raw energy total, which integrates power over fixed wall time
  and is insensitive to how many images that time produced.
- **Two separate check runs exist (v1, v2) — logged explicitly, not silently merged or treated as one
  replacing the other.** Both ran the identical 3-condition protocol, in the same uninterrupted boot
  session, ~1h45m apart:

  | | resnet18_fp32 hw | mobilenet_v3_small_fp32 hw | resnet18_int8 hw | resnet18_fp32 CC | mobilenet CC | resnet18_int8 CC |
  |---|---|---|---|---|---|---|
  | v1 | 297.04 | 129.92 | 694.81 | 580.09 | 418.43 | 573.69 |
  | v2 | 296.63 | 137.99 | 736.02 | 590.00 | 441.82 | 471.95 |

  **v1's raw data was overwritten by v2's rerun** (the recompute script's predecessor used `rm -rf`
  before rewriting each directory to add the per-component CodeCarbon fields) — a lapse against this
  project's own standing checklist item 12 (preserve raw data; don't overwrite without a determinism
  check first), not caught before the overwrite happened. **Both runs were stale-boot**: v1 at uptime
  85-89 min, v2 at uptime 186-193 min, same session throughout, both past the 30-min fresh-boot
  threshold. **v2 is not labelled as superseding v1** — both are feasibility-check-grade data from a
  stale-boot session, neither confirmatory.
  **Between-run vs. within-run variability, stated plainly because it matters:** the two conditions
  that moved between v1 and v2 (mobilenet hardware: 129.92→137.99, +6.2%; resnet18_int8 hardware:
  694.81→736.02, +5.9%) shifted by more than those conditions' own within-run CV (1.15% and 0.56%
  respectively, measured within v2) — i.e. the between-session difference is **larger than the
  within-session noise it would need to be explained by chance alone**. resnet18_fp32's hardware
  barely moved (+0.14%), smaller than its own 3.03% within-run CV. CodeCarbon's totals moved more:
  +1.7%, +5.6%, and **−17.7%** respectively, the last comparable to that condition's own 15.26%
  within-run CV. **This means between-session variability for this harness has not been measured and
  is not yet known to be small — it needs its own dedicated check (e.g. repeated short runs across
  several uptime points in one session, and across separate sessions), not assumed negligible because
  within-run CVs look tight.** Not yet done; flagged here as an open gap, not closed.
- **Could this change the conclusions?** Directly blocks RQ1 as currently specified — no instrument-
  agreement claim can be made from Stage 4 data without CodeCarbon-paired runs. Does not affect RQ2 or
  RQ3, which don't depend on CodeCarbon. D16 (RAPL double-counting) is the more consequential finding
  and could affect every reported CPU-side energy number pending its remediation. The v1/v2
  between-session gap is a separate, still-open question about how stable this harness's readings are
  across time within one long session — relevant to interpreting any single-session measurement in
  this project, not just the CodeCarbon checks.
- **4(e), proposed, not executed:** revised to **21 conditions** (18 compression states + the 3 D13
  FP32-CPU baselines, all of which need the same paired instrument-agreement check), batch=1,
  CodeCarbon on, kept as a separate result set (e.g. `results_stage4_codecarbon/`), **with a
  per-condition overhead audit** (hardware `energy_j`/`total_j_mean` with CodeCarbon on vs. the
  matching primary Stage 4 run, for all 21, not just the 3 feasibility-checked ones) rather than
  assuming the 3-condition feasibility check generalizes. **Explicitly: this pass is NOT expected to
  reproduce the primary matrix's numbers unchanged** — the 4(d) `batches`-per-window finding (−5.8% for
  `resnet18_fp32`) shows CodeCarbon can measurably reduce per-window throughput, which would show up as
  higher gross J/image under CodeCarbon for at least that model. The per-condition audit exists
  specifically to quantify this per condition, not to confirm a no-change assumption. **Wall-time
  estimate, from real timestamps:**
  per-rep wall time with CodeCarbon on was measured at 32.7 s/rep in two independent reruns — used here
  as an empirical basis, not as evidence of "no overhead" (see 4(d) above, that argument no longer
  applies) — matching the primary matrix's own 16.6-17.5 min/condition at 31 reps. 21 × ~16.6-17.5 min
  ≈ **5.8-6.1 hours total**. Awaiting approval before execution.
- **Mitigation:** gap disclosed here and in `stage5_analysis_plan.md` §3 rather than silently worked
  around; RQ1 reported as unanswerable from current data until the proposed pass runs.

## D16. `pilot.py`'s CPU/RAPL energy reading sums `package-0` and `psys` domains — ROOT CAUSE CONFIRMED

**Status: root cause confirmed 2026-10-06. Remediation decided: offline recompute + harness patch,
no rerun of already-collected data.**

- **Discovery:** surfaced while answering D15/item 4(c) ("does CodeCarbon read RAPL or fall back to
  TDP for the CPU"). Checking what CodeCarbon's RAPL path does differently from `pilot.py`'s required
  checking what `pilot.py`'s RAPL path actually reads — and it reads more than intended.
- **What `pilot.py` actually does:** `Sensor.__init__` (device='cpu') globs
  `/sys/class/powercap/intel-rapl:*/energy_uj`, filtered to paths one colon deep (correctly excluding
  the `intel-rapl:0:0` `core` sub-domain). On this machine that filter still matches **two** top-level
  domains: `intel-rapl:0` (`package-0`) and `intel-rapl:1` (`psys`) — confirmed by reading each
  domain's `name` file directly. `integrate()` sums every matched domain's energy delta into one
  number.
- **Root cause confirmed, three independent ways (2026-10-06):**
  1. **Documented intent contradicts the code.** This project's own `README.md` §"RAPL domain in use"
     states explicitly: "The harness reads `package-0` (CPU package energy), not `psys` (whole-system).
     This is a stated deviation, not a bug." The code does not do this — it reads both and sums them.
  2. **Fresh empirical check** (15s idle + 30s 16-core CPU load, explicit domain names, nothing else
     running): `psys` > `package-0` at both idle (40.93 W vs. 12.31 W) and under load (133.37 W vs.
     84.80 W) — consistent with `psys` being a superset platform-power domain (package + chipset +
     voltage regulators), not an independent additive rail.
  3. **Package-only recompute matches an independent instrument.** Recomputing `resnet18_int8`'s
     feasibility-check energy using only the `package-0` trace column gives 266.21 J (SD 3.07) —
     matching CodeCarbon's own, independently-read, package-only `cpu_energy_j` of 266.40 J (SD 3.16)
     to within **0.1%**. `pilot.py`'s combined (package+psys) reading for the same windows was 736.02 J
     — **2.765x** the package-only figure, i.e. `psys` very nearly triples the reported CPU energy for
     this workload.
- **Scope — 10 directories, all CPU, confirmed by checking `arguments.device` across every
  `environment.json` in the repo, not just the 6 originally flagged:** `results_stage2/pilot_proof_int8`,
  `results_stage3/resnet18_int8`, `results_stage4_preflight/resnet18_int8_b1`,
  `results_stage4/{resnet18,mobilenet_v3_small,efficientnet_b0}_int8`,
  `results_stage4/{resnet18,mobilenet_v3_small,efficientnet_b0}_fp32_cpu` (D13),
  `results_stage4/codecarbon_check/resnet18_int8`. Zero GPU/NVML-measured conditions are affected
  (confirmed: every GPU run's `trace` is a single-element list, one sensor, no RAPL involvement).
- **Timing:** discovered 2026-10-06, after all 10 affected directories were already collected and
  committed.
- **Could this change the conclusions?** Potentially significantly for any RQ1/RQ2 claim involving
  INT8 or the FP32-CPU baseline — the package+psys sum is confirmed inflated (not just "likely"), by a
  factor of ~2.8x for `resnet18_int8`'s feasibility-check windows specifically; the exact factor for
  other conditions is not assumed to be identical and is being recomputed per-directory, not scaled by
  this one ratio. Does not affect GPU-measured states or any conclusion that doesn't involve the CPU
  instrument.
- **Remediation decided (2026-10-06): offline recompute + harness patch, NO rerun.** `raw.jsonl`
  already stores the full per-domain trace for every window collected, so the package-only figure can
  be recovered exactly from already-collected data without repeating any measurement. Two parts:
  (1) `scripts/recompute_cpu_package_energy.py` recomputes package-only energy for all 10 affected
  directories from their existing `raw.jsonl`, writing `summary_package.csv` alongside the untouched
  originals; (2) `pilot.py` is patched going forward to log `package`/`psys` as separate columns, with
  package as primary, for all future CPU runs — not applied retroactively to existing `raw.jsonl`.
- **Part (1) executed 2026-10-06 (`8367662`).** 9/10 directories produced `summary_package.csv`
  (`pilot_proof_int8` has only 1 rep, below `summarize()`'s n≥3 floor — correctly produced no summary
  rather than a misleading one). Every directory's package+psys recompute reproduced the original
  recorded `energy_j` to <1e-4 relative error before the package-only split was trusted.
  `summary_package.csv` is now the corrected primary CPU energy figure everywhere it exists;
  `summary.csv`'s combined figure and `psys` are secondary. Part (2) (the harness patch) follows in
  its own commit.
- **Mitigation:** see the recompute script's commit and the `pilot.py` patch commit, both following
  this entry.

---

## Anomalies noted, not deviations

These are flagged discrepancies that remain unresolved but were not themselves decisions that changed
the plan — recorded for completeness per the same openness principle.

- **Stage 1 vs. Stage 3 FP32 gap:** Stage 3's FP32 measurement (0.03663 J/image) came in ~7.7% higher
  than Stage 1's exit test (0.0340 J/image), same corrected NVML backend, same measurement config
  (`results_stage3/stage3_pilot_report.md`, `stage3_deliverable.md`). Flagged at the time: Stage 1's
  exit test used seeded random weights, not a trained checkpoint, while Stage 3 used the real trained
  `resnet18_fp32.pt`. The most likely explanation offered was ordinary run-to-run GPU/thermal variance
  (weight values shouldn't affect dense-FP32 FLOP count or power draw), but this was never independently
  confirmed — the two runs are not a byte-for-byte repeat and the gap was never fully explained, only
  flagged as plausible noise.
- **Stage 4's batch=1 ratio vs. Stage 2's batch=128 speedup, pre-generalization:** before D7/D8's
  resolution, Stage 4 pre-flight's 3-point batch comparison (1/16/128) showed a non-monotonic pattern —
  batch=16 spiked above both neighbors for every compressed state tested, which was investigated at
  length (D7) and ultimately explained by the power-regime mechanism rather than remaining an open
  anomaly. Recorded here only because the *investigation path* (ruling out governor, thermal soak,
  boot-identity before landing on compute-threshold) is itself informative about how much effort an
  apparently-anomalous 3-point comparison required to resolve.
