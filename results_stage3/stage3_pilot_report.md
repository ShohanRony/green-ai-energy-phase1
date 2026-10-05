# Stage 3 Pilot Report — RQ1 Full-State Harness Validation

Energy-Aware Efficiency of Lightweight Vision Models Under Post-Training Compression.
Run 2026-10-05, x86 only (Lenovo LOQ 15IRX9, RTX 3050 6GB). Task 0 through Task 4, all complete.
`stage3_deliverable.md` in this directory holds the tables without the task narrative.

**GO.** See Task 4 below for the full reasoning; headline: all 6 states produced clean, physically
plausible numbers with zero plausibility flags across 60 reps, and the one real finding this pilot
surfaced (batch-size-dependent speedup ratio) is a Stage 4 design input, not a harness defect.

---

## Task 0 — Orient, confirm preconditions

Per the brief's §3, re-confirmed against the current working tree rather than assumed:

- **Corrected energy backend:** `pilot.py` line 18-29 — default branch is `nvmlDeviceGetPowerUsage`
  ("NVML sampled power"); `--legacy-cumulative-counter` is `action='store_true'`, opt-in only. Confirmed
  live in every `environment.json` this session (`backend: "NVML sampled power"` for all 5 GPU states).
- **NVML sampling floor:** `check_interval_floor` enforced in code (line 193-195), not just documented.
  Confirmed from actual trace timestamps: observed inter-sample gap 0.400-0.401s at `--interval 0.4`,
  inside the 0.3-0.5s floor.
- **RAPL access:** confirmed working — INT8's run used `backend: "RAPL package energy counters"`
  successfully, no fallback or error.
- **`--arch` flag:** confirmed present (`choices=['resnet18','mobilenet_v3_small','efficientnet_b0']`,
  default `resnet18`), and this session used it (implicitly, via default) across all 6 states without
  incident — every checkpoint format loaded (plain state_dict for FP32, TorchScript trace for FP16 and
  the 3 pruned states, TorchScript scripted/quantized for INT8).
- **All 18 Stage 2 checkpoints exist as files:** confirmed (`ls checkpoints/*.pt` → 18). Not re-proof-tested
  individually this session since `pilot.py`'s loading path wasn't touched — per the brief's own
  instruction ("re-verify any artifact whose loading path changes"), no change means no re-verification
  needed. The 6 checkpoints actually used this session (all ResNet-18) loaded and ran correctly, which is
  itself evidence the loading path still works.
- **FP16 input-dtype and INT8 StopIteration bugs:** both fixed in Stage 2 Task D, unchanged in the
  current tree (confirmed by `git log -p -- pilot.py` showing no edits since commit `888c585`). Both
  states ran cleanly this session with no dtype errors, consistent with the fixes still holding.
- **ARM/M1 confirmed out of scope:** not touched, not referenced in any command this session.

**Verdict:** all preconditions hold, no blockers, no time spent re-deriving any of them.

---

## Task 1 — Run ResNet-18 across all 6 states

10 reps/state (9 after cold-start discard, checklist item 3), x86 only, same invocation style as
`stage1_exit_test.md` for direct comparability: `--sizes 32 --batches 16 --windows 5 --interval 0.4
--warmup 3`, pointed at each state's actual Stage 2 checkpoint (not random weights — see Task 2 for the
one place this matters).

**Pre-run setup required manual intervention:** CPU governor was `powersave`, not the `performance`
required by checklist item 6. No passwordless sudo available in this environment; user ran
`echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` directly. Confirmed
`performance` across all cores and `platform_profile: performance` (item 14) before any measurement run
started, and re-confirmed in every run's `environment.json` through to the last (INT8) run — held for
the whole session.

Six runs, one per state:

| State | Device | Checkpoint | Wall-clock |
|---|---|---|---|
| FP32 | cuda | `resnet18_fp32.pt` (plain state_dict) | ~4 min |
| FP16 | cuda | `resnet18_fp16.pt` (TorchScript trace) | ~4 min |
| Pruned 30% | cuda | `resnet18_pruned30.pt` (TorchScript trace) | ~4 min |
| Pruned 50% | cuda | `resnet18_pruned50.pt` (TorchScript trace) | ~4 min |
| Pruned 70% | cuda | `resnet18_pruned70.pt` (TorchScript trace) | ~4 min |
| INT8 | **cpu** | `resnet18_int8.pt` (TorchScript scripted/quantized) | ~8 min (CPU-bound) |

**Discard/re-run log (checklist requirement — log what's discarded, don't silently exclude):** nothing
was discarded. All 6 states reached `pairs=9` cleanly from the first attempt; 0 of 240 active-phase
windows (6 states × 2 windows/rep × 10 reps, pre-cold-discard) tripped `flag_implausible_power`; 0 CUDA
OOM events (`skipped.jsonl` absent for every state — the harness only creates it on an actual OOM).

**Verdict:** pass, all 6 states, no discards needed.

---

## Task 2 — Per-state plausibility pass

Full table in `stage3_deliverable.md` §1. Headline numbers (gross J/image, mean of 9 paired reps):

| State | J/image |
|---|---|
| FP32 | 0.03663 |
| FP16 | 0.01221 |
| Pruned 30% | 0.01791 |
| Pruned 50% | 0.00659 |
| Pruned 70% | 0.00540 |
| INT8 | 0.18395 (CPU/RAPL — different power domain, not comparable to the GPU numbers) |

**Sanity-checked against the three criteria the brief named:**

1. **Stage 1's exit-test FP32 numbers:** 0.03663 J/image here vs. 0.0340 there, same corrected backend,
   same measurement config. ~7.7% higher — close, not identical. **Flagged rather than glossed over:**
   Stage 1's exit test used seeded random weights; this run used the real trained `resnet18_fp32.pt`
   checkpoint. Weight values shouldn't change FLOP count for dense FP32 compute, so the gap is more
   likely ordinary run-to-run GPU/thermal variance than a real weight-dependent effect, but the two runs
   are not a byte-for-byte repeat and that's stated plainly rather than assumed.
2. **The RTX 3050's 60W cap:** every GPU state's active power sits at 59.9-59.95W — pinned at the cap,
   never exceeding it, matching the established Stage 1 finding. 0 plausibility-flagged windows across
   all 5 GPU states.
3. **Expected ordering:** FP16 and all three pruned states show lower energy/image than FP32, with no
   exceptions. Pruned states decrease monotonically with sparsity (30%→50%→70%: 0.01791→0.00659→0.00540),
   tracking Stage 2's realized-MACs-reduction ordering (51.6%→74.8%→91.0%). INT8 correctly has no GPU
   number to compare against — confirmed structurally absent, not a zero or placeholder (Task 3, item 13).

**One pattern flagged, not explained away** (detailed in `stage3_deliverable.md` §3): every compressed
state's real energy improvement at this session's batch=16 is larger than Stage 2's batch=128 latency
speedup alone would predict. Consistent direction across all 4 compressed states, not scattered noise —
pointing at batch size as a real factor in the compression-vs-efficiency relationship, not a fixed
multiplier Stage 4 can assume carries over from Stage 2's single-batch-size check.

**Verdict:** all 6 states plausible and correctly labeled. One finding flagged for Stage 4 design, not
a harness problem.

---

## Task 3 — Checklist compliance report

Full 14-item table with actual observed values (not blanket "done") in `stage3_deliverable.md` §4, same
format as `stage1_exit_test.md`. All 14/14 pass. Item 13 (new for Stage 3) specifically confirmed: INT8's
`environment.json` shows `gpu: null` and `backend: "RAPL package energy counters"` — the GPU sensor is
never touched during the INT8 run, not populated with a meaningless or zeroed value.

**Verdict:** done, 14/14.

---

## Task 4 — Go/no-go recommendation

**GO.** Checked against every no-go trigger named in the brief's §7:

- *A state's energy numbers look physically implausible and the cause isn't understood yet* — did not
  occur. All 6 states' numbers fit the expected pattern (pinned power cap on GPU, correct ordering,
  correct RAPL-only path for INT8).
- *The harness silently falls back or masks a failure for any state* — did not occur. INT8's
  GPU-absent result is explicit (`gpu: null`), not a disguised zero. No dtype errors, no silent FP32
  fallback for FP16 or INT8 (the Stage 2 Task D fixes held).
- *Checkpoint loading or measurement differs in any unexplained way from Stage 2's proof-of-load* — did
  not occur. All 6 formats loaded via the same `torch.jit.load`-then-`build_model`-fallback path Task D
  built, with no errors.

Checked against the **go** criteria: all 6 states plausible and labeled (✅), no checklist item failed
silently (✅, 14/14 with real values), INT8's empty GPU column confirmed intentional (✅), variance not
wildly inconsistent with prior expectations (✅ — `idle_j_sd`/`idle_j_mean` ratio sits at 3.5-4.9% for
the 5 GPU states and ~10.5% for the one CPU/RAPL state; the CPU state's higher relative idle noise is
worth watching into Stage 4, but still single-digit-to-low-double-digit percent, not "wildly
inconsistent").

**Scoped, not open-ended:** the one real finding from this pilot (§3 of the deliverable — batch size
changes the realized energy/speedup ratio substantially, sometimes by 1.4-2x beyond what Stage 2's
single-batch-size check would predict) is a **quick design decision for Stage 4**, not a fix: pick one
batch size as Stage 4's headline condition (or add batch size as an explicit swept factor) before
starting the full matrix, rather than silently assuming Stage 2's batch-128 numbers transfer to whatever
batch size Stage 4 ends up using. This does not require its own mini-stage or further investigation — it
requires one explicit decision, logged, before Stage 4 starts.

**The harness is ready for Stage 4's 1,080-run matrix as-is.**
