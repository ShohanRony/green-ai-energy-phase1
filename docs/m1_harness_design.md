# M1-CPU arm harness design (draft — no code run on any Apple hardware)

**Status:** design only. Nothing in this document has been executed — this project has no access to
any Apple hardware or macOS install (consistent with `docs/m1_check_commands.md`'s own standing
disclosure). This design follows the x86-CPU block's structure (`stage5_analysis_plan.md` A7r1(e))
as closely as the M1's real constraints allow, and states plainly where it cannot.

## 1. States/conditions

Mirrors the x86-CPU block's fixed condition list (A7r1(e)) exactly, for comparability:
- **FP32-TorchScript** — block baseline.
- **FP32-eager** — collected alongside TS so D2 (runtime-confound equivalence) can be evaluated on
  this block directly, same reason the x86-CPU block carries both.
- **INT8**, via the **`qnnpack`** backend (`fbgemm` is x86-only — `stage5_analysis_plan.md` §10,
  confirmed again here). **Unverified: whether this project's existing `checkpoints/*_int8.pt` files
  (packed for `fbgemm`, per D2) load at all under `qnnpack`** — the two backends pack INT8 weights
  differently (`docs/m1_check_commands.md` §2); a fresh `qnnpack`-specific quantization pass may be
  required rather than a drop-in backend switch. Open question, §8.
- **Pruned 30/50/70**, each paired with its **`_bnrecal`** counterpart **where meaningful** — same
  "where meaningful" qualifier the x86-CPU block uses (A7r1(e)): skipped for states already at or
  near chance accuracy regardless of recalibration (carried over from the existing CPU-side
  `_bnrecal` findings, `docs/bnrecal_cpu_equivalence.md`, pending its own M1-specific re-check,
  §8).
- **No FP16 on this list** — matches the x86-CPU block's exclusion and the supervisor's own scope for
  this design, **but the underlying reason differs and is not yet known for M1.** The x86-CPU
  exclusion rests on a confirmed feasibility finding (54-65× slower, software-emulated,
  `docs/feasibility_x86_cpu_block.md`). Apple Silicon CPUs often have real ARMv8.2 FP16 hardware
  support — `docs/m1_check_commands.md` §3 explicitly warns not to assume the x86 result transfers.
  **This design excludes FP16 to match the x86-CPU block's structure and the scope given for this
  document, not because M1 FP16 has been checked and found infeasible — that check has simply never
  been run.** Flagged as an open question (§8), not a settled finding.

## 2. `powermetrics` sampling plan — unvalidated status

- **Sampler:** `powermetrics --samplers cpu_power` at minimum (per-cluster E/P active residency and
  frequency, `docs/m1_check_commands.md` §5); `thermal` sampler added for §4 below. **Exact sampler
  name spelling/availability is unverified on the actual machine/macOS version** — §1 of this
  project's own M1 check list already flags this; this design inherits that same uncertainty, not a
  new one.
- **Proposed interval: 0.4 s**, matching `pilot.py`'s own x86 default (`--interval`, default `.4`,
  confirmed from source). **Assumption, unverified for M1:** the 0.3-0.5 s NVML counter-telescoping
  floor that motivated this interval choice on x86 is a GPU/NVML-specific finding
  (`stage5_analysis_plan.md` §4's interval-sweep work) and does not mechanically transfer to
  `powermetrics`' own internal sampling behaviour — `powermetrics` may have a different practical
  floor, not yet measured. The 0.4 s figure here is a starting point for the same kind of
  interval-sweep check this project already ran for NVML, not a value carried over as validated.
- **Root requirement:** `powermetrics` needs root for every invocation. Per
  `docs/m1_check_commands.md` §6, this means a fixed-argument wrapper script plus a narrowly scoped
  `NOPASSWD` sudoers rule for that exact script — the same pattern as this project's existing x86
  `set_cpu_governor.sh` rule, not a wildcarded rule. **Not built** — this design states the pattern
  to follow; actually writing the wrapper script and sudoers rule requires access to the real
  machine (BLOCK I's own hard limits also forbid writing sudoers rules from this review, regardless).
- **Output parsing: unvalidated.** `powermetrics`' text output format and exact field names for the
  `cpu_power`/`thermal` samplers are not confirmed for this project's target macOS version
  (`docs/m1_check_commands.md` §5/§6's own "cannot verify" notes apply unchanged here). A parser
  cannot be written against a format that hasn't been captured from the real tool yet.

## 3. No core affinity on macOS, and how scheduling is recorded instead

- **x86's current approach does not transfer.** `pilot.py`'s `p_core_set()` (confirmed from source)
  pins measurement work to this machine's P-cores via Linux's `sched_setaffinity`/`taskset`-style
  mechanism. **macOS does not expose an equivalent user-level CPU-affinity API** — this is a platform
  fact, not a gap in this project's code specifically.
- **What macOS offers instead: a scheduling *request*, not a *guarantee*.** `taskpolicy -c <class>`
  (`docs/m1_check_commands.md` §4) can bias a process toward Efficiency or Performance cores via QoS
  class, but the OS scheduler retains final control — unlike Linux affinity, which is a hard pin.
  **Exact flag syntax unverified** (§4's own disclosure: "not fully confident `-c utility` is the
  exact current flag/value").
- **Recording, not controlling, is this design's approach:** every M1-CPU session logs (a) the
  `taskpolicy` QoS class requested at launch (if any was used) and (b) the actual per-cluster (E/P)
  active residency from `powermetrics`' `cpu_power` sampler output, captured concurrently
  (`docs/m1_check_commands.md` §5) — so which core type actually ran the workload is known from the
  measurement itself, independent of what was requested. **This is a real, disclosed asymmetry
  against the x86-CPU block:** x86 controls core placement directly; M1 can only request and then
  verify after the fact. Any session where the P-cluster residency is inconsistent across reps
  (analogous to x86's `power_regime` mixed/dip stability check, §7 of `stage5_analysis_plan.md`)
  would need its own stability-check gate — not yet designed, §8.

## 4. Thermal throttling (fanless)

- Target hardware class is fanless (per the proposal's own "Thermal observation" §3 clause,
  `stage5_analysis_plan.md` §10) — sustained compute load can throttle clock speed with no fan to
  delay it, unlike the x86 LOQ (actively cooled).
- **Logging, two tiers** (`docs/m1_check_commands.md` §7):
  - Coarse, no sudo: `pmset -g therm` — nominal/fair/serious/critical state, checked at minimum
    before and after each session.
  - Fine-grained, needs the same sudoers setup as §2: `powermetrics --samplers thermal`, logged
    continuously through the session alongside the `cpu_power` sampler.
- **Exclusion rule, mirroring this project's existing governor-not-performance exclusion (§8 of
  `stage5_analysis_plan.md`):** any window where the coarse thermal state is above "nominal," or
  where the fine-grained log shows a clock-frequency drop inconsistent with the rest of that
  condition's reps, is excluded and reported separately — not silently averaged in. **Exact
  frequency-drop threshold not yet defined** — needs a real thermal-throttle event captured first to
  calibrate against, same empirical-first approach this project used for the x86 power-regime
  threshold (`classify_power_regime`'s 0.9× cap boundary was set from observed data, not assumed).

## 5. Charger + wall-meter requirement

- **AC requirement, same standing rule as every other block in this project:** M1-CPU sessions
  require AC power throughout, for the same reason D26 enforces it on x86 (GPU/CPU power behaviour
  differs on battery) — **but M1 has no equivalent of D26's `check_ac_power()` yet.** `pilot.py`'s
  guard reads Linux's `/sys/class/power_supply`, which does not exist on macOS. **Open question,
  §8:** an M1 AC-detection equivalent (likely `pmset -g batt`, parsed for AC-connected status) needs
  to be designed and built before any M1-CPU session can run with the same hard-abort guarantee D26
  gives x86 — not yet done.
- **Wall meter:** if the P1 wall-meter protocol (`docs/p1_wall_meter_protocol.md`) is ever extended
  to the M1 arm, the same physical setup (meter between wall outlet and the M1's own charger) and the
  same D1-style comparison logic would apply, substituting `powermetrics`' CPU-power reading for
  `pilot.py`'s RAPL reading on the "backend" side of the comparison. **Not designed in detail here**
  — out of this document's scope; flagged so it isn't silently assumed covered by the x86-only
  protocol as written.

## 6. Session structure, matching the x86-CPU block

- **6 sessions**, same count as the x86-CPU block (A7r1(e)) and M1-CPU's own fixed count
  (A7r1(e): "M1-CPU: 6 sessions"). No conditional add/drop, same standing rule.
- **Fresh boot per session** — same discipline as every other block in this project
  (`stage4-implementation-brief.md` §5's precedent). M1 has no uptime-based fresh-boot guard
  equivalent to x86's `check_fresh_boot()` yet (reads `/proc/uptime`, Linux-only) — another open
  item, §8.
- **Randomised condition order within each session**, same principle as series R (A8/A8r1) and this
  project's general discipline against fixed-sequence confounds (`stage5_analysis_plan.md` §4's
  serial-reps limitation already flags this as a known weakness of the *original* Stage 4 matrix —
  M1-CPU's design avoids repeating it).
- **Seed convention: not yet fixed.** This project has three existing seed families — **2026**
  (all trained checkpoints, standing), **1001-1003** (P8 campaign, additive), **2000+N** (series R,
  A8/A8r1). An M1-CPU session-order randomisation seed would need its own value, distinct from all
  three to avoid provenance confusion (the same reasoning A8r1 gave for choosing 2000+N over reusing
  2026 or 1001-1003) — **no value is proposed here; left as an open decision for whoever finalises
  this design**, §8.

## 7. Which `docs/m1_check_commands.md` items must run first, and why

Not all seven checks block equally. Stated in priority order:

- **Required before any M1 session can start at all:**
  - **§1** (MPS build/availability) — establishes whether this PyTorch build can do anything on this
    machine; a hard prerequisite for FP32-eager/TS at minimum.
  - **§2** (`qnnpack` availability + the existing-checkpoint load smoke test) — determines whether
    INT8 needs fresh quantization work before this design's INT8 condition is runnable at all.
  - **§6** (sudoers + `powermetrics` wrapper setup) — without this, no energy/thermal data can be
    collected at all, regardless of which conditions are otherwise ready.
  - **§7** (thermal logging availability) — needed before any session, given the exclusion rule in
    §4 above depends on it.
- **Required before the scheduling-recording approach in §3 can be trusted:**
  - **§4** (`taskpolicy` flag syntax) and **§5** (per-cluster `powermetrics` output format) — both
    needed to actually implement §3's "record, don't control" approach; neither blocks a first
    smoke-test session, but both are needed before real session data collection begins.
- **Not required for this design's scope:**
  - **§3** (CPU FP16 timing) — FP16 is out of scope for this design (§1), so this check is not a
    blocker here, though running it anyway would resolve the open question in §1 about why FP16 is
    excluded.

## 8. Open questions, consolidated

- M1 CPU FP16 feasibility — never checked; excluded here by structural analogy to x86, not by
  evidence (§1).
- Whether existing `fbgemm`-packed INT8 checkpoints load under `qnnpack`, or need re-quantization
  (§1, `docs/m1_check_commands.md` §2).
- `powermetrics` sampler names, output format, and `-n 0`/unlimited-sample behaviour — all
  unconfirmed for the target macOS version (§2).
- Practical sampling-interval floor for `powermetrics` (the 0.4 s x86 default is a starting point,
  not a validated M1 value) (§2).
- `taskpolicy` exact flag/value syntax for the current macOS version (§3).
- Thermal-throttle exclusion threshold — needs a real throttle event captured first to calibrate
  (§4).
- M1-equivalent AC-detection guard (D26's `/sys/class/power_supply` read has no macOS analogue) —
  not yet designed or built (§5).
- M1-equivalent fresh-boot guard (`check_fresh_boot()`'s `/proc/uptime` read has no macOS analogue)
  — not yet designed or built (§6).
- Session-order randomisation seed value for the M1-CPU block — not yet chosen (§6).
- Whether the P1 wall-meter protocol extends to M1, and if so how `powermetrics`' CPU-power reading
  substitutes for RAPL on the comparison's "backend" side — not designed here (§5).
- `_bnrecal` CPU-functional-equivalence check (`docs/bnrecal_cpu_equivalence.md`) was run on x86 CPU
  only — whether the same 4-of-9 "FAIL by one image" pattern holds on M1's CPU (different
  floating-point kernel implementation entirely) is unknown and unchecked.
