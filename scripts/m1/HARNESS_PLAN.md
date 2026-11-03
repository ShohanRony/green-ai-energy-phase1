# M1-CPU harness plan — design only

**Design reviewed 2026-11-03; approved for implementation after the flush and marker tests.**

**Status: design only.** Nothing in this document has been executed, nothing has been
installed on the Mac beyond what `scripts/m1/setup_powermetrics_sudoers.sh` already
installed (verified in `~/m1_install_check.txt`), and nothing here is committed by this
document. This plan builds on `docs/m1_harness_design.md` (status: design, not executed)
and `m1_preflight/README.md` (status: executed, non-energy only) — it does not repeat
either, it extends them toward an actual energy-measurement harness.

All x86 line references below are to `pilot.py` and `scripts/run_cpu_block_session.py`
as they exist in this repo at commit `0e426d1b772b790bc2879586048f58c809715f38` (the
commit `m1_preflight/README.md` recorded as checked out on both machines).

---

## 1. x86 → M1 behaviour mapping

| Behaviour | x86 source | Stage 5 CPU-block effective value | M1 status | Notes |
|---|---|---|---|---|
| Sizes | `pilot.py:490` (`default=[32,64,128,224]`) | `[32]` — overridden by `run_cpu_block_session.py:79` (`--sizes 32`) | same | M1 session runner would pass the same `--sizes 32` override. |
| Batches | `pilot.py:491` (`default=[1,16,64,128,256]`) | `[1]` — overridden by `run_cpu_block_session.py:79` (`--batches 1`) | same | Same override on M1. |
| Windows | `pilot.py:492` (`default=[5]`) | `[5]` — not overridden by the session runner | same | M1 session runner would not need to override this either. |
| Repeats | `pilot.py:493` (`default=6`) | `6` — not overridden | same | |
| Interval | `pilot.py:493` (`default=.4`) | `0.4` — not overridden | **adapted** (floor unverified) | x86's 0.3–0.5s floor (`pilot.py:107-111,559-561`) is an NVML-telescoping finding; it does not apply to `powermetrics`. `docs/m1_harness_design.md` §2 already flags 0.4s as a starting point, not a validated M1 value. |
| Warmup | `pilot.py:511` (`default=3`), applied `pilot.py:622-626` | `3` — not overridden | same | Busy-loop warmup (`while ... work(); sync()`) ports directly; `sync()` is a no-op on CPU for both platforms (`pilot.py:600`). |
| Threads | `pilot.py:511` (`default=4`), set `pilot.py:585` (`torch.set_num_threads`) | `4` — not overridden | same | `torch.set_num_threads` exists identically on the M1 torch build (confirmed in `m1_preflight`: `torch threads: 4` was read back after setting it, same API). |
| Window timing (async poll thread) | `window()`, `pilot.py:348-373` | — | **not possible as-is (why)** | x86's `threading.Thread`-based poll-at-`interval` loop calls `sensor.read()` in-process every tick (a cheap sysfs/NVML read). M1 has no in-process sensor to read at all — `cpu_power`/`thermal` only exist via the external root `powermetrics` process. The in-process poll thread is removed entirely on M1, not adapted; see §4 for what replaces it (file-tail arrival timing) and §4's note that this is a deliberate, disclosed deviation from x86, not an equivalent mechanism. |
| Cold first repeat exclusion | `pilot.py:426-428,446` (`cold=rep==ids[0] and len(ids)>1`) | — | same | Logic is in `summarize()`, independent of sensor backend; ports unchanged. |
| Idle windows before/after | `pilot.py:622` (`phase in ['idle_before','a1','a2','idle_after']`), used `pilot.py:429,448` | — | same | Phase loop structure is sensor-independent; ports unchanged. |
| Energy integration method | `integrate()`, `pilot.py:200-222`; primary boundary selection `pilot.py:376-383` | CPU RAPL package-0, trapezoidal/delta-sum per domain | **not possible as-is** (replacement method, §2/§4) | x86 integrates a *continuously-readable* in-process counter (RAPL `energy_uj`, read every poll tick inside the same Python process). M1 has no such counter; `powermetrics` is an external root process that prints discrete power (mW) samples with its own "elapsed" time, not an integratable monotonic counter. Per the Decisions in §2, M1's "energy" is `power_mW × elapsed_ms` per sample, summed — a discrete-sample energy estimate, not RAPL-style delta integration. |
| AC guard (pre-run) | `check_ac_power()`, `pilot.py:123-134`; called `pilot.py:567-571` | — | **adapted** | No `/sys/class/power_supply` on macOS. M1 equivalent: parse `pmset -g batt` for `AC Power`/`Battery Power` and `charging`/`discharging` state (per `docs/m1_harness_design.md` §5's open question — this plan proposes the concrete parse target, not yet code). |
| AC guard (mid-window) | `sample()`'s per-tick check, `pilot.py:355-356` | — | **not possible as-is (deliberate deviation)** | x86 checks AC every poll tick via a cheap in-process sysfs read, with no cost to the measurement. M1 has no equivalent cheap in-process check, and §4 below requires **no subprocesses at all inside a measurement window** (spawning `pmset` mid-window would itself perturb CPU load during the window being measured). The M1 guard instead runs `pmset -g batt` once immediately before and once immediately after each window, not during it. This is a real, disclosed asymmetry against x86 (AC loss mid-window cannot be caught on M1 the instant it happens, only bounded to "sometime between the before- and after-checks") — to be stated plainly in the amendment that registers this design, not silently treated as equivalent. |
| AC guard (session level) | `run_cpu_block_session.py:138-140,166-168` | Pre-session + mid-session (after every condition) check | same (mechanism) | Logic ports directly once `check_ac_power()` has an M1 implementation; the session runner's call sites don't change. |
| Fresh-boot / uptime guard (per-invocation) | `check_fresh_boot()`, `pilot.py:180-194`, reads `/proc/uptime` | `max_uptime_min=180` via `run_cpu_block_session.py:80` (overriding `pilot.py`'s own default of 30 at `pilot.py:505-508`) | **adapted** | No `/proc/uptime` on macOS. M1 equivalent: `sysctl -n kern.boottime` (gives boot time, not elapsed uptime directly) combined with current time, or `sysctl -n kern.boottime` parsed and subtracted from `time.time()`. **Flag:** the two x86 uptime thresholds already differ from each other (180 min per-invocation vs. the session runner's own separate hard-coded 30 min check below) — this is an existing x86 inconsistency, not something this mapping introduces. |
| Fresh-boot / uptime guard (session level) | `run_cpu_block_session.py:59-60,117-121` (hard-coded `30 * 60`, not a flag) | 30 min, not configurable | **adapted** | Same `sysctl -n kern.boottime` approach. This 30-minute session-level check is stricter than and independent of the 180-minute per-invocation check passed to `pilot.py` — both would need M1 equivalents, and both would keep their current (different) thresholds unless a dated amendment changes them. |
| Governor check | `check_governor()`, `pilot.py:146-161`; called `pilot.py:565` and `run_cpu_block_session.py:126-129` | — | **not possible on M1 (why)** | macOS has no `scaling_governor` sysfs concept — CPU frequency scaling is entirely OS/XCPM-managed with no user-exposed governor selector. There is nothing to check or set. |
| Platform-profile check | `check_platform_profile()`, `pilot.py:136-144`; called `pilot.py:564` | — | **not possible on M1 (why)** | `/sys/firmware/acpi/platform_profile` is an ACPI/Linux concept; macOS has no equivalent control surface. **Note:** `run_cpu_block_session.py` itself never calls `check_platform_profile()` (only `check_governor()`, line 126-129) — so even on x86 the session runner does not enforce this particular pilot.py check independently; this asymmetry exists already. |
| CPU affinity | `p_core_set()` `pilot.py:13-26`; set `pilot.py:577-580`; requested via `run_cpu_block_session.py:78` (`--cpu-affinity pcores`) | pinned to P-cores | **not possible on M1 (why)** | No `sched_setaffinity`-equivalent on macOS (`docs/m1_harness_design.md` §3, confirmed from source here). No `taskpolicy` is used either — default OS scheduling only, no scheduling request of any kind at launch. Cluster residency is recorded, not requested: the `powermetrics` `cpu_power` sampler's own E-Cluster/P-Cluster active-residency fields (§5 parser spec), captured post-hoc from whichever cores the OS actually chose. |
| Thermal guard | `ThermalGuard` class `pilot.py:245-346`; `read_thermal_thresholds()` `pilot.py:225-243`, requires `config/thermal_thresholds.json` | P-core frequency vs. `T_mhz` threshold, throttle counters | **adapted** | No `thermal_throttle/core_throttle_count` sysfs, no per-core `scaling_cur_freq` in the same form. M1 equivalent: `pmset -g therm` (coarse, nominal/fair/serious/critical) plus `powermetrics`'s `thermal` sampler's "Current pressure level" line (§5) and per-cluster active frequency as a throttle proxy. **Hard gate, per `docs/stage5_consolidated_plan.md`:** "no primary CPU-block session may start until the thermal-throttle guard exists, has unit tests, and calibrated thresholds are registered in a dated amendment" — this applies to M1 exactly as it does to x86; no M1 session may start on this harness until that guard is built and calibrated, not just designed here. |
| environment.json fields | Built `pilot.py:552-556`, updated `pilot.py:602-609` | `platform, cpu, arguments, timestamp_utc, git_hash, thermal_sensor_type, cpuinfo, torch, torchvision, backend, gpu, nvidia_driver_version, gpu_power_cap_w, cpu_governors, platform_profile, uptime_s, cpu_affinity, ac_online, weights` | **adapted** (per-field) | `cpu_governors` → `None`/omitted (no concept). `platform_profile` → `None`/omitted (no concept). `gpu`/`nvidia_driver_version`/`gpu_power_cap_w` → `None` (CPU-only block on both platforms already; no change). `cpuinfo` → M1 equivalent is `sysctl -n machdep.cpu.brand_string` plus the powermetrics file header's `Machine model:`/`OS version:`/`Boot time:` lines (confirmed present in the real sample, lines 1-4). `backend` → descriptive string naming the `powermetrics` wrapper and samplers used, not RAPL/NVML. `uptime_s` → from the adapted `sysctl`-based guard above. `ac_online` → from the adapted `pmset -g batt` guard above. `cpu_affinity` → always `None` (no `taskpolicy`, no affinity request of any kind, per the CPU-affinity row above); the actual E-Cluster/P-Cluster residency is logged in per-window data, not this field. |
| Output files | `raw.jsonl` (`pilot.py:645`), `windows.csv`/`summary.csv` (`csv_write`, `pilot.py:473-477,651`), `resume_state.json` (`pilot.py:652`, read `528-536`), `skipped.jsonl` (`pilot.py:648`), `failure.txt` (`pilot.py:660`), `environment.json` (`pilot.py:556,610`) | — | same (format), **adapted** (contents per-row, matching the field adaptations above) | File names, JSON-lines/CSV format, and the overall write pattern (append raw, rewrite summary/windows every condition) port unchanged; only the per-row field set changes as above. |
| Seed / order logic | Session-order seed `run_cpu_block_session.py:23-26,47-56` (`CONDITION_SEED_BASE = 3000`, `random.Random(CONDITION_SEED_BASE + session_n)`); separate within-session grid shuffle `pilot.py:613` (`random.shuffle(grid)`, under the global `random.seed(2026)` set at `pilot.py:585`) | x86-CPU: `3000+N` | **adapted** (value only) | `docs/stage5_consolidated_plan.md`: "Seeds: x86-CPU `3000+N`, M1-CPU `4000+N`" — already registered. M1's session-order randomisation would use `CONDITION_SEED_BASE = 4000` in an M1 copy of `run_cpu_block_session.py`, distinct from 2026 (checkpoints), 1001-1003 (P8), 2000+N (series R), and 3000+N (x86-CPU) — same non-confusable-provenance reasoning as the existing families. The separate within-pilot grid shuffle (global seed 2026, `pilot.py:585,613`) is unaffected — it is not block-specific and would be identical on M1 since `pilot.py`'s own grid-shuffle code ports unchanged (sizes=[32], batches=[1], windows=[5] means this grid has exactly one entry per rep anyway, so the shuffle is a no-op for the CPU-block's own parameters — worth noting, not a meaningful source of variation here). |
| Void/rerun logic | `run_cpu_block_session.py:152-209` (`process_condition`, `reruns` list, `void_count > 2` abort, append-rerun-at-end) | AC-abort → void session; crash → void condition; thermal flag → void condition; >2 voids → void session | same (mechanism), **adapted** (thermal-flag source) | The control flow (`crashed`/`flagged` checks, `void_log`, rerun-at-end-of-order) is sensor-independent and ports unchanged. Only the thermal-flag check (`windows.csv`'s `thermal_flag` column, `run_cpu_block_session.py:178-188`) depends on the adapted `ThermalGuard` above — the column name and True/False semantics stay the same, only how it's computed changes. |
| Resume logic | `pilot.py:481-482,528-537` (`--resume`, restore args from `environment.json`, resume from `resume_state.json`'s `next_rep`); `power_state.py` whole file; checked `pilot.py:653-658` | — | same | Pure Python/filesystem logic (`.power_state/stop`, `resume_hint.txt`, JSON state files), no OS-specific dependency anywhere in `power_state.py` or the resume code path. Ports unchanged. Note `power_state.stop_requested()` itself has no M1-side *writer* yet — on x86 a separate `power_watchdog.py` process (not read in full for this plan; out of scope here) presumably writes `STOP_MARKER` on AC loss. An M1 watchdog process writing the same marker file, driven by the adapted AC guard above, would need to exist for this to actually trigger on M1 — not yet designed. |

---

## 2. Decisions already made

- Primary energy boundary: powermetrics "CPU Power"; GPU Power, ANE Power and Combined Power logged as secondary.
- INT8: ResNet-18 and EfficientNet-B0 only, existing *_int8.pt, torch.backends.quantized.engine="qnnpack" set and logged; MobileNetV3-Small INT8 structurally missing; 23 conditions per session.
- powermetrics only via: sudo -n /usr/local/sbin/gai_powermetrics.sh INTERVAL_MS COUNT TAG (samplers cpu_power,thermal; output in /var/gai-powermetrics; a run cannot be stopped early, it ends after COUNT samples).
- Energy per sample = power (integer mW) x the "elapsed" time printed for that sample, not the requested interval.
- Defaults pending validation: threads 4, interval 400 ms.

**Consistency check against §1's table:** "23 conditions per session" — the x86-CPU block's `conditions_for()` (`run_cpu_block_session.py:38-44`) yields, per architecture, 6 base conditions (`fp32_ts, fp32_eager, int8, pruned30, pruned50, pruned70`) plus up to 3 `_bnrecal` variants (minus `BNRECAL_SKIP` entries, `run_cpu_block_session.py:28-35`). For 3 architectures that's `3 × (6 + 3) − |BNRECAL_SKIP| = 27 − 3 = 24` on x86. M1's count of 23 is one less, consistent with MobileNetV3-Small INT8 being structurally missing.

**`BNRECAL_SKIP` on M1 — resolved, not unverified:** `run_cpu_block_session.py:28-35` skips exactly three `(arch, ratio)` pairs as known-collapsed: `(mobilenet_v3_small, pruned50)`, `(mobilenet_v3_small, pruned70)`, `(efficientnet_b0, pruned70)`. The M1 non-energy preflight already measured all three of these `_bnrecal` states directly (`m1_preflight/README.md`, "FP32 and pruned findings"): `mobilenet_v3_small pruned50_bnrecal` = 1000/10000 correct, `mobilenet_v3_small pruned70_bnrecal` = 1000/10000 correct, `efficientnet_b0 pruned70_bnrecal` = 1010/10000 correct — all at or within noise of chance accuracy on a 10-class task (1000/10000), i.e. collapsed on M1 exactly as on x86. The same skip set and the resulting 23-condition count both apply to M1 as given, confirmed by this project's own already-collected M1 data, not assumed by analogy.

---

## 3. Preflight checks that block a session

| Check | Blocks on | How |
|---|---|---|
| Fresh boot — session start | **The session must START within 30 minutes of boot** (mirroring `run_cpu_block_session.py:117-121`'s hard-coded, non-configurable `30 * 60` check) | `sysctl -n kern.boottime`, parsed, compared to `time.time()`, checked once before the session's first condition |
| Fresh boot — per-condition | **Each condition must start within 180 minutes of boot** (mirroring `pilot.py`'s `--max-uptime-min 180` passed by `run_cpu_block_session.py:80`) | Same `sysctl`-based check, re-evaluated before every condition within the session |
| Fresh boot — FileVault note | With FileVault enabled, the boot-to-login sequence requires the brother to physically restart and log in (no unattended reboot-and-resume) before `kern.boottime` is current and a session can start | Operationally: brother restarts and logs in, then the 30-minute session-start clock (above) begins running from that boot, not from when the harness happens to be launched |
| AC power | Not on AC, or state ambiguous | `pmset -g batt`, parsed for `AC Power`/charging state; checked once at session start, then immediately before and immediately after every window (§4) — never during one |
| Thermal nominal | `pmset -g therm` reports above "Nominal", or the `powermetrics` thermal sampler's "Current pressure level" is not "Nominal" | Both checked; either one failing blocks; the coarse `pmset -g therm` check runs immediately before/after every window (§4) alongside the AC check, the fine-grained `powermetrics` thermal sampler value comes from the sampler data itself, not a separate subprocess call during the window |
| No competing remote-access/screen-sharing processes | Any of: AnyDesk, OneDrive (actively syncing — not just the symlink/folder existing), Chrome Remote Desktop, macOS Screen Sharing (`com.apple.screensharing`) running | `ps aux` / `pgrep` for known process names; this check exists directly because of the `.anydesk` directory flagged during the `m1_install_check.txt` review — a stray remote-access tool actively running during a measurement session would contaminate CPU load exactly like concurrent GPU work does for `check_no_concurrent_gpu()` on x86 (`pilot.py:113-121`) |
| Memory pressure | Not blocking by itself, but recorded | `memory_pressure` output and swap-used (`sysctl vm.swapusage` or equivalent) logged into `environment.json`; given the 8GB RAM already observed to run low on free pages during prior checks in this session, this is recorded for every session even though no fixed threshold is proposed yet (open question, §6) |
| Clean git tree at expected commit | `git status --porcelain` non-empty, or `git rev-parse HEAD` != the commit pinned for the session | Mirrors `pilot.py:544-546`'s `git_hash` capture, but as a hard precondition rather than just a logged fact — the x86 code only *records* the hash, it does not *block* on a dirty tree; this plan proposes blocking for M1 sessions specifically, a stricter rule than x86's own, stated as a deliberate addition, not something copied from x86 |
| Checkpoint integrity | Any checkpoint's sha256 doesn't match `m1_preflight/results/m1_checkpoint_hashes.txt` | Re-run the same `shasum -a 256` check already done once in `m1_preflight`, before every session, not just once at setup |
| caffeinate ownership | — | The harness itself must start its own `caffeinate` process (not rely on one already running, as was observed and flagged during an earlier check in this session) and must stop it when the session ends or aborts — so a session's sleep-prevention is accounted for and torn down deterministically, not left running indefinitely by an unrelated prior invocation |

---

## 4. Timing alignment between inference windows and powermetrics samples

**The problem:** `powermetrics` is a separate root process (launched via the sudoers-wrapped
`gai_powermetrics.sh`) with a fixed `COUNT`; it cannot be stopped early, and it does not
know anything about the harness's own `window()` phase boundaries (`idle_before`/`a1`/`a2`/
`idle_after`). The harness process and the `powermetrics` process are two independent
clocks that must be reconciled after the fact, not synchronized live.

**Why wall-clock bucketing by the printed timestamp alone is NOT acceptable:** the sample
header timestamp has only **1-second resolution** (`Tue Nov  3 21:38:52 2026 +0900` —
whole seconds only, no sub-second field, confirmed from the real captured sample). It is
also **unknown whether this timestamp marks the start or the end of the sample's coverage
window** — `powermetrics`' own documentation does not say, and no experiment distinguishing
the two has been run. For this project's 5-second windows, a timestamp with only 1-second
resolution and an unknown start/end convention is not precise enough to assign samples to
windows correctly near a boundary — the error could be comparable to the window length
itself in the worst case. Bucketing by timestamp alone, as originally proposed, is rejected.

**Revised method:**

**(a) Primary — file-tail arrival timing.** The harness tails the `powermetrics` output
file at ~20ms polling resolution for the entire duration of a run, and records its own
`perf_counter()` time at the moment each new `*** Sampled system activity` header line
first appears in the file. Call this arrival time `arrival_k` for sample `k`. Sample `k`'s
own printed "elapsed" field (`elapsed_k`, e.g. `1005.14` ms) gives its covered duration, so
the sample is attributed to the wall-clock interval `[arrival_k - elapsed_k, arrival_k]` —
anchored to when the harness *observed* the sample land, not to the sample's own
low-resolution, ambiguous internal timestamp. **This is valid only if `powermetrics`
flushes its output to disk per-sample (not buffered and written in a batch at process
exit or only periodically)** — this must be tested first (see "flush test" below) before
any real measurement relies on it.

The file-tail polling runs in the harness process during measurement windows as well, so
its CPU cost is included in measured energy. It is present equally in every condition and
in the idle windows (comparable to the x86 poll thread), but it is a disclosed overhead.
The flush test will run at both 20 ms and 50 ms poll intervals; the coarser one is used if
it still resolves every sample.

**(b) Fallback — idle-to-load step marker.** If the flush test fails (output is buffered,
so arrival timing cannot distinguish samples), fall back to an explicit marker: at the
start of each `powermetrics` run, the workload performs a deliberate idle period followed
by a deliberate load period (see "marker test" below), producing a sharp, recognizable step
in the `CPU Power` series. The harness's own known wall-clock time for when it started the
load step is aligned to wherever that step actually appears in the parsed `CPU Power`
sequence, giving one single-point time correction for that run (coarser than per-sample
arrival timing, but not dependent on flush behaviour).

**Energy attribution across windows:** once each sample is placed on the wall-clock axis
(by (a) or (b)), a sample that overlaps more than one window boundary is **split by time
overlap, not assigned whole** — e.g. if sample `k` covers `[t0, t1]` and a window boundary
falls at `t_b` inside that range, the portion `[t0, t_b]` contributes
`power_k × (t_b - t0)` to the window ending at `t_b`, and `[t_b, t1]` contributes
`power_k × (t1 - t_b)` to the next window. This replaces the earlier "assign sample whole
to the overlapping window" approach.

**One `powermetrics` run per condition, not per session.** `COUNT` is sized from the
expected duration of one condition, not an entire session: 6 repeats × 4 phases
(`idle_before`, `a1`, `a2`, `idle_after`) × (3s warm-up/settle + 5s window) = `6 × 4 × 8s =
192s` of measured phases, plus model-load time, plus a margin (exact margin not fixed
here — open question, §6). The harness waits for the previous `powermetrics` process to
fully exit before starting the next condition's `powermetrics` invocation (sequential, not
overlapping). Any surplus tail samples beyond what a condition actually needed (because
the real run finished faster than the `COUNT`-derived estimate) are idle and are recorded
as such, not discarded — consistent with this project's "no silently dropped data" rule.

**Rejected alternative: one long-lived `powermetrics` run per session.** Originally
proposed, now rejected. Reason: a single session-spanning `COUNT` would need to be
estimated far in advance of knowing the session's actual total duration (warm-up timing
variance, voided/rerun conditions each adding unpredictable extra time), making the
margin much harder to size safely than a per-condition estimate; a per-condition run also
bounds the blast radius of a mis-timed or corrupted `powermetrics` invocation to one
condition rather than an entire session.

**Deliberate deviation from x86, to state in the amendment:** x86 checks AC power every
poll tick via a cheap in-process sysfs read with no cost to the measurement
(`pilot.py:355-356`). M1 has no equivalent cheap in-process check and, per this section,
**no subprocess of any kind runs inside a measurement window** — `pmset -g batt` (AC) and
`pmset -g therm` (coarse thermal) are both run once immediately before and once
immediately after every window, never during. Fine-grained thermal pressure for the window
itself comes from the `powermetrics` sampler data already being collected, not a separate
call. This means an AC loss or thermal excursion occurring strictly inside a window (and
resolving before the after-check) would not be caught — a real, disclosed gap against x86's
tighter guarantee, not an equivalent mechanism under a different name.

**How this will be tested, in order:**
1. **Flush test (first, before anything else in this section is relied on):** run
   `gai_powermetrics.sh 400 20 flushtest` while tailing the output file every ~20ms from a
   separate process, recording the arrival time of each of the 20 expected sample headers.
   Pass criterion: the 20 arrival times are spread over approximately 8 seconds (20 samples
   × 400ms), not clustered at the very end of the run — a cluster-at-the-end result means
   output is buffered and method (a) is invalid, forcing fallback to method (b).
2. **Marker test:** a workload that idles for 3s, runs full CPU load for 3s, then idles for
   3s again, while `powermetrics` samples throughout. Check that the parsed `CPU Power`
   series shows a step increase at approximately the expected time (load start) and a step
   decrease at approximately the expected time (load end) — validates that the harness's
   own wall-clock marks for "load started"/"load ended" actually correspond to a visible
   change in the sampled data, which both method (a) and method (b) depend on.
3. **Synthetic bucketing test (after 1 and 2 pass):** using the one real captured
   `powermetrics` sample file (`smoketest_20261103T123851Z_70699.txt`, already committed
   research artifact from `m1_preflight`), replay it against a fabricated set of window
   boundaries with known wall-clock start/end times chosen to land (a) fully inside one
   sample, (b) exactly on a sample boundary, and (c) spanning two samples — asserting the
   overlap-splitting energy attribution produces the expected per-window energy in each
   case. This is a parser/alignment unit test against already-captured real data, not a
   new capture — it requires no further access to the Mac beyond what's already been
   saved.

Tests 1 and 2 require running the already-installed wrapper on the Mac (no new install,
no sudo rule change — `gai_powermetrics.sh` is already in place per `~/m1_install_check.txt`)
and are not performed by this document; they are specified here as the first things to
actually run before any real session, not run now.

---

## 5. Parser spec for the real powermetrics text format

Based on `/var/gai-powermetrics/smoketest_20261103T123851Z_70699.txt` (3 samples, `COUNT=3`,
`INTERVAL_MS=1000`, samplers `cpu_power,thermal` — the exact invocation used in the Step 5
smoke test earlier this session).

**File-level header (once, lines 1-4 of the real sample):**
- `Machine model: <string>` (e.g. `MacBookAir10,1`)
- `OS version: <build>` (e.g. `25G83`)
- `Boot arguments: <string, possibly empty>`
- `Boot time: <ctime-style string>` (e.g. `Sat Aug 29 02:26:39 2026`)

**Per-sample block, repeated `COUNT` times, each starting with:**
- `*** Sampled system activity (<ctime-style string with UTC-offset suffix>) (<float>ms elapsed) ***`
  — confirmed exact format: `Tue Nov  3 21:38:52 2026 +0900`, and elapsed `1005.14ms`
  (not equal to the requested 1000ms interval — confirms §4's point that the nominal
  interval must not be assumed equal to actual elapsed time).

**Within each sample block (`**** Processor usage ****` section):**
- `E-Cluster HW active frequency: <int> MHz`
- `E-Cluster HW active residency: <float>% (<breakdown by frequency bucket>)`
- `E-Cluster idle residency: <float>%`
- Per-core, repeated for each E-core: `CPU <N> frequency: <int> MHz`, `CPU <N> active
  residency: <float>% (...)`, `CPU <N> idle residency: <float>%`
- `P-Cluster HW active frequency: <int> MHz`
- `P-Cluster HW active residency: <float>% (<breakdown>)`
- `P-Cluster idle residency: <float>%`
- Per-core, repeated for each P-core: same three lines as E-core.
- `CPU Power: <int> mW`
- `GPU Power: <int> mW`
- `ANE Power: <int> mW`
- `Combined Power (CPU + GPU + ANE): <int> mW`

**Within each sample block (`**** Thermal pressure ****` section):**
- `Current pressure level: <Nominal|...>` — only `Nominal` observed in the captured sample;
  other values (`Fair`/`Serious`/`Critical`, per `pmset -g therm`'s own vocabulary) are
  assumed by name analogy, not confirmed from a real throttled sample (none captured yet —
  same gap `docs/m1_harness_design.md` §4 already flags for the throttle-threshold
  calibration).

**Proposed parser function:** `parse_powermetrics_file(path) -> list[dict]`, one dict per
sample block, with keys: `timestamp` (parsed datetime, keeping the printed UTC offset),
`elapsed_ms` (float), `cpu_power_mw` (int), `gpu_power_mw` (int), `ane_power_mw` (int),
`combined_power_mw` (int), `thermal_pressure` (str), `e_cluster_active_freq_mhz` (int),
`e_cluster_active_residency_pct` (float), `e_cluster_idle_residency_pct` (float), and the
P-Cluster equivalents.

**Unit tests to write against the saved sample file** (none written yet — this is a spec):
1. `test_sample_count_matches_count_arg` — asserts exactly 3 sample blocks are parsed from
   the file (matching the `COUNT=3` used when it was captured).
2. `test_cpu_power_values_match_known_sample` — asserts the three `cpu_power_mw` values
   parsed are exactly `67, 84, 57` (read directly from the file at lines 45, 93, 141).
3. `test_elapsed_ms_not_equal_to_requested_interval` — asserts none of the three
   `elapsed_ms` values equals `1000.0` exactly, and all are within some small tolerance of
   it (guards against a future regression where the parser is fed a synthetic/rounded value
   instead of the real printed one).
4. `test_thermal_pressure_nominal_on_known_sample` — asserts all three `thermal_pressure`
   values parse as `"Nominal"`.
5. `test_timestamp_parses_with_utc_offset` — asserts the parsed timestamp for sample 1
   correctly carries the `+0900` offset (not silently dropped or assumed to be UTC — this
   matters directly for §4's wall-clock bucketing).
6. `test_file_header_fields_present` — asserts `Machine model`, `OS version`, `Boot time`
   are all parsed from the file-level header, for inclusion in `environment.json` (§1 table,
   `cpuinfo` row).
7. `test_malformed_block_raises_not_silently_skips` — a deliberately truncated/corrupted
   copy of the sample file (e.g. missing the `CPU Power:` line from one block) should raise,
   not silently produce a dict with a missing/null field — matching this project's standing
   "no generated measurements or silent sensor fallback" rule (`pilot.py:1`'s own module
   docstring).

---

## 6. Open questions (separate from the decisions in §2)

- Exact `powermetrics` sampling-interval practical floor on this machine — 400ms is a
  starting point carried over from x86's NVML-specific finding, not independently measured
  for `powermetrics` (`docs/m1_harness_design.md` §2; §1 table, Interval row).
- `pmset -g batt` exact parse target (field names/format) for the AC guard — not yet
  captured from a battery-powered run on this Mac (only AC-powered output seen so far, in
  the Stage 5 pmset checks earlier this session).
- `sysctl -n kern.boottime` exact output format and parsing approach for the uptime guard —
  not yet captured/tested on this Mac.
- Thermal-throttle exclusion threshold — no throttled sample has been captured; the "above
  Nominal" rule has no frequency-drop calibration yet, and `docs/stage5_consolidated_plan.md`
  states this is a **hard gate**: no primary CPU-block session (x86 or M1) may start until
  this guard exists, has unit tests, and calibrated thresholds are registered in a dated
  amendment.
- No M1-side process has been designed to *write* the `power_state.py` `STOP_MARKER` file
  on AC loss (x86's watchdog process, `power_watchdog.py`, was not read for this plan and
  is out of scope here) — resume logic (§1 table) ports, but nothing currently triggers it
  on M1.
- Memory-pressure threshold for blocking a session (§3) — recorded but not yet assigned a
  concrete pass/fail value; this Mac's 8GB RAM was observed with very low free pages during
  an earlier unrelated check this session, so a real threshold may need to be stricter than
  "just record it," but that threshold is not chosen here.
- Which remote-access/screen-sharing process names (§3) are exhaustive — `.anydesk` was the
  one actually observed; whether Chrome Remote Desktop, macOS Screen Sharing, or others are
  installed/could start on this Mac has not been independently checked.
- Exact margin to add on top of the 192s-per-condition estimate (§4) for model-load time and
  safety buffer when sizing each condition's `powermetrics` `COUNT` — not fixed here; too
  small risks the process exiting before the condition finishes (losing tail coverage), too
  large just means more idle tail samples (harmless but wasteful of session wall-clock time
  against the 180-minute per-condition uptime limit; the 30-minute limit applies only to the
  session start. With 23 conditions of roughly 4 minutes each, a session already takes 90+
  minutes, so the margin must be small enough that the last condition still starts well
  inside 180 minutes).
- Whether 20ms is actually a fine enough file-tail poll resolution for the flush test (§4)
  to reliably distinguish "flushed per-sample" from "buffered" — not validated; the flush
  test itself is designed to answer this empirically, but the poll interval chosen to run
  that test is itself a guess pending the test's own result.
