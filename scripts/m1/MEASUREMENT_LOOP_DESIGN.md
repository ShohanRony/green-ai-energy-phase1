# M1 per-condition measurement loop — design only, no code

**Status: design only.** Nothing in this document has been executed, nothing has been
installed, and this is not an implementation of the loop — it maps the x86 loop
(`pilot.py`) onto M1's constraints, building on `scripts/m1/HARNESS_PLAN.md` (the
general M1 harness plan), `scripts/m1/guards.py`, `scripts/m1/m1_pipeline.py`, and
`scripts/m1/m1_session_runner.py` (already committed, `e49cf5e`). Per your explicit
instruction: do not implement this loop yet.

All x86 line references are to `pilot.py` at commit `0e426d1b772b790bc2879586048f58c809715f38`.

---

## 1. One condition, start to finish

### 1.1 Preflight, before anything else for this condition

- Checkpoint sha256 matches `m1_preflight/results/m1_checkpoint_hashes.txt` (re-checked
  per condition, not just once at session setup — a checkpoint file could change between
  conditions in a long session).
- `git status --porcelain` is empty and `git rev-parse HEAD` equals the commit pinned for
  this session. Mirrors `pilot.py:544-546`'s `git_hash` capture, but as a hard precondition
  here (x86 only records the hash, it doesn't block on it — a deliberate M1-specific
  addition, stated in `HARNESS_PLAN.md` §3).
- No blocked background app is running — checked by exact process name via `ps`/`pgrep`,
  **blocking if the process is running at all** (no "actively syncing" or similar
  activity-level nuance; presence alone blocks): `AnyDesk`, `OneDrive`, `remoting_host`
  (Chrome Remote Desktop's host process), `screensharingd` (macOS's built-in Screen
  Sharing/VNC service). Per `HARNESS_PLAN.md` §3, which exists directly because of the
  `.anydesk` directory found running during this session's own earlier inventory.
- `caffeinate` is started by the harness itself for this condition (not relying on one
  already running) — torn down at the end of the condition regardless of success/failure.
- AC guard (`guards.check_ac_power_m1`) and thermal guard (`guards.check_thermal_nominal_m1`)
  — both fail-closed (A1). Uptime guard: condition must start within 180 minutes of boot
  (`guards.check_uptime_guard`, `max_condition_uptime_s=10800.0`).

### 1.2 Model load and warm-up (mirrors `pilot.py:583-599`, before any sensor/powermetrics activity)

- Set `torch.set_num_threads(4)` (`pilot.py:585`, `a.threads` default 4 — unchanged on M1,
  confirmed identical API in `m1_preflight`).
- Load the checkpoint: `torch.jit.load` first, `build_model` + `load_state_dict` fallback
  on `RuntimeError` (`pilot.py:589-594`) — unchanged, same two-path logic.
- For INT8 conditions (ResNet-18, EfficientNet-B0 only — MobileNetV3-Small INT8 is
  structurally missing under qnnpack, `m1_preflight/README.md`): explicitly set
  `torch.backends.quantized.engine = "qnnpack"` **before** loading the INT8 checkpoint, and
  log the engine value into this condition's `environment.json` (mirrors the already-run
  preflight check in `m1_preflight/results/mbv3_int8_diag.txt`, where the engine value was
  printed and verified for every INT8 attempt).
- CIFAR-10 load: `torchvision.datasets.CIFAR10(..., download=False, ...)` (`pilot.py:598`)
  — unchanged; data already present and sha256-verified in `m1_preflight`.
- No separate "warm-up" matmul phase here at the model level — x86's own warm-up happens
  per-phase, per-repeat (§1.4 below), not once at condition start. (This differs from
  `edge_test.py`'s standalone calibration design, which does one batch of untimed
  warm-up matmuls before its own synthetic powermetrics run — that is a calibration tool,
  not this loop.)

### 1.3 Start `powermetrics`, wait for first header

- Launch `sudo -n /usr/local/sbin/gai_powermetrics.sh INTERVAL_MS COUNT TAG` via
  `subprocess.Popen` (per-condition, not per-session or per-window — `HARNESS_PLAN.md` §4's
  "one run per condition" decision, confirmed in that document).
  **`COUNT = ceil(expected_duration_s / 0.405) + 5`**, where `expected_duration_s` is this
  condition's expected total duration (6 repeats × 4 phases × (3s settle + 5s window) = 192s,
  plus model-load time already elapsed before `powermetrics` starts). The divisor 0.405
  (not the nominal requested 0.400s interval) is deliberately close to, but slightly below,
  the measured mean actual sample interval of **410.36 ms** (calibration run, Test 2
  duty-cycle, `calibration/m1_thermal_20261104/`: requested interval 400ms, actual mean
  410.36ms across 6782 samples) — dividing by a value below the true actual interval
  over-provisions `COUNT` rather than under-provisions it, and the fixed `+5` adds a small
  further buffer. This closes the open margin question from `HARNESS_PLAN.md` §6, replacing
  "exact margin still open" there.
  **The harness must wait for the previous condition's `powermetrics` process to fully exit
  before starting the next condition's `powermetrics` invocation** — sequential, never
  overlapping (already stated as the per-condition design's intent elsewhere in this
  document; restated here explicitly next to the `COUNT` formula since both govern the same
  process lifecycle).
- Locate the output file under `/var/gai-powermetrics/{TAG}_*.txt` created after this
  process's own start time (same glob-and-mtime-check pattern already used in
  `scripts/m1/flush_test.py` and `scripts/m1/edge_test.py`).
- Start the P2 poller (§1.5) immediately, and **wait for its first-header event** before
  proceeding to phase 1 — this establishes the `arrival_perf ↔ elapsed_ms` anchor the
  overlap-splitter (`m1_pipeline.process_condition_energy`, already committed) needs for
  every window boundary in this condition.

### 1.4 Phase loop: 4 phases × 6 repeats, 3s settle + 5s window

Mirrors `pilot.py:612-651`, `window()` (`pilot.py:348-413`), and `summarize()`
(`pilot.py:415-471`) structurally:

- Outer loop: `repeat in range(6)` (`pilot.py:493` default; unchanged on M1).
- Inner loop, per repeat, phases in order: `idle_before`, `a1`, `a2`, `idle_after`
  (`pilot.py:622`, unchanged order and naming).
- For `idle_before`/`idle_after`: 3s settle (`time.sleep(a.warmup)`, `pilot.py:626` —
  "same declared settling delay, no thermal equilibrium claim", same comment applies
  unchanged on M1), then a 5s **idle window** (no model call).
- For `a1`/`a2`: a 3s busy-loop warm-up (`pilot.py:624-625`, `while ... work(); sync()`;
  `sync()` is a no-op on CPU for both platforms, `pilot.py:600`), then a 5s **active
  window** running `model(x)` in a tight loop, counting completed batches
  (`pilot.py:138-141`'s pattern, reused as-is — `count` incremented per completed
  model forward call (`model(x)`) inside the window, not per matmul operation inside
  the model).
- Window boundaries (start/end `perf_counter()` marks) are recorded for every phase, same
  shape as x86's `window()` return dict minus the sensor-specific fields (no `trace`, no
  RAPL `ranges`) — energy for each window is **not** measured synchronously here; it is
  attributed after the fact by `m1_pipeline.process_condition_energy`'s overlap-splitter
  against the powermetrics sample stream collected independently in §1.3/§1.5.
- Cold first repeat (`repeat == 0`) is still measured and still appears in `windows.csv`,
  but excluded from `summary.csv` averages — identical to x86's `cold` flag
  (`pilot.py:426-428,446`), already implemented unchanged in `m1_pipeline.summarize_condition`.
- Batch counting: `imgs = batches * batch` (`pilot.py:182`'s equivalent, already in
  `m1_pipeline.summarize_condition`) — `batch` is fixed at 1 for this block
  (`run_cpu_block_session.py:79`'s `--batches 1` override, same on M1).

### 1.5 P2 poller lifecycle

- **Runs as a separate OS process, not a thread in the inference process.** The edge test
  (`edge_test.py`) ran P2 as a `threading.Thread` inside the same process as the
  `torch.matmul` load; that was acceptable for a calibration tool measuring timing
  structure, but inside the real measurement loop a poller thread would contend for the
  GIL with the model's forward-call loop (§1.4), perturbing the very inference timing being
  measured. The real loop instead spawns P2 as its own child process (e.g. a small
  standalone script, started via `subprocess.Popen`, communicating sample arrivals back to
  the main process through a file or pipe) so inference and polling never compete for the
  same interpreter lock.
- Started once per condition, immediately after the `powermetrics` process is confirmed
  running and its output file is located (§1.3) — not per-window, not per-phase.
- Implementation per `HARNESS_PLAN.md` §4 method (a), as flush-tested in
  `~/m1_flush_test.txt` and validated again (P20 vs P2 arrival comparison) in
  `~/m1_edge_test.txt`: `os.stat` every ~2ms, reads only bytes after the last-seen offset,
  carries over any unmatched trailing text across polls (header can straddle a read
  boundary).
- Runs continuously through **all 24 windows** of this condition (4 phases × 6 repeats),
  not restarted per window — a single long-lived poller process for the condition's full
  duration, feeding one shared arrival record keyed by sample number.
- Stopped only after the `powermetrics` subprocess itself has exited and a final full-file
  read confirms every sample the file actually contains is captured — same pattern as
  `edge_test.py`'s worker trial, adapted for a separate process rather than a thread.
- **No offset correction applied** (delta = 0, per the two-poller edge test's superseding
  finding over the marker-test memo's withdrawn -15ms recommendation) — raw P2 arrival
  times are used directly as each sample's window-placement anchor.
- CPU cost of the poller process itself is **included** in measured energy during windows
  (disclosed overhead, `HARNESS_PLAN.md` §4) — present equally in every phase and every
  condition, not something this loop tries to subtract out. Running it as a separate
  process rather than a thread does not remove this cost, only removes GIL contention.

**Required pre-check, not yet run: `time.perf_counter()` comparability across processes
on this macOS.** `perf_counter()` is documented as a per-process monotonic clock; Python's
own docs do not guarantee its reference point is identical across two separate processes
on every platform, only that it is monotonic within one process. The edge test never
needed to check this (single process, one `perf_counter()` origin for both pollers). A
separate-process P2 poller does need it, because the main (inference) process's window
boundaries and the poller process's arrival timestamps must be combined on one shared time
axis. **Design of the check (not run here):**
1. Main process and the poller process are started together, each immediately records
   `time.perf_counter()` and `time.time()` (wall clock) in a tight loop for a few seconds,
   writing `(perf_counter, time)` pairs to its own file.
2. After both stop, align the two processes' records by wall-clock `time.time()` (which
   *is* comparable across processes) to find, for each process, the offset between its own
   `perf_counter()` origin and wall-clock time.
3. Compare the two processes' offsets. If they agree to within a small tolerance (on the
   order of the P2 poll interval, ~2ms) and stay stable across the few seconds of
   recording, `perf_counter()` values from the two processes can be combined directly by
   subtracting their respective offsets. If they disagree or drift, `perf_counter()` is
   **not** directly comparable across these two processes on this machine/OS version, and
   the loop must use `time.time()` (wall clock) as the shared axis instead, accepting
   wall-clock's coarser resolution.
4. This check must be run and pass (or the fallback adopted) before the separate-process
   P2 design in this section is implemented — not assumed from Python's general
   documentation, which doesn't promise cross-process agreement one way or the other.

**Void rule: missing live arrival.** If any sample overlapping a measurement window was
**not** captured by a live P2 poll during the window (i.e. its arrival was only recovered
by the final full-file read after `powermetrics` exited, §1.5's "Stopped" bullet), the
condition is void. A sample recovered only at final-read time means the poller missed it
live — for any reason (poller process stalled, scheduling delay, a poll tick landed on the
wrong side of a header write) — and this loop does not treat a late-recovered arrival as
equivalent to a live one for energy attribution during that window.

### 1.6 No subprocess inside windows; pmset checks only between windows

- Zero subprocess calls of any kind during any of the 24 windows' active measurement
  interval (`idle_before`/`a1`/`a2`/`idle_after`) — matches the edge test's own "ZERO
  subprocess calls during the train" rule and `HARNESS_PLAN.md` §4's explicit requirement.
- `pmset -g batt` (AC) and `pmset -g therm` (coarse thermal) are each run **once per phase,
  at the START of that phase's 3s settle** — not immediately before the window, and never
  within 2.5s of a window actually starting. This gives every check at least 2.5s of
  margin between the subprocess call returning and any measurement window beginning,
  so the check's own process-spawn overhead cannot bleed into a window even in a slow
  case. This is the same deliberate deviation from x86 already disclosed in
  `HARNESS_PLAN.md` §1's table (x86 checks AC every poll tick via a cheap in-process
  sysfs read at zero cost; M1 cannot do that without spawning a subprocess mid-window, so
  it checks once per phase, as early as possible within the settle period, instead —
  which means an AC loss or thermal excursion occurring after the phase-start check but
  before the window ends would not be caught until the next phase's check).
- Per-condition thermal check does **not** re-run `pmset -g therm` between every single
  window on top of the per-phase checks above — instead, it reads the `thermal_pressure`
  field already present in each `powermetrics` sample for that window's time range (parsed
  by `powermetrics_parser.py`, already committed) and requires it to read `"Nominal"` for
  every sample overlapping the window. This avoids adding *more* subprocess calls than the
  per-phase `pmset` checks already require, while still getting fine-grained thermal
  coverage from data already being collected for energy attribution.
  **This per-sample "Nominal" check is provisional.** The actual throttle rule (what
  frequency drop or pressure pattern counts as a violation, beyond the sample simply not
  reading "Nominal") and the action taken on violation (voiding the condition, as stated
  elsewhere in this document for other rules, is the expected direction but not yet
  decided here) both await the A7r4(b) calibration gate in `docs/stage5_consolidated_plan.md`
  ("no primary CPU-block session may start until the thermal-throttle guard exists, has
  unit tests, and calibrated thresholds are registered in a dated amendment"). No numbers
  are proposed here for that calibration.

### 1.7 Per-condition outputs

Matching x86's `windows.csv`/`summary.csv` column set (`csv_write`, `pilot.py:473-477,651`;
`summarize()`, `pilot.py:415-471`), **plus** `arch` and `condition` columns that x86's
`windows.csv` does not actually carry (confirmed by inspecting a real x86 `windows.csv`
header during A5 — those two fields are implicit in the directory name on x86, not present
in the CSV itself; M1's `m1_pipeline.summarize_condition` requires them explicitly, so they
are added as real columns here rather than left for the caller to inject after the fact):

- `windows.csv`: one row per phase per repeat (24 rows per condition) — `device`, `arch`,
  `condition`, `size`, `batch`, `requested_s`, `repeat`, `phase`, `start_t`, `end_t`,
  `duration_s`, `batches`, `energy_j`, `mean_power_w`, plus the same
  `estimated_idle_j`/`above_idle_j`/`above_idle_fraction`/`gross_j_per_image` fields x86
  attaches to `a1`/`a2` rows during summarization (already implemented in
  `m1_pipeline.summarize_condition`).
- `summary.csv`: one row per condition — same field set already verified bit-exact against
  real x86 data in A5 (`device`, `arch`, `condition`, `size`, `batch`, `window_s`, `pairs`,
  `gross_j_per_image_mean`, `paired_difference_sd`, `paired_order_bias`,
  `approx_mde_j_per_image`, `conservative_screen_fraction`, `candidate_for_confirmation`
  — structurally unreachable at n_pairs<30, documented A4 — `total_active_j_mean`,
  `above_idle_j_mean`, `above_idle_fraction_mean`, `idle_power_w_mean`,
  `images_per_s_mean`). Fields x86 has that M1 does not compute
  (`cpu_package_j_per_image_mean`, `gpu_j_per_image_mean`, `system_j_per_image_mean`,
  `power_regime`, `idle_j_sd`, `total_j_sd`) are omitted, not fabricated as null/zero.
- `environment.json` per condition: platform/cpu/torch/torchvision/arguments/timestamp/
  git_hash fields adapted per `HARNESS_PLAN.md` §1's table, plus the qnnpack engine value
  (§1.2 above) logged explicitly for INT8 conditions.
- `overlap_diagnostics` block (`total_sample_energy_j`, `total_accounted_energy_j`,
  `outside_energy_j`, `relative_discrepancy`) — already part of
  `m1_pipeline.process_condition_energy`'s return value, carried into the condition's
  summary unchanged.
- **Raw `powermetrics` output file** for this condition is kept (not deleted after
  parsing) — the same file `/var/gai-powermetrics/{TAG}_*.txt` referenced in §1.3,
  retained as the primary source record.
- **`arrivals.csv`** per condition, one row per sample: `sample_k`, `arrival_perf`,
  `elapsed_ms`, `cpu_power_mw`, `thermal_pressure` — the P2 poller's own arrival record
  (§1.5), kept alongside the raw file so the exact arrival timestamps used for window
  attribution are auditable independently of re-parsing the raw text later.

---

## 2. Every deviation from x86, consolidated

| # | x86 behaviour | M1 deviation | Why |
|---|---|---|---|
| 1 | In-process poll thread reads RAPL every tick, zero extra cost | No in-process sensor at all; a separate P2 poller **process** reads the `powermetrics` output file from disk every ~2ms | M1 has no RAPL/NVML equivalent; `powermetrics` is an external root process (§1.3, §1.5) |
| 2 | AC checked every poll tick, in-process, free | AC (and coarse thermal) checked only immediately before/after each window, via `pmset` subprocess calls | No cheap in-process check exists; subprocesses are banned inside windows (§1.6) |
| 3 | Energy read synchronously as a continuous counter | Energy attributed after the fact via time-overlap splitting against discrete `powermetrics` samples | `powermetrics` samples are discrete power readings, not an integratable counter |
| 4 | `check_governor()` / `check_platform_profile()` enforce x86-specific OS controls | Both omitted entirely — no concept exists on macOS | Not a gap in the code; the control surface doesn't exist (`HARNESS_PLAN.md` §1) |
| 5 | `--cpu-affinity pcores` hard-pins the process | No pinning; default scheduling only, no `taskpolicy` request either | No `sched_setaffinity` equivalent on macOS; cluster residency recorded from `powermetrics`, not controlled |
| 6 | `/proc/uptime` read directly | `sysctl -n kern.boottime` parsed and subtracted from current time | No `/proc/uptime` on macOS |
| 7 | `/sys/class/power_supply` read for AC state | `pmset -g batt` parsed | No sysfs equivalent on macOS |
| 8 | Thermal guard reads per-core `scaling_cur_freq` + throttle counters | Coarse `pmset -g therm` + `powermetrics` thermal-pressure field per sample | No equivalent sysfs thermal interface on macOS |
| 9 | `windows.csv` has no `arch`/`condition` columns (implicit in directory name) | M1's `windows.csv`/`summary.csv` carry them as explicit columns | `m1_pipeline.summarize_condition` requires them directly; confirmed missing on real x86 data during A5 |
| 10 | Session-order condition list built base-states-then-bnrecal, seed `3000+N` | Same base-then-bnrecal order (A6 fix), seed `4000+N` | Distinct seed family, same structural shape, for direct comparability |
| 11 | AC checked once per condition (after only) at the session-runner level | AC checked twice per condition (before AND after) at the session-runner level | `m1_session_runner.py`'s existing design, stricter than x86's, not reduced to match |
| 12 | No offset correction needed (sensor is synchronous) | Explicitly delta=0 after the two-poller edge test superseded the marker-test memo's withdrawn -15ms recommendation | Empirically re-derived this session, not carried over from the single-poller marker test |
| 13 | `gpu`/`nvidia_driver_version`/`gpu_power_cap_w` always null (CPU-only block already) | Same — no change | Not an M1-specific deviation; both blocks are CPU-only |
| 14 | No MobileNetV3-Small INT8 condition is missing | MobileNetV3-Small INT8 structurally absent (qnnpack cannot run this checkpoint, `m1_preflight/results/mbv3_int8_diag.txt`) | 23 conditions per session on M1 vs 24 on x86, for this reason only |
| 15 | Energy boundary: RAPL package-0 counter (`pilot.py`'s `integrate()`, `D16` primary boundary) | Energy boundary: `powermetrics`'s `CPU Power` field | No RAPL on Apple Silicon; `CPU Power` is the closest available analogue, not the same measurement principle |
| 16 | INT8 backend: `fbgemm` (x86-only) | INT8 backend: `qnnpack` | `fbgemm` does not exist on ARM; confirmed x86-only in `docs/m1_harness_design.md` and this session's own INT8 diagnostics |
| 17 | RAPL counter read is a passive sysfs read with no separate process | `powermetrics` is itself a separate root **process** running continuously for the condition's duration, consuming its own CPU/scheduling slice | Overhead from the measurement instrument itself is present on M1 in a way it isn't on x86; disclosed, not subtracted out |
| 18 | RAPL `energy_uj` is a continuously-accumulating microjoule counter, read at the exact moment needed | `powermetrics` samples are discrete, integer-mW power readings at a **requested** 400ms interval that is **not** what's actually observed (the real captured sample headers show ~414ms actual elapsed per sample, not 400ms) | Coarser time resolution, coarser power resolution (integer mW vs effectively-continuous RAPL deltas), and a nominal-vs-actual interval gap that doesn't exist on x86 |
| 19 | x86 sessions were run with the operator physically present | FileVault requires the brother to restart and log in on the keyboard before each fresh-boot session on M1 | No remote/unattended reboot path exists for a FileVault-enabled Mac; `m1_preflight`/earlier session notes already established this operational constraint |

## 3. Preflight items carried over unchanged from x86 (not deviations)

- Checkpoint sha256 verification (already run once in `m1_preflight`; re-verified per
  condition here, which is *stricter* than x86's one-time check, not a deviation from it
  per se — x86 doesn't re-verify per condition either, so this is an M1-specific addition
  in the same spirit as item 11 above, not a parity requirement).
- Clean git tree at the pinned commit — new on M1 (§1.1), not present as a blocking check
  in x86's `pilot.py` or `run_cpu_block_session.py` at all (x86 only *records* the hash).
- `caffeinate` ownership and blocked-background-app checks have no x86 analogue to deviate
  from — they exist purely because of macOS-specific risks (sleep, remote-access tools)
  that don't apply to the x86 LOQ.

## 4. Open items — must be resolved before the dated amendment, not deferred further

- **`powermetrics` `COUNT` margin (§1.3) — RESOLVED.** Now
  `COUNT = ceil(expected_duration_s / 0.405) + 5`, derived from the measured mean actual
  sample interval (410.36ms, not the nominal 400ms) in the Test 2 duty-cycle calibration
  run (`calibration/m1_thermal_20261104/`). No longer an open item.
- The `time.perf_counter()` cross-process comparability pre-check (§1.5) must be run, and
  its result (direct comparison vs wall-clock fallback) must be known, before the
  separate-process P2 design can be implemented as written.
- The A7r4(b) thermal-throttle calibration (§1.6) remains the standing hard gate from
  `docs/stage5_consolidated_plan.md` — unresolved here, not newly introduced by this
  document.
