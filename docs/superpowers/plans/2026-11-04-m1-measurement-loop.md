# M1 Per-Condition Measurement Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the production M1-CPU per-condition measurement loop and integrate it into the M1 session runner, adhering strictly to Amendment A13 and `scripts/m1/MEASUREMENT_LOOP_DESIGN.md`.

**Architecture:** A standalone child process (`m1_poller.py`) samples the `powermetrics` output via `os.stat` every ~2ms to eliminate Python GIL contention with model forward passes. The main execution engine (`m1_measurement_loop.py`) runs preflight guards (git status, checkpoint SHA-256, blocked apps, AC/thermal nominal, caffeinate), coordinates powermetrics and the poller, executes the 24-window phase loop with zero mid-window subprocesses, validates live arrivals, attributes overlap energy via `m1_pipeline.py`, and records complete condition artifacts.

**Tech Stack:** Python 3.9+, PyTorch (CPU, 4 threads, qnnpack for INT8), torchvision (CIFAR-10), macOS `powermetrics`, `pmset`, `caffeinate`, `sysctl`.

**Spec:** `scripts/m1/MEASUREMENT_LOOP_DESIGN.md`, `docs/stage5_analysis_plan.md` (Amendment A13), and `scripts/m1/HANDOVER_TO_AGY_20261104.md`.

## Global Constraints

- Platform: Apple Silicon M1 (ARM64) running macOS.
- Threading: Fixed to 4 threads (`torch.set_num_threads(4)`).
- Quantization: Explicit `torch.backends.quantized.engine = "qnnpack"` for INT8 conditions.
- Powermetrics: Invoked via `sudo -n /usr/local/sbin/gai_powermetrics.sh 400 COUNT TAG` with sequential process lifecycle (previous process must exit before next starts).
- Powermetrics sizing: `COUNT = ceil(expected_duration_s / 0.405) + 5`.
- Poller: Standalone OS process polling `os.stat` every ~2ms, reading only incremental bytes, recording `perf_counter` arrivals and `elapsed_ms`.
- Timing offset: $\delta = 0.0$ (no artificial shift).
- Subprocess isolation: ZERO subprocess calls during any 5s measurement window; `pmset -g batt` and `pmset -g therm` run only at start of 3s settle ($\ge 2.5\text{s}$ margin before window start).
- Missing live arrival rule: Condition is void if any sample overlapping a measurement window was not captured live by P2 poller.
- Thermal guard rule: Windows `a1`/`a2` with `repeat > 0` flagged if mean frequency < $T = 2881.11\text{ MHz}$ or any overlapping sample thermal pressure != "Nominal". Any flagged window voids condition.
- Uptime guard: Condition must start within 180 min of boot (`max_condition_uptime_s = 10800.0`).
- Blocked processes: Immediate fail-closed abort if `AnyDesk`, `OneDrive`, `remoting_host`, or `screensharingd` is running.
- Clean git tree: Hard block if `git status --porcelain` is not empty.
- Git controls: No push to `origin/master`, no git tags without explicit operator instruction, generic AI commit trailers only.

## Review Focus

1. *Late powermetrics sample arrival during window*: A sample spans into a window but was only parsed on final file read (poller stalled or missed tick). Verify condition is flagged void with missing live arrival error.
2. *Non-Nominal thermal pressure with normal frequency*: P-core frequency stays above 3000 MHz, but one overlapping sample reports `thermal_pressure: Moderate`. Verify window is flagged throttled and condition is voided.
3. *Process cleanup on sudden exception / SIGINT*: Inference loop throws an unexpected runtime error mid-condition. Verify `caffeinate` daemon and poller process are cleanly killed in `finally` and powermetrics process is stopped.
4. *P2 poller buffer straddling across chunks*: A `*** Sampled system activity ...` line is split halfway across two `os.stat` read increments. Verify poller correctly stitches the boundary via the carry-over buffer and attributes correct sample count and timestamps.
5. *Pre-window pmset check latency margin*: If `pmset` subprocess takes 300ms, ensure measurement window start is delayed so window duration remains exactly 5.0s and subprocess never overlaps window.

---

### Task 1: Standalone Out-of-Process P2 Poller

**Files:**
- Create: `scripts/m1/m1_poller.py`
- Create: `scripts/m1/test_m1_poller.py`

**Interfaces:**
- CLI / Process Invocation:
  `python3 scripts/m1/m1_poller.py --output-file <powermetrics_out> --ipc-file <arrivals_ipc> [--ready-file <sentinel>] [--poll-interval-s 0.002]`
- Produces:
  `arrivals_ipc`: JSON lines or CSV containing records `{"sample_k": int, "arrival_perf": float, "elapsed_ms": float, "is_live": bool}`
  Signals readiness via creating `ready-file` (or writing "READY\n" to stdout) after initial file check, and signals first header via sentinel file or IPC entry.
  Exit on EOF when `powermetrics` finishes and SIGTERM/stop signal sent, performing final file read to flag any non-live arrivals.

- [ ] **Step 1: Write the failing test for P2 poller**

Write `scripts/m1/test_m1_poller.py`:
- Test incremental polling on an actively growing synthetic powermetrics text file.
- Verify headers straddling a chunk boundary are parsed without corruption.
- Verify samples captured while poller runs are recorded with `is_live=True`.
- Verify samples appended after poller receives stop signal and performs final read are recorded with `is_live=False`.

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/shohan/green-ai-research/measurement_env/bin/python3 -m unittest scripts/m1/test_m1_poller.py`
Expected: FAIL (script does not exist).

- [ ] **Step 3: Implement `scripts/m1/m1_poller.py`**

Implement standalone poller logic:
- Parse arguments (`--output-file`, `--ipc-file`, `--ready-file`, `--first-header-file`).
- Loop: `os.stat` every 2ms, read new bytes from `last_offset`.
- Match `HEADER_RE = re.compile(r'\*\*\* Sampled system activity .*? \(([\d.]+)ms elapsed\) \*\*\*')`.
- Write arrivals immediately to `ipc-file` (flushed).
- Handle signal `SIGTERM` / `SIGINT` to break loop and execute final full-file read, writing any newly found samples with `is_live=False`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/shohan/green-ai-research/measurement_env/bin/python3 -m unittest scripts/m1/test_m1_poller.py`
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add scripts/m1/m1_poller.py scripts/m1/test_m1_poller.py
git commit -m "feat(m1): implement standalone out-of-process P2 powermetrics poller"
```

---

### Task 2: Preflight Checks and Environment Guarding

**Files:**
- Create: `scripts/m1/m1_preflight_checks.py`
- Create: `scripts/m1/test_m1_preflight_checks.py`

**Interfaces:**
- Consumes: `m1_preflight/results/m1_checkpoint_hashes.txt`, `guards.py` (`check_ac_power_m1`, `check_thermal_nominal_m1`, `check_uptime_guard`).
- Produces:
  `verify_checkpoint_sha256(checkpoint_path: Path, expected_hashes_file: Path) -> str`
  `check_git_clean_tree(repo_dir: Path, expected_commit: Optional[str] = None) -> str`
  `check_blocked_processes(blocked: Tuple[str, ...]) -> None`
  `caffeinate_scope(target_pid: Optional[int] = None)` (context manager)

- [ ] **Step 1: Write the failing test for preflight checks**

Write `scripts/m1/test_m1_preflight_checks.py`:
- Test SHA-256 checkpoint verification against real hash table and corrupted hash table.
- Test git status clean tree checker (clean passes, modified/untracked fails when required).
- Test blocked processes detection (`AnyDesk`, `OneDrive`, `remoting_host`, `screensharingd`).
- Test `caffeinate_scope` process lifecycle (starts process, terminates cleanly on exit).

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/shohan/green-ai-research/measurement_env/bin/python3 -m unittest scripts/m1/test_m1_preflight_checks.py`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/m1/m1_preflight_checks.py`**

Implement:
- `verify_checkpoint_sha256`: Computes hashlib.sha256 of checkpoint, verifies against expected map.
- `check_git_clean_tree`: Runs `git status --porcelain`. If output non-empty (excluding untracked handover/reports if configured, or fail-closed), raises `RuntimeError`. Checks `git rev-parse HEAD`.
- `check_blocked_processes`: Uses `ps -A -o comm=` to find exact process names. Raises `GuardError` if any running.
- `caffeinate_scope`: Spawns `caffeinate -dimsu -w <pid>`, handles `SIGTERM` cleanup in `finally`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/shohan/green-ai-research/measurement_env/bin/python3 -m unittest scripts/m1/test_m1_preflight_checks.py`
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add scripts/m1/m1_preflight_checks.py scripts/m1/test_m1_preflight_checks.py
git commit -m "feat(m1): implement preflight checkpoint, git, process, and caffeinate guards"
```

---

### Task 3: Per-Condition Measurement Loop

**Files:**
- Modify: `scripts/m1/m1_pipeline.py` (add `write_summary_csv`)
- Create: `scripts/m1/m1_measurement_loop.py`
- Create: `scripts/m1/test_m1_measurement_loop.py`

**Interfaces:**
- Consumes:
  `m1_poller.py`, `m1_preflight_checks.py`, `m1_pipeline.py`, `powermetrics_parser.py`, `guards.py`.
- Produces:
  `measure_condition(arch: str, condition: str, out_dir: Path, checkpoints_dir: Path, ...)`:
  Emits:
  - `windows.csv` (24 rows, with `thermal_flag`, `energy_j`, `batches`, etc.)
  - `summary.csv` (1 row, exact schema)
  - `summary.json` (complete summary with overlap diagnostics)
  - `environment.json` (platform, torch, qnnpack engine, uptime, boottime, args)
  - `arrivals.csv` (all samples with `arrival_perf`, `elapsed_ms`, `cpu_power_mw`, `is_live`)
  - `windows_raw.jsonl` (incremental live boundary logs)
  - raw powermetrics txt copy
  Returns summary dictionary.

- [ ] **Step 1: Add `write_summary_csv` and write failing tests in `test_m1_measurement_loop.py`**

- In `m1_pipeline.py`, add `write_summary_csv(summary_row: Dict[str, Any], out_path: Union[str, Path])`.
- In `test_m1_measurement_loop.py`:
  - Mock powermetrics runner and poller.
  - Run synthetic condition (e.g. 6 repeats $\times$ 4 phases).
  - Verify all 24 window phases recorded with exact timing.
  - Verify `windows.csv`, `summary.csv`, `arrivals.csv`, `environment.json`, `summary.json` generated.
  - Verify missing live arrival raises void error.
  - Verify non-Nominal thermal pressure flags window and voids condition.

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/shohan/green-ai-research/measurement_env/bin/python3 -m unittest scripts/m1/test_m1_measurement_loop.py`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/m1/m1_measurement_loop.py`**

Implement complete loop:
- Model and dataset initialization:
  - `torch.set_num_threads(4)`.
  - For INT8: `torch.backends.quantized.engine = 'qnnpack'`.
  - CIFAR-10 evaluation batch creation (batch=1, size 32x32, normalized).
- Powermetrics execution:
  - Sequential process guard: wait for any running `gai_powermetrics.sh` / `powermetrics` to terminate.
  - `COUNT = math.ceil(expected_duration_s / 0.405) + 5`.
  - Launch powermetrics wrapper, find output file.
  - Launch P2 poller process (`m1_poller.py`), wait for first header event.
- Phase loop:
  - 6 repeats $\times$ 4 phases (`idle_before`, `a1`, `a2`, `idle_after`).
  - Settle (3.0s):
    - Run `check_ac_power_m1()` and `check_thermal_nominal_m1()` at the START of settle.
    - Sleep or warm-up forward loop for remaining settle time ($\ge 2.5\text{s}$ margin before window start).
  - Window (5.0s):
    - Zero subprocess calls.
    - If idle: `time.sleep(5.0)`.
    - If active: loop `model(x)` counting completed forward passes.
    - Append to `WindowPhase` and write incrementally to `windows_raw.jsonl`.
- Post-run:
  - Terminate poller process, wait for powermetrics completion.
  - Read `arrivals_raw.jsonl` / `arrivals.csv`.
  - Live arrival verification: verify all samples overlapping windows have `is_live=True`. If not, raise `RuntimeError("Missing live arrival in measurement window")`.
  - Parse powermetrics output, match samples with poller `arrival_perf`.
  - Attribute energy with `m1_pipeline.process_condition_energy` and `summarize_condition`.
  - Write all output artifacts.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/shohan/green-ai-research/measurement_env/bin/python3 -m unittest scripts/m1/test_m1_measurement_loop.py`
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add scripts/m1/m1_pipeline.py scripts/m1/m1_measurement_loop.py scripts/m1/test_m1_measurement_loop.py
git commit -m "feat(m1): implement M1 per-condition measurement loop and output pipeline"
```

---

### Task 4: Integration with `m1_session_runner.py` and CLI Interface

**Files:**
- Modify: `scripts/m1/m1_session_runner.py`
- Modify: `scripts/m1/test_m1_session_runner.py`

**Interfaces:**
- Connects: `m1_measurement_loop.measure_condition` as default `condition_runner`.
- CLI Interface:
  `python3 scripts/m1/m1_session_runner.py --session <N> --out-base <dir> --checkpoints-dir <dir> [--data-dir <dir>] [--single-condition ARCH COND]`

- [ ] **Step 1: Write tests for CLI and default runner in `test_m1_session_runner.py`**

- Test `--single-condition` CLI dispatch for protocol checking.
- Test default runner wiring when no custom `condition_runner` is supplied.

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/shohan/green-ai-research/measurement_env/bin/python3 -m unittest scripts/m1/test_m1_session_runner.py`
Expected: FAIL on new CLI tests.

- [ ] **Step 3: Implement CLI and default runner in `m1_session_runner.py`**

- Add argument parser in `main()`.
- Add support for `--single-condition <arch> <condition>` (for protocol check and isolated condition runs).
- Wire `measure_condition` as default runner.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/shohan/green-ai-research/measurement_env/bin/python3 -m unittest scripts/m1/test_m1_session_runner.py`
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add scripts/m1/m1_session_runner.py scripts/m1/test_m1_session_runner.py
git commit -m "feat(m1): wire production measurement loop and CLI entrypoint into m1_session_runner"
```

---

### Task 5: Protocol-Check Execution on M1 Mac (`m1-tailscale`)

**Files:**
- Deploy files to `m1-tailscale:~/green-ai-m1/`
- Target command:
  `python3 scripts/m1/m1_session_runner.py --session 0 --out-base ~/protocol_check --checkpoints-dir ~/green-ai-m1/checkpoints --single-condition resnet18 fp32_ts`

- [ ] **Step 1: Sync latest scripts and test suite to M1 Mac**

Sync `scripts/m1/*.py`, `config/thermal_thresholds_m1.json`, `m1_preflight/results/m1_checkpoint_hashes.txt` to `m1-tailscale:~/green-ai-m1/`.
Run unit tests remotely on M1 Mac:
`ssh m1-tailscale "cd ~/green-ai-m1 && ./venv/bin/python -m unittest discover scripts/m1"`
Verify all tests pass on Darwin ARM64.

- [ ] **Step 2: Execute protocol-check trial on M1 Mac**

Run one protocol-check condition under session runner exact arguments:
`ssh m1-tailscale "cd ~/green-ai-m1 && ./venv/bin/python scripts/m1/m1_session_runner.py --session 0 --out-base ~/m1_protocol_check --checkpoints-dir ~/green-ai-m1/checkpoints --single-condition resnet18 fp32_ts"`
Check that:
- Preflight guards succeed (AC, thermal, git status, checkpoint hash, no blocked apps).
- Powermetrics runs with correct `COUNT` and sequential process lifecycle.
- P2 poller captures live arrivals with zero missing live arrivals.
- Exactly 24 rows in `windows.csv`.
- `summary.csv`, `summary.json`, `arrivals.csv`, `environment.json` written cleanly.
- Thermal flag evaluates to `False` (nominal).

- [ ] **Step 3: Verify protocol-check data properties**

Inspect `~/m1_protocol_check/resnet18_fp32_ts/`:
- `summary.json` overlap diagnostics (relative discrepancy $< 10^{-4}$).
- Verify no subprocess ran inside 5s measurement windows.
- Confirm this protocol-check is guard-validation only (not deposited, not analyzed as confirmatory).

- [ ] **Step 4: Update handover checklist and commit**

Update `scripts/m1/HANDOVER_TO_AGY_20261104.md` marking the measurement loop and protocol check items completed.
Commit changes locally (no push, no tag).
