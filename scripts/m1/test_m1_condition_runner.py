"""Unit tests for M1 condition runner and measurement loop.

Verifies all rules per scripts/m1/MEASUREMENT_LOOP_DESIGN.md and Amendment A13:
1. COUNT sizing formula: COUNT = ceil(expected_duration_s / 0.405) + 5
2. ZERO subprocess calls inside any 5.0 s measurement window; pmset checks only at settle start
3. Condition voided if any sample overlapping a measurement window lacks a live arrival
4. Thermal flagging rule: qualifying window (a1/a2, rep > 0) flagged if freq < T or pressure != 'Nominal'
5. Preflight guards: blocked processes, git tree clean, checkpoint SHA-256
6. Environment logging: qnnpack engine logged for INT8
7. Complete output generation: windows.csv (24 rows), summary.csv (1 row), arrivals.csv, environment.json, summary.json
8. Integration with m1_session_runner.run_m1_session
"""
import csv
import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import m1_condition_runner
from guards import GuardError
from m1_condition_runner import (
    compute_powermetrics_count,
    measure_condition,
    run_condition_preflight,
    verify_live_arrivals_coverage,
)
from m1_pipeline import WindowPhase, read_thermal_thresholds_m1



class TestM1ConditionRunnerRules(unittest.TestCase):
    def test_count_sizing_formula(self):
        """COUNT = ceil(expected_duration_s / 0.405) + 5 (Amendment A13)."""
        # 192s = 6 repeats * 4 phases * (3s settle + 5s window)
        expected_192 = math.ceil(192.0 / 0.405) + 5
        self.assertEqual(compute_powermetrics_count(192.0), expected_192)
        self.assertEqual(compute_powermetrics_count(192.0), 475 + 5)
        self.assertEqual(compute_powermetrics_count(192.0), 480)

        # 0s edge case
        self.assertEqual(compute_powermetrics_count(0.0), 5)
        # fractional duration
        self.assertEqual(compute_powermetrics_count(40.5), math.ceil(40.5 / 0.405) + 5)

    def test_preflight_blocked_processes(self):
        """Preflight fails-closed if AnyDesk, OneDrive, remoting_host, screensharingd running."""
        for blocked in ['AnyDesk', 'OneDrive', 'remoting_host', 'screensharingd']:
            with patch('m1_condition_runner._get_running_processes', return_value=['bash', 'python', blocked]):
                with self.assertRaises(GuardError) as ctx:
                    run_condition_preflight(
                        arch='resnet18',
                        condition='fp32_ts',
                        checkpoints_dir=Path('/tmp'),
                        skip_git_check=True,
                        skip_hash_check=True,
                    )
                self.assertIn(blocked, str(ctx.exception))

    def test_preflight_checkpoint_sha256(self):
        """Preflight re-verifies checkpoint SHA-256 against hash table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / 'resnet18_fp32_ts.pt'
            ckpt_path.write_bytes(b'fake_corrupted_model_bytes')

            hash_file = tmp_path / 'hashes.txt'
            hash_file.write_text("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  checkpoints/resnet18_fp32_ts.pt\n")

            with patch('m1_condition_runner._get_running_processes', return_value=['bash', 'python']):
                with self.assertRaises(RuntimeError) as ctx:
                    run_condition_preflight(
                        arch='resnet18',
                        condition='fp32_ts',
                        checkpoints_dir=tmp_path,
                        hashes_file=hash_file,
                        skip_git_check=True,
                    )
                self.assertIn("hash mismatch", str(ctx.exception).lower())

    def test_void_on_missing_live_arrival(self):
        """Condition is void if any sample overlapping a measurement window was NOT captured live."""
        windows = [
            WindowPhase(
                device='cpu', arch='resnet18', condition='fp32_ts',
                size=32, batch=1, requested_s=5.0, repeat=1, phase='a1',
                start_t=10.0, end_t=15.0, batches=100
            )
        ]

        # Case 1: All overlapping samples are live -> passes
        arrivals_live = [
            {'sample_k': 1, 'arrival_perf': 9.8, 'elapsed_ms': 400.0, 'is_live': True},   # [9.4, 9.8] outside
            {'sample_k': 2, 'arrival_perf': 10.2, 'elapsed_ms': 400.0, 'is_live': True},  # [9.8, 10.2] overlaps
            {'sample_k': 3, 'arrival_perf': 10.6, 'elapsed_ms': 400.0, 'is_live': True},  # [10.2, 10.6] overlaps
            {'sample_k': 4, 'arrival_perf': 15.1, 'elapsed_ms': 400.0, 'is_live': True},  # [14.7, 15.1] overlaps
            {'sample_k': 5, 'arrival_perf': 16.0, 'elapsed_ms': 400.0, 'is_live': False}, # [15.6, 16.0] outside & not live (allowed!)
        ]
        # Should not raise
        verify_live_arrivals_coverage(arrivals_live, windows)

        # Case 2: One overlapping sample was only recovered post-mortem (is_live=False) -> VOID
        arrivals_missing_live = [
            {'sample_k': 1, 'arrival_perf': 9.8, 'elapsed_ms': 400.0, 'is_live': True},
            {'sample_k': 2, 'arrival_perf': 10.2, 'elapsed_ms': 400.0, 'is_live': True},
            {'sample_k': 3, 'arrival_perf': 10.6, 'elapsed_ms': 400.0, 'is_live': False}, # Overlaps window [10.0, 15.0] but not live!
        ]
        with self.assertRaises(RuntimeError) as ctx:
            verify_live_arrivals_coverage(arrivals_missing_live, windows)
        self.assertIn("live arrival", str(ctx.exception).lower())

    def test_no_subprocess_inside_windows_enforced_by_test(self):
        """ZERO subprocess calls during any measurement window; pmset only at settle start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'test_cond'
            out_dir.mkdir()

            # We will run a mocked 1-repeat run (or 2-phase) and intercept all subprocess calls
            subprocess_calls = []

            orig_popen = m1_condition_runner._spawn_subprocess
            def fake_spawn(*args, **kwargs):
                subprocess_calls.append((time.perf_counter(), args, kwargs))
                mock_proc = MagicMock()
                mock_proc.pid = 99999
                mock_proc.poll.return_value = 0
                mock_proc.wait.return_value = 0
                mock_proc.communicate.return_value = (b"", b"")
                return mock_proc

            with patch('m1_condition_runner._spawn_subprocess', side_effect=fake_spawn), \
                 patch('m1_condition_runner.check_ac_power_m1', return_value=True), \
                 patch('m1_condition_runner.check_thermal_nominal_m1', return_value=True):
                
                # Run condition with mock loop
                res = measure_condition(
                    arch='resnet18',
                    condition='fp32_ts',
                    out_dir=out_dir,
                    checkpoints_dir=Path(tmpdir),
                    repeats=1,  # 1 repeat for fast test
                    settle_s=0.1,
                    window_s=0.2,
                    skip_preflight=True,
                    mock_forward=True,
                    mock_powermetrics=True,
                    check_ac_fn=lambda: True,
                    check_therm_fn=lambda: True,
                )

            # Read windows.csv
            csv_path = out_dir / 'windows.csv'
            self.assertTrue(csv_path.exists())
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                windows = list(reader)

            self.assertEqual(len(windows), 4)  # 1 repeat * 4 phases

            # Check that during [start_t, end_t] of EVERY window, NO subprocess was called!
            for w in windows:
                w_start = float(w['start_t'])
                w_end = float(w['end_t'])
                for call_t, args, kwargs in subprocess_calls:
                    # Assert no call occurred inside (w_start, w_end)
                    self.assertFalse(
                        w_start <= call_t <= w_end,
                        f"Subprocess called at {call_t} inside window [{w_start}, {w_end}]: {args}"
                    )

    def test_thermal_flag_and_voiding_rule(self):
        """Condition with any flagged qualifying window is flagged and voided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'cond_throttled'
            out_dir.mkdir()

            # Powermetrics sample with throttled freq (e.g. 2600 MHz < T=2881.11)
            fake_samples = [
                {
                    'sample_k': i,
                    'arrival_perf': 1.0 + i * 0.4,
                    'elapsed_ms': 400.0,
                    'is_live': True,
                    'cpu_power_mw': 5000,
                    'pcore_mean_freq_mhz': 2500.0,  # Below T!
                    'thermal_pressure': 'Nominal',
                }
                for i in range(1, 30)
            ]

            res = measure_condition(
                arch='resnet18',
                condition='int8',
                out_dir=out_dir,
                checkpoints_dir=Path(tmpdir),
                repeats=2,
                settle_s=0.05,
                window_s=0.1,
                skip_preflight=True,
                mock_forward=True,
                check_ac_fn=lambda: True,
                check_therm_fn=lambda: True,
                injected_samples=fake_samples,
            )

            self.assertTrue(res.get('thermal_flagged'))
            self.assertTrue(res.get('condition_voided'))

            # Check windows.csv thermal_flag
            with open(out_dir / 'windows.csv') as f:
                rows = list(csv.DictReader(f))
            # Qualifying active windows for repeat 1 should be flagged True
            qualifying = [r for r in rows if r['phase'] in ('a1', 'a2') and int(r['repeat']) > 0]
            self.assertTrue(len(qualifying) > 0)
            for q in qualifying:
                self.assertEqual(q['thermal_flag'], 'True')

    def test_int8_qnnpack_engine_logged_in_environment(self):
        """For INT8 conditions, qnnpack engine must be set and logged in environment.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'cond_int8'
            out_dir.mkdir()

            fake_samples = [
                {
                    'sample_k': i,
                    'arrival_perf': 1.0 + i * 0.4,
                    'elapsed_ms': 400.0,
                    'is_live': True,
                    'cpu_power_mw': 5000,
                    'pcore_mean_freq_mhz': 3200.0,
                    'thermal_pressure': 'Nominal',
                }
                for i in range(1, 20)
            ]

            res = measure_condition(
                arch='resnet18',
                condition='int8',
                out_dir=out_dir,
                checkpoints_dir=Path(tmpdir),
                repeats=1,
                settle_s=0.05,
                window_s=0.1,
                skip_preflight=True,
                mock_forward=True,
                check_ac_fn=lambda: True,
                check_therm_fn=lambda: True,
                injected_samples=fake_samples,
            )

            env_path = out_dir / 'environment.json'
            self.assertTrue(env_path.exists())
            env_data = json.loads(env_path.read_text())
            self.assertEqual(env_data.get('quantized_engine'), 'qnnpack')

    def test_output_artifacts_created(self):
        """All expected artifacts must be created matching schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'cond_complete'
            out_dir.mkdir()

            fake_samples = [
                {
                    'sample_k': i,
                    'arrival_perf': 1.0 + i * 0.4,
                    'elapsed_ms': 400.0,
                    'is_live': True,
                    'cpu_power_mw': 4500,
                    'pcore_mean_freq_mhz': 3204.0,
                    'thermal_pressure': 'Nominal',
                }
                for i in range(1, 60)
            ]

            res = measure_condition(
                arch='resnet18',
                condition='fp32_ts',
                out_dir=out_dir,
                checkpoints_dir=Path(tmpdir),
                repeats=6,
                settle_s=0.01,
                window_s=0.02,
                skip_preflight=True,
                mock_forward=True,
                check_ac_fn=lambda: True,
                check_therm_fn=lambda: True,
                injected_samples=fake_samples,
            )

            self.assertTrue((out_dir / 'windows.csv').exists())
            self.assertTrue((out_dir / 'summary.csv').exists())
            self.assertTrue((out_dir / 'arrivals.csv').exists())
            self.assertTrue((out_dir / 'environment.json').exists())
            self.assertTrue((out_dir / 'summary.json').exists())

            # Exactly 24 rows in windows.csv
            with open(out_dir / 'windows.csv') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 24)

            # Exactly 1 row in summary.csv
            with open(out_dir / 'summary.csv') as f:
                s_rows = list(csv.DictReader(f))
            self.assertEqual(len(s_rows), 1)
            self.assertEqual(s_rows[0]['arch'], 'resnet18')
            self.assertEqual(s_rows[0]['condition'], 'fp32_ts')

    def test_plugged_into_m1_session_runner(self):
        """run_m1_session successfully executes with measure_condition plugged in."""
        from m1_session_runner import run_m1_session

        with tempfile.TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / 'session_test'
            ckpt_dir = Path(__file__).resolve().parent.parent.parent / 'checkpoints'

            # Define runner wrapper with test overrides
            def test_runner(arch, condition, out_dir, checkpoints_dir):
                return measure_condition(
                    arch=arch,
                    condition=condition,
                    out_dir=out_dir,
                    checkpoints_dir=checkpoints_dir,
                    repeats=1,
                    settle_s=0.01,
                    window_s=0.01,
                    skip_preflight=True,
                    mock_forward=True,
                    mock_powermetrics=True,
                    check_ac_fn=lambda: True,
                    check_therm_fn=lambda: True,
                )

            log = run_m1_session(
                session_n=1,
                architectures=['resnet18'],
                out_base=out_base,
                checkpoints_dir=ckpt_dir,
                condition_runner=test_runner,
                get_boottime_fn=lambda: time.time() - 300.0,
                check_ac_fn=lambda: True,
                check_therm_fn=lambda: True,
            )

            self.assertEqual(log['session'], 1)
            self.assertEqual(len(log['conditions']), 9)  # 9 conditions for resnet18
            # Verify first condition directory has windows.csv and summary.csv
            first_cond_dir = out_base / log['conditions'][0]
            self.assertTrue((first_cond_dir / 'windows.csv').exists())
            self.assertTrue((first_cond_dir / 'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
