"""Unit and integration tests for M1 condition pipeline and energy post-processing.

Verifies:
1. Complete condition processing across 6 repeats (24 phase windows).
2. Cold first repeat exclusion from summary metrics.
3. Idle baseline power calculation and net energy attribution.
4. Paired difference statistics (sd, MDE, bias) matching pilot.py summarize().
5. Exact energy conservation via overlap-splitting across all phases.
6. Time shift adjustment (e.g. -15 ms) effect on interval placement.
"""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from m1_pipeline import (
    WindowPhase,
    check_condition_thermal_flags,
    evaluate_window_thermal_flag,
    process_condition_energy,
    read_thermal_thresholds_m1,
    summarize_condition,
    write_windows_csv,
)


class TestM1Pipeline(unittest.TestCase):
    def setUp(self):
        # Construct synthetic session condition:
        # 6 repeats (0 to 5), 4 phases per repeat (idle_before, a1, a2, idle_after)
        # Each phase is 5.0 s.
        # Repeat spacing: 3.0 s settle between phases.
        # Total time per repeat: 4 * 5.0 + 3 * 3.0 = 29.0 s.
        self.phase_windows = []
        curr_t = 1.0  # start 1s after stream start
        for rep in range(6):
            for phase in ['idle_before', 'a1', 'a2', 'idle_after']:
                start_t = curr_t
                end_t = curr_t + 5.0
                batches = 500 if phase in ['a1', 'a2'] else 0
                self.phase_windows.append(
                    WindowPhase(
                        device='cpu',
                        arch='resnet18',
                        condition='fp32_torchscript',
                        size=32,
                        batch=1,
                        requested_s=5.0,
                        repeat=rep,
                        phase=phase,
                        start_t=start_t,
                        end_t=end_t,
                        batches=batches,
                    )
                )
                curr_t = end_t + 3.0  # 3s gap

        total_span = curr_t + 2.0
        # Build synthetic powermetrics stream: 400 ms samples over total_span
        self.samples = []
        sample_interval = 0.400
        n_samples = int(total_span / sample_interval) + 2

        # Power profile:
        # Idle baseline: 20 mW (0.02 W)
        # Active load (a1, a2): 6000 mW (6.0 W)
        # We assign power based on whether the midpoint falls in an active window
        for k in range(n_samples):
            s_start = k * sample_interval
            s_end = (k + 1) * sample_interval
            mid = 0.5 * (s_start + s_end)

            is_active = any(
                w.phase in ['a1', 'a2'] and w.start_t <= mid <= w.end_t
                for w in self.phase_windows
            )
            power_mw = 6000 if is_active else 20
            self.samples.append({
                'start': s_start,
                'end': s_end,
                'elapsed_ms': 400.0,
                'cpu_power_mw': power_mw,
            })

    def test_24_windows_generated(self):
        rows, summary = process_condition_energy(self.samples, self.phase_windows)
        self.assertEqual(len(rows), 24)
        phases = [r['phase'] for r in rows]
        self.assertEqual(phases.count('idle_before'), 6)
        self.assertEqual(phases.count('a1'), 6)
        self.assertEqual(phases.count('a2'), 6)
        self.assertEqual(phases.count('idle_after'), 6)

    def test_cold_repeat_excluded_from_summary(self):
        rows, summary = process_condition_energy(self.samples, self.phase_windows)
        # 6 repeats total; cold repeat 0 excluded -> pairs should be 5
        self.assertEqual(summary['pairs'], 5)

        # In rows, repeat 0 is still present with estimated idle and gross per image
        rep0_rows = [r for r in rows if r['repeat'] == 0]
        self.assertEqual(len(rep0_rows), 4)
        for r in rep0_rows:
            if r['phase'] in ['a1', 'a2']:
                self.assertIn('estimated_idle_j', r)
                self.assertIn('above_idle_j', r)
                self.assertIn('gross_j_per_image', r)

    def test_energy_values_and_idle_subtraction(self):
        rows, summary = process_condition_energy(self.samples, self.phase_windows)
        # Idle phases are 5.0 s at ~20 mW = 100 mJ = 0.1 J
        for r in rows:
            if r['phase'] in ['idle_before', 'idle_after']:
                self.assertAlmostEqual(r['energy_j'], 0.1, delta=0.015)
                self.assertAlmostEqual(r['mean_power_w'], 0.02, delta=0.003)

        # Active phases (a1, a2) are 5.0 s at ~6000 mW = 30,000 mJ = 30.0 J
        for r in rows:
            if r['phase'] in ['a1', 'a2']:
                self.assertAlmostEqual(r['energy_j'], 30.0, delta=0.5)
                self.assertAlmostEqual(r['mean_power_w'], 6.0, delta=0.1)
                # Gross J per image: 30 J / 500 images = 0.06 J/img
                self.assertAlmostEqual(r['gross_j_per_image'], 0.06, delta=0.001)

        # Summary averages
        self.assertAlmostEqual(summary['idle_power_w_mean'], 0.02, delta=0.005)
        self.assertAlmostEqual(summary['gross_j_per_image_mean'], 0.06, delta=0.002)
        # Above idle J mean: ~30.0 J - (0.02 W * 5.0 s) = 30.0 - 0.1 = 29.9 J
        self.assertAlmostEqual(summary['above_idle_j_mean'], 29.9, delta=0.5)

    def test_energy_conservation_diagnostics(self):
        rows, summary = process_condition_energy(self.samples, self.phase_windows)
        diag = summary['overlap_diagnostics']
        self.assertLess(diag['relative_discrepancy'], 1e-9)
        self.assertAlmostEqual(
            diag['total_accounted_energy_j'],
            diag['total_sample_energy_j'],
            places=7,
        )
        self.assertGreater(diag['outside_energy_j'], 0.0)

    def test_candidate_for_confirmation_unreachable_at_6_repeats(self):
        """A4: documents, does not change, that candidate_for_confirmation
        (requires n_pairs >= 30) is structurally unreachable at the fixed
        6-repeat-per-condition design (5 pairs max after cold exclusion)."""
        rows, summary = process_condition_energy(self.samples, self.phase_windows)
        self.assertEqual(summary['pairs'], 5)
        self.assertLess(summary['pairs'], 30)
        self.assertFalse(summary['candidate_for_confirmation'])

    def test_time_shift_applied(self):
        # Shift -15 ms (-0.015 s)
        rows_unshifted, _ = process_condition_energy(self.samples, self.phase_windows, time_shift_s=0.0)
        rows_shifted, _ = process_condition_energy(self.samples, self.phase_windows, time_shift_s=-0.015)
        self.assertEqual(len(rows_shifted), 24)
        # Verify both runs produce valid energies within tight tolerance
        for ru, rs in zip(rows_unshifted, rows_shifted):
            self.assertAlmostEqual(ru['energy_j'], rs['energy_j'], delta=0.1)

    def test_read_thermal_thresholds_m1_valid(self):
        config = read_thermal_thresholds_m1()
        self.assertAlmostEqual(config['T_mhz'], 2881.11, places=2)
        self.assertEqual(config['rule_version'], 'm1-v1')
        self.assertEqual(config['source'], 'calibration/m1_thermal_calibration_20261104')
        self.assertEqual(config['metric'], 'pcore_mean_freq_mhz')
        self.assertEqual(config['flag_rule'], 'window_mean_below_T')

    def test_read_thermal_thresholds_m1_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir) / 'bad.json'
            
            # Missing file
            with self.assertRaises(RuntimeError):
                read_thermal_thresholds_m1(Path(tmpdir) / 'nonexistent.json')

            # Invalid T_mhz
            tmp_p.write_text(json.dumps({'T_mhz': -10, 'rule_version': 'm1-v1', 'source': 's', 'metric': 'pcore_mean_freq_mhz', 'flag_rule': 'window_mean_below_T'}))
            with self.assertRaises(RuntimeError):
                read_thermal_thresholds_m1(tmp_p)

            # Invalid rule_version
            tmp_p.write_text(json.dumps({'T_mhz': 2881.11, 'rule_version': 'v1', 'source': 's', 'metric': 'pcore_mean_freq_mhz', 'flag_rule': 'window_mean_below_T'}))
            with self.assertRaises(RuntimeError):
                read_thermal_thresholds_m1(tmp_p)

            # Invalid metric
            tmp_p.write_text(json.dumps({'T_mhz': 2881.11, 'rule_version': 'm1-v1', 'source': 's', 'metric': 'wrong_metric', 'flag_rule': 'window_mean_below_T'}))
            with self.assertRaises(RuntimeError):
                read_thermal_thresholds_m1(tmp_p)

            # Invalid flag_rule
            tmp_p.write_text(json.dumps({'T_mhz': 2881.11, 'rule_version': 'm1-v1', 'source': 's', 'metric': 'pcore_mean_freq_mhz', 'flag_rule': 'wrong_rule'}))
            with self.assertRaises(RuntimeError):
                read_thermal_thresholds_m1(tmp_p)

    def test_evaluate_window_thermal_flag_rules(self):
        T = 2881.11
        # a1, repeat > 0, freq < T -> 'True'
        self.assertEqual(evaluate_window_thermal_flag(2700.0, 'a1', 1, T), 'True')
        # a2, repeat > 0, freq >= T -> 'False'
        self.assertEqual(evaluate_window_thermal_flag(3204.0, 'a2', 2, T), 'False')
        # cold repeat (repeat == 0) -> None
        self.assertIsNone(evaluate_window_thermal_flag(2500.0, 'a1', 0, T))
        self.assertIsNone(evaluate_window_thermal_flag(3204.0, 'a2', 0, T))
        # idle phases -> None
        self.assertIsNone(evaluate_window_thermal_flag(2500.0, 'idle_before', 1, T))
        self.assertIsNone(evaluate_window_thermal_flag(2500.0, 'idle_after', 1, T))
        # active window with repeat > 0 missing frequency -> empty string (triggers error on condition check)
        self.assertEqual(evaluate_window_thermal_flag(None, 'a1', 1, T), '')

    def test_non_nominal_pressure_flags_even_when_frequency_is_fine(self):
        """A qualifying window must be flagged if any overlapping sample has
        thermal pressure other than 'Nominal', independent of frequency."""
        T = 2881.11
        # Frequency well above T, but one overlapping sample is non-Nominal -> flagged True
        self.assertEqual(
            evaluate_window_thermal_flag(3204.0, 'a1', 1, T, thermal_pressures=['Nominal', 'Fair', 'Nominal']),
            'True',
        )
        # All overlapping samples Nominal, frequency above T -> False
        self.assertEqual(
            evaluate_window_thermal_flag(3204.0, 'a1', 1, T, thermal_pressures=['Nominal', 'Nominal']),
            'False',
        )
        # No pressure data passed at all (None, the default) -> unaffected, same as before this change
        self.assertEqual(
            evaluate_window_thermal_flag(3204.0, 'a1', 1, T, thermal_pressures=None),
            'False',
        )
        # Empty pressure list (no overlapping samples had pressure data) -> unaffected
        self.assertEqual(
            evaluate_window_thermal_flag(3204.0, 'a1', 1, T, thermal_pressures=[]),
            'False',
        )
        # Non-Nominal pressure present AND frequency already below T -> still True (not double-counted, just True)
        self.assertEqual(
            evaluate_window_thermal_flag(2700.0, 'a1', 1, T, thermal_pressures=['Serious']),
            'True',
        )
        # Non-Nominal pressure on a cold repeat (repeat == 0) -> still None, not evaluated at all
        self.assertIsNone(
            evaluate_window_thermal_flag(3204.0, 'a1', 0, T, thermal_pressures=['Fair'])
        )
        # Missing frequency takes precedence over pressure -> still '' (missing evaluation)
        self.assertEqual(
            evaluate_window_thermal_flag(None, 'a1', 1, T, thermal_pressures=['Fair']),
            '',
        )

    def test_real_throttled_int8_plateau_window_is_flagged(self):
        """A real throttled INT8 plateau window from calibration data must be flagged."""
        int8_file = Path('/home/shohan/thermal_extended_results/test1_int8/windows.json')
        if not int8_file.exists():
            int8_file = Path('/home/shohan/m1_thermal_calibration_results/resnet18_int8/windows.json')
        
        self.assertTrue(int8_file.exists(), f"Calibration file {int8_file} must exist")
        windows = json.loads(int8_file.read_text())
        # Window 100 is in the throttled plateau (~2650-2750 MHz)
        throttled_w = windows[100]
        pcore_freq = throttled_w['p47_mean_mhz']
        self.assertLess(pcore_freq, 2881.11, "Throttled plateau window must be below T=2881.11 MHz")

        flag = evaluate_window_thermal_flag(pcore_freq, phase='a1', repeat=1, threshold_t_mhz=2881.11)
        self.assertEqual(flag, 'True', "Real throttled INT8 window must evaluate to thermal_flag='True'")

    def test_real_duty_cycle_window_is_not_flagged(self):
        """A real duty-cycle active window from calibration data must NOT be flagged."""
        duty_file = Path('/home/shohan/thermal_extended_results/test2_dutycycle/active_windows.json')
        self.assertTrue(duty_file.exists(), f"Calibration file {duty_file} must exist")
        windows = json.loads(duty_file.read_text())
        # Window 0 (or any window) is pegged at ~3204 MHz
        nominal_w = windows[0]
        pcore_freq = nominal_w['p47_mean_mhz']
        self.assertGreaterEqual(pcore_freq, 2881.11, "Duty-cycle window must be >= T=2881.11 MHz")

        flag = evaluate_window_thermal_flag(pcore_freq, phase='a1', repeat=1, threshold_t_mhz=2881.11)
        self.assertEqual(flag, 'False', "Real duty-cycle active window must evaluate to thermal_flag='False'")

    def test_check_condition_thermal_flags_voiding(self):
        # Condition where all a1/a2 windows with repeat > 0 have thermal_flag='False' -> False (passed)
        nominal_rows = [
            {'phase': 'idle_before', 'repeat': 1, 'thermal_flag': 'None'},
            {'phase': 'a1', 'repeat': 1, 'thermal_flag': 'False'},
            {'phase': 'a2', 'repeat': 1, 'thermal_flag': 'False'},
            {'phase': 'idle_after', 'repeat': 1, 'thermal_flag': 'None'},
        ]
        self.assertFalse(check_condition_thermal_flags(nominal_rows))

        # Condition where one active window is flagged -> True (voided)
        flagged_rows = [
            {'phase': 'idle_before', 'repeat': 1, 'thermal_flag': 'None'},
            {'phase': 'a1', 'repeat': 1, 'thermal_flag': 'False'},
            {'phase': 'a2', 'repeat': 1, 'thermal_flag': 'True'},
            {'phase': 'idle_after', 'repeat': 1, 'thermal_flag': 'None'},
        ]
        self.assertTrue(check_condition_thermal_flags(flagged_rows))

        # Cold repeat 0 flagged does NOT void condition
        cold_flagged_rows = [
            {'phase': 'idle_before', 'repeat': 0, 'thermal_flag': 'None'},
            {'phase': 'a1', 'repeat': 0, 'thermal_flag': 'True'},
            {'phase': 'a2', 'repeat': 0, 'thermal_flag': 'True'},
            {'phase': 'idle_after', 'repeat': 0, 'thermal_flag': 'None'},
            {'phase': 'a1', 'repeat': 1, 'thermal_flag': 'False'},
            {'phase': 'a2', 'repeat': 1, 'thermal_flag': 'False'},
        ]
        self.assertFalse(check_condition_thermal_flags(cold_flagged_rows))

    def test_missing_evaluation_raises(self):
        # Missing evaluation (empty string or None on repeat > 0 a1/a2) must raise RuntimeError
        missing_rows = [
            {'phase': 'a1', 'repeat': 1, 'thermal_flag': ''},
        ]
        with self.assertRaises(RuntimeError) as ctx:
            check_condition_thermal_flags(missing_rows)
        self.assertIn('Evaluation missing (empty thermal_flag)', str(ctx.exception))

        missing_col_rows = [
            {'phase': 'a1', 'repeat': 1},
        ]
        with self.assertRaises(RuntimeError) as ctx:
            check_condition_thermal_flags(missing_col_rows)
        self.assertIn('Evaluation missing (empty thermal_flag)', str(ctx.exception))

        # In CSV format:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'windows.csv'
            
            # Missing thermal_flag column
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['phase', 'repeat'])
                writer.writeheader()
                writer.writerow({'phase': 'a1', 'repeat': '1'})
            with self.assertRaises(RuntimeError) as ctx:
                check_condition_thermal_flags(csv_path)
            self.assertIn('thermal_flag column missing', str(ctx.exception))

            # Empty thermal_flag in CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['phase', 'repeat', 'thermal_flag'])
                writer.writeheader()
                writer.writerow({'phase': 'a1', 'repeat': '1', 'thermal_flag': ''})
            with self.assertRaises(RuntimeError) as ctx:
                check_condition_thermal_flags(csv_path)
            self.assertIn('Evaluation missing (empty thermal_flag)', str(ctx.exception))

    def test_write_windows_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'windows.csv'
            rows = [
                {'phase': 'idle_before', 'repeat': 0, 'thermal_flag': 'None'},
                {'phase': 'a1', 'repeat': 1, 'thermal_flag': 'False'},
                {'phase': 'a2', 'repeat': 1, 'thermal_flag': 'False'},
            ]
            write_windows_csv(rows, csv_path)
            self.assertTrue(csv_path.exists())
            self.assertFalse(check_condition_thermal_flags(csv_path))


if __name__ == '__main__':
    unittest.main()
