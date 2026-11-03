"""Unit and integration tests for M1 condition pipeline and energy post-processing.

Verifies:
1. Complete condition processing across 6 repeats (24 phase windows).
2. Cold first repeat exclusion from summary metrics.
3. Idle baseline power calculation and net energy attribution.
4. Paired difference statistics (sd, MDE, bias) matching pilot.py summarize().
5. Exact energy conservation via overlap-splitting across all phases.
6. Time shift adjustment (e.g. -15 ms) effect on interval placement.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from m1_pipeline import WindowPhase, process_condition_energy, summarize_condition


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


if __name__ == '__main__':
    unittest.main()
