"""Unit tests for powermetrics_parser.py and synthetic overlap-splitting.

Specs: scripts/m1/HARNESS_PLAN.md sections 4 and 5.
"""
import copy
import re
import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from powermetrics_parser import (
    MalformedPowermetricsError,
    parse_powermetrics_file,
    parse_powermetrics_header,
    split_energy_by_overlap,
)

SAMPLE_FILE = Path(__file__).parent / "smoketest_20261103T123851Z_70699.txt"


class TestPowermetricsParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SAMPLE_FILE.exists():
            raise FileNotFoundError(f"Missing smoketest sample file: {SAMPLE_FILE}")
        with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
            cls.raw_text = f.read()
        cls.parsed_run = parse_powermetrics_file(SAMPLE_FILE)

    def test_sample_count_matches_count_arg(self):
        """1. Asserts exactly 3 sample blocks are parsed (COUNT=3)."""
        self.assertEqual(len(self.parsed_run), 3)

    def test_cpu_power_values_match_known_sample(self):
        """2. Asserts the three cpu_power_mw values are exactly 67, 84, 57."""
        cpu_powers = [s["cpu_power_mw"] for s in self.parsed_run]
        self.assertEqual(cpu_powers, [67, 84, 57])

    def test_elapsed_ms_not_equal_to_requested_interval(self):
        """3. Asserts none of elapsed_ms equals 1000.0 exactly, but all near it."""
        for s in self.parsed_run:
            self.assertNotEqual(s["elapsed_ms"], 1000.0)
            self.assertAlmostEqual(s["elapsed_ms"], 1000.0, delta=25.0)

    def test_thermal_pressure_nominal_on_known_sample(self):
        """4. Asserts all three thermal_pressure values parse as 'Nominal'."""
        for s in self.parsed_run:
            self.assertEqual(s["thermal_pressure"], "Nominal")

    def test_timestamp_parses_with_utc_offset(self):
        """5. Asserts parsed timestamp for sample 1 carries +0900 offset."""
        s1 = self.parsed_run[0]
        ts = s1["timestamp"]
        self.assertIsInstance(ts, datetime)
        self.assertIsNotNone(ts.tzinfo)
        offset = ts.utcoffset()
        self.assertEqual(offset, timedelta(hours=9))
        self.assertEqual(ts.year, 2026)
        self.assertEqual(ts.month, 11)
        self.assertEqual(ts.day, 3)
        self.assertEqual(ts.hour, 21)
        self.assertEqual(ts.minute, 38)
        self.assertEqual(ts.second, 52)

    def test_file_header_fields_present(self):
        """6. Asserts Machine model, OS version, Boot time are parsed from file header."""
        header = parse_powermetrics_header(SAMPLE_FILE)
        self.assertEqual(header["machine_model"], "MacBookAir10,1")
        self.assertEqual(header["os_version"], "25G83")
        self.assertEqual(header["boot_time"], "Sat Aug 29 02:26:39 2026")
        # Also check on run.header
        self.assertEqual(self.parsed_run.header["machine_model"], "MacBookAir10,1")
        self.assertEqual(self.parsed_run.header["os_version"], "25G83")
        self.assertEqual(self.parsed_run.header["boot_time"], "Sat Aug 29 02:26:39 2026")

    def test_malformed_block_raises_not_silently_skips(self):
        """7. Truncated/corrupted copy missing CPU Power line must raise rather than skip."""
        corrupted_text = re.sub(r'CPU Power: \d+ mW\n', '', self.raw_text, count=1)
        self.assertNotEqual(corrupted_text, self.raw_text)
        with self.assertRaises(MalformedPowermetricsError):
            parse_powermetrics_file(corrupted_text)

        # Missing thermal line
        corrupted_therm = re.sub(r'Current pressure level: Nominal\n', '', self.raw_text, count=1)
        with self.assertRaises(MalformedPowermetricsError):
            parse_powermetrics_file(corrupted_therm)

        # Missing header line
        corrupted_hdr = re.sub(r'Machine model:.*?\n', '', self.raw_text, count=1)
        with self.assertRaises(MalformedPowermetricsError):
            parse_powermetrics_file(corrupted_hdr)


class TestSyntheticOverlapSplitting(unittest.TestCase):
    def test_analytic_power_step_and_conservation(self):
        """Test on synthetic streams with an analytically known power step.
        
        Requires:
        - Total energy conservation to 1e-9 relative
        - Exact analytic window energies
        - Case with a sample straddling a window boundary
        - Case with samples entirely outside every window
        """
        # Synthetic stream: 10 samples of 1.0s each across [0.0, 10.0]s
        # Step: idle at 100 mW, active at 1000 mW between [3.0, 7.0]s
        samples = [
            {"start": 0.0, "end": 1.0, "power": 100.0},    # sample 0
            {"start": 1.0, "end": 2.0, "power": 100.0},    # sample 1
            {"start": 2.0, "end": 3.0, "power": 100.0},    # sample 2
            {"start": 3.0, "end": 4.0, "power": 1000.0},   # sample 3 (step up)
            {"start": 4.0, "end": 5.0, "power": 1000.0},   # sample 4
            {"start": 5.0, "end": 6.0, "power": 1000.0},   # sample 5
            {"start": 6.0, "end": 7.0, "power": 1000.0},   # sample 6
            {"start": 7.0, "end": 8.0, "power": 100.0},    # sample 7 (step down)
            {"start": 8.0, "end": 9.0, "power": 100.0},    # sample 8
            {"start": 9.0, "end": 10.0, "power": 100.0},   # sample 9 (entirely outside)
        ]
        # Total energy in stream:
        # 3 * 100 + 4 * 1000 + 3 * 100 = 300 + 4000 + 300 = 4600.0 mJ

        windows = [
            {"name": "baseline", "start": 0.5, "end": 2.5},
            {"name": "load_phase", "start": 2.5, "end": 6.2},
            {"name": "post_phase", "start": 6.2, "end": 8.5},
        ]
        # Notice:
        # Sample 6 [6.0, 7.0] straddles the boundary at t=6.2 between load_phase and post_phase!
        # - overlap with load_phase: [6.0, 6.2] = 0.2s * 1000 mW = 200.0 mJ
        # - overlap with post_phase: [6.2, 7.0] = 0.8s * 1000 mW = 800.0 mJ
        # Sample 9 [9.0, 10.0] is ENTIRELY outside every window.
        #
        # Analytic expectations:
        # Window "baseline" [0.5, 2.5]:
        # - Sample 0 [0.5, 1.0]: 0.5s * 100 mW = 50.0 mJ
        # - Sample 1 [1.0, 2.0]: 1.0s * 100 mW = 100.0 mJ
        # - Sample 2 [2.0, 2.5]: 0.5s * 100 mW = 50.0 mJ
        # Total baseline energy = 200.0 mJ.
        #
        # Window "load_phase" [2.5, 6.2]:
        # - Sample 2 [2.5, 3.0]: 0.5s * 100 mW = 50.0 mJ
        # - Sample 3 [3.0, 4.0]: 1.0s * 1000 mW = 1000.0 mJ
        # - Sample 4 [4.0, 5.0]: 1.0s * 1000 mW = 1000.0 mJ
        # - Sample 5 [5.0, 6.0]: 1.0s * 1000 mW = 1000.0 mJ
        # - Sample 6 [6.0, 6.2]: 0.2s * 1000 mW = 200.0 mJ
        # Total load_phase energy = 50 + 1000 + 1000 + 1000 + 200 = 3250.0 mJ.
        #
        # Window "post_phase" [6.2, 8.5]:
        # - Sample 6 [6.2, 7.0]: 0.8s * 1000 mW = 800.0 mJ
        # - Sample 7 [7.0, 8.0]: 1.0s * 100 mW = 100.0 mJ
        # - Sample 8 [8.0, 8.5]: 0.5s * 100 mW = 50.0 mJ
        # Total post_phase energy = 800 + 100 + 50 = 950.0 mJ.
        #
        # Outside energy:
        # - Sample 0 [0.0, 0.5]: 0.5s * 100 = 50.0 mJ
        # - Sample 8 [8.5, 9.0]: 0.5s * 100 = 50.0 mJ
        # - Sample 9 [9.0, 10.0]: 1.0s * 100 = 100.0 mJ (entirely outside)
        # Total outside energy = 50 + 50 + 100 = 200.0 mJ.

        res = split_energy_by_overlap(samples, windows)

        # 1. Exact analytic window energies
        self.assertAlmostEqual(res["window_energies"]["baseline"], 200.0, places=9)
        self.assertAlmostEqual(res["window_energies"]["load_phase"], 3250.0, places=9)
        self.assertAlmostEqual(res["window_energies"]["post_phase"], 950.0, places=9)

        # 2. Exact analytic outside energy
        self.assertAlmostEqual(res["outside_energy"], 200.0, places=9)

        # 3. Total energy conservation to 1e-9 relative
        expected_total = 4600.0
        self.assertAlmostEqual(res["total_sample_energy"], expected_total, places=9)
        self.assertAlmostEqual(res["total_accounted_energy"], expected_total, places=9)
        self.assertLess(res["relative_discrepancy"], 1e-9)

    def test_nonuniform_intervals_straddling(self):
        """Test with realistic non-uniform sample intervals (e.g. ~415 ms)."""
        samples = [
            {"start": 0.000, "end": 0.405, "power": 50.0},
            {"start": 0.405, "end": 0.820, "power": 50.0},
            {"start": 0.820, "end": 1.235, "power": 5000.0},
            {"start": 1.235, "end": 1.650, "power": 5000.0},
            {"start": 1.650, "end": 2.065, "power": 50.0},
        ]
        # Window starts at 0.500 (straddles sample 1) and ends at 1.500 (straddles sample 3)
        windows = [
            {"name": "w1", "start": 0.500, "end": 1.500}
        ]
        res = split_energy_by_overlap(samples, windows)
        # Overlap breakdown:
        # Sample 0: 0 overlap
        # Sample 1: [0.500, 0.820] = 0.320s * 50 = 16.0
        # Sample 2: [0.820, 1.235] = 0.415s * 5000 = 2075.0
        # Sample 3: [1.235, 1.500] = 0.265s * 5000 = 1325.0
        # Sample 4: 0 overlap
        # Total w1 = 16.0 + 2075.0 + 1325.0 = 3416.0
        self.assertAlmostEqual(res["window_energies"]["w1"], 3416.0, places=7)
        self.assertLess(res["relative_discrepancy"], 1e-9)


if __name__ == "__main__":
    unittest.main()
