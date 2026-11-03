"""Unit tests for M1 system guards (AC, Uptime, Thermal, Memory).

Tests all positive, boundary, and negative/failure cases.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from guards import (
    GuardError,
    check_ac_power_m1,
    check_thermal_nominal_m1,
    check_uptime_guard,
    compute_uptime,
    parse_boottime,
    parse_pmset_batt,
    parse_pmset_therm,
    parse_vm_stat,
)


class TestACGuard(unittest.TestCase):
    def test_ac_finishing_charge(self):
        sample = (
            "Now drawing from 'AC Power'\n"
            " -InternalBattery-0 (id=34865251)\t100%; finishing charge; 0:00 remaining present: true\n"
        )
        parsed = parse_pmset_batt(sample)
        self.assertTrue(parsed["is_ac"])
        self.assertEqual(parsed["battery_pct"], 100.0)
        self.assertEqual(parsed["battery_state"], "finishing charge")
        self.assertTrue(parsed["is_present"])
        self.assertTrue(check_ac_power_m1(sample))

    def test_ac_charging(self):
        sample = (
            "Now drawing from 'AC Power'\n"
            " -InternalBattery-0 (id=34865251)\t82%; charging; 0:42 remaining present: true\n"
        )
        parsed = parse_pmset_batt(sample)
        self.assertTrue(parsed["is_ac"])
        self.assertEqual(parsed["battery_pct"], 82.0)
        self.assertEqual(parsed["battery_state"], "charging")
        self.assertTrue(check_ac_power_m1(sample))

    def test_ac_charged(self):
        sample = (
            "Now drawing from 'AC Power'\n"
            " -InternalBattery-0 (id=34865251)\t100%; charged; 0:00 remaining present: true\n"
        )
        parsed = parse_pmset_batt(sample)
        self.assertTrue(parsed["is_ac"])
        self.assertTrue(check_ac_power_m1(sample))

    def test_battery_discharging_raises(self):
        sample = (
            "Now drawing from 'Battery Power'\n"
            " -InternalBattery-0 (id=34865251)\t95%; discharging; 5:10 remaining present: true\n"
        )
        parsed = parse_pmset_batt(sample)
        self.assertFalse(parsed["is_ac"])
        self.assertEqual(parsed["battery_state"], "discharging")
        with self.assertRaises(GuardError) as ctx:
            check_ac_power_m1(sample)
        self.assertIn("AC power is not connected", str(ctx.exception))

    def test_ac_but_discharging_raises(self):
        sample = (
            "Now drawing from 'AC Power'\n"
            " -InternalBattery-0 (id=34865251)\t75%; discharging; 2:10 remaining present: true\n"
        )
        with self.assertRaises(GuardError) as ctx:
            check_ac_power_m1(sample)
        self.assertIn("System is discharging", str(ctx.exception))

    def test_empty_or_malformed_raises(self):
        with self.assertRaises(GuardError):
            check_ac_power_m1("")
        with self.assertRaises(GuardError):
            check_ac_power_m1("Garbage output without power status")


class TestUptimeGuard(unittest.TestCase):
    def test_parse_boottime_darwin_format(self):
        sample = "{ sec = 1756405599, usec = 852783 } Sat Aug 29 02:26:39 2026"
        ts = parse_boottime(sample)
        self.assertAlmostEqual(ts, 1756405599.852783, places=6)

    def test_parse_boottime_real_mac_capture(self):
        # Captured from M1 Mac on 2026-11-03
        sample = "{ sec = 1787937999, usec = 349992 } Sat Aug 29 02:26:39 2026"
        ts = parse_boottime(sample)
        self.assertAlmostEqual(ts, 1787937999.349992, places=6)

    def test_parse_boottime_numeric(self):
        sample = "1756405599\n"
        self.assertEqual(parse_boottime(sample), 1756405599.0)

    def test_compute_uptime(self):
        boottime = 1000.0
        now = 1900.0
        uptime = compute_uptime(boottime, current_time=now)
        self.assertEqual(uptime, 900.0)

    def test_negative_uptime_raises(self):
        with self.assertRaises(GuardError):
            compute_uptime(2000.0, current_time=1000.0)

    def test_session_start_threshold(self):
        boottime = 10000.0
        # 20 min uptime -> PASS
        upt = check_uptime_guard(boottime, max_uptime_seconds=1800.0, current_time=10000.0 + 1200.0)
        self.assertEqual(upt, 1200.0)

        # 31 min uptime -> FAIL
        with self.assertRaises(GuardError) as ctx:
            check_uptime_guard(boottime, max_uptime_seconds=1800.0, current_time=10000.0 + 1860.0)
        self.assertIn("31.0 min (> 30.0 min)", str(ctx.exception))

    def test_per_condition_threshold(self):
        boottime = 10000.0
        # 120 min uptime -> PASS (< 180 min)
        upt = check_uptime_guard(boottime, max_uptime_seconds=10800.0, current_time=10000.0 + 7200.0, context="condition start")
        self.assertEqual(upt, 7200.0)

        # 185 min uptime -> FAIL (> 180 min)
        with self.assertRaises(GuardError) as ctx:
            check_uptime_guard(boottime, max_uptime_seconds=10800.0, current_time=10000.0 + 11100.0, context="condition start")
        self.assertIn("185.0 min (> 180.0 min)", str(ctx.exception))


class TestThermalGuard(unittest.TestCase):
    """A1 fail-closed contract: parse_pmset_therm itself raises GuardError for
    anything other than exactly the three expected nominal lines -- it no
    longer returns a dict with is_nominal=False for bad/unknown input. All
    sample strings below are SYNTHETIC test fixtures, not captured from any
    real machine (the one real capture is test_thermal_nominal_real_mac_capture,
    labelled as such, reusing the exact text recorded from the Mac elsewhere
    in this session)."""

    def test_nominal_output_synthetic(self):
        sample = (  # synthetic, matches the exact expected nominal form
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n"
            "Note: No CPU power status has been recorded\n"
        )
        parsed = parse_pmset_therm(sample)
        self.assertTrue(parsed["is_nominal"])
        self.assertTrue(check_thermal_nominal_m1(sample))

    def test_thermal_nominal_real_mac_capture(self):
        # Captured from the M1 Mac via `pmset -g therm` on 2026-11-03 (same
        # session; re-verified live immediately before the edge test in this
        # same conversation). Real data, not synthetic.
        sample = (
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n"
            "Note: No CPU power status has been recorded\n"
        )
        parsed = parse_pmset_therm(sample)
        self.assertTrue(parsed["is_nominal"])
        self.assertTrue(check_thermal_nominal_m1(sample))

    def test_thermal_warning_present_raises_synthetic(self):
        sample = (  # synthetic: a made-up non-nominal line, not a real capture
            "Thermal warning level has been recorded: 1\n"
            "Note: No performance warning level has been recorded\n"
        )
        with self.assertRaises(GuardError):
            parse_pmset_therm(sample)
        with self.assertRaises(GuardError):
            check_thermal_nominal_m1(sample)

    def test_perf_warning_present_raises_synthetic(self):
        sample = (  # synthetic
            "Note: No thermal warning level has been recorded\n"
            "Performance warning level has been recorded: 2\n"
        )
        with self.assertRaises(GuardError):
            parse_pmset_therm(sample)
        with self.assertRaises(GuardError):
            check_thermal_nominal_m1(sample)

    def test_throttled_pressure_line_raises_synthetic(self):
        sample = "Current pressure level: Serious\n"  # synthetic, format not observed for real
        with self.assertRaises(GuardError):
            parse_pmset_therm(sample)
        with self.assertRaises(GuardError):
            check_thermal_nominal_m1(sample)

    def test_extra_unexpected_line_raises_synthetic(self):
        # Fail-closed must also reject the nominal three lines PLUS anything
        # extra -- an unknown additional line should not be silently ignored.
        sample = (  # synthetic
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n"
            "Note: No CPU power status has been recorded\n"
            "Some new line this parser has never seen\n"
        )
        with self.assertRaises(GuardError):
            parse_pmset_therm(sample)

    def test_empty_raises_synthetic(self):
        with self.assertRaises(GuardError):
            parse_pmset_therm("")

    def test_garbage_raises_synthetic(self):
        with self.assertRaises(GuardError):
            parse_pmset_therm("completely unrecognized garbage output")


class TestMemoryTelemetry(unittest.TestCase):
    def test_vm_stat_m1_page_size(self):
        sample = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10000.
Pages active:                            200000.
Pages inactive:                          100000.
Pages speculative:                         5000.
Pages throttled:                              0.
Pages wired down:                         50000.
Pages purgeable:                           2000.
"Translation faults":                 123456789.
Pages copy-on-write:                    1234567.
Pages zero filled:                     23456789.
Pages reactivated:                         1234.
Pages purged:                              5678.
File-backed pages:                        40000.
Anonymous pages:                         160000.
Pages stored in compressor:               10000.
Pages occupied by compressor:              4000.
"""
        parsed = parse_vm_stat(sample)
        self.assertEqual(parsed["page_size_bytes"], 16384)
        self.assertEqual(parsed["pages_free"], 10000)
        self.assertEqual(parsed["pages_speculative"], 5000)
        # This synthetic sample has no Swapins/Swapouts/Compressions/Decompressions
        # lines at all -- confirms they default to 0 rather than raising/missing.
        self.assertEqual(parsed["swapins"], 0)
        self.assertEqual(parsed["swapouts"], 0)
        self.assertEqual(parsed["compressions"], 0)
        self.assertEqual(parsed["decompressions"], 0)

        # 16384 bytes = 0.015625 MB per page
        # free_mb = 15000 * 16384 / (1024*1024) = 234.375 MB
        self.assertAlmostEqual(parsed["free_mb"], 234.375, places=3)
        # active_mb = 200000 * 16384 / (1024*1024) = 3125.0 MB
        self.assertAlmostEqual(parsed["active_mb"], 3125.0, places=3)
        # compressed_mb = 4000 * 16384 / (1024*1024) = 62.5 MB
        self.assertAlmostEqual(parsed["compressed_mb"], 62.5, places=3)

    def test_vm_stat_real_mac_capture(self):
        # Captured from M1 Mac on 2026-11-03
        sample = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     3737.
Pages active:                                 137250.
Pages inactive:                               135189.
Pages speculative:                               869.
Pages throttled:                                   0.
Pages wired down:                              89676.
Pages purgeable:                                2797.
"Translation faults":                     2230437913.
Pages copy-on-write:                        36240688.
Pages zero filled:                        1200717237.
Pages reactivated:                         555746244.
Pages purged:                              49512373.
File-backed pages:                             83513.
Anonymous pages:                              189795.
Pages stored in compressor:                   371849.
Pages occupied by compressor:                 121664.
Decompressions:                            624072172.
Compressions:                              657415860.
Pageins:                                    85640546.
Pageouts:                                    1970594.
Swapins:                                     3247646.
Swapouts:                                    3881501.
"""
        parsed = parse_vm_stat(sample)
        self.assertEqual(parsed["page_size_bytes"], 16384)
        self.assertEqual(parsed["pages_free"], 3737)
        self.assertEqual(parsed["pages_speculative"], 869)
        self.assertEqual(parsed["pages_active"], 137250)
        self.assertEqual(parsed["pages_occupied_by_compressor"], 121664)
        self.assertEqual(parsed["decompressions"], 624072172)
        self.assertEqual(parsed["compressions"], 657415860)
        self.assertEqual(parsed["swapins"], 3247646)
        self.assertEqual(parsed["swapouts"], 3881501)
        self.assertAlmostEqual(parsed["free_mb"], 71.96875, places=4)
        self.assertAlmostEqual(parsed["compressed_mb"], 1901.0, places=4)
        self.assertAlmostEqual(parsed["available_mb"], 2184.296875, places=4)


if __name__ == "__main__":
    unittest.main()
