#!/usr/bin/env python3
"""Unit tests for repeat-level energy extraction, averaging, and A11 throttle exclusion."""

import unittest
from scripts.repeat_level_energy import (
    compute_condition_value,
    extract_repeat_values,
    audit_condition_throttles,
)


def _make_window_row(
    phase: str,
    repeat: int,
    energy_j: float = 100.0,
    batches: int = 1000,
    batch: int = 1,
    throttle_delta: int = 0,
    cpu_psys_energy_j: float = 200.0,
    above_idle_j: float = 80.0,
) -> dict[str, str]:
    return {
        "device": "cpu",
        "size": "32",
        "batch": str(batch),
        "requested_s": "5.0",
        "repeat": str(repeat),
        "phase": phase,
        "energy_j": str(energy_j),
        "batches": str(batches),
        "cpu_psys_energy_j": str(cpu_psys_energy_j),
        "above_idle_j": str(above_idle_j),
        "thermal_pkg_throttle_count_delta": str(throttle_delta),
    }


def _make_repeat_rows(
    repeat: int,
    a1_energy: float,
    a2_energy: float,
    a1_batches: int = 1000,
    a2_batches: int = 1000,
    a1_throttle: int = 0,
    a2_throttle: int = 0,
    a1_psys: float = 200.0,
    a2_psys: float = 200.0,
    a1_above: float = 80.0,
    a2_above: float = 80.0,
) -> list[dict[str, str]]:
    return [
        _make_window_row("idle_before", repeat, energy_j=30.0, batches=0),
        _make_window_row(
            "a1",
            repeat,
            energy_j=a1_energy,
            batches=a1_batches,
            throttle_delta=a1_throttle,
            cpu_psys_energy_j=a1_psys,
            above_idle_j=a1_above,
        ),
        _make_window_row(
            "a2",
            repeat,
            energy_j=a2_energy,
            batches=a2_batches,
            throttle_delta=a2_throttle,
            cpu_psys_energy_j=a2_psys,
            above_idle_j=a2_above,
        ),
        _make_window_row("idle_after", repeat, energy_j=30.0, batches=0),
    ]


class RepeatLevelEnergyTests(unittest.TestCase):
    def test_cold_repeat_exclusion(self):
        """Repeat 0 (cold) must be excluded when multiple repeats exist."""
        rows = []
        # Repeat 0 (cold): per-image = 1.0 (mean of 1000/1000)
        rows.extend(_make_repeat_rows(0, a1_energy=1000.0, a2_energy=1000.0))
        # Repeat 1 (warm): per-image = 0.2 (mean of 200/1000)
        rows.extend(_make_repeat_rows(1, a1_energy=200.0, a2_energy=200.0))
        # Repeat 2 (warm): per-image = 0.4 (mean of 400/1000)
        rows.extend(_make_repeat_rows(2, a1_energy=400.0, a2_energy=400.0))

        repeat_vals = extract_repeat_values(rows, column="energy_j")
        self.assertEqual(len(repeat_vals), 2)
        rep_ids = [r[0] for r in repeat_vals]
        self.assertEqual(rep_ids, [1, 2])
        self.assertAlmostEqual(repeat_vals[0][1], 0.2)
        self.assertAlmostEqual(repeat_vals[1][1], 0.4)

        cond_val = compute_condition_value(rows, column="energy_j")
        self.assertAlmostEqual(cond_val, 0.3)

    def test_a1_a2_averaging(self):
        """Each repeat value is the mean of active phases a1 and a2 per-image energy."""
        rows = []
        # Cold repeat 0
        rows.extend(_make_repeat_rows(0, a1_energy=100.0, a2_energy=100.0))
        # Warm repeat 1:
        # a1: 150 J / 1000 images = 0.15 J/image
        # a2: 250 J / 1000 images = 0.25 J/image
        # repeat mean: (0.15 + 0.25) / 2 = 0.20 J/image
        rows.extend(_make_repeat_rows(1, a1_energy=150.0, a2_energy=250.0))
        # Warm repeat 2:
        # a1: 300 J / 1000 images = 0.30 J/image
        # a2: 500 J / 1000 images = 0.50 J/image
        # repeat mean: (0.30 + 0.50) / 2 = 0.40 J/image
        rows.extend(_make_repeat_rows(2, a1_energy=300.0, a2_energy=500.0))

        repeat_vals = extract_repeat_values(rows, column="energy_j")
        self.assertEqual(len(repeat_vals), 2)
        self.assertAlmostEqual(repeat_vals[0][1], 0.20)
        self.assertAlmostEqual(repeat_vals[1][1], 0.40)

        cond_val = compute_condition_value(rows, column="energy_j")
        self.assertAlmostEqual(cond_val, 0.30)

    def test_a11_throttle_exclusion(self):
        """Under A11, any warm repeat with thermal_pkg_throttle_count_delta > 0 in a1 or a2 is excluded."""
        rows = []
        rows.extend(_make_repeat_rows(0, a1_energy=100.0, a2_energy=100.0))  # cold
        # Repeat 1: unthrottled (mean 0.10)
        rows.extend(_make_repeat_rows(1, a1_energy=100.0, a2_energy=100.0, a1_throttle=0, a2_throttle=0))
        # Repeat 2: a1 throttled (mean 0.20)
        rows.extend(_make_repeat_rows(2, a1_energy=200.0, a2_energy=200.0, a1_throttle=1, a2_throttle=0))
        # Repeat 3: a2 throttled (mean 0.30)
        rows.extend(_make_repeat_rows(3, a1_energy=300.0, a2_energy=300.0, a1_throttle=0, a2_throttle=2))
        # Repeat 4: unthrottled (mean 0.40)
        rows.extend(_make_repeat_rows(4, a1_energy=400.0, a2_energy=400.0, a1_throttle=0, a2_throttle=0))

        # Without A11 exclusion
        vals_all = extract_repeat_values(rows, column="energy_j", exclude_throttled=False)
        self.assertEqual(len(vals_all), 4)
        self.assertEqual([r[0] for r in vals_all], [1, 2, 3, 4])
        self.assertAlmostEqual(compute_condition_value(rows, exclude_throttled=False), 0.25)

        # With A11 exclusion
        vals_a11 = extract_repeat_values(rows, column="energy_j", exclude_throttled=True)
        self.assertEqual(len(vals_a11), 2)
        self.assertEqual([r[0] for r in vals_a11], [1, 4])
        self.assertAlmostEqual(vals_a11[0][1], 0.10)
        self.assertAlmostEqual(vals_a11[1][1], 0.40)
        self.assertAlmostEqual(compute_condition_value(rows, exclude_throttled=True), 0.25)

        # Audit throttles
        audit = audit_condition_throttles(rows)
        self.assertEqual(audit["warm_repeats_total"], 4)
        self.assertEqual(audit["warm_repeats_excluded"], 2)
        self.assertEqual(audit["warm_active_windows_total"], 8)
        self.assertEqual(audit["warm_active_windows_throttled"], 2)

    def test_alternative_supported_columns(self):
        """Psys gross and package net columns must compute per-image values accurately."""
        rows = []
        rows.extend(_make_repeat_rows(0, a1_energy=100.0, a2_energy=100.0))  # cold
        # Repeat 1: a1_psys=300 J (0.3 J/img), a2_psys=500 J (0.5 J/img) -> mean=0.4
        #           a1_above=70 J (0.07 J/img), a2_above=90 J (0.09 J/img) -> mean=0.08
        rows.extend(
            _make_repeat_rows(
                1,
                a1_energy=100.0,
                a2_energy=100.0,
                a1_psys=300.0,
                a2_psys=500.0,
                a1_above=70.0,
                a2_above=90.0,
            )
        )

        psys_val = compute_condition_value(rows, column="cpu_psys_energy_j")
        self.assertAlmostEqual(psys_val, 0.4)

        net_val = compute_condition_value(rows, column="above_idle_j")
        self.assertAlmostEqual(net_val, 0.08)


if __name__ == "__main__":
    unittest.main()
