#!/usr/bin/env python3
"""Unit tests for accuracy table metrics and tier assignment."""

import unittest
import numpy as np
from analysis.accuracy_table import wilson_ci, exact_mcnemar, determine_tier


class AccuracyTableTests(unittest.TestCase):
    def test_wilson_ci(self):
        """Wilson interval should match expected binomial score bounds."""
        lo, hi = wilson_ci(9309, 10000, z=1.96)
        self.assertAlmostEqual(lo, 0.9257616972, places=6)
        self.assertAlmostEqual(hi, 0.9357073608, places=6)

        # Extreme proportions
        lo0, hi0 = wilson_ci(0, 100, z=1.96)
        self.assertGreaterEqual(lo0, 0.0)
        self.assertLess(hi0, 0.1)

    def test_exact_mcnemar(self):
        """Exact McNemar test computes discordant counts and two-sided binomial p-value."""
        # Identical predictions -> b=0, c=0, p=1.0
        s_corr = np.array([True, True, False, False])
        r_corr = np.array([True, True, False, False])
        b, c, p = exact_mcnemar(s_corr, r_corr)
        self.assertEqual((b, c), (0, 0))
        self.assertEqual(p, 1.0)

        # Discordant pairs: b=1, c=1 -> p=1.0
        s_corr = np.array([True, False, False])
        r_corr = np.array([False, True, False])
        b, c, p = exact_mcnemar(s_corr, r_corr)
        self.assertEqual((b, c), (1, 1))
        self.assertEqual(p, 1.0)

        # Discordant pairs: b=0, c=6 -> p = 2 * (0.5^6) = 0.03125
        s_corr = np.array([False] * 6)
        r_corr = np.array([True] * 6)
        b, c, p = exact_mcnemar(s_corr, r_corr)
        self.assertEqual((b, c), (0, 6))
        self.assertAlmostEqual(p, 0.03125, places=5)

    def test_determine_tier(self):
        """Tier determination according to Plan v1 section 9 thresholds."""
        ref_acc = 0.9000
        # 99% threshold = 0.8910, 99.9% threshold = 0.8991

        # Strict: >= 99.9% of ref
        tier_strict = determine_tier(0.9000, 0.8950, 0.9050, ref_acc)
        self.assertEqual(tier_strict, "strict")

        # Deployable: >= 99% of ref, does not straddle
        tier_deployable = determine_tier(0.8950, 0.8920, 0.8980, ref_acc)
        self.assertEqual(tier_deployable, "deployable")

        # Borderline: Wilson interval straddles 99% threshold
        tier_borderline = determine_tier(0.8920, 0.8880, 0.8960, ref_acc)
        self.assertEqual(tier_borderline, "borderline")

        # Not deployable: < 99% of ref and interval below threshold
        tier_not = determine_tier(0.8500, 0.8400, 0.8600, ref_acc)
        self.assertEqual(tier_not, "not deployable")


if __name__ == "__main__":
    unittest.main()
