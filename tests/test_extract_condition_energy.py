#!/usr/bin/env python3
"""Tests for scripts/extract_condition_energy.py on synthetic data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.extract_condition_energy import (
    DEFAULT_ENERGY_COL,
    extract_all_condition_energies,
    main,
)


def create_summary_csv(
    cond_dir: Path,
    energy_val: float,
    energy_col: str = DEFAULT_ENERGY_COL,
    extra_rows: int = 0,
) -> None:
    cond_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cond_dir / "summary.csv"
    fieldnames = [energy_col, "cpu_package_j_per_image_mean", "pairs"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                energy_col: str(energy_val),
                "cpu_package_j_per_image_mean": str(energy_val),
                "pairs": "10",
            }
        )
        for _ in range(extra_rows):
            writer.writerow(
                {
                    energy_col: str(energy_val),
                    "cpu_package_j_per_image_mean": str(energy_val),
                    "pairs": "10",
                }
            )


class TestExtractConditionEnergy(unittest.TestCase):
    def test_extract_synthetic_energies(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            s1 = base / "session_01"
            s1.mkdir()
            with open(s1 / "session_log.json", "w") as f:
                json.dump({"session": 1, "order": ["condA", "condB"]}, f)

            create_summary_csv(s1 / "condA", 12.5)
            create_summary_csv(s1 / "condB", 25.0)

            rows = extract_all_condition_energies([s1])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["session"], 1)
            self.assertEqual(rows[0]["condition"], "condA")
            self.assertEqual(rows[0]["gross_j_per_image_mean"], 12.5)
            self.assertEqual(rows[1]["condition"], "condB")
            self.assertEqual(rows[1]["gross_j_per_image_mean"], 25.0)

    def test_corrupt_summary_raises_error(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            s1 = base / "session_01"
            s1.mkdir()
            create_summary_csv(s1 / "condA", 10.0, extra_rows=1)

            with self.assertRaises(ValueError):
                extract_all_condition_energies([s1])

    def test_cli_output(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            s1 = base / "session_01"
            s1.mkdir()
            create_summary_csv(s1 / "c1", 5.0)
            out_csv = base / "out.csv"

            ret = main([str(s1), "-o", str(out_csv)])
            self.assertEqual(ret, 0)
            self.assertTrue(out_csv.is_file())

            with open(out_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["condition"], "c1")
            self.assertEqual(float(rows[0]["gross_j_per_image_mean"]), 5.0)


if __name__ == "__main__":
    unittest.main()
