#!/usr/bin/env python3
"""Tests for scripts/extract_h2_input.py on synthetic data only."""

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.extract_h2_input import (
    DEFAULT_ENERGY_COL,
    MAC_REDUCTIONS,
    MODELS,
    STATES,
    extract_h2_data,
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


def build_synthetic_session(
    session_dir: Path,
    session_n: int,
    base_energy: float = 100.0,
    omit_cells: set[str] | None = None,
    void_cells: set[str] | None = None,
    include_bnrecal: bool = True,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    with open(session_dir / "session_log.json", "w", encoding="utf-8") as f:
        json.dump({"session": session_n, "seed": 3000 + session_n}, f)

    if void_cells:
        with open(session_dir / "void_log.json", "w", encoding="utf-8") as f:
            json.dump({"session": session_n, "voids": list(void_cells)}, f)
    else:
        with open(session_dir / "void_log.json", "w", encoding="utf-8") as f:
            json.dump({"session": session_n, "voids": []}, f)

    omit = omit_cells or set()

    for model in MODELS:
        # Baseline
        base_dir = session_dir / f"{model}_fp32_ts"
        if f"{model}_fp32_ts" not in omit:
            create_summary_csv(base_dir, base_energy)

        # 3 pruned states
        for state in STATES:
            cond = f"{model}_{state}"
            if cond not in omit:
                mac_ratio = 1.0 - MAC_REDUCTIONS[model][state]
                cell_energy = base_energy * mac_ratio * 1.05  # slightly higher for d > 0
                create_summary_csv(session_dir / cond, cell_energy)

            if include_bnrecal:
                bn_cond = f"{model}_{state}_bnrecal"
                create_summary_csv(session_dir / bn_cond, base_energy * 0.8)


class TestExtractH2Input(unittest.TestCase):
    def test_known_energies_exact_log_ratio_and_log_mac(self):
        with tempfile.TemporaryDirectory() as td:
            s_dir = Path(td) / "session_01"
            s_dir.mkdir()
            with open(s_dir / "session_log.json", "w") as f:
                json.dump({"session": 1}, f)

            base_val = 200.0
            cell_val = 100.0  # r = 0.5, log_ratio = log(0.5)

            for m in MODELS:
                create_summary_csv(s_dir / f"{m}_fp32_ts", base_val)
                for s in STATES:
                    create_summary_csv(s_dir / f"{m}_{s}", cell_val)

            rows = extract_h2_data([s_dir])
            self.assertEqual(len(rows), 9)

            for row in rows:
                self.assertEqual(row["session"], 1)
                m = row["model"]
                s = row["state"]
                expected_log_ratio = math.log(0.5)
                expected_log_mac = math.log(1.0 - MAC_REDUCTIONS[m][s])
                self.assertAlmostEqual(row["log_ratio"], expected_log_ratio, places=9)
                self.assertAlmostEqual(row["log_mac"], expected_log_mac, places=9)

    def test_bnrecal_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            s_dir = Path(td) / "session_01"
            build_synthetic_session(s_dir, 1, include_bnrecal=True)

            rows = extract_h2_data([s_dir])
            self.assertEqual(len(rows), 9)
            for row in rows:
                self.assertNotIn("bnrecal", row["state"])
                self.assertIn(row["state"], STATES)

    def test_missing_cell_omitted_and_reported(self):
        with tempfile.TemporaryDirectory() as td:
            s_dir = Path(td) / "session_01"
            # Omit efficientnet_b0_pruned70
            build_synthetic_session(
                s_dir, 1, omit_cells={"efficientnet_b0_pruned70"}
            )

            rows = extract_h2_data([s_dir])
            self.assertEqual(len(rows), 8)
            found = {(r["model"], r["state"]) for r in rows}
            self.assertNotIn(("efficientnet_b0", "pruned70"), found)

    def test_voided_cell_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            s_dir = Path(td) / "session_01"
            # Mark resnet18_pruned30 as voided
            build_synthetic_session(
                s_dir, 1, void_cells={"resnet18_pruned30"}
            )

            rows = extract_h2_data([s_dir])
            self.assertEqual(len(rows), 8)
            found = {(r["model"], r["state"]) for r in rows}
            self.assertNotIn(("resnet18", "pruned30"), found)

    def test_missing_baseline_raises_error(self):
        with tempfile.TemporaryDirectory() as td:
            s_dir = Path(td) / "session_01"
            build_synthetic_session(s_dir, 1, omit_cells={"resnet18_fp32_ts"})

            with self.assertRaises(FileNotFoundError):
                extract_h2_data([s_dir])

    def test_summary_csv_with_multiple_or_zero_rows_raises_error(self):
        with tempfile.TemporaryDirectory() as td:
            s_dir = Path(td) / "session_01"
            build_synthetic_session(s_dir, 1)

            # Corrupt summary.csv of resnet18_pruned30 with 2 rows
            create_summary_csv(
                s_dir / "resnet18_pruned30", 50.0, extra_rows=1
            )
            with self.assertRaises(ValueError):
                extract_h2_data([s_dir])

    def test_cli_execution_and_r_gap_model_integration(self):
        """Build 6 synthetic sessions, extract CSV, and run fit_h2_primary in R."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            session_dirs = []
            for s in range(1, 7):
                s_path = base_dir / f"session_0{s}"
                build_synthetic_session(s_path, s)
                session_dirs.append(s_path)

            csv_out = base_dir / "h2_synthetic_input.csv"

            # Run extract_h2_input.py via CLI
            cli_code = main(
                [str(p) for p in session_dirs] + ["-o", str(csv_out)]
            )
            self.assertEqual(cli_code, 0)
            self.assertTrue(csv_out.is_file())

            # Read CSV and verify 54 rows
            with open(csv_out, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 54)

            # Run R integration check
            r_script = f"""
            suppressMessages(library(lme4))
            suppressMessages(library(lmerTest))
            source('{ROOT}/analysis/rq2_gap_model.R')
            df <- read.csv('{csv_out}')
            validate_h2_gap_input_v1(df)
            res <- fit_h2_primary(df)
            cat(paste(res$primary_method, res$complete_design, sep=";"))
            """

            proc = subprocess.run(
                ["Rscript", "-e", r_script],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"Rscript failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}",
            )
            out = proc.stdout.strip().split(";")
            primary_method = out[0]
            complete_design = out[1]
            self.assertEqual(primary_method, "session_mean_t")
            self.assertEqual(complete_design, "TRUE")

    def test_incomplete_design_r_primary_fallback(self):
        """Test with 1 missing cell: fit_h2_primary should use cell_fixed_mixed."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            session_dirs = []
            for s in range(1, 7):
                s_path = base_dir / f"session_0{s}"
                # Omit 1 cell in session 1
                omit = {"mobilenet_v3_small_pruned50"} if s == 1 else None
                build_synthetic_session(s_path, s, omit_cells=omit)
                session_dirs.append(s_path)

            csv_out = base_dir / "h2_incomplete_synthetic.csv"
            cli_code = main(
                [str(p) for p in session_dirs] + ["-o", str(csv_out)]
            )
            self.assertEqual(cli_code, 0)

            with open(csv_out, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 53)

            r_script = f"""
            suppressMessages(library(lme4))
            suppressMessages(library(lmerTest))
            source('{ROOT}/analysis/rq2_gap_model.R')
            df <- read.csv('{csv_out}')
            validate_h2_gap_input_v1(df)
            res <- fit_h2_primary(df)
            cat(paste(res$primary_method, res$complete_design, sep=";"))
            """

            proc = subprocess.run(
                ["Rscript", "-e", r_script],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"R error:\n{proc.stderr}")
            out = proc.stdout.strip().split(";")
            primary_method = out[0]
            complete_design = out[1]
            self.assertEqual(primary_method, "cell_fixed_mixed")
            self.assertEqual(complete_design, "FALSE")


if __name__ == "__main__":
    unittest.main()
