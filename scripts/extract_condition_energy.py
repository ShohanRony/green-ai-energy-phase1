#!/usr/bin/env python3
"""Extract per-condition gross energy from session directories.

Reads every summary.csv across sessions and outputs:
  session, condition, gross_j_per_image_mean

Raises an error and exits non-zero if summary.csv has != 1 data row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_ENERGY_COL = "gross_j_per_image_mean"


def extract_session_id(session_dir: Path) -> int | str:
    slog = session_dir / "session_log.json"
    if slog.is_file():
        try:
            with open(slog, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "session" in d:
                    return int(d["session"])
        except Exception:
            pass
    m = re.search(r"(\d+)", session_dir.name)
    if m:
        return int(m.group(1))
    return session_dir.name


def read_condition_energy(summary_csv_path: Path, energy_col: str, condition_desc: str) -> float:
    if not summary_csv_path.is_file():
        raise FileNotFoundError(f"Missing summary.csv for {condition_desc} at {summary_csv_path}")

    with open(summary_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly 1 data row in {summary_csv_path} for {condition_desc}, found {len(rows)}"
        )

    row = rows[0]
    if energy_col not in row:
        raise KeyError(
            f"Column '{energy_col}' not found in {summary_csv_path}. Available: {list(row.keys())}"
        )

    val_str = row[energy_col]
    if val_str is None or val_str == "":
        raise ValueError(f"Empty value for '{energy_col}' in {summary_csv_path}")

    val = float(val_str)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"Non-positive or non-finite energy {val} in {summary_csv_path}")

    return val


def extract_all_condition_energies(
    session_dirs: list[Path],
    energy_col: str = DEFAULT_ENERGY_COL,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []

    for s_dir in session_dirs:
        if not s_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {s_dir}")

        sess_id = extract_session_id(s_dir)

        # Look for subdirectories containing summary.csv
        # If session_log.json has an order list, use that to preserve order; else sort subdirs
        cond_names = []
        slog = s_dir / "session_log.json"
        if slog.is_file():
            try:
                with open(slog, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if "order" in d and isinstance(d["order"], list):
                        cond_names = d["order"]
            except Exception:
                pass

        if not cond_names:
            cond_names = sorted([d.name for d in s_dir.iterdir() if d.is_dir()])

        for c_name in cond_names:
            c_dir = s_dir / c_name
            if not c_dir.is_dir():
                continue
            summary_path = c_dir / "summary.csv"
            if not summary_path.is_file():
                continue

            energy_val = read_condition_energy(
                summary_path, energy_col, f"{c_name} in session {sess_id}"
            )
            out_rows.append(
                {
                    "session": sess_id,
                    "condition": c_name,
                    energy_col: energy_val,
                }
            )

    return out_rows


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-condition gross energy across session directories."
    )
    parser.add_argument(
        "session_dirs",
        nargs="*",
        type=Path,
        help="Explicit session directories to process.",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help="Parent directory containing session folders (e.g. results_stage5/cpu).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Path to write output CSV.",
    )
    parser.add_argument(
        "--energy-col",
        type=str,
        default=DEFAULT_ENERGY_COL,
        help=f"Energy column name (default: {DEFAULT_ENERGY_COL}).",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    ns = parse_args(args)

    if ns.session_dirs:
        session_dirs = ns.session_dirs
    elif ns.sessions_dir:
        session_dirs = sorted(
            [d for d in ns.sessions_dir.glob("session_*") if d.is_dir() and not d.name.startswith("voided")]
        )
        if not session_dirs:
            sys.stderr.write(f"Error: no session directories found under {ns.sessions_dir}\n")
            return 1
    else:
        default_parent = Path("results_stage5/cpu")
        if default_parent.is_dir():
            session_dirs = sorted(
                [d for d in default_parent.glob("session_*") if d.is_dir() and not d.name.startswith("voided")]
            )
        else:
            session_dirs = []
        if not session_dirs:
            sys.stderr.write("Error: no session directories specified and default has none.\n")
            return 1

    try:
        rows = extract_all_condition_energies(session_dirs, energy_col=ns.energy_col)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    fieldnames = ["session", "condition", ns.energy_col]

    if ns.out is None or str(ns.out) == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        with open(ns.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
