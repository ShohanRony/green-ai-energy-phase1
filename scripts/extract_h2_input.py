#!/usr/bin/env python3
"""Extract H2 gap model inputs from CPU block session directories.

For each session and each of the 9 zero-finetune pruned cells:
  {resnet18, mobilenet_v3_small, efficientnet_b0} x {pruned30, pruned50, pruned70}
computes:
  r = E(cell) / E({model}_fp32_ts)
  log_ratio = log(r)
  log_mac = log(1 - mac_reduction)

Output CSV columns:
  session, model, state, log_ratio, log_mac

Zero-finetune only; never _bnrecal.
Missing or voided cells are omitted (listed on stderr).
Exits non-zero if a session's fp32_ts baseline is missing or summary.csv != 1 data row.
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

MODELS = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
STATES = ["pruned30", "pruned50", "pruned70"]

# Realised MAC reductions from Stage 2 registered specification
MAC_REDUCTIONS: dict[str, dict[str, float]] = {
    "resnet18": {
        "pruned30": 0.516,
        "pruned50": 0.748,
        "pruned70": 0.910,
    },
    "mobilenet_v3_small": {
        "pruned30": 0.467,
        "pruned50": 0.691,
        "pruned70": 0.868,
    },
    "efficientnet_b0": {
        "pruned30": 0.484,
        "pruned50": 0.712,
        "pruned70": 0.881,
    },
}

DEFAULT_ENERGY_COL = "gross_j_per_image_mean"


def extract_session_id(session_dir: Path, raw_names: bool = False) -> int | str:
    """Extract integer session index or raw folder name."""
    if raw_names:
        return session_dir.name
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


def read_summary_energy(summary_csv_path: Path, energy_col: str, condition_desc: str) -> float:
    """Read energy from summary.csv. Must contain exactly 1 data row."""
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
            f"Column '{energy_col}' not found in {summary_csv_path}. Available columns: {list(row.keys())}"
        )

    val_str = row[energy_col]
    if val_str is None or val_str == "":
        raise ValueError(f"Empty value for '{energy_col}' in {summary_csv_path}")

    val = float(val_str)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"Non-positive or non-finite energy {val} in {summary_csv_path}")

    return val


def get_voided_conditions(session_dir: Path) -> set[str]:
    """Return set of voided condition names from void_log.json if present."""
    void_file = session_dir / "void_log.json"
    if not void_file.is_file():
        return set()
    try:
        with open(void_file, "r", encoding="utf-8") as f:
            d = json.load(f)
            voids = d.get("voids", [])
            void_set = set()
            for v in voids:
                if isinstance(v, str):
                    void_set.add(v)
                elif isinstance(v, dict) and "condition" in v:
                    void_set.add(v["condition"])
            return void_set
    except Exception as e:
        sys.stderr.write(f"Warning: could not parse {void_file}: {e}\n")
        return set()


def extract_h2_data(
    session_dirs: list[Path],
    energy_col: str = DEFAULT_ENERGY_COL,
    raw_session_names: bool = False,
) -> list[dict[str, Any]]:
    """Extract (session, model, state, log_ratio, log_mac) rows across sessions."""
    out_rows: list[dict[str, Any]] = []

    for s_dir in session_dirs:
        if not s_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {s_dir}")

        sess_id = extract_session_id(s_dir, raw_names=raw_session_names)
        voided = get_voided_conditions(s_dir)

        # 1. Read baselines: fp32_ts for each model
        baselines: dict[str, float] = {}
        for model in MODELS:
            base_cond = f"{model}_fp32_ts"
            if base_cond in voided:
                raise RuntimeError(
                    f"Baseline {base_cond} is marked voided in {s_dir / 'void_log.json'}"
                )
            base_dir = s_dir / base_cond
            if not base_dir.is_dir():
                raise FileNotFoundError(
                    f"Required baseline condition directory missing: {base_dir}"
                )
            summary_path = base_dir / "summary.csv"
            base_energy = read_summary_energy(
                summary_path, energy_col, f"baseline {base_cond} in session {sess_id}"
            )
            baselines[model] = base_energy

        # 2. Extract 9 zero-finetune pruned cells
        for model in MODELS:
            base_energy = baselines[model]
            for state in STATES:
                cell_cond = f"{model}_{state}"

                # Voided check
                if cell_cond in voided:
                    sys.stderr.write(
                        f"Warning: cell {cell_cond} in session {sess_id} is voided; omitting.\n"
                    )
                    continue

                cell_dir = s_dir / cell_cond
                summary_path = cell_dir / "summary.csv"

                if not cell_dir.is_dir() or not summary_path.is_file():
                    sys.stderr.write(
                        f"Warning: cell {cell_cond} in session {sess_id} is missing; omitting.\n"
                    )
                    continue

                # Read cell energy (raises if summary.csv has != 1 row or missing col)
                cell_energy = read_summary_energy(
                    summary_path, energy_col, f"cell {cell_cond} in session {sess_id}"
                )

                ratio = cell_energy / base_energy
                log_ratio = math.log(ratio)

                mac_red = MAC_REDUCTIONS[model][state]
                mac_ratio = 1.0 - mac_red
                log_mac = math.log(mac_ratio)

                out_rows.append(
                    {
                        "session": sess_id,
                        "model": model,
                        "state": state,
                        "log_ratio": log_ratio,
                        "log_mac": log_mac,
                    }
                )

    return out_rows


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract H2 gap model inputs from CPU block session directories."
    )
    parser.add_argument(
        "session_dirs",
        nargs="*",
        type=Path,
        help="Explicit session directories to process (in order).",
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
        help="Path to write output CSV. If omitted or '-', writes to stdout.",
    )
    parser.add_argument(
        "--energy-col",
        type=str,
        default=DEFAULT_ENERGY_COL,
        help=f"Column to read energy from in summary.csv (default: {DEFAULT_ENERGY_COL}).",
    )
    parser.add_argument(
        "--raw-session-names",
        action="store_true",
        help="Use exact directory names for session column rather than integer index.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    ns = parse_args(args)

    if ns.session_dirs:
        session_dirs = ns.session_dirs
    elif ns.sessions_dir:
        session_dirs = sorted(
            [d for d in ns.sessions_dir.glob("session_*") if d.is_dir()]
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
            sys.stderr.write(
                "Error: no session directories specified and default results_stage5/cpu has none.\n"
            )
            return 1

    try:
        rows = extract_h2_data(
            session_dirs,
            energy_col=ns.energy_col,
            raw_session_names=ns.raw_session_names,
        )
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    fieldnames = ["session", "model", "state", "log_ratio", "log_mac"]

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
