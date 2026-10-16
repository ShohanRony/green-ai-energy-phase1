#!/usr/bin/env python3
"""Repeat-level energy extraction and condition-level aggregation for x86-CPU.

Follows pilot.py summarize() logic (pilot.py:421-454):
  - Identifies repeat IDs (pilot.py:421: ids=sorted({r['repeat'] for r in group}))
  - Discards the cold first repeat when len(ids) > 1 (pilot.py:428, 446: cold=rep==ids[0] and len(ids)>1)
  - Checks complete phase structure (pilot.py:425: set(d) == {'idle_before','a1','a2','idle_after'})
  - Computes per-image energy for active phases a1 and a2: column / (batches * batch) (pilot.py:431-434)
  - Computes repeat value as the mean of active phases a1 and a2 (pilot.py:449: pairs.append(statistics.mean(per)))
  - Computes condition value as the mean over warm repeats (pilot.py:454: mean=statistics.mean(pairs))

Supported columns:
  - package gross: 'energy_j' (or 'cpu_package_energy_j')
  - psys gross: 'cpu_psys_energy_j'
  - package net: 'above_idle_j'
  - psys net: unavailable (no psys-specific idle quantity in windows.csv)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

MODELS = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
PRUNED_STATES = ["pruned30", "pruned50", "pruned70"]

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

COLUMN_MAP = {
    "package_gross": "energy_j",
    "psys_gross": "cpu_psys_energy_j",
    "package_net": "above_idle_j",
}


def load_windows_csv(path: Path | str) -> list[dict[str, str]]:
    """Read windows.csv into a list of row dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_repeat_values(
    rows: list[dict[str, str]],
    column: str = "energy_j",
    exclude_throttled: bool = False,
) -> list[tuple[int, float]]:
    """Compute per-repeat energy per image for warm repeats.

    Following pilot.py:421-454:
      - Groups by (device, size, batch, requested_s)
      - Repeat 0 is discarded if len(repeats) > 1 (cold-repeat rule)
      - Computes mean of a1 and a2 per-image energy: column / (batches * batch)
      - Under A11 (exclude_throttled=True), any repeat where a1 or a2 has
        thermal_pkg_throttle_count_delta > 0 is excluded.
    Returns list of (repeat_id, repeat_mean_energy).
    """
    if not rows:
        return []

    keys = sorted(
        {
            (
                r.get("device", ""),
                int(r.get("size", 0)),
                int(r.get("batch", 1)),
                float(r.get("requested_s", 0.0)),
            )
            for r in rows
        }
    )

    out: list[tuple[int, float]] = []

    for key in keys:
        group = [
            r
            for r in rows
            if (
                r.get("device", ""),
                int(r.get("size", 0)),
                int(r.get("batch", 1)),
                float(r.get("requested_s", 0.0)),
            )
            == key
        ]
        ids = sorted({int(r["repeat"]) for r in group})
        if not ids:
            continue

        cold_rep = ids[0] if len(ids) > 1 else None

        for rep in ids:
            d = {r["phase"]: r for r in group if int(r["repeat"]) == rep}
            if set(d.keys()) != {"idle_before", "a1", "a2", "idle_after"}:
                continue

            # pilot.py:428, 446: cold first repeat discarded
            if rep == cold_rep:
                continue

            # A11 throttle exclusion rule
            if exclude_throttled:
                a1_throttle = float(d["a1"].get("thermal_pkg_throttle_count_delta", 0.0) or 0.0)
                a2_throttle = float(d["a2"].get("thermal_pkg_throttle_count_delta", 0.0) or 0.0)
                if a1_throttle > 0 or a2_throttle > 0:
                    continue

            per: list[float] = []
            for p in ["a1", "a2"]:
                r = d[p]
                raw_val = float(r[column])
                imgs = int(r["batches"]) * key[2]
                if imgs <= 0:
                    raise ValueError(f"Zero images processed in phase {p}, repeat {rep}")
                per.append(raw_val / imgs)

            # pilot.py:449: pairs.append(statistics.mean(per))
            out.append((rep, statistics.mean(per)))

    return out


def compute_condition_value(
    rows: list[dict[str, str]],
    column: str = "energy_j",
    exclude_throttled: bool = False,
) -> float | None:
    """Compute condition-level mean energy over warm repeats (pilot.py:454)."""
    repeat_vals = extract_repeat_values(rows, column=column, exclude_throttled=exclude_throttled)
    if not repeat_vals:
        return None
    return statistics.mean(v for _, v in repeat_vals)


def audit_condition_throttles(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count warm repeats, excluded repeats, and warm active windows with throttle."""
    keys = sorted(
        {
            (
                r.get("device", ""),
                int(r.get("size", 0)),
                int(r.get("batch", 1)),
                float(r.get("requested_s", 0.0)),
            )
            for r in rows
        }
    )
    total_warm_repeats = 0
    excluded_warm_repeats = 0
    total_warm_active_windows = 0
    throttled_warm_active_windows = 0

    for key in keys:
        group = [
            r
            for r in rows
            if (
                r.get("device", ""),
                int(r.get("size", 0)),
                int(r.get("batch", 1)),
                float(r.get("requested_s", 0.0)),
            )
            == key
        ]
        ids = sorted({int(r["repeat"]) for r in group})
        if not ids:
            continue
        cold_rep = ids[0] if len(ids) > 1 else None

        for rep in ids:
            d = {r["phase"]: r for r in group if int(r["repeat"]) == rep}
            if set(d.keys()) != {"idle_before", "a1", "a2", "idle_after"}:
                continue
            if rep == cold_rep:
                continue

            total_warm_repeats += 1
            a1_th = float(d["a1"].get("thermal_pkg_throttle_count_delta", 0.0) or 0.0)
            a2_th = float(d["a2"].get("thermal_pkg_throttle_count_delta", 0.0) or 0.0)

            total_warm_active_windows += 2
            if a1_th > 0:
                throttled_warm_active_windows += 1
            if a2_th > 0:
                throttled_warm_active_windows += 1

            if a1_th > 0 or a2_th > 0:
                excluded_warm_repeats += 1

    return {
        "warm_repeats_total": total_warm_repeats,
        "warm_repeats_excluded": excluded_warm_repeats,
        "warm_active_windows_total": total_warm_active_windows,
        "warm_active_windows_throttled": throttled_warm_active_windows,
    }


def run_reconciliation(sessions_dir: Path | str, tolerance: float = 1e-9) -> tuple[bool, float, int, list[str]]:
    """Check package gross from windows.csv against summary.csv gross_j_per_image_mean.

    Returns (passed, max_rel_diff, checked_count, failure_details).
    """
    sessions_path = Path(sessions_dir)
    max_rel_diff = 0.0
    checked = 0
    failures: list[str] = []

    for sdir in sorted(sessions_path.glob("session_*")):
        if not sdir.is_dir():
            continue
        for cdir in sorted(sdir.iterdir()):
            if not cdir.is_dir():
                continue
            wpath = cdir / "windows.csv"
            spath = cdir / "summary.csv"
            if not wpath.is_file() or not spath.is_file():
                continue

            wrows = load_windows_csv(wpath)
            with open(spath, "r", encoding="utf-8") as f:
                srows = list(csv.DictReader(f))
            if len(srows) != 1:
                failures.append(f"{cdir}: summary.csv does not have exactly 1 data row ({len(srows)})")
                continue

            ref_val = float(srows[0]["gross_j_per_image_mean"])
            calc_val = compute_condition_value(wrows, column="energy_j", exclude_throttled=False)
            if calc_val is None:
                failures.append(f"{cdir}: no warm repeats extracted from windows.csv")
                continue

            rel_diff = abs(calc_val - ref_val) / ref_val if ref_val > 0 else abs(calc_val - ref_val)
            if rel_diff > max_rel_diff:
                max_rel_diff = rel_diff

            if rel_diff > tolerance:
                failures.append(
                    f"{cdir}: rel_diff={rel_diff:.2e} > {tolerance:.2e} (calc={calc_val}, ref={ref_val})"
                )
            checked += 1

    passed = (len(failures) == 0) and (checked == 144)
    return passed, max_rel_diff, checked, failures


def generate_condition_energy_specs(sessions_dir: Path | str, out_path: Path | str) -> list[dict[str, Any]]:
    """Generate condition_energy_specs.csv containing all 144 condition values across supported columns."""
    sessions_path = Path(sessions_dir)
    records: list[dict[str, Any]] = []

    for sdir in sorted(sessions_path.glob("session_*")):
        if not sdir.is_dir():
            continue
        # Extract integer session index from folder name
        sid_str = sdir.name.split("_")[-1].lstrip("0")
        sid = int(sid_str) if sid_str else 0

        for cdir in sorted(sdir.iterdir()):
            if not cdir.is_dir():
                continue
            wpath = cdir / "windows.csv"
            if not wpath.is_file():
                continue
            wrows = load_windows_csv(wpath)

            pkg_gross = compute_condition_value(wrows, column="energy_j", exclude_throttled=False)
            pkg_net = compute_condition_value(wrows, column="above_idle_j", exclude_throttled=False)
            psys_gross = compute_condition_value(wrows, column="cpu_psys_energy_j", exclude_throttled=False)

            records.append(
                {
                    "session": sid,
                    "condition": cdir.name,
                    "pkg_gross": pkg_gross,
                    "pkg_net": pkg_net,
                    "psys_gross": psys_gross,
                }
            )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["session", "condition", "pkg_gross", "pkg_net", "psys_gross"]
        )
        writer.writeheader()
        writer.writerows(records)

    return records


def generate_a11_h2_input(sessions_dir: Path | str, out_path: Path | str) -> list[dict[str, Any]]:
    """Build H2 input under A11 throttle exclusion rule (A11, A12)."""
    sessions_path = Path(sessions_dir)
    records: list[dict[str, Any]] = []

    # Map (session, condition) -> a11_pkg_gross
    data: dict[tuple[int, str], float | None] = {}
    for sdir in sorted(sessions_path.glob("session_*")):
        if not sdir.is_dir():
            continue
        sid = int(sdir.name.split("_")[-1].lstrip("0"))
        for cdir in sorted(sdir.iterdir()):
            if not cdir.is_dir():
                continue
            wpath = cdir / "windows.csv"
            if not wpath.is_file():
                continue
            wrows = load_windows_csv(wpath)
            # A11 exclusion
            val = compute_condition_value(wrows, column="energy_j", exclude_throttled=True)
            data[(sid, cdir.name)] = val

    for sid in range(1, 7):
        for model in MODELS:
            base_key = (sid, f"{model}_fp32_ts")
            e_base = data.get(base_key)
            if e_base is None:
                print(f"Warning: Baseline {base_key} has no remaining repeats under A11; omitting session {sid} for {model}", file=sys.stderr)
                continue

            for state in PRUNED_STATES:
                cell_key = (sid, f"{model}_{state}")
                e_cell = data.get(cell_key)
                if e_cell is None:
                    print(f"Warning: Cell {cell_key} has no remaining repeats under A11; omitting", file=sys.stderr)
                    continue

                r = e_cell / e_base
                log_ratio = math.log(r)
                mac_reduction = MAC_REDUCTIONS[model][state]
                log_mac = math.log(1.0 - mac_reduction)

                records.append(
                    {
                        "session": sid,
                        "model": model,
                        "state": state,
                        "log_ratio": log_ratio,
                        "log_mac": log_mac,
                    }
                )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["session", "model", "state", "log_ratio", "log_mac"]
        )
        writer.writeheader()
        writer.writerows(records)

    return records


def generate_throttle_audit(sessions_dir: Path | str) -> list[dict[str, Any]]:
    """Produce per-cell throttle audit across all 6 sessions."""
    sessions_path = Path(sessions_dir)
    # Group by cell name: {model}_{state}
    cells = [f"{m}_{s}" for m in MODELS for s in PRUNED_STATES]
    # Also include baselines
    baselines = [f"{m}_fp32_ts" for m in MODELS]
    all_targets = cells + baselines

    audit_summary: dict[str, dict[str, int]] = {
        c: {
            "warm_repeats_total": 0,
            "warm_repeats_excluded": 0,
            "warm_active_windows_total": 0,
            "warm_active_windows_throttled": 0,
        }
        for c in all_targets
    }

    for sdir in sorted(sessions_path.glob("session_*")):
        if not sdir.is_dir():
            continue
        for cname in all_targets:
            cdir = sdir / cname
            wpath = cdir / "windows.csv"
            if not wpath.is_file():
                continue
            wrows = load_windows_csv(wpath)
            res = audit_condition_throttles(wrows)
            for k in audit_summary[cname]:
                audit_summary[cname][k] += res[k]

    rows_out: list[dict[str, Any]] = []
    for cname in all_targets:
        s = audit_summary[cname]
        share = (
            s["warm_active_windows_throttled"] / s["warm_active_windows_total"]
            if s["warm_active_windows_total"] > 0
            else 0.0
        )
        rows_out.append(
            {
                "cell": cname,
                "warm_repeats_total": s["warm_repeats_total"],
                "warm_repeats_excluded": s["warm_repeats_excluded"],
                "warm_active_windows_total": s["warm_active_windows_total"],
                "warm_active_windows_throttled": s["warm_active_windows_throttled"],
                "throttled_active_share": share,
            }
        )
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat-level energy extraction and reconciliation.")
    parser.add_argument("--sessions-dir", default="results_stage5/cpu", help="Path to CPU sessions directory")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="Reconciliation relative tolerance")
    parser.add_argument("--export-specs", default="analysis/results/condition_energy_specs.csv", help="Export condition specs CSV")
    parser.add_argument("--export-a11-h2", default="analysis/results/h2_input_a11.csv", help="Export A11 H2 input CSV")
    args = parser.parse_args()

    passed, max_rel_diff, checked, failures = run_reconciliation(args.sessions_dir, tolerance=args.tolerance)
    print(f"Reconciliation checked: {checked} condition pairs")
    print(f"Max relative difference: {max_rel_diff:.2e}")
    if not passed:
        print(f"RECONCILIATION GATE: FAILED with {len(failures)} failures:")
        for fail in failures[:10]:
            print(f"  {fail}")
        sys.exit(1)
    print("RECONCILIATION GATE: PASSED")

    if args.export_specs:
        generate_condition_energy_specs(args.sessions_dir, args.export_specs)
        print(f"Exported condition energy specs to {args.export_specs}")

    if args.export_a11_h2:
        generate_a11_h2_input(args.sessions_dir, args.export_a11_h2)
        print(f"Exported A11 H2 input to {args.export_a11_h2}")


if __name__ == "__main__":
    main()
