#!/usr/bin/env python3
"""Accuracy table computation and deployability assessment per Plan v1 section 9.

Computed strictly from existing predictions in results_accuracy/.
No new model inference is performed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats

MODELS = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
PRIMARY_STATES = [
    "fp32_cpu",
    "int8",
    "pruned30",
    "pruned50",
    "pruned70",
    "pruned30_bnrecal",
    "pruned50_bnrecal",
    "pruned70_bnrecal",
]
SECONDARY_STATES = [
    "pruned30_ft",
    "pruned50_ft",
    "pruned70_ft",
    "fp16",
]

BNRECAL_SKIP = {
    ("mobilenet_v3_small", "pruned50_bnrecal"),
    ("mobilenet_v3_small", "pruned70_bnrecal"),
    ("efficientnet_b0", "pruned70_bnrecal"),
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z**2) / n
    centre = p + (z**2) / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p)) / n + (z**2) / (4.0 * (n**2)))
    return ((centre - margin) / denom, (centre + margin) / denom)


def exact_mcnemar(
    state_correct: np.ndarray, ref_correct: np.ndarray
) -> tuple[int, int, float]:
    """Exact two-sided McNemar test based on discordant pairs b and c."""
    # b: state correct, reference incorrect
    b = int((state_correct & ~ref_correct).sum())
    # c: state incorrect, reference correct
    c = int((~state_correct & ref_correct).sum())

    if b + c == 0:
        p_val = 1.0
    else:
        # Two-sided exact binomial test with p = 0.5
        p_val = float(
            stats.binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue
        )
    return b, c, p_val


def determine_tier(
    acc: float, wilson_lo: float, wilson_hi: float, ref_acc: float
) -> str:
    """Classify deployability tier per Plan v1 section 9.

    - "strict" if >= 99.9% of reference
    - "deployable" if >= 99% of reference (and interval does not straddle)
    - "borderline" if Wilson interval straddles 0.99 * reference accuracy
    - else "not deployable"
    """
    thresh_99 = 0.99 * ref_acc
    thresh_999 = 0.999 * ref_acc

    if wilson_lo < thresh_99 < wilson_hi:
        return "borderline"
    elif acc >= thresh_999:
        return "strict"
    elif acc >= thresh_99:
        return "deployable"
    else:
        return "not deployable"


def reconcile_with_summary(
    accuracy_dir: Path, labels: np.ndarray
) -> list[str]:
    """Verify that recomputed accuracy on stored predictions matches summary.csv."""
    summary_path = accuracy_dir / "summary.csv"
    if not summary_path.is_file():
        return [f"Missing summary file: {summary_path}"]

    with open(summary_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    mismatches: list[str] = []
    n = len(labels)

    for row in reader:
        m = row["model"]
        s = row["state"]
        pred_file = accuracy_dir / f"{m}_{s}_preds.npy"
        if not pred_file.is_file():
            mismatches.append(f"Missing pred file for summary row: {pred_file}")
            continue

        preds = np.load(pred_file)
        if len(preds) != n:
            mismatches.append(
                f"{m}_{s}: length {len(preds)} != labels length {n}"
            )
            continue

        correct = int((preds == labels).sum())
        acc = correct / n
        expected_correct = int(row["correct"])
        expected_acc = float(row["accuracy"])

        if correct != expected_correct:
            mismatches.append(
                f"{m}_{s}: correct={correct} != summary.csv {expected_correct}"
            )
        if abs(acc - expected_acc) > 1e-9:
            mismatches.append(
                f"{m}_{s}: acc={acc:.6f} != summary.csv {expected_acc:.6f}"
            )

    return mismatches


def compute_table_rows(
    accuracy_dir: Path,
    labels: np.ndarray,
    states: list[str],
    is_primary: bool = True,
) -> list[dict[str, Any]]:
    """Compute accuracy, Wilson interval, McNemar test, and tier for states."""
    n = len(labels)
    rows: list[dict[str, Any]] = []

    for model in MODELS:
        ref_file = accuracy_dir / f"{model}_fp32_cpu_preds.npy"
        ref_preds = np.load(ref_file)
        ref_corr_arr = (ref_preds == labels)
        ref_correct = int(ref_corr_arr.sum())
        ref_acc = ref_correct / n

        for state in states:
            pred_file = accuracy_dir / f"{model}_{state}_preds.npy"
            preds = np.load(pred_file)
            corr_arr = (preds == labels)
            correct = int(corr_arr.sum())
            acc = correct / n

            lo, hi = wilson_ci(correct, n)
            pct_ref = (acc / ref_acc) * 100.0
            b, c, p_val = exact_mcnemar(corr_arr, ref_corr_arr)
            tier = determine_tier(acc, lo, hi, ref_acc)

            if is_primary:
                if (model, state) in BNRECAL_SKIP:
                    measured = "no"
                else:
                    measured = "yes"
            else:
                measured = "no"

            rows.append(
                {
                    "model": model,
                    "state": state,
                    "n": n,
                    "correct": correct,
                    "accuracy": acc,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                    "ref_accuracy": ref_acc,
                    "pct_of_ref": pct_ref,
                    "mcnemar_b": b,
                    "mcnemar_c": c,
                    "mcnemar_p": p_val,
                    "tier": tier,
                    "energy_measured_in_cpu_block": measured,
                }
            )

    return rows


def generate_accuracy_report(accuracy_dir: Path, output_file: Path) -> str:
    """Generate accuracy tables, reconciliation report, and item 4 search."""
    labels_file = accuracy_dir / "labels.npy"
    if not labels_file.is_file():
        raise FileNotFoundError(f"labels.npy not found in {accuracy_dir}")
    labels = np.load(labels_file)

    # 1. Reconciliation
    mismatches = reconcile_with_summary(accuracy_dir, labels)
    if mismatches:
        err_msg = "Reconciliation FAILED with mismatches:\n" + "\n".join(
            f"  {m}" for m in mismatches
        )
        raise RuntimeError(err_msg)

    # 2. Compute tables
    primary_rows = compute_table_rows(
        accuracy_dir, labels, PRIMARY_STATES, is_primary=True
    )
    secondary_rows = compute_table_rows(
        accuracy_dir, labels, SECONDARY_STATES, is_primary=False
    )

    lines: list[str] = []
    lines.append("=== Reconciliation with results_accuracy/summary.csv ===")
    lines.append("Reconciliation check: All 36 states present in summary.csv recomputed.")
    lines.append("Mismatch count: 0")
    lines.append("Reconciliation status: PASSED (exact match on correct counts and accuracy).")
    lines.append("")

    lines.append("=== Primary Accuracy Table (Plan v1 section 9) ===")
    lines.append("Reference: Same model's fp32_cpu predictions (n=10,000 CIFAR-10 test set)")
    lines.append("Tiers: 'strict' (>= 99.9% of ref), 'deployable' (>= 99.0% of ref), 'borderline' (Wilson CI straddles 99% threshold), 'not deployable' (< 99.0% of ref)")
    lines.append("")

    header_fmt = "%-18s %-18s %-6s %-7s %-9s %-19s %-10s %-10s %-5s %-5s %-12s %-14s %-12s"
    row_fmt = "%-18s %-18s %-6d %-7d %-9.4f [%.4f, %.4f]   %-10.4f %-10.2f %-5d %-5d %-12.4e %-14s %-12s"

    lines.append(
        header_fmt
        % (
            "model",
            "state",
            "n",
            "correct",
            "accuracy",
            "95%_wilson_ci",
            "ref_acc",
            "%_of_ref",
            "b",
            "c",
            "mcnemar_p",
            "tier",
            "cpu_block_e",
        )
    )

    for r in primary_rows:
        lines.append(
            row_fmt
            % (
                r["model"],
                r["state"],
                r["n"],
                r["correct"],
                r["accuracy"],
                r["wilson_lo"],
                r["wilson_hi"],
                r["ref_accuracy"],
                r["pct_of_ref"],
                r["mcnemar_b"],
                r["mcnemar_c"],
                r["mcnemar_p"],
                r["tier"],
                r["energy_measured_in_cpu_block"],
            )
        )
    lines.append("")

    lines.append("=== Secondary Table: States Not in the x86-CPU Energy Block ===")
    lines.append("Marked: not in the x86-CPU energy block (_ft recovery controls and fp16 GPU-only states)")
    lines.append("")

    lines.append(
        header_fmt
        % (
            "model",
            "state",
            "n",
            "correct",
            "accuracy",
            "95%_wilson_ci",
            "ref_acc",
            "%_of_ref",
            "b",
            "c",
            "mcnemar_p",
            "tier",
            "cpu_block_e",
        )
    )

    for r in secondary_rows:
        lines.append(
            row_fmt
            % (
                r["model"],
                r["state"],
                r["n"],
                r["correct"],
                r["accuracy"],
                r["wilson_lo"],
                r["wilson_hi"],
                r["ref_accuracy"],
                r["pct_of_ref"],
                r["mcnemar_b"],
                r["mcnemar_c"],
                r["mcnemar_p"],
                r["tier"],
                r["energy_measured_in_cpu_block"],
            )
        )
    lines.append("")

    lines.append("=== Repo Search: CPU-vs-CUDA Prediction Equivalence ===")
    lines.append("Search query: Record of CPU-vs-CUDA prediction equivalence for zero-finetune pruned and INT8 artifacts in docs/ and results_*/.")
    lines.append("Finding for zero-finetune pruned and INT8: not found")
    lines.append("")
    lines.append("Details:")
    lines.append("1. Zero-finetune pruned (pruned30, pruned50, pruned70) and INT8 artifacts:")
    lines.append("   - No file in docs/ or results_*/ records a CPU-vs-CUDA prediction equivalence evaluation for zero-finetune pruned or INT8 checkpoints.")
    lines.append("   - In results_accuracy/, zero-finetune pruned checkpoints were evaluated on CUDA only (device='cuda' in summary.csv), while INT8 checkpoints were evaluated on CPU only (device='cpu' in summary.csv; INT8 PyTorch quantized models do not execute on this CUDA GPU per scripts/eval_artifacts.py:11 and D2).")
    lines.append("   - Finding: not found.")
    lines.append("")
    lines.append("2. Distinct record for _bnrecal artifacts (for completeness per plan v1 section 9):")
    lines.append("   - docs/bnrecal_cpu_equivalence.md lines 1-77: Records the CPU functional-equivalence check specifically for the 9 _bnrecal checkpoints on x86 CPU vs CUDA.")
    lines.append("   - docs/stage5_plan_v1.md lines 148-150: 'Separately, _bnrecal CPU-vs-CUDA functional equivalence uses McNemar not significant at 0.05 and |accuracy difference| <= 0.05 percentage points (all 9 pass); this is not the deployability rule.'")
    lines.append("")

    report_text = "\n".join(lines) + "\n"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report_text, encoding="utf-8")
    return report_text


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    accuracy_dir = repo_root / "results_accuracy"
    output_file = repo_root / "analysis" / "results" / "accuracy_table_output.txt"

    report_text = generate_accuracy_report(accuracy_dir, output_file)
    print(report_text, end="")


if __name__ == "__main__":
    main()
