# Paper Requirements Checklist

Date: 2026-10-16

This checklist records items that must be included, disclosed, or addressed in the manuscript, along with their respective source files and commit references.

---

### 1. Registration Status and Deposit Timing
- **Item:** The git tag `stage5-a7-frozen-20261012` (commit `70f84f0`) predates data collection (which took place 13–14 October 2026). The Zenodo archive (DOI [10.5281/zenodo.23357783](https://zenodo.org/records/23357783)) was deposited on 16 October 2026 after data collection and prior to confirmatory analysis (deviation D34). The manuscript must describe the registration as git-tag-based and state explicitly that the Zenodo deposit occurred after data collection; it must never be characterized as a prospective registration. Note: Deviation D34 cites "plan v1 section 11 / A11" for the deposit-order rule; the actual source of the rule is `docs/a7_freeze_checklist.md`.
- **Sources:** Commit `70f84f0`, `docs/a7_freeze_checklist.md`, `docs/deviation_log.md` (D34).

### 2. Primary H2 Confirmatory Result
- **Item:** Confirmatory H2 test result: mean gap $d = \log(r) - \log(\text{MAC ratio}) = 0.824$, 95% CI $[0.815, 0.833]$, $t = 243.16$, $p = 1.1 \times 10^{-11}$, df = 5. Exponentiated point estimate: $\exp(0.824) = 2.28\times$ (95% CI $[2.26, 2.30]$), demonstrating that theoretical MAC reductions overstate realized energy savings by 2.28-fold on average. Scope: x86-CPU, RAPL package-0, batch size 1, $32 \times 32$ image inputs, zero-finetune structured magnitude pruning.
- **Sources:** Commit `6c1818b`, `analysis/results/h2_primary_output.txt`.

### 3. Model Accuracy and Deployability Rule
- **Item:** Top-1 accuracy evaluated on the CIFAR-10 test set ($n = 10{,}000$): only ResNet-18 INT8 meets the registered 99% relative deployability threshold (clearing the 99.9% strict tier at 100.03% of reference). All 9 zero-finetune pruned states fail the 99% deployability threshold, with several collapsing near chance performance (10.0%–17.6%). The paper must clarify that H2 evaluates computational scaling vs. hardware energy draw, not deployable models in production, and must present the full accuracy table directly alongside the energy results.
- **Sources:** Commit `3d8377b`, `analysis/results/accuracy_table_output.txt`.

### 4. Recorded Omission of the `_bnrecal` Separate Fit
- **Item:** The registered separate mixed-model fit on the `_bnrecal` cells was not computed. Required statement:
  > "The registered separate fit for the _bnrecal cells could not be computed as specified because only 6 of the 9 cells exist by design (known-collapsed combinations were not measured); the question it addresses, whether energy depends on weights at fixed architecture, is answered by decision rule D3r1, under which all six measured pairs were equivalent within +/-5%."
- **Sources:** `analysis/results/sensitivity_part2_output.txt`, `docs/deviation_log.md` (D36), researcher decision recorded 2026-10-16.

### 5. Runtime Confound (D2r1) and Baseline Dominance in Specification Curve
- **Item:** Under decision rule D2r1, FP32-TorchScript and FP32-eager are not equivalent within $\pm 5\%$ for any architecture (eager draws substantially more energy: ResNet-18 $+5.4\%$, MobileNetV3-Small $+46.2\%$, EfficientNet-B0 $+36.4\%$). The primary CPU baseline remains FP32-TorchScript as registered. In the specification curve analysis, runtime baseline choice dominates the specification spread, while instrument (package-0 vs. psys) and quantity (gross vs. net) exert minimal influence. Within each architecture, $sd_{\text{spec}}$ is nearly constant across pruning states because shifting the baseline from TorchScript to eager mechanically shifts all states of that architecture by the same constant log ratio.
- **Sources:** Commit `48479a3` (`analysis/results/sensitivity_part1_output.txt`), commit `73f435b` (`analysis/results/sensitivity_part2_output.txt`).

### 6. Sensitivity Diagnostics and Model Interpretation
- **Item:** Diagnostic checks reveal that per-architecture slopes differ substantially (heterogeneity LRT $p = 1.49 \times 10^{-38}$; ResNet-18 slope 0.812, EfficientNet-B0 slope 0.222, MobileNetV3-Small slope 0.129), and mild curvature is present ($p = 0.0287$). These sensitivity diagnostics never alter or substitute for the primary H2 confirmatory test. The reported coefficient $\beta$ represents the best-fit log-linear slope over the empirically observed pruning range (A7r8), and the pooled slope is a Frisch-Waugh-Lovell weighted average of the individual architecture slopes.
- **Sources:** Commit `48479a3`, `analysis/results/sensitivity_part1_output.txt`.

### 7. Thermal Throttle Guard Execution (A11)
- **Item:** In all 6 CPU confirmatory sessions (144 conditions, 1,440 warm active windows), zero package-throttle events occurred (`thermal_pkg_throttle_count_delta = 0` in every window). The manuscript must state openly that the initial pre-calibration throttle-counter criterion was removed based on calibration evidence prior to confirmatory data collection, leaving the calibrated busy-core frequency threshold and post-hoc package-throttle sensitivity check in place.
- **Sources:** Commit `73f435b`, `analysis/results/sensitivity_part2_output.txt`, `docs/stage5_analysis_plan.md` (A11).

### 8. Specification Curve Missing Points
- **Item:** Out of 8 potential specification points on x86-CPU (2 instruments $\times$ 2 runtimes $\times$ 2 quantities), exactly 6 points are available. The 2 psys net points (`(psys, fp32_ts, net)` and `(psys, fp32_eager, net)`) are unavailable because no psys-specific idle quantity or psys net column was logged in `windows.csv`.
- **Sources:** Commit `73f435b`, `scripts/repeat_level_energy.py`, `analysis/results/sensitivity_part2_output.txt`.

### 9. Device Discrepancy in Prediction vs. Energy Measurement
- **Item:** Zero-finetune pruned checkpoints were evaluated for accuracy on CUDA, whereas their energy was measured on CPU. CPU-vs-CUDA prediction equivalence was empirically verified only for the `_bnrecal` checkpoints (`docs/bnrecal_cpu_equivalence.md`); no record exists in the repository for zero-finetune pruned or INT8 checkpoints (deviation D37). Status: OPEN, with the researcher deciding whether to run a post-hoc CPU prediction check.
- **Sources:** Commit `3d8377b`, `analysis/results/accuracy_table_output.txt`, `docs/deviation_log.md` (D37).

### 10. Complete Deviation Table
- **Item:** The manuscript must include a comprehensive deviation table covering deviations D21 through D37, highlighting:
  - D32: Voided session 1 attempt 1 due to TorchScript checkpoint loading error and subsequent clean rerun.
  - D29: CPU scaling governor reset to `performance` and enforcement of governor gate.
  - D30: Configuration fix restricting `pilot.py` to batch 1 at $32 \times 32$ inputs.
  - D31: Fresh-boot uptime check relocation to session start.
- **Sources:** `docs/deviation_log.md` (D21–D37).

### 11. Known Design Limitations
- **Item:** The discussion must explicitly report known design limitations:
  - Single training seed (2026) used for initial checkpoint training.
  - Epoch count disparity for ResNet-18 (30 vs. 60 epochs).
  - Distinct quantization pipelines across architectures for INT8.
  - Absence of FP16 on x86-CPU.
  - Single physical hardware testbed device.
- **Sources:** `docs/stage5_plan_v1.md` sections 3, 7, 9; `docs/deviation_log.md`.

### 12. Cross-Platform and Exploratory Scope Boundaries
- **Item:** The x86-GPU block data are designated as supplementary only (due to the eager-baseline caveat). The M1-CPU block is deferred to subsequent work, and Series R (runtime-bridging experiment) is deferred.
- **Sources:** `docs/stage5_plan_v1.md` sections 1, 2, 3; `docs/stage5_analysis_plan.md`.

### 13. AI-Use Disclosure
- **Item:** The Methods section must include an AI-use disclosure. The exact wording will be decided by the researcher prior to submission, taking into account target journal policies. The statement must accurately describe AI assistance in authoring harness code and analysis scripts, as well as in analysis design and sanity-checking, while confirming that all empirical results and statistical outputs were verified independently.
- **Sources:** Target journal submission guidelines and author disclosures.

### 14. Session Logs and Console Evidence
- **Item:** The raw console logs (`console.log`) across all CPU sessions contain only benign third-party library deprecation warnings. Confirmatory session evidence rests upon the structured machine-generated logs: `session_log.json`, `void_log.json`, and per-condition `windows.csv`, `summary.csv`, `raw.jsonl`, and `environment.json`.
- **Sources:** `results_stage5/cpu/session_01..06/console.log`, `session_log.json`, `void_log.json`.
