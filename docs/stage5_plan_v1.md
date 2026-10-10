# v1.0 DRAFT, awaiting researcher confirmation

## Scope and Blocks
- **x86-CPU**: The single confirmatory block.
- **M1-CPU**: Registered replication with identical analysis, reported whether or not it replicates.
- **x86-GPU**: Supplementary block (FP32-eager baseline, per-state ratios with the eager-baseline caveat).
- **Series R**: Deferred (GPU spec-curve runtime axis is unavailable).

## Research Questions
- **RQ1a (Coverage)**: How much of a wall-meter-derived energy footprint software-based energy estimation captures.
- **RQ1b (Estimation Mode)**: CodeCarbon with hardware counters available vs. forced estimation mode (TDP heuristics).
- **RQ2 (literal)**: FLOPs/MAC reduction overstates energy savings.
- **RQ3**: Is the rank ordering of compression states architecture-dependent?

## H2 and Tests
- **H2 Definition**: Per state, `d = log(energy ratio) - log(MAC ratio)`. Computed using the 9 zero-finetune pruned states (3 architectures: ResNet-18, MobileNetV3-Small, EfficientNet-B0 x pruned30/50/70) against the same-session FP32-TorchScript baseline.
- **Primary Test**: Compute each session's mean of its 9 `d` values. Perform a one-sample t-test across the 6 session means (df = 5), one-sided, H2: mean d > 0, alpha = 0.05.
- **Completeness Rule**: The primary test requires all 9 cells in all 6 sessions after the registered void/rerun procedure. If any cell is missing, the primary becomes the cell-fixed mixed model `d ~ 0 + cell + (1|session) + (1|session:model)`, REML, Satterthwaite, equal-weight contrast of the 9 cell means. The trigger is data completeness only, decided before results.
- **Headline**: `exp(mean d)` with its 95% CI. Per-cell `d` with 95% CIs are descriptive, no p-values. A non-rejection is reported as inconclusive, never as "proportional".
- **Cross-check (always reported)**: The cell-fixed mixed model above. Singular or non-converged flags are reported but never change the primary result (the old non-convergence fallback A7r6(b)/A7r9(a) no longer applies).
- **Secondary (not confirmatory)**: Anchored per-architecture slopes through the FP32 anchor (descriptive, known to be biased under an energy floor).
- **Sensitivity (not confirmatory)**: Free-intercept slope model with `(1|session) + (1|session:model)`. Curvature and heterogeneity diagnostics are reported with corrected framing (pooled slope = Frisch-Waugh-weighted average; heterogeneity causes power loss, not type I error inflation).

## Baselines
- CPU: same-session FP32-TorchScript baseline.
- GPU: FP32-eager baseline.

## Sessions and Seeds
- **Replicate Unit**: Session is the replicate unit (6 sessions per block).
- **Seeds**: Fixed seeds as previously registered before any session execution.

## Void/Rerun Rules and Exclusions
- **Thermal Guard**: Hard gate. Any thermal guard abort voids the condition, and it must be rerun appended at the end of the same session's run order.
- **AC Guard**: Abort voids the entire session.
- **Multiple Voids**: More than 2 voided conditions in one session voids the whole session.
- **Exclusions**: The cold-start rep is an exclusion (not a void-and-rerun trigger).

## Accuracy Rule
- **Rule**: McNemar test not significant at alpha = 0.05 AND |accuracy difference| <= 0.05 percentage points. 

## D1r1-D4r1 Status
- **Redaction**: None. D22/D23/D25 deposited as written.

## Spec Curve
- Spec curve generated per block.

## "Looked At" Statement
- No statistical test was changed after looking at the corresponding real energy measurement results.

## Superseded Amendment Rule Mapping
| Superseded Rule | v1.0 Section |
|---|---|
| A1-A9 | Scope and Blocks / Void Rules |
| A7r1-A7r9 | H2 and Tests / Completeness Rule |
| A8-A8r2 | Accuracy Rule / Baselines |
