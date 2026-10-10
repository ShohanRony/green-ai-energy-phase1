# RQ2 Model Rebuild Notes
*(Derived note, not a registered amendment)*

## Context and Decision
The researcher decided that Hypothesis 2 (H2) will be tested directly on the "gap" $d = \log(r) - \log(\text{MAC ratio})$, asking if $d > 0$. The primary estimand is the equal-weight mean of $d$ across the 9 prune-level/architecture cells.

## Model Refinement
- **Primary Model**: `lmerTest::lmer(d ~ 0 + cell + (1|session) + (1|session:model), REML = TRUE)` where `cell` is the 9-level interaction of model and state.
- **Why this formulation?**: Introducing the `(1|session:model)` nested random effect properly accounts for the baseline noise that is SHARED by the three prune states of the same architecture within the same session (since $\log(r) = \log(\text{pruned energy}) - \log(\text{baseline energy})$). Without it, the model assumes independent observations within a session and underestimates the standard error (SE), as demonstrated by the simulation SE/SD ratios. The 9-cell fixed effect ensures that architecture-specific differences in average MAC ratio ranges do not implicitly weight the estimator (addressing the Frisch-Waugh weighting issue).

## Robustness and Fallback Rule
If the mixed model fails to converge or throws an error, we now explicitly fall back to a simple session-mean t-test (`session_mean_t_test`). This fallback strictly requires a complete design (all 9 cells present per session) to compute a valid equal-weight mean per session before applying the t-test (df = n_sessions - 1). This ensures that even under severe fitting issues, a robust, conservative fallback exists.

## Simulation Design and Acceptance Criteria
We generated synthetic data mirroring the actual design constants (MAC ratios). The generator constructs $\log(r)$ directly as the difference of state energy and shared baseline energy, explicitly injecting shared baseline noise per `(session, model)`.
We evaluated 13 scenarios across proportional, power law, energy floor, and heterogeneous truths.
**Acceptance Criteria for true mean d = 0**:
- Rejection rate (type I error) in [0.03, 0.08]
- 95% CI coverage in [0.93, 0.97]
- Mean SE / empirical SD in [0.90, 1.10]

