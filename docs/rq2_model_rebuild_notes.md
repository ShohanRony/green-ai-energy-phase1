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

## Simulation Results

### Primary Model (fit_h2_gap_model)

| Scenario | True d | Bias | Emp SD | Mean SE | SE/SD | Coverage | Rej Rate | Non-conv | Singular |
|---|---|---|---|---|---|---|---|---|---|
| prop_c0.8 | -0.223 | 0.0010 | 0.0205 | 0.0198 | 0.965 | 0.953 | 0.000 | 115 | 114 |
| prop_c1 | 0.000 | 0.0004 | 0.0197 | 0.0196 | 0.995 | 0.952 | 0.044 | 145 | 143 |
| prop_c1.1 | 0.095 | 0.0013 | 0.0198 | 0.0197 | 0.995 | 0.959 | 0.992 | 110 | 106 |
| prop_c1.25 | 0.223 | 0.0001 | 0.0207 | 0.0200 | 0.966 | 0.943 | 1.000 | 98 | 95 |
| pow_b0.8 | 0.275 | -0.0001 | 0.0205 | 0.0195 | 0.950 | 0.946 | 1.000 | 110 | 105 |
| pow_b0.9 | 0.138 | -0.0005 | 0.0203 | 0.0196 | 0.968 | 0.954 | 1.000 | 124 | 122 |
| pow_b1 | 0.000 | 0.0005 | 0.0206 | 0.0198 | 0.961 | 0.945 | 0.063 | 114 | 110 |
| floor_f0.1 | 0.302 | 0.0004 | 0.0199 | 0.0196 | 0.989 | 0.960 | 1.000 | 112 | 102 |
| floor_f0.3 | 0.680 | 0.0010 | 0.0202 | 0.0196 | 0.969 | 0.945 | 1.000 | 118 | 109 |
| floor_f0.5 | 0.938 | -0.0005 | 0.0199 | 0.0199 | 1.001 | 0.956 | 1.000 | 105 | 99 |
| hetero_floor | 0.626 | -0.0004 | 0.0210 | 0.0195 | 0.929 | 0.938 | 1.000 | 164 | 103 |
| hetero_mean0 | 0.000 | 0.0002 | 0.0206 | 0.0194 | 0.944 | 0.942 | 0.054 | 124 | 123 |
| ctrl_prop1.0 | 0.000 | -0.0005 | 0.0180 | 0.0170 | 0.944 | 0.947 | 0.050 | 534 | 534 |
| prop1.0_noise1 | 0.000 | 0.0005 | 0.0089 | 0.0096 | 1.076 | 0.970 | 0.036 | 627 | 627 |
| prop1.0_noise2 | 0.000 | -0.0000 | 0.0213 | 0.0208 | 0.976 | 0.958 | 0.041 | 162 | 161 |
| prop1.0_noise3 | 0.000 | 0.0001 | 0.0127 | 0.0134 | 1.061 | 0.972 | 0.030 | 460 | 455 |

### Robustness Fallback (session_mean_t_test)

| Scenario | True d | Bias | Emp SD | Mean SE | SE/SD | Coverage | Rej Rate |
|---|---|---|---|---|---|---|---|
| prop_c0.8 | -0.223 | 0.0010 | 0.0205 | 0.0195 | 0.952 | 0.953 | 0.000 |
| prop_c1 | 0.000 | 0.0004 | 0.0197 | 0.0193 | 0.979 | 0.948 | 0.047 |
| prop_c1.1 | 0.095 | 0.0013 | 0.0198 | 0.0195 | 0.983 | 0.957 | 0.992 |
| prop_c1.25 | 0.223 | 0.0001 | 0.0207 | 0.0198 | 0.955 | 0.946 | 1.000 |
| pow_b0.8 | 0.275 | -0.0001 | 0.0205 | 0.0193 | 0.938 | 0.948 | 1.000 |
| pow_b0.9 | 0.138 | -0.0005 | 0.0203 | 0.0193 | 0.953 | 0.953 | 1.000 |
| pow_b1 | 0.000 | 0.0005 | 0.0206 | 0.0195 | 0.949 | 0.947 | 0.064 |
| floor_f0.1 | 0.302 | 0.0004 | 0.0199 | 0.0194 | 0.980 | 0.960 | 1.000 |
| floor_f0.3 | 0.680 | 0.0010 | 0.0202 | 0.0193 | 0.957 | 0.951 | 1.000 |
| floor_f0.5 | 0.938 | -0.0005 | 0.0199 | 0.0196 | 0.989 | 0.959 | 1.000 |
| hetero_floor | 0.626 | -0.0004 | 0.0210 | 0.0194 | 0.925 | 0.946 | 1.000 |
| hetero_mean0 | 0.000 | 0.0002 | 0.0206 | 0.0191 | 0.931 | 0.937 | 0.059 |
| ctrl_prop1.0 | 0.000 | -0.0005 | 0.0180 | 0.0170 | 0.941 | 0.949 | 0.050 |
| prop1.0_noise1 | 0.000 | 0.0005 | 0.0089 | 0.0088 | 0.986 | 0.955 | 0.051 |
| prop1.0_noise2 | 0.000 | -0.0000 | 0.0213 | 0.0204 | 0.960 | 0.954 | 0.043 |
| prop1.0_noise3 | 0.000 | 0.0001 | 0.0127 | 0.0122 | 0.962 | 0.952 | 0.041 |

## Acceptance Criteria Check

For scenarios where true mean d = 0 (prop_c1, hetero_mean0, ctrl_prop1.0, prop1.0_noise*):
- **prop_c1**: PASS (Rej: 0.044 pass, Cov: 0.952 pass, SE/SD: 0.995 pass)
- **hetero_mean0**: PASS (Rej: 0.054 pass, Cov: 0.942 pass, SE/SD: 0.944 pass)
- **ctrl_prop1.0**: PASS (Rej: 0.050 pass, Cov: 0.947 pass, SE/SD: 0.944 pass)
- **prop1.0_noise1**: PASS (Rej: 0.036 pass, Cov: 0.970 pass, SE/SD: 1.076 pass)
- **prop1.0_noise2**: PASS (Rej: 0.041 pass, Cov: 0.958 pass, SE/SD: 0.976 pass)
- **prop1.0_noise3**: FAIL (Rej: 0.030 pass, Cov: 0.972 FAIL, SE/SD: 1.061 pass)

**Overall Acceptance Criteria: FAILED**
