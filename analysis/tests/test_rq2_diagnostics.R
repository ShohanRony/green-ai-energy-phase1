# Plain-assertion tests for analysis/rq2_diagnostics.R, on synthetic data only.
# Run standalone: Rscript analysis/tests/test_rq2_diagnostics.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE),
                                                   value = TRUE))), "..", "rq2_diagnostics.R"))

pass_count <- 0
check <- function(expr, msg) {
  if (!isTRUE(expr)) stop("FAILED: ", msg)
  pass_count <<- pass_count + 1
}

models <- c("resnet18", "mobilenet_v3_small", "efficientnet_b0")
mac <- c(0.484, 0.252, 0.090, 0.533, 0.309, 0.132, 0.516, 0.288, 0.119)
states <- rep(c("pruned30", "pruned50", "pruned70"), 3)
mdl <- rep(models, each = 3)
log_mac <- log(mac)

make_data <- function(beta, gamma = 0, model_offset = c(0, 0.1, -0.1), session_sd = 0.02,
                       residual_sd = 0.02, seed) {
  set.seed(seed)
  beta_vec <- if (length(beta) == 1) rep(beta, length(log_mac)) else beta[mdl]
  rows <- lapply(seq_len(6), function(s) {
    sess_eff <- rnorm(1, 0, session_sd)
    lr <- beta_vec * log_mac + gamma * log_mac^2 + model_offset[match(mdl, models)] +
      sess_eff + rnorm(9, 0, residual_sd)
    data.frame(session = s, model = mdl, state = states, log_ratio = lr, log_mac = log_mac)
  })
  do.call(rbind, rows)
}

# --- fit_curvature_diagnostic: known curvature detected ---
d_curv <- make_data(beta = 0.8, gamma = 0.15, seed = 42)
res_curv <- fit_curvature_diagnostic(d_curv)
check(nrow(res_curv) == 1, "fit_curvature_diagnostic: returns exactly one row")
check(all(c("quad_estimate", "quad_se", "quad_df", "quad_ci_lower", "quad_ci_upper",
            "lrt_chisq", "lrt_df", "lrt_p", "converged", "singular",
            "n_obs", "n_sessions", "n_models") %in% names(res_curv)),
      "fit_curvature_diagnostic: output has all required columns")
check(abs(res_curv$quad_estimate - 0.15) < 0.05,
      "fit_curvature_diagnostic: quad_estimate recovers strong known curvature (+-0.05)")
check(res_curv$quad_ci_lower > 0,
      "fit_curvature_diagnostic: CI excludes 0 for strong known curvature")
check(res_curv$lrt_p < 0.001, "fit_curvature_diagnostic: LRT strongly rejects for known curvature")
check(res_curv$lrt_df == 1, "fit_curvature_diagnostic: LRT has 1 degree of freedom")

# --- fit_curvature_diagnostic: no curvature, CI should include 0 (spot check) ---
d_lin <- make_data(beta = 0.8, gamma = 0, seed = 42)
res_lin <- fit_curvature_diagnostic(d_lin)
check(res_lin$quad_ci_lower < 0 && res_lin$quad_ci_upper > 0,
      "fit_curvature_diagnostic: CI includes 0 for a genuinely linear relationship (spot check)")
check(res_lin$lrt_p > 0.05,
      "fit_curvature_diagnostic: LRT does not reject for a genuinely linear relationship (spot check)")

# --- fit_heterogeneity_diagnostic: known heterogeneity detected ---
het_true <- c(resnet18 = 0.4, mobilenet_v3_small = 0.8, efficientnet_b0 = 1.2)
d_het <- make_data(beta = het_true, seed = 7)
res_het <- fit_heterogeneity_diagnostic(d_het)
check(nrow(res_het$per_model) == 3, "fit_heterogeneity_diagnostic: one row per architecture")
check(all(c("model", "slope", "se", "df", "ci_lower", "ci_upper") %in% names(res_het$per_model)),
      "fit_heterogeneity_diagnostic: per_model has all required columns")
for (m in names(het_true)) {
  est <- res_het$per_model$slope[res_het$per_model$model == m]
  check(abs(est - het_true[[m]]) < 0.1,
        paste0("fit_heterogeneity_diagnostic: recovers ", m, "'s known slope (+-0.1)"))
}
check(res_het$lrt$lrt_p < 0.001,
      "fit_heterogeneity_diagnostic: LRT strongly rejects for known heterogeneity")
check(res_het$lrt$lrt_df == 2, "fit_heterogeneity_diagnostic: interaction LRT has 2 degrees of freedom")

# --- fit_heterogeneity_diagnostic: no heterogeneity, LRT should not reject (spot check) ---
d_hom <- make_data(beta = 0.8, seed = 7)
res_hom <- fit_heterogeneity_diagnostic(d_hom)
check(res_hom$lrt$lrt_p > 0.05,
      "fit_heterogeneity_diagnostic: LRT does not reject for genuinely homogeneous slopes (spot check)")

# --- input validation, both functions (same guards as fit_rq2_model) ---
check(inherits(tryCatch(fit_curvature_diagnostic(data.frame(session = 1)), error = function(e) e),
               "error"),
      "fit_curvature_diagnostic: must error on missing required columns")
check(inherits(tryCatch(fit_heterogeneity_diagnostic(data.frame(session = 1)), error = function(e) e),
               "error"),
      "fit_heterogeneity_diagnostic: must error on missing required columns")

d_bnrecal <- d_lin
d_bnrecal$state[1] <- "pruned30_bnrecal"
check(inherits(tryCatch(fit_curvature_diagnostic(d_bnrecal), error = function(e) e), "error"),
      "fit_curvature_diagnostic: must error when bnrecal states are present")
check(inherits(tryCatch(fit_heterogeneity_diagnostic(d_bnrecal), error = function(e) e), "error"),
      "fit_heterogeneity_diagnostic: must error when bnrecal states are present")

d_dup <- rbind(d_lin, d_lin[1, ])
check(inherits(tryCatch(fit_curvature_diagnostic(d_dup), error = function(e) e), "error"),
      "fit_curvature_diagnostic: must error when a (session, model, state) combination repeats")
check(inherits(tryCatch(fit_heterogeneity_diagnostic(d_dup), error = function(e) e), "error"),
      "fit_heterogeneity_diagnostic: must error when a (session, model, state) combination repeats")

# --- nominal-rate check over many replicates, pure null (no curvature, no heterogeneity) ---
# Smaller n and looser tolerance than the full simulation study (analysis/simulate_rq2.R) --
# this is a fast sanity check against a grossly broken LRT (e.g. always rejecting), not a
# precise calibration claim; the full-scale calibration check lives in the simulation study.
n_null_reps <- 100
quad_rejections <- 0
interaction_rejections <- 0
for (i in seq_len(n_null_reps)) {
  d_null <- make_data(beta = 0.8, gamma = 0, seed = 5000 + i)
  r_curv <- suppressMessages(suppressWarnings(fit_curvature_diagnostic(d_null)))
  r_het <- suppressMessages(suppressWarnings(fit_heterogeneity_diagnostic(d_null)))
  if (!is.na(r_curv$lrt_p) && r_curv$lrt_p < 0.05) quad_rejections <- quad_rejections + 1
  if (!is.na(r_het$lrt$lrt_p) && r_het$lrt$lrt_p < 0.05) interaction_rejections <- interaction_rejections + 1
}
quad_rate <- quad_rejections / n_null_reps
interaction_rate <- interaction_rejections / n_null_reps
cat(sprintf("null-case quad LRT rejection rate: %.3f (%d/%d)\n",
            quad_rate, quad_rejections, n_null_reps))
cat(sprintf("null-case interaction LRT rejection rate: %.3f (%d/%d)\n",
            interaction_rate, interaction_rejections, n_null_reps))
check(quad_rate <= 0.15,
      "fit_curvature_diagnostic: null-case LRT does not fire far more than the nominal 0.05 rate")
check(interaction_rate <= 0.15,
      "fit_heterogeneity_diagnostic: null-case LRT does not fire far more than the nominal 0.05 rate")

cat("test_rq2_diagnostics.R: ", pass_count, " checks passed\n", sep = "")
