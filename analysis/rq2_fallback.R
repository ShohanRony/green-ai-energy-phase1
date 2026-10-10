# RQ2 non-convergence fallback, registered in stage5_analysis_plan.md A7r6(b)/A7r9(a).
# analysis/rq2_model.R's fit_rq2_model() is UNCHANGED by this file -- this wraps it, never
# edits it.
#
# Operational definition of "fails to converge" (A7r9(a)): the fallback triggers if
# fit_rq2_model() raises an error OR returns converged == FALSE. A singular-but-converged fit
# stays on the primary (lmer) path -- singular alone never triggers the fallback (A7r6(a)).

suppressMessages(library(lmerTest))
# Sourced relative to the repository root, matching every other Rscript invocation in this
# project (run_all.R, the simulate_rq2*.R scripts) -- not auto-detected from commandArgs(),
# which does not reliably resolve a sibling file's own path when this file is itself source()'d
# from a test rather than run directly.
source("analysis/rq2_model.R")

#' Same input validation as fit_rq2_model()'s guards -- duplicated rather than sourced-and-
#' reused, matching analysis/rq2_diagnostics.R's own convention, so a validation failure (bad
#' columns, bnrecal states, duplicate rows) is rejected immediately here and never silently
#' routed into the lm() fallback as if it were a convergence problem.
.validate_rq2_input <- function(data, caller) {
  required_cols <- c("session", "model", "state", "log_ratio", "log_mac")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(caller, ": missing required column(s): ", paste(missing_cols, collapse = ", "))
  }
  if (any(grepl("bnrecal", data$state, ignore.case = TRUE))) {
    stop(caller, ": input contains _bnrecal state(s) -- zero-finetune pruned states only ",
         "(A7r4(f)).")
  }
  combo_counts <- table(interaction(data$session, data$model, data$state, drop = TRUE))
  if (any(combo_counts != 1)) {
    stop(caller, ": every (session, model, state) combination must occur at most once and ",
         "not repeat. Session is the replicate unit (A7 point 3).")
  }
}

#' Fit RQ2 via the primary model, falling back to session-as-fixed-factor OLS if the primary
#' fit errors or does not converge (A7r6(b)/A7r9(a)).
#'
#' @param data same shape as fit_rq2_model()'s input: session, model, state, log_ratio, log_mac.
#'   Unlike fit_rq2_model(), a missing (session, model, state) combination is accepted (reported
#'   via `complete_design`, not rejected) -- only a *repeated* combination is still an error,
#'   same as the primary model's own rule.
#' @param primary_fit_fn the primary fitting function to attempt first; defaults to
#'   fit_rq2_model(). Overridable for testing the fallback path deterministically (e.g. a mock
#'   that throws an error), without needing to engineer genuinely lmer-failing synthetic data.
#' @return a one-row data.frame with fit_rq2_model()'s columns (beta_hat, se, df, t_stat,
#'   p_one_sided, ci_lower, ci_upper, n_obs, n_sessions, n_models, converged, singular) plus
#'   `model_used` ("lmer" or "lm_fixed_session") and `complete_design` (TRUE only if
#'   n_obs == n_sessions * n_models * 3, the registered design's full cell count).
fit_rq2_with_fallback <- function(data, primary_fit_fn = fit_rq2_model) {
  .validate_rq2_input(data, "fit_rq2_with_fallback")

  n_sessions <- length(unique(data$session))
  n_models <- length(unique(data$model))
  n_obs <- nrow(data)
  complete_design <- isTRUE(n_obs == n_sessions * n_models * 3)

  primary_result <- tryCatch(primary_fit_fn(data), error = function(e) e)
  primary_errored <- inherits(primary_result, "error")
  primary_nonconverged <- !primary_errored && isFALSE(primary_result$converged)

  if (!primary_errored && !primary_nonconverged) {
    out <- primary_result
    out$model_used <- "lmer"
    out$complete_design <- complete_design
    return(out)
  }

  # Fallback (A7r6(b)): session as a fixed factor, ordinary least squares.
  data$model <- factor(data$model)
  data$session <- factor(data$session)
  fit <- lm(log_ratio ~ log_mac + model + session, data = data)
  coefs <- summary(fit)$coefficients
  if (!"log_mac" %in% rownames(coefs)) {
    stop("fit_rq2_with_fallback: log_mac coefficient not found in the fallback lm() fit -- ",
         "check input data (rank deficiency?).")
  }
  beta_hat <- coefs["log_mac", "Estimate"]
  se <- coefs["log_mac", "Std. Error"]
  df <- fit$df.residual
  t_stat <- (beta_hat - 1) / se
  p_one_sided <- pt(t_stat, df = df, lower.tail = TRUE)
  t_crit <- qt(0.975, df = df)
  ci_lower <- beta_hat - t_crit * se
  ci_upper <- beta_hat + t_crit * se

  data.frame(
    beta_hat = beta_hat,
    se = se,
    df = df,
    t_stat = t_stat,
    p_one_sided = p_one_sided,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    n_obs = n_obs,
    n_sessions = n_sessions,
    n_models = n_models,
    converged = TRUE,  # the OLS closed-form solution exists whenever lm() returns without
                        # error -- no iterative optimiser to fail to converge
    singular = NA,      # lme4::isSingular()'s boundary-variance-component concept does not
                         # apply to an OLS fixed-effects fit; left explicitly NA, not guessed
    model_used = "lm_fixed_session",
    complete_design = complete_design
  )
}
