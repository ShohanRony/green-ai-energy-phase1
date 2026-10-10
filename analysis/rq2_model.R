# RQ2 model, exactly as registered in stage5_analysis_plan.md A7r3(f)/A7r4(f)/A7r7(e):
#   log(r) ~ beta*log(MAC ratio) + model (fixed, 3 levels) + (1|session)
#   fit via lmerTest::lmer, REML, Satterthwaite df
#   H2: beta < 1, one-sided, t = (beta_hat - 1)/SE, left-tail p-value
#   report beta_hat with its 95% CI (same Satterthwaite df)
#   input unit: exactly one row per (session, model, state) -- session is the replicate unit
#   (A7 point 3), not the rep; rep-level rows are not accepted (A7r7(e)).
# Primary fit uses zero-finetune pruned states only (A7r4(f)) -- bnrecal states are a separate
# sensitivity fit, not this function's concern; this function refuses bnrecal input rather than
# silently including it.

suppressMessages(library(lmerTest))

#' Fit the registered RQ2 model and return the one-row H2 test result.
#'
#' @param data data.frame with columns: session, model, state, log_ratio, log_mac.
#'   `model` must have exactly the 3 registered levels (ResNet-18, MobileNetV3-Small,
#'   EfficientNet-B0, any labelling consistent within the data). `state` must not contain
#'   any `_bnrecal` entries -- those belong to the separate sensitivity fit, not this one.
#'   Every (session, model, state) combination must occur exactly once -- session is the
#'   replicate unit (A7 point 3); a data frame with multiple rows per combination (e.g. one row
#'   per rep, rather than one already-summarised row per session) is rejected.
#' @return a one-row data.frame: beta_hat, se, df, t_stat, p_one_sided, ci_lower, ci_upper,
#'   n_obs, n_sessions, n_models, converged, singular. `converged` and `singular` are
#'   independent flags (A7r6(a)/A7r7): `singular` is `lme4::isSingular()`; `converged` is the
#'   optimiser's own return code (0) with no non-singularity warning -- a singular fit can still
#'   be `converged`.
fit_rq2_model <- function(data) {
  required_cols <- c("session", "model", "state", "log_ratio", "log_mac")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop("fit_rq2_model: missing required column(s): ", paste(missing_cols, collapse = ", "))
  }
  if (any(grepl("bnrecal", data$state, ignore.case = TRUE))) {
    stop("fit_rq2_model: input contains _bnrecal state(s) -- the primary RQ2 fit uses ",
         "zero-finetune pruned states only (A7r4(f)); bnrecal belongs to the separate ",
         "sensitivity fit, not this function.")
  }
  combo_counts <- table(interaction(data$session, data$model, data$state, drop = TRUE))
  if (any(combo_counts != 1)) {
    stop("fit_rq2_model: every (session, model, state) combination must occur exactly once. ",
         "Session is the replicate unit (A7 point 3) -- this function expects one already-",
         "summarised row per session, not one row per repetition. Pre-aggregate rep-level data ",
         "to one log_ratio per (session, model, state) before calling this function.")
  }
  n_models <- length(unique(data$model))
  if (n_models != 3) {
    warning("fit_rq2_model: expected 3 model levels, found ", n_models,
             " -- proceeding, but this is not the registered design.")
  }

  data$model <- factor(data$model)
  data$session <- factor(data$session)

  fit <- lmerTest::lmer(log_ratio ~ log_mac + model + (1 | session), data = data, REML = TRUE)

  coefs <- summary(fit)$coefficients
  if (!"log_mac" %in% rownames(coefs)) {
    stop("fit_rq2_model: log_mac coefficient not found in fitted model -- check input data.")
  }
  beta_hat <- coefs["log_mac", "Estimate"]
  se <- coefs["log_mac", "Std. Error"]
  df <- coefs["log_mac", "df"]

  # H2: beta < 1 -- NOT the package's default test (which is against 0). Computed explicitly.
  t_stat <- (beta_hat - 1) / se
  p_one_sided <- pt(t_stat, df = df, lower.tail = TRUE)

  t_crit <- qt(0.975, df = df)
  ci_lower <- beta_hat - t_crit * se
  ci_upper <- beta_hat + t_crit * se

  # `converged` and `singular` are independent (A7r6(a)/A7r7): a singular fit is not by itself
  # a non-convergence -- the optimiser can report success (code 0) with no warning other than
  # the boundary-singular message, which is filtered out here and tracked only via `singular`.
  opt_code <- fit@optinfo$conv$opt
  conv_messages <- fit@optinfo$conv$lme4$messages
  if (is.null(conv_messages)) conv_messages <- character(0)
  non_singular_messages <- conv_messages[!grepl("^boundary \\(singular\\) fit", conv_messages)]
  converged <- isTRUE(opt_code == 0) && length(non_singular_messages) == 0
  singular <- lme4::isSingular(fit)

  data.frame(
    beta_hat = beta_hat,
    se = se,
    df = df,
    t_stat = t_stat,
    p_one_sided = p_one_sided,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    n_obs = nrow(data),
    n_sessions = length(unique(data$session)),
    n_models = n_models,
    converged = converged,
    singular = singular
  )
}
