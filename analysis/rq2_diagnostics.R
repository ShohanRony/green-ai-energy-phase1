# RQ2 sensitivity diagnostics, registered in stage5_analysis_plan.md A7r8.
# These NEVER replace the primary RQ2 fit and carry no multiplicity adjustment (A7r8).
# Note (BLOCK U): The researcher decided the primary H2 test is now on the gap d = log_ratio - log_mac.
# These curvature and heterogeneity diagnostics now apply strictly as sensitivity checks alongside
# the secondary/sensitivity models in analysis/rq2_gap_model.R. They have been updated to include
# the (1|session:model) term.
#
# (a) Curvature diagnostic: log_ratio ~ log_mac + I(log_mac^2) + model + (1|session) + (1|session:model)
# (b) Slope-heterogeneity diagnostic: log_ratio ~ 0 + model + log_mac:model + (1|session) + (1|session:model)

suppressMessages(library(lmerTest))

.validate_rq2_diag_input <- function(data, caller) {
  required_cols <- c("session", "model", "state", "log_ratio", "log_mac")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(caller, ": missing required column(s): ", paste(missing_cols, collapse = ", "))
  }
  if (any(grepl("bnrecal", data$state, ignore.case = TRUE))) {
    stop(caller, ": input contains _bnrecal state(s).")
  }
  combo_counts <- table(interaction(data$session, data$model, data$state, drop = TRUE))
  if (any(combo_counts != 1)) {
    stop(caller, ": every (session, model, state) combination must occur exactly once.")
  }
}

.assess_fit_quality <- function(fit) {
  opt_code <- fit@optinfo$conv$opt
  conv_messages <- fit@optinfo$conv$lme4$messages
  if (is.null(conv_messages)) conv_messages <- character(0)
  non_singular_messages <- conv_messages[!grepl("^boundary \\(singular\\) fit", conv_messages)]
  list(converged = isTRUE(opt_code == 0) && length(non_singular_messages) == 0,
       singular = lme4::isSingular(fit))
}

fit_curvature_diagnostic <- function(data) {
  .validate_rq2_diag_input(data, "fit_curvature_diagnostic")
  data$model <- factor(data$model)
  data$session <- factor(data$session)

  fit_reml <- lmerTest::lmer(log_ratio ~ log_mac + I(log_mac^2) + model + (1 | session) + (1 | session:model),
                              data = data, REML = TRUE)
  coefs <- summary(fit_reml)$coefficients
  quad_row <- "I(log_mac^2)"
  if (!quad_row %in% rownames(coefs)) stop("fit_curvature_diagnostic: quadratic coefficient not found.")
  
  quad_estimate <- coefs[quad_row, "Estimate"]
  quad_se <- coefs[quad_row, "Std. Error"]
  quad_df <- coefs[quad_row, "df"]
  t_crit <- qt(0.975, df = quad_df)
  quad_ci_lower <- quad_estimate - t_crit * quad_se
  quad_ci_upper <- quad_estimate + t_crit * quad_se

  fit_reduced_ml <- lme4::lmer(log_ratio ~ log_mac + model + (1 | session) + (1 | session:model), data = data, REML = FALSE)
  fit_full_ml <- lme4::lmer(log_ratio ~ log_mac + I(log_mac^2) + model + (1 | session) + (1 | session:model), data = data, REML = FALSE)
  
  lrt <- anova(fit_reduced_ml, fit_full_ml)
  quality <- .assess_fit_quality(fit_reml)

  data.frame(
    quad_estimate = quad_estimate, quad_se = quad_se, quad_df = quad_df,
    quad_ci_lower = quad_ci_lower, quad_ci_upper = quad_ci_upper,
    lrt_chisq = lrt[["Chisq"]][2], lrt_df = lrt[["Df"]][2], lrt_p = lrt[["Pr(>Chisq)"]][2],
    converged = quality$converged, singular = quality$singular,
    n_obs = nrow(data), n_sessions = length(unique(data$session)), n_models = length(unique(data$model))
  )
}

fit_heterogeneity_diagnostic <- function(data) {
  .validate_rq2_diag_input(data, "fit_heterogeneity_diagnostic")
  data$model <- factor(data$model)
  data$session <- factor(data$session)
  model_levels <- levels(data$model)

  fit_reml <- lmerTest::lmer(log_ratio ~ 0 + model + log_mac:model + (1 | session) + (1 | session:model),
                              data = data, REML = TRUE)
  coefs <- summary(fit_reml)$coefficients
  per_model <- do.call(rbind, lapply(model_levels, function(m) {
    row_name <- paste0("model", m, ":log_mac")
    if (!row_name %in% rownames(coefs)) stop("Not found: ", row_name)
    est <- coefs[row_name, "Estimate"]
    se <- coefs[row_name, "Std. Error"]
    df <- coefs[row_name, "df"]
    t_crit <- qt(0.975, df = df)
    data.frame(model = m, slope = est, se = se, df = df, ci_lower = est - t_crit * se, ci_upper = est + t_crit * se)
  }))
  rownames(per_model) <- NULL

  fit_reduced_ml <- lme4::lmer(log_ratio ~ log_mac + model + (1 | session) + (1 | session:model), data = data, REML = FALSE)
  fit_full_ml <- lme4::lmer(log_ratio ~ log_mac * model + (1 | session) + (1 | session:model), data = data, REML = FALSE)
  
  lrt <- anova(fit_reduced_ml, fit_full_ml)
  quality <- .assess_fit_quality(fit_reml)

  list(
    per_model = per_model,
    lrt = data.frame(lrt_chisq = lrt[["Chisq"]][2], lrt_df = lrt[["Df"]][2], lrt_p = lrt[["Pr(>Chisq)"]][2]),
    converged = quality$converged, singular = quality$singular,
    n_obs = nrow(data), n_sessions = length(unique(data$session)), n_models = length(model_levels)
  )
}
