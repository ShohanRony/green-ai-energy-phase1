# RQ2 sensitivity diagnostics, registered in stage5_analysis_plan.md A7r8.
# These NEVER replace the primary RQ2 fit (analysis/rq2_model.R, unchanged by this file) and
# carry no multiplicity adjustment (A7r8) -- they are reported alongside the primary fit's
# result, not in place of it.
#
# (a) Curvature diagnostic: adds a quadratic term in log_mac to check whether the primary
#     model's linearity assumption holds over the observed MAC-ratio range.
# (b) Slope-heterogeneity diagnostic: lets the log_mac slope vary by architecture, to check
#     whether the primary model's single pooled slope is a reasonable summary across the three
#     architectures.
#
# Both use the same input guards as fit_rq2_model() (same required columns, bnrecal rejection,
# exactly-one-row-per-combination rule) and the same converged/singular definitions
# (A7r6(a)/A7r7): `singular` is lme4::isSingular(); `converged` is the optimiser's own return
# code (0) with no non-singularity warning.

suppressMessages(library(lmerTest))

#' Shared input validation, identical to fit_rq2_model()'s guards (analysis/rq2_model.R) --
#' duplicated here rather than sourced, so this file has no load-order dependency on
#' rq2_model.R and so the primary fit's file is never touched by this diagnostics work.
.validate_rq2_input <- function(data, caller) {
  required_cols <- c("session", "model", "state", "log_ratio", "log_mac")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(caller, ": missing required column(s): ", paste(missing_cols, collapse = ", "))
  }
  if (any(grepl("bnrecal", data$state, ignore.case = TRUE))) {
    stop(caller, ": input contains _bnrecal state(s) -- these diagnostics use zero-finetune ",
         "pruned states only, matching the primary fit's own input (A7r4(f)).")
  }
  combo_counts <- table(interaction(data$session, data$model, data$state, drop = TRUE))
  if (any(combo_counts != 1)) {
    stop(caller, ": every (session, model, state) combination must occur exactly once. ",
         "Session is the replicate unit (A7 point 3) -- pre-aggregate rep-level data first.")
  }
}

#' Same converged/singular assessment as fit_rq2_model() (A7r6(a)/A7r7), applied to a fitted
#' lmerMod/lmerModLmerTest object.
.assess_fit_quality <- function(fit) {
  opt_code <- fit@optinfo$conv$opt
  conv_messages <- fit@optinfo$conv$lme4$messages
  if (is.null(conv_messages)) conv_messages <- character(0)
  non_singular_messages <- conv_messages[!grepl("^boundary \\(singular\\) fit", conv_messages)]
  list(converged = isTRUE(opt_code == 0) && length(non_singular_messages) == 0,
       singular = lme4::isSingular(fit))
}

#' (a) Curvature diagnostic: log_ratio ~ log_mac + I(log_mac^2) + model + (1|session), REML,
#' Satterthwaite df for the quadratic term's CI; likelihood-ratio test of the quadratic term
#' from ML refits of the model with and without it.
#'
#' @param data same shape as fit_rq2_model()'s input: session, model, state, log_ratio, log_mac;
#'   zero-finetune pruned states only, one row per (session, model, state).
#' @return one-row data.frame: quad_estimate, quad_se, quad_df, quad_ci_lower, quad_ci_upper,
#'   lrt_chisq, lrt_df, lrt_p, converged, singular, n_obs, n_sessions, n_models.
fit_curvature_diagnostic <- function(data) {
  .validate_rq2_input(data, "fit_curvature_diagnostic")
  data$model <- factor(data$model)
  data$session <- factor(data$session)

  fit_reml <- lmerTest::lmer(log_ratio ~ log_mac + I(log_mac^2) + model + (1 | session),
                              data = data, REML = TRUE)
  coefs <- summary(fit_reml)$coefficients
  quad_row <- "I(log_mac^2)"
  if (!quad_row %in% rownames(coefs)) {
    stop("fit_curvature_diagnostic: quadratic coefficient not found -- check input data.")
  }
  quad_estimate <- coefs[quad_row, "Estimate"]
  quad_se <- coefs[quad_row, "Std. Error"]
  quad_df <- coefs[quad_row, "df"]
  t_crit <- qt(0.975, df = quad_df)
  quad_ci_lower <- quad_estimate - t_crit * quad_se
  quad_ci_upper <- quad_estimate + t_crit * quad_se

  # LRT: ML refits of the nested models (reduced = primary model, full = + quadratic term).
  fit_reduced_ml <- lme4::lmer(log_ratio ~ log_mac + model + (1 | session), data = data,
                                REML = FALSE)
  fit_full_ml <- lme4::lmer(log_ratio ~ log_mac + I(log_mac^2) + model + (1 | session),
                             data = data, REML = FALSE)
  # anova()'s Df column is already the degrees-of-freedom DIFFERENCE between the two models
  # (not cumulative npar) -- row 2 holds it directly, row 1 is NA (nothing to compare against).
  lrt <- anova(fit_reduced_ml, fit_full_ml)
  lrt_chisq <- lrt[["Chisq"]][2]
  lrt_df <- lrt[["Df"]][2]
  lrt_p <- lrt[["Pr(>Chisq)"]][2]

  quality <- .assess_fit_quality(fit_reml)

  data.frame(
    quad_estimate = quad_estimate,
    quad_se = quad_se,
    quad_df = quad_df,
    quad_ci_lower = quad_ci_lower,
    quad_ci_upper = quad_ci_upper,
    lrt_chisq = lrt_chisq,
    lrt_df = lrt_df,
    lrt_p = lrt_p,
    converged = quality$converged,
    singular = quality$singular,
    n_obs = nrow(data),
    n_sessions = length(unique(data$session)),
    n_models = length(unique(data$model))
  )
}

#' (b) Slope-heterogeneity diagnostic: per-architecture log_mac slopes
#' (log_ratio ~ 0 + model + log_mac:model + (1|session), REML, Satterthwaite df per slope --
#' algebraically the same fixed-effects space as log_mac*model + (1|session), just parameterised
#' to return each architecture's own slope directly instead of a reference-level + deltas) and a
#' likelihood-ratio test of the interaction (common slope vs. per-architecture slopes) from ML
#' refits of the two nested models.
#'
#' @param data same shape as fit_rq2_model()'s input.
#' @return list(per_model = data.frame with one row per architecture: model, slope, se, df,
#'   ci_lower, ci_upper; lrt = one-row data.frame: lrt_chisq, lrt_df, lrt_p; converged, singular
#'   -- both from the per-architecture-slope REML fit; n_obs, n_sessions, n_models).
fit_heterogeneity_diagnostic <- function(data) {
  .validate_rq2_input(data, "fit_heterogeneity_diagnostic")
  data$model <- factor(data$model)
  data$session <- factor(data$session)
  model_levels <- levels(data$model)

  # Per-architecture intercepts and slopes directly, no reference-level coding.
  fit_reml <- lmerTest::lmer(log_ratio ~ 0 + model + log_mac:model + (1 | session),
                              data = data, REML = TRUE)
  coefs <- summary(fit_reml)$coefficients
  per_model <- do.call(rbind, lapply(model_levels, function(m) {
    row_name <- paste0("model", m, ":log_mac")
    if (!row_name %in% rownames(coefs)) {
      stop("fit_heterogeneity_diagnostic: expected coefficient '", row_name,
           "' not found -- check model factor levels.")
    }
    est <- coefs[row_name, "Estimate"]
    se <- coefs[row_name, "Std. Error"]
    df <- coefs[row_name, "df"]
    t_crit <- qt(0.975, df = df)
    data.frame(model = m, slope = est, se = se, df = df,
               ci_lower = est - t_crit * se, ci_upper = est + t_crit * se)
  }))
  rownames(per_model) <- NULL

  # LRT: ML refits, reduced = common slope (the primary model's own formula), full = the same
  # fixed-effects space as per-architecture slopes, coded as log_mac*model for a clean nesting.
  fit_reduced_ml <- lme4::lmer(log_ratio ~ log_mac + model + (1 | session), data = data,
                                REML = FALSE)
  fit_full_ml <- lme4::lmer(log_ratio ~ log_mac * model + (1 | session), data = data,
                             REML = FALSE)
  lrt <- anova(fit_reduced_ml, fit_full_ml)
  lrt_df <- data.frame(
    lrt_chisq = lrt[["Chisq"]][2],
    lrt_df = lrt[["Df"]][2],  # anova()'s Df is already the difference, not cumulative npar
    lrt_p = lrt[["Pr(>Chisq)"]][2]
  )

  quality <- .assess_fit_quality(fit_reml)

  list(
    per_model = per_model,
    lrt = lrt_df,
    converged = quality$converged,
    singular = quality$singular,
    n_obs = nrow(data),
    n_sessions = length(unique(data$session)),
    n_models = length(model_levels)
  )
}
