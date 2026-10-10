library(lme4)
library(lmerTest)
library(stats)

# Distinctively named validation helper
validate_h2_gap_input_v1 <- function(data) {
  required_cols <- c("session", "model", "state", "log_ratio", "log_mac")
  if (!all(required_cols %in% colnames(data))) {
    stop("Input must contain columns: session, model, state, log_ratio, log_mac")
  }
  
  if (any(grepl("_bnrecal", data$state))) {
    stop("Input must not contain _bnrecal states. Filter these out before modeling.")
  }
  
  dups <- data[duplicated(data[, c("session", "model", "state")]), ]
  if (nrow(dups) > 0) {
    stop("Input contains duplicate rows for (session, model, state). Exactly one row per cell is allowed.")
  }
}

fit_h2_gap_model <- function(data) {
  validate_h2_gap_input_v1(data)
  
  fit <- tryCatch({
    data$d <- as.numeric(data$log_ratio) - as.numeric(data$log_mac)
    data$cell <- as.factor(paste(data$model, data$state, sep = "_"))
    data$session <- as.factor(data$session)
    data$model <- as.factor(data$model)
    lmerTest::lmer(d ~ 0 + cell + (1|session) + (1|session:model), data = data, REML = TRUE)
  }, error = function(e) {
    return(e)
  })
  
  if (inherits(fit, "error")) {
    complete_design <- tryCatch({
       length(unique(paste(data$model, data$state))) == 9 && nrow(data) == 9 * length(unique(data$session))
    }, error=function(e) FALSE)
    return(list(converged = FALSE, singular = NA, error = fit$message, complete_design = complete_design))
  }
  
  opt_info <- fit@optinfo
  converged <- (opt_info$conv$opt == 0) && length(opt_info$conv$lme4$messages) == 0
  singular <- lme4::isSingular(fit)
  
  data$cell <- as.factor(paste(data$model, data$state, sep = "_"))
  complete_design <- length(unique(data$cell)) == 9 && nrow(data) == 9 * length(unique(data$session))

  n_cells <- length(unique(data$cell))
  if (n_cells != 9) {
    # If not 9 cells, we can't reliably compute the 1/9 mean
    return(list(converged = converged, singular = singular, complete_design = complete_design, error = "Not all 9 cells are present in the data to compute the equal-weight mean"))
  }
  
  # 1/9 weights for the cells
  L <- rep(1/9, 9)
  
  contest_res <- lmerTest::contest1D(fit, L, rhs = 0)
  
  mu_hat <- contest_res$Estimate
  se <- contest_res$`Std. Error`
  df <- contest_res$df
  
  ci_lower <- mu_hat - qt(0.975, df) * se
  ci_upper <- mu_hat + qt(0.975, df) * se
  
  t_stat <- mu_hat / se
  p_one_sided <- pt(t_stat, df, lower.tail = FALSE)
  
  # Per-cell estimates
  cell_names <- levels(data$cell)
  cell_estimates <- data.frame(cell = cell_names, estimate = NA, ci_lower = NA, ci_upper = NA)
  for (i in 1:9) {
    L_cell <- rep(0, 9)
    L_cell[i] <- 1
    cell_res <- lmerTest::contest1D(fit, L_cell, rhs = 0)
    cell_estimates$estimate[i] <- cell_res$Estimate
    cell_estimates$ci_lower[i] <- cell_res$Estimate - qt(0.975, cell_res$df) * cell_res$`Std. Error`
    cell_estimates$ci_upper[i] <- cell_res$Estimate + qt(0.975, cell_res$df) * cell_res$`Std. Error`
  }
  
  list(
    mu_hat = mu_hat,
    se = se,
    df = df,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    t_stat = t_stat,
    p_one_sided = p_one_sided,
    exp_mu_hat = exp(mu_hat),
    exp_ci_lower = exp(ci_lower),
    exp_ci_upper = exp(ci_upper),
    converged = converged,
    singular = singular,
    complete_design = complete_design,
    cell_estimates = cell_estimates
  )
}

session_mean_t_test <- function(data) {
  validate_h2_gap_input_v1(data)
  
  data$d <- as.numeric(data$log_ratio) - as.numeric(data$log_mac)
  
  session_counts <- tapply(data$d, data$session, length)
  if (any(session_counts != 9) || length(unique(data$model)) != 3 || length(unique(data$state)) != 3) {
    stop("Session mean t-test requires all 9 cells to be present in each session.")
  }
  
  session_means <- tapply(data$d, data$session, mean)
  n_sessions <- length(session_means)
  
  mu_hat <- mean(session_means)
  se <- sd(session_means) / sqrt(n_sessions)
  df <- n_sessions - 1
  
  ci_lower <- mu_hat - qt(0.975, df) * se
  ci_upper <- mu_hat + qt(0.975, df) * se
  
  t_stat <- mu_hat / se
  p_one_sided <- pt(t_stat, df, lower.tail = FALSE)
  
  list(
    mu_hat = mu_hat,
    se = se,
    df = df,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    t_stat = t_stat,
    p_one_sided = p_one_sided,
    exp_mu_hat = exp(mu_hat),
    exp_ci_lower = exp(ci_lower),
    exp_ci_upper = exp(ci_upper)
  )
}

fit_h2_with_fallback <- function(data) {
  t_res <- tryCatch({
    session_mean_t_test(data)
  }, error = function(e) {
    list(error = e$message)
  })
  
  mixed_res <- fit_h2_gap_model(data)
  
  if (!is.null(mixed_res$error) || isFALSE(mixed_res$converged)) {
    model_used <- "session_mean_t"
    primary_res <- if (!is.null(t_res$error)) list(error = paste("Mixed failed and t-test failed:", t_res$error)) else t_res
    if (!is.null(t_res$error)) {
        return(list(
          model_used = "error",
          error = primary_res$error,
          mixed_converged = if(!is.null(mixed_res$converged)) mixed_res$converged else FALSE,
          mixed_singular = if(!is.null(mixed_res$singular)) mixed_res$singular else NA,
          complete_design = if(!is.null(mixed_res$complete_design)) mixed_res$complete_design else FALSE
        ))
    }
    
    primary_res$model_used <- model_used
    primary_res$mixed_converged <- if(!is.null(mixed_res$converged)) mixed_res$converged else FALSE
    primary_res$mixed_singular <- if(!is.null(mixed_res$singular)) mixed_res$singular else NA
    primary_res$complete_design <- if(!is.null(mixed_res$complete_design)) mixed_res$complete_design else FALSE
    primary_res$fallback_triggered <- TRUE
  } else {
    model_used <- "mixed"
    primary_res <- mixed_res
    primary_res$model_used <- model_used
    primary_res$fallback_triggered <- FALSE
    primary_res$mixed_converged <- mixed_res$converged
    primary_res$mixed_singular <- mixed_res$singular
  }
  
  # Always return t_test results in extra columns
  if (is.null(t_res$error)) {
    primary_res$t_test_mu_hat <- t_res$mu_hat
    primary_res$t_test_se <- t_res$se
    primary_res$t_test_p <- t_res$p_one_sided
    primary_res$t_test_df <- t_res$df
  }
  
  primary_res
}

# SECONDARY FIT: Not confirmatory
fit_anchored_slopes <- function(data) {
  validate_h2_gap_input_v1(data)
  
  data$session <- as.factor(data$session)
  data$model <- as.factor(data$model)
  
  fit <- tryCatch({
    lmerTest::lmer(log_ratio ~ 0 + model:log_mac + (1|session) + (1|session:model), data = data, REML = TRUE)
  }, error = function(e) { return(e) })
  
  if (inherits(fit, "error")) return(list(error = fit$message))
  
  # Extract the 3 slopes
  models <- sort(unique(data$model))
  if (length(models) != 3) return(list(error = "Requires exactly 3 models"))
  
  L <- c(1/3, 1/3, 1/3)
  contest_res <- lmerTest::contest1D(fit, L, rhs = 0)
  
  mu_hat <- contest_res$Estimate
  se <- contest_res$`Std. Error`
  df <- contest_res$df
  
  ci_lower <- mu_hat - qt(0.975, df) * se
  ci_upper <- mu_hat + qt(0.975, df) * se
  
  t_stat <- (mu_hat - 1) / se  # test slope < 1
  p_one_sided <- pt(t_stat, df, lower.tail = TRUE) # H0: slope >= 1
  
  list(
    mean_slope = mu_hat,
    se = se,
    df = df,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    t_stat = t_stat,
    p_one_sided = p_one_sided
  )
}

# SENSITIVITY FIT: Not confirmatory
fit_sensitivity_free_intercept <- function(data) {
  validate_h2_gap_input_v1(data)
  
  data$session <- as.factor(data$session)
  data$model <- as.factor(data$model)
  
  fit <- tryCatch({
    lmerTest::lmer(log_ratio ~ log_mac + model + (1|session) + (1|session:model), data = data, REML = TRUE)
  }, error = function(e) { return(e) })
  
  if (inherits(fit, "error")) return(list(error = fit$message))
  
  sum_fit <- summary(fit)
  beta_hat <- sum_fit$coefficients["log_mac", "Estimate"]
  se <- sum_fit$coefficients["log_mac", "Std. Error"]
  df <- sum_fit$coefficients["log_mac", "df"]
  
  t_stat <- (beta_hat - 1) / se
  p_one_sided <- pt(t_stat, df, lower.tail = TRUE)
  
  list(
    beta_hat = beta_hat,
    se = se,
    df = df,
    t_stat = t_stat,
    p_one_sided = p_one_sided
  )
}

fit_h2_primary <- function(data) {
  mixed_res <- fit_h2_gap_model(data)
  
  if (!is.null(mixed_res$complete_design) && mixed_res$complete_design) {
    t_res <- session_mean_t_test(data)
    list(
      primary_method = "session_mean_t",
      complete_design = TRUE,
      estimate = t_res$mu_hat,
      se = t_res$se,
      df = t_res$df,
      p_one_sided = t_res$p_one_sided,
      ci_lower = t_res$ci_lower,
      ci_upper = t_res$ci_upper,
      exp_estimate = t_res$exp_mu_hat,
      exp_ci_lower = t_res$exp_ci_lower,
      exp_ci_upper = t_res$exp_ci_upper
    )
  } else {
    list(
      primary_method = "cell_fixed_mixed",
      complete_design = FALSE,
      estimate = mixed_res$mu_hat,
      se = mixed_res$se,
      df = mixed_res$df,
      p_one_sided = mixed_res$p_one_sided,
      ci_lower = mixed_res$ci_lower,
      ci_upper = mixed_res$ci_upper,
      exp_estimate = mixed_res$exp_mu_hat,
      exp_ci_lower = mixed_res$exp_ci_lower,
      exp_ci_upper = mixed_res$exp_ci_upper
    )
  }
}
