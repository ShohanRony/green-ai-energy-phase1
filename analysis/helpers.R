# Helper functions for the decision rules and metrics registered in
# stage5_analysis_plan.md (A7r2/A7r3/A7r4/A8r1/A8r2). Each function implements exactly the
# formula already written out in the plan -- nothing invented here.

#' (a) D2-style TOST on per-session log ratios.
#' 90% interval (two one-sided tests at alpha=0.05 each), margin ln(1.05), t with df = n-1.
#'
#' @param log_ratios numeric vector, one per-session log ratio (e.g. log(FP32-TS/FP32-eager)).
#' @param margin equivalence margin in log units, default ln(1.05) (the project's standing +-5%).
#' @return list(mean, sd, n, df, margin, ci_lower, ci_upper, bounded).
tost_equivalence <- function(log_ratios, margin = log(1.05)) {
  n <- length(log_ratios)
  if (n < 2) stop("tost_equivalence: need at least 2 sessions, got ", n)
  mean_x <- mean(log_ratios)
  sd_x <- sd(log_ratios)
  df <- n - 1
  t_crit <- qt(0.95, df = df)
  half_width <- t_crit * sd_x / sqrt(n)
  ci_lower <- mean_x - half_width
  ci_upper <- mean_x + half_width
  bounded <- (ci_lower >= -margin) && (ci_upper <= margin)
  list(mean = mean_x, sd = sd_x, n = n, df = df, margin = margin,
       ci_lower = ci_lower, ci_upper = ci_upper, bounded = bounded)
}

#' (b) Best-case (true difference = 0) SD threshold below which a 90% CI can fit inside the
#' equivalence margin at all, for n sessions. Same derivation as A8r2/A7r3(b):
#' t(0.95, n-1) * SD / sqrt(n) <= margin  =>  SD <= margin * sqrt(n) / t(0.95, n-1).
#'
#' @param n number of sessions.
#' @param margin equivalence margin in log units, default ln(1.05).
#' @return the threshold SD (log units).
sd_threshold <- function(n, margin = log(1.05)) {
  if (n < 2) stop("sd_threshold: need at least 2 sessions, got ", n)
  t_crit <- qt(0.95, df = n - 1)
  margin * sqrt(n) / t_crit
}

#' (c) Bridged ratio on the log scale: bridged = ratio_eager * factor, so
#' log(bridged) = log(ratio_eager) + log(factor); for independent terms,
#' Var(log(bridged)) = Var(log(ratio_eager)) + Var(log(factor)). Degrees of freedom for the
#' combined interval via Welch-Satterthwaite (A7r4(e)).
#'
#' IMPORTANT, stated explicitly because mixing the two up is an easy off-by-a-factor-of-n error:
#' `var_r_eager` and `var_factor` are the **variances of the estimates** (i.e. of the sample
#' *mean* log-ratio across sessions: `sample_variance_of_per_session_values / n`, the square of
#' the usual standard error of the mean) -- **not** the raw per-session sample variances
#' themselves. If `log_r_eager` is `mean(per_session_log_ratios)` over `df1 + 1` sessions, then
#' `var_r_eager = var(per_session_log_ratios) / (df1 + 1)`, and likewise for `var_factor` over
#' `df2 + 1` sessions. Passing a raw sample variance instead of a mean's variance here silently
#' inflates `var_bridged`, `se_bridged`, and the resulting interval width by whatever factor of n
#' was omitted.
#'
#' @param log_r_eager log of the within-session eager-baseline ratio (a sample mean across
#'   sessions).
#' @param var_r_eager variance **of that sample mean** (sample variance / n, not the sample
#'   variance itself), estimated with df1 degrees of freedom.
#' @param df1 degrees of freedom backing var_r_eager (GPU block: 5, from 6 sessions).
#' @param log_factor log of the eager/TS factor from series R (a sample mean across sessions).
#' @param var_factor variance **of that sample mean** (sample variance / n, not the sample
#'   variance itself), estimated with df2 degrees of freedom.
#' @param df2 degrees of freedom backing var_factor (series R: 3, from 4 sessions).
#' @param conf confidence level for the reported interval, default 0.95.
#' @return list(log_bridged, var_bridged, se_bridged, df_eff, ci_lower, ci_upper).
bridged_ratio <- function(log_r_eager, var_r_eager, df1, log_factor, var_factor, df2,
                           conf = 0.95) {
  if (var_r_eager <= 0 || var_factor <= 0) {
    stop("bridged_ratio: variances must be positive")
  }
  log_bridged <- log_r_eager + log_factor
  var_bridged <- var_r_eager + var_factor
  se_bridged <- sqrt(var_bridged)
  df_eff <- (var_r_eager + var_factor)^2 /
    ((var_r_eager^2) / df1 + (var_factor^2) / df2)
  t_crit <- qt(1 - (1 - conf) / 2, df = df_eff)
  list(log_bridged = log_bridged, var_bridged = var_bridged, se_bridged = se_bridged,
       df_eff = df_eff, ci_lower = log_bridged - t_crit * se_bridged,
       ci_upper = log_bridged + t_crit * se_bridged)
}

#' (d) Specification-curve headline metrics (A7r3(g)/A7r4(a)), descriptive only.
#' headline_2's denominator is floored at the equivalence margin (A7r4(a)) so a near-null
#' primary effect doesn't blow the ratio up.
#'
#' @param log_r_spec numeric vector, log ratio under every specification combination.
#' @param w_star half-width of the primary ratio's own 95% log-r interval.
#' @param log_r_star the primary-specification log ratio itself.
#' @param margin equivalence margin in log units, default ln(1.05), used as the headline_2 floor.
#' @return list(sd_spec, headline_1, headline_2).
spec_curve_metrics <- function(log_r_spec, w_star, log_r_star, margin = log(1.05)) {
  if (length(log_r_spec) < 2) stop("spec_curve_metrics: need at least 2 specification points")
  if (w_star <= 0) stop("spec_curve_metrics: w_star must be positive")
  sd_spec <- sd(log_r_spec)
  headline_1 <- sd_spec / w_star
  headline_2 <- sd_spec / max(abs(log_r_star), margin)
  list(sd_spec = sd_spec, headline_1 = headline_1, headline_2 = headline_2)
}

#' (e) Holm-Bonferroni across p-values (this project's confirmatory family is 2 tests,
#' A7r3(b)/A7r1(b), but this accepts any length). Wraps base R's own p.adjust -- no
#' reimplementation of a correction base R already provides correctly.
#'
#' @param p_values numeric vector of raw p-values.
#' @return numeric vector of Holm-adjusted p-values, same order as input.
holm_adjust <- function(p_values) {
  stats::p.adjust(p_values, method = "holm")
}
