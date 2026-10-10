# Plain-assertion tests for analysis/helpers.R. No framework -- stopifnot, matching this
# project's existing minimalist test style (one assertion per check, no fixtures).
# Run standalone: Rscript analysis/tests/test_helpers.R
# Each check increments `pass_count` on success; a failed stopifnot aborts the script, which
# run_all.R treats as a failure for this file.

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE),
                                                   value = TRUE))), "..", "helpers.R"))

pass_count <- 0
check <- function(expr, msg) {
  if (!isTRUE(expr)) stop("FAILED: ", msg)
  pass_count <<- pass_count + 1
}

# --- tost_equivalence ---
res <- tost_equivalence(c(0.0, 0.0, 0.0, 0.0))
check(res$bounded == TRUE, "tost_equivalence: zero-spread data must be bounded")
check(abs(res$mean - 0) < 1e-10, "tost_equivalence: mean of zeros is zero")
check(res$df == 3, "tost_equivalence: df = n-1 for n=4")

res2 <- tost_equivalence(c(0.5, 0.5, 0.5, 0.5))  # far outside +-5% margin, zero spread
check(res2$bounded == FALSE, "tost_equivalence: large constant offset must not be bounded")

check(inherits(tryCatch(tost_equivalence(c(0.1)), error = function(e) e), "error"),
      "tost_equivalence: must error with n<2")

# --- sd_threshold ---
check(abs(sd_threshold(4) - 0.0415) < 0.001, "sd_threshold(4) ~= 0.0415")
check(abs(sd_threshold(6) - 0.0593) < 0.001, "sd_threshold(6) ~= 0.0593")
check(inherits(tryCatch(sd_threshold(1), error = function(e) e), "error"),
      "sd_threshold: must error with n<2")
# monotonicity sanity: more sessions -> looser (larger) threshold, for fixed margin
check(sd_threshold(6) > sd_threshold(4), "sd_threshold: should increase with n (looser bound)")

# --- bridged_ratio ---
br <- bridged_ratio(log(1.0), 0.01, 5, log(1.0), 0.02, 3)
check(abs(br$log_bridged - 0) < 1e-10, "bridged_ratio: log(1)+log(1) = 0")
check(abs(br$var_bridged - 0.03) < 1e-10, "bridged_ratio: variances add (0.01+0.02=0.03)")
check(abs(br$se_bridged - sqrt(0.03)) < 1e-10, "bridged_ratio: se = sqrt(var)")
# hand-computed Welch-Satterthwaite for this exact input
expected_df <- (0.01 + 0.02)^2 / ((0.01^2) / 5 + (0.02^2) / 3)
check(abs(br$df_eff - expected_df) < 1e-8, "bridged_ratio: df_eff matches hand-computed W-S value")
check(br$ci_lower < br$log_bridged && br$log_bridged < br$ci_upper,
      "bridged_ratio: point estimate must fall inside its own CI")
check(inherits(tryCatch(bridged_ratio(0, 0, 5, 0, 0.01, 3), error = function(e) e), "error"),
      "bridged_ratio: must error on non-positive variance")

# --- bridged_ratio: inputs are variances OF THE MEAN (sample var / n), demonstrated with
#     different n for the two sides, since mixing this up is an easy off-by-n error ---
per_session_1 <- c(0.10, 0.12, 0.09, 0.11, 0.08, 0.13)  # n1 = 6 sessions (GPU block)
per_session_2 <- c(0.02, -0.01, 0.015, 0.00)              # n2 = 4 sessions (series R)
n1 <- length(per_session_1); n2 <- length(per_session_2)
mean_log_r_eager <- mean(per_session_1)
mean_log_factor <- mean(per_session_2)
var_of_mean_1 <- var(per_session_1) / n1   # variance of the SAMPLE MEAN, not var(per_session_1)
var_of_mean_2 <- var(per_session_2) / n2
br2 <- bridged_ratio(mean_log_r_eager, var_of_mean_1, n1 - 1,
                      mean_log_factor, var_of_mean_2, n2 - 1)
check(abs(br2$var_bridged - (var_of_mean_1 + var_of_mean_2)) < 1e-10,
      "bridged_ratio: with different n1!=n2, var_bridged is the sum of the two means' variances")
check(abs(br2$var_bridged - (var(per_session_1) + var(per_session_2))) > 1e-6,
      "bridged_ratio: raw sample variances instead of mean-variances give a different (wrong) answer")

# --- spec_curve_metrics ---
sc <- spec_curve_metrics(c(0.1, 0.1, 0.1, 0.1), w_star = 0.05, log_r_star = 0.1)
check(abs(sc$sd_spec - 0) < 1e-10, "spec_curve_metrics: zero spread for identical inputs")
check(abs(sc$headline_1 - 0) < 1e-10, "spec_curve_metrics: headline_1 zero when sd_spec is zero")

# floor behaviour: log_r_star smaller than the margin -> denominator is the margin, not |log_r_star|
sc2 <- spec_curve_metrics(c(0.01, 0.02, 0.03), w_star = 0.01, log_r_star = 0.001)
margin_default <- log(1.05)
check(abs(sc2$headline_2 - sd(c(0.01, 0.02, 0.03)) / margin_default) < 1e-10,
      "spec_curve_metrics: headline_2 denominator floored at margin when |log_r_star| is tiny")
check(inherits(tryCatch(spec_curve_metrics(1, 0.1, 0.1), error = function(e) e), "error"),
      "spec_curve_metrics: must error with fewer than 2 specification points")

# --- holm_adjust ---
h <- holm_adjust(c(0.01, 0.04))
check(abs(h[1] - 0.02) < 1e-10, "holm_adjust: smaller p doubled (m=2 Holm step 1)")
check(abs(h[2] - 0.04) < 1e-10, "holm_adjust: larger p unchanged at step 2")
check(all(holm_adjust(c(0.5, 0.5)) <= 1), "holm_adjust: adjusted p-values never exceed 1")

cat("test_helpers.R: ", pass_count, " checks passed\n", sep = "")
