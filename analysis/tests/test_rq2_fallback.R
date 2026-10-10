# Plain-assertion tests for analysis/rq2_fallback.R, on synthetic data only.
# Run standalone: Rscript analysis/tests/test_rq2_fallback.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE),
                                                   value = TRUE))), "..", "rq2_fallback.R"))

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

make_data <- function(beta, model_offset = c(0, 0.1, -0.1), session_sd, residual_sd, seed) {
  set.seed(seed)
  rows <- lapply(seq_len(6), function(s) {
    sess_eff <- rnorm(1, 0, session_sd)
    lr <- beta * log_mac + model_offset[match(mdl, models)] + sess_eff + rnorm(9, 0, residual_sd)
    data.frame(session = s, model = mdl, state = states, log_ratio = lr, log_mac = log_mac)
  })
  do.call(rbind, rows)
}

# --- normal path: well-behaved data uses the primary (lmer) fit ---
d_normal <- make_data(beta = 0.8, session_sd = 0.04, residual_sd = 0.02, seed = 42)
res_normal <- fit_rq2_with_fallback(d_normal)
check(nrow(res_normal) == 1, "fit_rq2_with_fallback: returns exactly one row")
check(all(c("beta_hat", "se", "df", "t_stat", "p_one_sided", "ci_lower", "ci_upper", "n_obs",
            "n_sessions", "n_models", "converged", "singular", "model_used",
            "complete_design") %in% names(res_normal)),
      "fit_rq2_with_fallback: output has all required columns")
check(res_normal$model_used == "lmer", "fit_rq2_with_fallback: normal path uses lmer")
check(isTRUE(res_normal$complete_design), "fit_rq2_with_fallback: full 54-row design is complete")

# --- forced failure path via mock: fit_rq2_model() throwing an error triggers the fallback,
#     with df = 45 (the registered design's OLS residual df) ---
mock_error <- function(data) stop("simulated optimiser failure for testing")
res_error <- fit_rq2_with_fallback(d_normal, primary_fit_fn = mock_error)
check(res_error$model_used == "lm_fixed_session",
      "fit_rq2_with_fallback: an error from the primary fit triggers the fallback")
check(res_error$df == 45, "fit_rq2_with_fallback: fallback df is 45 for the full registered design")
check(isTRUE(res_error$converged), "fit_rq2_with_fallback: fallback result is marked converged")
check(is.na(res_error$singular), "fit_rq2_with_fallback: fallback's singular flag is NA, not guessed")

# --- mock returning converged == FALSE (no error) also triggers the fallback ---
mock_nonconverged <- function(data) {
  data.frame(beta_hat = 0.5, se = 0.1, df = 10, t_stat = -5, p_one_sided = 0.001,
             ci_lower = 0.3, ci_upper = 0.7, n_obs = nrow(data),
             n_sessions = length(unique(data$session)), n_models = length(unique(data$model)),
             converged = FALSE, singular = FALSE)
}
res_nc <- fit_rq2_with_fallback(d_normal, primary_fit_fn = mock_nonconverged)
check(res_nc$model_used == "lm_fixed_session",
      "fit_rq2_with_fallback: converged==FALSE from the primary fit (no error) triggers the fallback")
check(res_nc$df == 45, "fit_rq2_with_fallback: fallback df is 45 here too")

# --- a mock returning converged == TRUE (even if singular) must NOT trigger the fallback ---
mock_singular_converged <- function(data) {
  data.frame(beta_hat = 0.8, se = 0.01, df = 44, t_stat = -20, p_one_sided = 1e-20,
             ci_lower = 0.78, ci_upper = 0.82, n_obs = nrow(data),
             n_sessions = length(unique(data$session)), n_models = length(unique(data$model)),
             converged = TRUE, singular = TRUE)
}
res_sc_mock <- fit_rq2_with_fallback(d_normal, primary_fit_fn = mock_singular_converged)
check(res_sc_mock$model_used == "lmer",
      "fit_rq2_with_fallback: converged==TRUE (even if singular) from a mock stays on lmer")

# --- singular-but-converged REAL fit (near-zero session variance) stays on lmer, not the
#     fallback -- A7r6(a): singular alone never triggers it ---
d_singular <- make_data(beta = 0.8, session_sd = 0.001, residual_sd = 0.05, seed = 1)
res_singular <- suppressMessages(suppressWarnings(fit_rq2_with_fallback(d_singular)))
check(res_singular$model_used == "lmer",
      "fit_rq2_with_fallback: a real singular-but-converged fit stays on lmer, not the fallback")
check(isTRUE(res_singular$singular),
      "fit_rq2_with_fallback: the singular flag is still TRUE and reported on the lmer path")

# --- incomplete design: complete_design is FALSE, n_obs reported, no error raised ---
d_incomplete <- d_normal[-1, ]
res_incomplete <- fit_rq2_with_fallback(d_incomplete)
check(isFALSE(res_incomplete$complete_design),
      "fit_rq2_with_fallback: a missing (session, model, state) cell sets complete_design FALSE")
check(res_incomplete$n_obs == 53, "fit_rq2_with_fallback: n_obs reflects the actual row count (53)")
check(!is.null(res_incomplete$beta_hat) && !is.na(res_incomplete$beta_hat),
      "fit_rq2_with_fallback: an incomplete design is still fit, not rejected")

# --- input validation: a genuinely malformed input errors immediately, never silently
#     routed into the lm() fallback ---
check(inherits(tryCatch(fit_rq2_with_fallback(data.frame(session = 1)), error = function(e) e),
               "error"),
      "fit_rq2_with_fallback: must error on missing required columns, not fall back")
d_bnrecal <- d_normal
d_bnrecal$state[1] <- "pruned30_bnrecal"
check(inherits(tryCatch(fit_rq2_with_fallback(d_bnrecal), error = function(e) e), "error"),
      "fit_rq2_with_fallback: must error when bnrecal states are present, not fall back")
d_dup <- rbind(d_normal, d_normal[1, ])
check(inherits(tryCatch(fit_rq2_with_fallback(d_dup), error = function(e) e), "error"),
      "fit_rq2_with_fallback: must error when a (session, model, state) combination repeats")

cat("test_rq2_fallback.R: ", pass_count, " checks passed\n", sep = "")
