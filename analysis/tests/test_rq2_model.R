# Plain-assertion tests for analysis/rq2_model.R, on synthetic data only.
# Run standalone: Rscript analysis/tests/test_rq2_model.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE),
                                                   value = TRUE))), "..", "rq2_model.R"))

pass_count <- 0
check <- function(expr, msg) {
  if (!isTRUE(expr)) stop("FAILED: ", msg)
  pass_count <<- pass_count + 1
}

make_synthetic <- function(beta_true, seed = 1) {
  set.seed(seed)
  models <- c("resnet18", "mobilenet_v3_small", "efficientnet_b0")
  mac <- c(0.484, 0.252, 0.090, 0.533, 0.309, 0.132, 0.516, 0.288, 0.119)
  states <- rep(c("pruned30", "pruned50", "pruned70"), 3)
  mdl <- rep(models, each = 3)
  rows <- lapply(seq_len(6), function(s) {
    sess_eff <- rnorm(1, 0, 0.02)
    lr <- beta_true * log(mac) + c(0, 0.1, -0.1)[match(mdl, models)] + sess_eff +
      rnorm(9, 0, 0.02)
    data.frame(session = s, model = mdl, state = states, log_ratio = lr, log_mac = log(mac))
  })
  do.call(rbind, rows)
}

# --- basic fit, strong signal ---
d <- make_synthetic(beta_true = 0.5, seed = 42)
res <- fit_rq2_model(d)

check(nrow(res) == 1, "fit_rq2_model: returns exactly one row")
check(all(c("beta_hat", "se", "df", "t_stat", "p_one_sided", "ci_lower", "ci_upper",
            "n_obs", "n_sessions", "n_models", "converged", "singular") %in% names(res)),
      "fit_rq2_model: output has all required columns")
check(abs(res$beta_hat - 0.5) < 0.1, "fit_rq2_model: beta_hat recovers the true value (+-0.1)")
check(res$ci_lower < res$beta_hat && res$beta_hat < res$ci_upper,
      "fit_rq2_model: point estimate falls inside its own 95% CI")
check(res$p_one_sided >= 0 && res$p_one_sided <= 1, "fit_rq2_model: p-value in [0,1]")
check(res$p_one_sided < 0.05, "fit_rq2_model: strong beta=0.5 signal rejects H2 at alpha=0.05")
check(res$n_obs == 54, "fit_rq2_model: 6 sessions x 9 states = 54 observations")
check(res$n_sessions == 6, "fit_rq2_model: n_sessions counted correctly")
check(res$n_models == 3, "fit_rq2_model: n_models counted correctly")
check(is.logical(res$converged) && is.logical(res$singular),
      "fit_rq2_model: converged/singular are logical flags")

# --- one-sided direction: t_stat should be negative when beta_hat < 1 ---
check(res$t_stat < 0, "fit_rq2_model: t_stat negative when beta_hat well below 1")

# --- null-ish case: beta_true = 1 should not reliably reject (spot check, not a power claim) ---
d_null <- make_synthetic(beta_true = 1.0, seed = 7)
res_null <- fit_rq2_model(d_null)
check(res_null$p_one_sided >= 0 && res_null$p_one_sided <= 1,
      "fit_rq2_model: p-value well-formed under beta_true=1 too")

# --- input validation ---
check(inherits(tryCatch(fit_rq2_model(data.frame(session = 1)), error = function(e) e), "error"),
      "fit_rq2_model: must error on missing required columns")

d_bnrecal <- d
d_bnrecal$state[1] <- "pruned30_bnrecal"
check(inherits(tryCatch(fit_rq2_model(d_bnrecal), error = function(e) e), "error"),
      "fit_rq2_model: must error when bnrecal states are present")

# --- (session, model, state) must occur exactly once: duplicate row rejected (A7r7(e)) ---
d_dup <- rbind(d, d[1, ])
check(inherits(tryCatch(fit_rq2_model(d_dup), error = function(e) e), "error"),
      "fit_rq2_model: must error when a (session, model, state) combination repeats")

# --- converged and singular are independent flags (A7r6(a)/A7r7) ---
make_singular_prone <- function(seed) {
  set.seed(seed)
  models <- c("resnet18", "mobilenet_v3_small", "efficientnet_b0")
  mac <- c(0.484, 0.252, 0.090, 0.533, 0.309, 0.132, 0.516, 0.288, 0.119)
  states <- rep(c("pruned30", "pruned50", "pruned70"), 3)
  mdl <- rep(models, each = 3)
  rows <- lapply(seq_len(6), function(s) {
    sess_eff <- rnorm(1, 0, 0.001)  # near-zero session variance provokes a boundary fit
    lr <- 0.8 * log(mac) + c(0, 0.1, -0.1)[match(mdl, models)] + sess_eff + rnorm(9, 0, 0.05)
    data.frame(session = s, model = mdl, state = states, log_ratio = lr, log_mac = log(mac))
  })
  do.call(rbind, rows)
}
res_sing <- suppressMessages(suppressWarnings(fit_rq2_model(make_singular_prone(seed = 1))))
check(isTRUE(res_sing$singular), "fit_rq2_model: near-zero session variance produces a singular fit")
check(isTRUE(res_sing$converged),
      "fit_rq2_model: a singular fit can still be converged -- the two flags are independent")

cat("test_rq2_model.R: ", pass_count, " checks passed\n", sep = "")
