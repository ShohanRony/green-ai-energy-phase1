# BLOCK P extension of the RQ2 simulation study (analysis/simulate_rq2.R from BLOCK O).
# Synthetic data only -- no results_* file is read anywhere in this script.
#
# Deliberately a separate file, not an edit to simulate_rq2.R: that script's original
# 18-scenario run (analysis/sim_output/summary.csv, raw_replicates.csv) is kept exactly as
# BLOCK O produced it, untouched. This file reuses the same design constants (restated here,
# not sourced from simulate_rq2.R, since sourcing that file would re-run its own full
# 18-scenario x 1000-replicate simulation as a side effect) and the same rq2_model.R fitting
# function, and writes to new output files only.
#
# Usage: Rscript analysis/simulate_rq2_block_p.R [n_reps]   (default n_reps = 1000)

suppressMessages(library(lmerTest))
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "analysis"
source(file.path(script_dir, "rq2_model.R"))

args <- commandArgs(trailingOnly = TRUE)
n_reps <- if (length(args) >= 1) as.integer(args[1]) else 1000

out_dir <- file.path(script_dir, "sim_output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Same realised-MAC-ratio design constants as simulate_rq2.R (stage5_analysis_plan.md section 5).
mac_reduction <- c(
  resnet18.pruned30 = 0.516, resnet18.pruned50 = 0.748, resnet18.pruned70 = 0.910,
  mobilenet_v3_small.pruned30 = 0.467, mobilenet_v3_small.pruned50 = 0.691,
  mobilenet_v3_small.pruned70 = 0.868,
  efficientnet_b0.pruned30 = 0.484, efficientnet_b0.pruned50 = 0.712,
  efficientnet_b0.pruned70 = 0.881
)
mac_ratio <- 1 - mac_reduction
models <- c("resnet18", "mobilenet_v3_small", "efficientnet_b0")
states <- c("pruned30", "pruned50", "pruned70")
design <- expand.grid(model = models, state = states, stringsAsFactors = FALSE)
design$key <- paste(design$model, design$state, sep = ".")
design$mac_ratio <- mac_ratio[design$key]
design$log_mac <- log(design$mac_ratio)
n_sessions <- 6
model_offset <- c(resnet18 = 0, mobilenet_v3_small = 0.15, efficientnet_b0 = -0.1)

cat(sprintf("log_mac range: [%.3f, %.3f]; log_mac^2 range: [%.3f, %.3f]\n",
            min(design$log_mac), max(design$log_mac),
            min(design$log_mac^2), max(design$log_mac^2)))

#' Generic synthetic-data generator.
#' @param beta either a single number (same slope for every model) or a named vector
#'   (one slope per model, e.g. c(resnet18=0.6, mobilenet_v3_small=0.8, efficientnet_b0=1.0)).
#' @param gamma coefficient on log_mac^2 (curvature); 0 for the linear-only scenarios.
simulate_one <- function(beta, session_sd, residual_sd, gamma = 0) {
  beta_vec <- if (length(beta) == 1) rep(beta, nrow(design)) else beta[design$model]
  session_eff <- rnorm(n_sessions, mean = 0, sd = session_sd)
  rows <- do.call(rbind, lapply(seq_len(n_sessions), function(s) {
    noise <- rnorm(nrow(design), mean = 0, sd = residual_sd)
    log_ratio <- beta_vec * design$log_mac + gamma * design$log_mac^2 +
      model_offset[design$model] + session_eff[s] + noise
    data.frame(session = s, model = design$model, state = design$state,
               log_ratio = log_ratio, log_mac = design$log_mac)
  }))
  rows
}

run_scenario <- function(label, beta, beta_true_for_test, session_sd, residual_sd, gamma,
                          seed, n_reps) {
  set.seed(seed)
  beta_hats <- numeric(n_reps)
  covered <- logical(n_reps)
  rejected <- logical(n_reps)
  converged <- logical(n_reps)
  singular <- logical(n_reps)

  for (r in seq_len(n_reps)) {
    d <- simulate_one(beta, session_sd, residual_sd, gamma)
    fit_res <- tryCatch(
      suppressMessages(suppressWarnings(fit_rq2_model(d))),
      error = function(e) NULL
    )
    if (is.null(fit_res)) {
      beta_hats[r] <- NA; covered[r] <- NA; rejected[r] <- NA
      converged[r] <- FALSE; singular[r] <- NA
    } else {
      beta_hats[r] <- fit_res$beta_hat
      covered[r] <- (beta_true_for_test >= fit_res$ci_lower) &&
        (beta_true_for_test <= fit_res$ci_upper)
      rejected[r] <- fit_res$p_one_sided < 0.05
      converged[r] <- fit_res$converged
      singular[r] <- fit_res$singular
    }
  }

  data.frame(
    label = label,
    beta_true = beta_true_for_test,
    session_sd = session_sd,
    residual_sd = residual_sd,
    gamma = gamma,
    n_reps = n_reps,
    bias = mean(beta_hats, na.rm = TRUE) - beta_true_for_test,
    coverage_95 = mean(covered, na.rm = TRUE),
    rejection_rate = mean(rejected, na.rm = TRUE),
    n_fit_failed = sum(is.na(beta_hats)),
    n_nonconverged = sum(!converged, na.rm = TRUE),
    n_singular = sum(singular, na.rm = TRUE)
  )
}

results <- list()
seed_base <- 2000  # distinct range from simulate_rq2.R's 1000+scenario_id, no seed collision

# --- (a) near-boundary scenarios: beta_true in {0.90, 0.95}, session_sd=0.04,
#     residual_sd in {0.02, 0.05} ---
near_boundary <- expand.grid(beta_true = c(0.90, 0.95), residual_sd = c(0.02, 0.05))
for (i in seq_len(nrow(near_boundary))) {
  bt <- near_boundary$beta_true[i]; rsd <- near_boundary$residual_sd[i]
  results[[length(results) + 1]] <- run_scenario(
    label = "near_boundary", beta = bt, beta_true_for_test = bt,
    session_sd = 0.04, residual_sd = rsd, gamma = 0,
    seed = seed_base + i, n_reps = n_reps
  )
  cat(sprintf("near_boundary done: beta_true=%.2f, residual_sd=%.2f\n", bt, rsd))
}

# --- (b) heterogeneous slopes, two cases, session_sd=0.04, residual_sd=0.02 (representative) ---
het_low <- c(resnet18 = 0.6, mobilenet_v3_small = 0.8, efficientnet_b0 = 1.0)   # pooled mean 0.8
het_high <- c(resnet18 = 0.9, mobilenet_v3_small = 1.0, efficientnet_b0 = 1.1)  # pooled mean 1.0
stopifnot(abs(mean(het_low) - 0.8) < 1e-10, abs(mean(het_high) - 1.0) < 1e-10)

results[[length(results) + 1]] <- run_scenario(
  label = "heterogeneous_pooled_0.8", beta = het_low, beta_true_for_test = mean(het_low),
  session_sd = 0.04, residual_sd = 0.02, gamma = 0, seed = seed_base + 10, n_reps = n_reps
)
cat("heterogeneous (0.6,0.8,1.0), pooled mean 0.8, done\n")

results[[length(results) + 1]] <- run_scenario(
  label = "heterogeneous_pooled_1.0", beta = het_high, beta_true_for_test = mean(het_high),
  session_sd = 0.04, residual_sd = 0.02, gamma = 0, seed = seed_base + 11, n_reps = n_reps
)
cat("heterogeneous (0.9,1.0,1.1), pooled mean 1.0, done\n")

# --- (c) mild curvature: gamma justified from the range of the real MAC ratios ---
# log_mac spans [-2.408, -0.629] (resnet18 pruned70 to mobilenet_v3_small pruned30);
# log_mac^2 spans [0.396, 5.798]. gamma = 0.03 makes the quadratic term's contribution at the
# most extreme observed point (0.03 * 5.798 = 0.174) about 9% of the linear term's magnitude
# there (0.8 * 2.408 = 1.926) -- a real but mild departure from linearity, not a dominant one.
# Representative setting matches simulate_rq2.R's scenario 5 (beta_true=0.8, session_sd=0.04,
# residual_sd=0.02) for direct comparability against the linear-only result.
gamma_curv <- 0.03
results[[length(results) + 1]] <- run_scenario(
  label = "curvature", beta = 0.8, beta_true_for_test = 0.8,
  session_sd = 0.04, residual_sd = 0.02, gamma = gamma_curv,
  seed = seed_base + 20, n_reps = n_reps
)
cat(sprintf("curvature done: gamma=%.2f\n", gamma_curv))

summary_df <- do.call(rbind, results)
write.csv(summary_df, file.path(out_dir, "summary_block_p.csv"), row.names = FALSE)

cat("\n=== Block P summary ===\n")
print(summary_df, row.names = FALSE)
cat("\nWritten to", file.path(out_dir, "summary_block_p.csv"), "\n")
cat("(simulate_rq2.R's original summary.csv/raw_replicates.csv untouched)\n")
