# Simulation study for the RQ2 model (analysis/rq2_model.R), on synthetic data only.
# Validates the fitting/testing pipeline before it is ever pointed at real Stage 4b data --
# no results_* file is read anywhere in this script.
#
# Design: 6 sessions (the registered CPU-block session count, A7r1(e)), 3 architectures x 3
# prune levels = 9 states per session. Predictor values are the realised MAC ratios from Stage
# 2's torch-pruning dependency-graph report (stage5_analysis_plan.md section 5's table) -- a
# fixed design constant, not outcome data, reused here exactly as registered.
#
# Usage: Rscript analysis/simulate_rq2.R [n_reps]   (default n_reps = 1000)

suppressMessages(library(lmerTest))
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "analysis"
source(file.path(script_dir, "rq2_model.R"))

args <- commandArgs(trailingOnly = TRUE)
n_reps <- if (length(args) >= 1) as.integer(args[1]) else 1000

out_dir <- file.path(script_dir, "sim_output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Realised MAC ratios (= 1 - realised MACs reduction), from stage5_analysis_plan.md section 5's
# table -- a design constant.
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

# Arbitrary, fixed simulation-only per-architecture offsets (not derived from any real data --
# just enough to exercise the fixed-effect `model` term in the fitted formula).
model_offset <- c(resnet18 = 0, mobilenet_v3_small = 0.15, efficientnet_b0 = -0.1)

simulate_one <- function(beta_true, session_sd, residual_sd) {
  session_eff <- rnorm(n_sessions, mean = 0, sd = session_sd)
  rows <- do.call(rbind, lapply(seq_len(n_sessions), function(s) {
    noise <- rnorm(nrow(design), mean = 0, sd = residual_sd)
    log_ratio <- beta_true * design$log_mac + model_offset[design$model] +
      session_eff[s] + noise
    data.frame(session = s, model = design$model, state = design$state,
               log_ratio = log_ratio, log_mac = design$log_mac)
  }))
  rows
}

scenarios <- expand.grid(
  beta_true = c(0.6, 0.8, 1.0),
  session_sd = c(0.02, 0.04, 0.06),
  residual_sd = c(0.02, 0.05)
)
scenarios$scenario_id <- seq_len(nrow(scenarios))

summary_rows <- list()
raw_rows <- list()

for (i in seq_len(nrow(scenarios))) {
  sc <- scenarios[i, ]
  set.seed(1000 + sc$scenario_id)  # fixed, scenario-specific seed -- reproducible
  beta_hats <- numeric(n_reps)
  covered <- logical(n_reps)
  rejected <- logical(n_reps)
  converged <- logical(n_reps)
  singular <- logical(n_reps)

  for (r in seq_len(n_reps)) {
    d <- simulate_one(sc$beta_true, sc$session_sd, sc$residual_sd)
    fit_res <- tryCatch(
      suppressMessages(suppressWarnings(fit_rq2_model(d))),
      error = function(e) NULL
    )
    if (is.null(fit_res)) {
      beta_hats[r] <- NA
      covered[r] <- NA
      rejected[r] <- NA
      converged[r] <- FALSE
      singular[r] <- NA
    } else {
      beta_hats[r] <- fit_res$beta_hat
      covered[r] <- (sc$beta_true >= fit_res$ci_lower) && (sc$beta_true <= fit_res$ci_upper)
      rejected[r] <- fit_res$p_one_sided < 0.05
      converged[r] <- fit_res$converged
      singular[r] <- fit_res$singular
    }
    raw_rows[[length(raw_rows) + 1]] <- data.frame(
      scenario_id = sc$scenario_id, rep = r, beta_hat = beta_hats[r],
      covered = covered[r], rejected = rejected[r], converged = converged[r],
      singular = singular[r]
    )
  }

  n_fit_failed <- sum(is.na(beta_hats))
  n_nonconverged <- sum(!converged, na.rm = TRUE)
  n_singular <- sum(singular, na.rm = TRUE)

  summary_rows[[i]] <- data.frame(
    scenario_id = sc$scenario_id,
    beta_true = sc$beta_true,
    session_sd = sc$session_sd,
    residual_sd = sc$residual_sd,
    n_reps = n_reps,
    bias = mean(beta_hats, na.rm = TRUE) - sc$beta_true,
    coverage_95 = mean(covered, na.rm = TRUE),
    rejection_rate = mean(rejected, na.rm = TRUE),
    n_fit_failed = n_fit_failed,
    n_nonconverged = n_nonconverged,
    n_singular = n_singular
  )
  cat(sprintf("scenario %d/%d done (beta_true=%.1f, session_sd=%.2f, residual_sd=%.2f)\n",
              i, nrow(scenarios), sc$beta_true, sc$session_sd, sc$residual_sd))
}

summary_df <- do.call(rbind, summary_rows)
raw_df <- do.call(rbind, raw_rows)

write.csv(summary_df, file.path(out_dir, "summary.csv"), row.names = FALSE)
write.csv(raw_df, file.path(out_dir, "raw_replicates.csv"), row.names = FALSE)

cat("\n=== Summary ===\n")
print(summary_df, row.names = FALSE)
cat("\nWritten to", file.path(out_dir, "summary.csv"), "and",
    file.path(out_dir, "raw_replicates.csv"), "\n")
