# A7r8 simulation extension (BLOCK R): separates bias driven by curvature (model
# misspecification against a genuinely curved truth) from the registered RQ2 model's own
# sampling noise, by comparing beta_hat against both the true linear coefficient (beta_true)
# AND the population best-linear-projection (BLP) slope -- the slope the primary model would
# recover even with infinite data, given its own linear specification and the real MAC-ratio
# design. When gamma=0 these two targets coincide exactly (verified below); under curvature they
# diverge, and coverage against beta_true is then NOT the coverage of the registered estimand.
#
# Synthetic data only -- no results_* file is read anywhere in this script. A separate file from
# analysis/simulate_rq2.R and analysis/simulate_rq2_block_p.R (same convention as BLOCK P/Q: new
# scenarios in a new file, writing new output files, not touching the earlier scripts' already-
# validated outputs).
#
# Usage: Rscript analysis/simulate_rq2_diagnostics.R [n_reps]   (default n_reps = 1000)

suppressMessages(library(lmerTest))
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "analysis"
source(file.path(script_dir, "rq2_model.R"))
source(file.path(script_dir, "rq2_diagnostics.R"))

args <- commandArgs(trailingOnly = TRUE)
n_reps <- if (length(args) >= 1) as.integer(args[1]) else 1000

out_dir <- file.path(script_dir, "sim_output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Same realised-MAC-ratio design constants as simulate_rq2.R/simulate_rq2_block_p.R
# (stage5_analysis_plan.md section 5).
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

#' Population best-linear-projection slope of E[log_ratio] on log_mac, controlling for the
#' model fixed effect (i.e. the within-model-demeaned regression slope -- Frisch-Waugh-Lovell),
#' over the actual design points, equal weights. When gamma=0 this equals beta exactly; a
#' nonzero gamma (curvature) or per-model beta vector (heterogeneity) makes it diverge from a
#' single "beta_true" number, which is the whole point of reporting it separately.
population_blp_slope <- function(beta, gamma, design) {
  beta_vec <- if (length(beta) == 1) rep(beta, nrow(design)) else beta[design$model]
  ey <- beta_vec * design$log_mac + gamma * design$log_mac^2
  dx <- design$log_mac - ave(design$log_mac, design$model)
  dy <- ey - ave(ey, design$model)
  sum(dx * dy) / sum(dx^2)
}

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

# --- Curvature sweep: gamma in {0, 0.01, 0.03, 0.06}, beta_true=0.8, session_sd=0.04,
#     residual_sd=0.02 (same representative setting as the earlier single-gamma curvature
#     scenario in simulate_rq2_block_p.R, for comparability). ---
curvature_gammas <- c(0, 0.01, 0.03, 0.06)
curvature_beta_true <- 0.8
curvature_session_sd <- 0.04
curvature_residual_sd <- 0.02

curvature_results <- list()
next_id <- 200  # distinct id range from simulate_rq2.R (1-18) and simulate_rq2_block_p.R (19-25)

for (gamma in curvature_gammas) {
  set.seed(9000 + next_id)
  blp <- population_blp_slope(curvature_beta_true, gamma, design)

  beta_hats <- numeric(n_reps)
  covered_bt <- logical(n_reps); covered_blp <- logical(n_reps)
  rejected <- logical(n_reps)
  quad_detected <- logical(n_reps)
  converged <- logical(n_reps); singular <- logical(n_reps)
  diag_converged <- logical(n_reps); diag_singular <- logical(n_reps)

  for (r in seq_len(n_reps)) {
    d <- simulate_one(curvature_beta_true, curvature_session_sd, curvature_residual_sd, gamma)
    primary <- tryCatch(suppressMessages(suppressWarnings(fit_rq2_model(d))),
                         error = function(e) NULL)
    diag <- tryCatch(suppressMessages(suppressWarnings(fit_curvature_diagnostic(d))),
                      error = function(e) NULL)
    if (is.null(primary)) {
      beta_hats[r] <- NA; covered_bt[r] <- NA; covered_blp[r] <- NA; rejected[r] <- NA
      converged[r] <- FALSE; singular[r] <- NA
    } else {
      beta_hats[r] <- primary$beta_hat
      covered_bt[r] <- (curvature_beta_true >= primary$ci_lower) && (curvature_beta_true <= primary$ci_upper)
      covered_blp[r] <- (blp >= primary$ci_lower) && (blp <= primary$ci_upper)
      rejected[r] <- primary$p_one_sided < 0.05
      converged[r] <- primary$converged; singular[r] <- primary$singular
    }
    if (is.null(diag)) {
      quad_detected[r] <- NA; diag_converged[r] <- FALSE; diag_singular[r] <- NA
    } else {
      quad_detected[r] <- !is.na(diag$lrt_p) && diag$lrt_p < 0.05
      diag_converged[r] <- diag$converged; diag_singular[r] <- diag$singular
    }
  }

  curvature_results[[length(curvature_results) + 1]] <- data.frame(
    scenario_id = next_id,
    label = "curvature_sweep",
    gamma = gamma,
    beta_true = curvature_beta_true,
    blp_slope = blp,
    session_sd = curvature_session_sd,
    residual_sd = curvature_residual_sd,
    n_reps = n_reps,
    bias_vs_beta_true = mean(beta_hats, na.rm = TRUE) - curvature_beta_true,
    bias_vs_blp_slope = mean(beta_hats, na.rm = TRUE) - blp,
    coverage_vs_beta_true = mean(covered_bt, na.rm = TRUE),
    coverage_vs_blp_slope = mean(covered_blp, na.rm = TRUE),
    rejection_rate = mean(rejected, na.rm = TRUE),
    quad_lrt_detection_rate = mean(quad_detected, na.rm = TRUE),
    n_fit_failed = sum(is.na(beta_hats)),
    n_nonconverged = sum(!converged, na.rm = TRUE),
    n_singular = sum(singular, na.rm = TRUE),
    diag_n_nonconverged = sum(!diag_converged, na.rm = TRUE),
    diag_n_singular = sum(diag_singular, na.rm = TRUE)
  )
  cat(sprintf("curvature gamma=%.2f done (scenario_id=%d, BLP slope=%.4f)\n",
              gamma, next_id, blp))
  next_id <- next_id + 1
}

curvature_df <- do.call(rbind, curvature_results)
write.csv(curvature_df, file.path(out_dir, "summary_curvature_sweep.csv"), row.names = FALSE)

# --- Heterogeneous slopes (2 scenarios) + homogeneous null (1 scenario): interaction LRT
#     detection rate, gamma=0, session_sd=0.04, residual_sd=0.02. ---
het_low <- c(resnet18 = 0.6, mobilenet_v3_small = 0.8, efficientnet_b0 = 1.0)
het_high <- c(resnet18 = 0.9, mobilenet_v3_small = 1.0, efficientnet_b0 = 1.1)
null_beta <- 1.0  # homogeneous null: same slope for every architecture, at the H2 boundary

het_scenarios <- list(
  list(label = "heterogeneous_pooled_0.8", beta = het_low, beta_true_for_test = mean(het_low)),
  list(label = "heterogeneous_pooled_1.0", beta = het_high, beta_true_for_test = mean(het_high)),
  list(label = "homogeneous_null", beta = null_beta, beta_true_for_test = null_beta)
)

het_results <- list()
for (sc in het_scenarios) {
  set.seed(9000 + next_id)
  beta_hats <- numeric(n_reps)
  covered <- logical(n_reps)
  rejected <- logical(n_reps)
  interaction_detected <- logical(n_reps)
  converged <- logical(n_reps); singular <- logical(n_reps)
  diag_converged <- logical(n_reps); diag_singular <- logical(n_reps)

  for (r in seq_len(n_reps)) {
    d <- simulate_one(sc$beta, 0.04, 0.02, gamma = 0)
    primary <- tryCatch(suppressMessages(suppressWarnings(fit_rq2_model(d))),
                         error = function(e) NULL)
    diag <- tryCatch(suppressMessages(suppressWarnings(fit_heterogeneity_diagnostic(d))),
                      error = function(e) NULL)
    if (is.null(primary)) {
      beta_hats[r] <- NA; covered[r] <- NA; rejected[r] <- NA
      converged[r] <- FALSE; singular[r] <- NA
    } else {
      beta_hats[r] <- primary$beta_hat
      covered[r] <- (sc$beta_true_for_test >= primary$ci_lower) &&
        (sc$beta_true_for_test <= primary$ci_upper)
      rejected[r] <- primary$p_one_sided < 0.05
      converged[r] <- primary$converged; singular[r] <- primary$singular
    }
    if (is.null(diag)) {
      interaction_detected[r] <- NA; diag_converged[r] <- FALSE; diag_singular[r] <- NA
    } else {
      interaction_detected[r] <- !is.na(diag$lrt$lrt_p) && diag$lrt$lrt_p < 0.05
      diag_converged[r] <- diag$converged; diag_singular[r] <- diag$singular
    }
  }

  het_results[[length(het_results) + 1]] <- data.frame(
    scenario_id = next_id,
    label = sc$label,
    beta_true = sc$beta_true_for_test,
    session_sd = 0.04,
    residual_sd = 0.02,
    n_reps = n_reps,
    bias = mean(beta_hats, na.rm = TRUE) - sc$beta_true_for_test,
    coverage = mean(covered, na.rm = TRUE),
    rejection_rate = mean(rejected, na.rm = TRUE),
    interaction_lrt_detection_rate = mean(interaction_detected, na.rm = TRUE),
    n_fit_failed = sum(is.na(beta_hats)),
    n_nonconverged = sum(!converged, na.rm = TRUE),
    n_singular = sum(singular, na.rm = TRUE),
    diag_n_nonconverged = sum(!diag_converged, na.rm = TRUE),
    diag_n_singular = sum(diag_singular, na.rm = TRUE)
  )
  cat(sprintf("%s done (scenario_id=%d)\n", sc$label, next_id))
  next_id <- next_id + 1
}

het_df <- do.call(rbind, het_results)
write.csv(het_df, file.path(out_dir, "summary_heterogeneity_and_null.csv"), row.names = FALSE)

cat("\n=== Curvature sweep ===\n")
print(curvature_df, row.names = FALSE)
cat("\n=== Heterogeneous slopes + homogeneous null ===\n")
print(het_df, row.names = FALSE)
cat("\nWritten to", file.path(out_dir, "summary_curvature_sweep.csv"), "and",
    file.path(out_dir, "summary_heterogeneity_and_null.csv"), "\n")
cat("(simulate_rq2.R's and simulate_rq2_block_p.R's existing outputs untouched)\n")
