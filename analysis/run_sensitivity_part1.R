# Registered sensitivity analyses part 1 and decision rules D2r1, D3r1
# Plan v1 sections 4, 6, 7

suppressMessages(library(lme4))
suppressMessages(library(lmerTest))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "analysis"
repo_root <- normalizePath(file.path(script_dir, ".."))

source(file.path(repo_root, "analysis", "rq2_gap_model.R"))
source(file.path(repo_root, "analysis", "rq2_diagnostics.R"))
source(file.path(repo_root, "analysis", "helpers.R"))

h2_input_path <- file.path(repo_root, "analysis", "results", "h2_input.csv")
condition_energy_path <- file.path(repo_root, "analysis", "results", "condition_energy.csv")
output_path <- file.path(repo_root, "analysis", "results", "sensitivity_part1_output.txt")

h2_data <- read.csv(h2_input_path, stringsAsFactors = FALSE)
cond_energy <- read.csv(condition_energy_path, stringsAsFactors = FALSE)

lines <- c()

# ==============================================================================
# Environment
# ==============================================================================
lines <- c(lines, "=== Environment ===")
lines <- c(lines, paste("R version:", R.version.string))
lines <- c(lines, paste("lme4 version:", as.character(packageVersion("lme4"))))
lines <- c(lines, paste("lmerTest version:", as.character(packageVersion("lmerTest"))))
lines <- c(lines, "")

# ==============================================================================
# 3a. Sensitivity and Secondary Fits on h2_input.csv
# ==============================================================================
lines <- c(lines, "=== 3a. Sensitivity & Secondary Fits (rq2_gap_model.R) ===")

# fit_sensitivity_free_intercept
sens_free <- fit_sensitivity_free_intercept(h2_data)
lines <- c(lines, "-- fit_sensitivity_free_intercept --")
lines <- c(lines, paste("beta_hat (log_mac slope):", sprintf("%.8f", sens_free$beta_hat)))
lines <- c(lines, paste("se:", sprintf("%.8f", sens_free$se)))
lines <- c(lines, paste("df:", sprintf("%.4f", sens_free$df)))
lines <- c(lines, paste("t_stat (H0: beta >= 1):", sprintf("%.8f", sens_free$t_stat)))
lines <- c(lines, paste("p_one_sided:", sprintf("%.8e", sens_free$p_one_sided)))
lines <- c(lines, "")

# fit_anchored_slopes
anchored_slopes <- fit_anchored_slopes(h2_data)
lines <- c(lines, "-- fit_anchored_slopes --")
lines <- c(lines, paste("mean_slope:", sprintf("%.8f", anchored_slopes$mean_slope)))
lines <- c(lines, paste("se:", sprintf("%.8f", anchored_slopes$se)))
lines <- c(lines, paste("df:", sprintf("%.4f", anchored_slopes$df)))
lines <- c(lines, paste("ci_lower (95%):", sprintf("%.8f", anchored_slopes$ci_lower)))
lines <- c(lines, paste("ci_upper (95%):", sprintf("%.8f", anchored_slopes$ci_upper)))
lines <- c(lines, paste("t_stat (H0: mean_slope >= 1):", sprintf("%.8f", anchored_slopes$t_stat)))
lines <- c(lines, paste("p_one_sided:", sprintf("%.8e", anchored_slopes$p_one_sided)))
lines <- c(lines, "")

# ==============================================================================
# 3b. Curvature and Heterogeneity Diagnostics on h2_input.csv
# ==============================================================================
lines <- c(lines, "=== 3b. Diagnostics (rq2_diagnostics.R) ===")
lines <- c(lines, "Note: No column adaptation required: h2_input.csv already provides required columns (session, model, state, log_ratio, log_mac).")
lines <- c(lines, "")

curv_diag <- fit_curvature_diagnostic(h2_data)
lines <- c(lines, "-- fit_curvature_diagnostic --")
lines <- c(lines, paste("quad_estimate:", sprintf("%.8f", curv_diag$quad_estimate)))
lines <- c(lines, paste("quad_se:", sprintf("%.8f", curv_diag$quad_se)))
lines <- c(lines, paste("quad_df:", sprintf("%.4f", curv_diag$quad_df)))
lines <- c(lines, paste("quad_ci_lower (95%):", sprintf("%.8f", curv_diag$quad_ci_lower)))
lines <- c(lines, paste("quad_ci_upper (95%):", sprintf("%.8f", curv_diag$quad_ci_upper)))
lines <- c(lines, paste("lrt_chisq:", sprintf("%.8f", curv_diag$lrt_chisq)))
lines <- c(lines, paste("lrt_df:", sprintf("%d", as.integer(curv_diag$lrt_df))))
lines <- c(lines, paste("lrt_p:", sprintf("%.8e", curv_diag$lrt_p)))
lines <- c(lines, paste("converged:", curv_diag$converged))
lines <- c(lines, paste("singular:", curv_diag$singular))
lines <- c(lines, paste("n_obs:", curv_diag$n_obs))
lines <- c(lines, paste("n_sessions:", curv_diag$n_sessions))
lines <- c(lines, paste("n_models:", curv_diag$n_models))
lines <- c(lines, "")

hetero_diag <- fit_heterogeneity_diagnostic(h2_data)
lines <- c(lines, "-- fit_heterogeneity_diagnostic --")
lines <- c(lines, "per_model slopes:")
lines <- c(lines, sprintf("  %-20s %-12s %-12s %-8s %-12s %-12s",
                          "model", "slope", "se", "df", "ci_lower", "ci_upper"))
for (i in seq_len(nrow(hetero_diag$per_model))) {
  r <- hetero_diag$per_model[i, ]
  lines <- c(lines, sprintf("  %-20s %-12.8f %-12.8f %-8.2f %-12.8f %-12.8f",
                            r$model, r$slope, r$se, r$df, r$ci_lower, r$ci_upper))
}
lines <- c(lines, paste("lrt_chisq:", sprintf("%.8f", hetero_diag$lrt$lrt_chisq)))
lines <- c(lines, paste("lrt_df:", sprintf("%d", as.integer(hetero_diag$lrt$lrt_df))))
lines <- c(lines, paste("lrt_p:", sprintf("%.8e", hetero_diag$lrt$lrt_p)))
lines <- c(lines, paste("converged:", hetero_diag$converged))
lines <- c(lines, paste("singular:", hetero_diag$singular))
lines <- c(lines, paste("n_obs:", hetero_diag$n_obs))
lines <- c(lines, paste("n_sessions:", hetero_diag$n_sessions))
lines <- c(lines, paste("n_models:", hetero_diag$n_models))
lines <- c(lines, "")

# ==============================================================================
# 3c. Separate fit of fit_h2_gap_model on bnrecal subset
# ==============================================================================
lines <- c(lines, "=== 3c. bnrecal Subset Fit (fit_h2_gap_model) ===")
lines <- c(lines, "Note: 6 conditions exist (3 missing by design under BNRECAL_SKIP).")

mac_reductions <- list(
  resnet18 = c(pruned30 = 0.516, pruned50 = 0.748, pruned70 = 0.910),
  mobilenet_v3_small = c(pruned30 = 0.467, pruned50 = 0.691, pruned70 = 0.868),
  efficientnet_b0 = c(pruned30 = 0.484, pruned50 = 0.712, pruned70 = 0.881)
)

# Extract baselines for fp32_ts per session and model
baselines_ts <- cond_energy[grepl("_fp32_ts$", cond_energy$condition), ]
baselines_ts$model <- sub("_fp32_ts$", "", baselines_ts$condition)
ts_map <- setNames(baselines_ts$gross_j_per_image_mean, paste(baselines_ts$session, baselines_ts$model, sep = ":"))

bn_subset <- cond_energy[grepl("_bnrecal$", cond_energy$condition), ]
bn_subset$model <- sub("_(pruned[0-9]+)_bnrecal$", "", bn_subset$condition)
bn_subset$state <- sub(".*_(pruned[0-9]+)_bnrecal$", "\\1", bn_subset$condition)
bn_subset$baseline_energy <- ts_map[paste(bn_subset$session, bn_subset$model, sep = ":")]
bn_subset$log_ratio <- log(bn_subset$gross_j_per_image_mean / bn_subset$baseline_energy)
bn_subset$log_mac <- mapply(function(m, s) log(1 - mac_reductions[[m]][[s]]), bn_subset$model, bn_subset$state)

bn_model_data <- bn_subset[, c("session", "model", "state", "log_ratio", "log_mac")]
bn_fit <- fit_h2_gap_model(bn_model_data)

lines <- c(lines, paste("converged:", bn_fit$converged))
lines <- c(lines, paste("singular:", bn_fit$singular))
lines <- c(lines, paste("complete_design:", bn_fit$complete_design))
lines <- c(lines, paste("error:", bn_fit$error))
lines <- c(lines, "")

# ==============================================================================
# 3d. Decision Rule D2r1 (Plan v1 section 7)
# ==============================================================================
lines <- c(lines, "=== 3d. Decision Rule D2r1 (FP32-TS vs FP32-eager TOST) ===")
lines <- c(lines, "Equivalence margin: log(1.05) = +/-0.04879016, df = 5, SD threshold = 0.0593")
archs <- c("resnet18", "mobilenet_v3_small", "efficientnet_b0")

d2r1_results <- list()
for (arch in archs) {
  ts_sub <- cond_energy[cond_energy$condition == paste0(arch, "_fp32_ts"), ]
  eager_sub <- cond_energy[cond_energy$condition == paste0(arch, "_fp32_eager"), ]
  m <- merge(ts_sub, eager_sub, by = "session", suffixes = c("_ts", "_eager"))
  log_r <- log(m$gross_j_per_image_mean_ts / m$gross_j_per_image_mean_eager)
  
  tost <- tost_equivalence(log_r, margin = log(1.05))
  sd_thresh <- 0.0593
  sd_below <- (tost$sd <= sd_thresh)
  
  d2r1_results[[arch]] <- tost
  
  lines <- c(lines, paste("Architecture:", arch))
  lines <- c(lines, paste("  n_sessions:", tost$n))
  lines <- c(lines, paste("  mean log ratio:", sprintf("%.8f", tost$mean)))
  lines <- c(lines, paste("  sd:", sprintf("%.8f", tost$sd)))
  lines <- c(lines, paste("  90% CI: [", sprintf("%.8f", tost$ci_lower), ",", sprintf("%.8f", tost$ci_upper), "]"))
  lines <- c(lines, paste("  bounded within [-log(1.05), +log(1.05)]:", tost$bounded))
  lines <- c(lines, paste("  sd <= 0.0593 threshold:", sprintf("%.8f <= %.4f is %s", tost$sd, sd_thresh, sd_below)))
  lines <- c(lines, "")
}

# ==============================================================================
# 3e. Decision Rule D3r1 (Plan v1 section 7)
# ==============================================================================
lines <- c(lines, "=== 3e. Decision Rule D3r1 (Weight Dependence: bnrecal vs zero-finetune TOST) ===")
lines <- c(lines, "Equivalence margin: log(1.05) = +/-0.04879016, df = 5")

bn_pairs <- list(
  list(arch = "resnet18", level = "pruned30"),
  list(arch = "resnet18", level = "pruned50"),
  list(arch = "resnet18", level = "pruned70"),
  list(arch = "mobilenet_v3_small", level = "pruned30"),
  list(arch = "efficientnet_b0", level = "pruned30"),
  list(arch = "efficientnet_b0", level = "pruned50")
)

all_d3r1_bounded <- TRUE
for (p in bn_pairs) {
  cond_bn <- paste0(p$arch, "_", p$level, "_bnrecal")
  cond_zf <- paste0(p$arch, "_", p$level)
  
  sub_bn <- cond_energy[cond_energy$condition == cond_bn, ]
  sub_zf <- cond_energy[cond_energy$condition == cond_zf, ]
  m <- merge(sub_bn, sub_zf, by = "session", suffixes = c("_bn", "_zf"))
  log_r <- log(m$gross_j_per_image_mean_bn / m$gross_j_per_image_mean_zf)
  
  tost <- tost_equivalence(log_r, margin = log(1.05))
  if (!tost$bounded) all_d3r1_bounded <- FALSE
  
  lines <- c(lines, paste("Pair:", cond_bn, "vs", cond_zf))
  lines <- c(lines, paste("  n_sessions:", tost$n))
  lines <- c(lines, paste("  mean log ratio:", sprintf("%.8f", tost$mean)))
  lines <- c(lines, paste("  sd:", sprintf("%.8f", tost$sd)))
  lines <- c(lines, paste("  90% CI: [", sprintf("%.8f", tost$ci_lower), ",", sprintf("%.8f", tost$ci_upper), "]"))
  lines <- c(lines, paste("  bounded within [-log(1.05), +log(1.05)]:", tost$bounded))
  lines <- c(lines, "")
}

lines <- c(lines, paste("Overall D3r1 verdict (YES only if all 6 are bounded):", ifelse(all_d3r1_bounded, "YES", "NO")))
lines <- c(lines, "")

# ==============================================================================
# 3f. Section 6 Per-State Ratios (21 non-fp32_ts conditions)
# ==============================================================================
lines <- c(lines, "=== 3f. Section 6 Per-State Ratios (relative to same-session arch fp32_ts) ===")
lines <- c(lines, "Geometric mean over 6 sessions, 95% t-interval on log r (df = 5), exp transformed.")
lines <- c(lines, "")

non_ts_data <- cond_energy[!grepl("_fp32_ts$", cond_energy$condition), ]
unique_conditions <- sort(unique(non_ts_data$condition))

lines <- c(lines, sprintf("%-36s %-18s %-4s %-12s %-12s %-12s %-12s %-12s",
                          "condition", "model", "n", "geom_mean_r", "ci_lower_95", "ci_upper_95", "mean_log_r", "se_log_r"))

for (cname in unique_conditions) {
  arch <- if (startsWith(cname, "resnet18")) "resnet18"
          else if (startsWith(cname, "mobilenet_v3_small")) "mobilenet_v3_small"
          else if (startsWith(cname, "efficientnet_b0")) "efficientnet_b0"
          else stop("Unknown architecture for condition: ", cname)
  
  sub <- non_ts_data[non_ts_data$condition == cname, ]
  base_vals <- sapply(sub$session, function(s) ts_map[[paste0(s, ":", arch)]])
  
  log_r <- log(sub$gross_j_per_image_mean / base_vals)
  n <- length(log_r)
  mean_log_r <- mean(log_r)
  se_log_r <- sd(log_r) / sqrt(n)
  df <- n - 1
  ci_lower_log <- mean_log_r - qt(0.975, df) * se_log_r
  ci_upper_log <- mean_log_r + qt(0.975, df) * se_log_r
  
  geom_mean_r <- exp(mean_log_r)
  ci_lower_r <- exp(ci_lower_log)
  ci_upper_r <- exp(ci_upper_log)
  
  lines <- c(lines, sprintf("%-36s %-18s %-4d %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f",
                            cname, arch, n, geom_mean_r, ci_lower_r, ci_upper_r, mean_log_r, se_log_r))
}

# Write output file
writeLines(lines, output_path)

# Print full output to stdout
cat(paste(lines, collapse = "\n"), "\n")
