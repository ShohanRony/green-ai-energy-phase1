# Primary H2 analysis script (Plan v1 section 4)
# Registered computation only.

suppressMessages(library(lme4))
suppressMessages(library(lmerTest))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "analysis"
repo_root <- normalizePath(file.path(script_dir, ".."))

source(file.path(repo_root, "analysis", "rq2_gap_model.R"))

input_path <- file.path(repo_root, "analysis", "results", "h2_input.csv")
output_path <- file.path(repo_root, "analysis", "results", "h2_primary_output.txt")

data <- read.csv(input_path, stringsAsFactors = FALSE)
validate_h2_gap_input_v1(data)

# Compute d
data$d <- as.numeric(data$log_ratio) - as.numeric(data$log_mac)

# 1. Primary test
primary_res <- fit_h2_primary(data)

# 2. Cross-check mixed model
mixed_res <- fit_h2_gap_model(data)

# 3. Per-cell descriptive statistics
models <- c("resnet18", "mobilenet_v3_small", "efficientnet_b0")
states <- c("pruned30", "pruned50", "pruned70")

cell_rows <- list()
for (m in models) {
  for (s in states) {
    cell_d <- data$d[data$model == m & data$state == s]
    n_s <- length(cell_d)
    mean_d <- mean(cell_d)
    se_d <- sd(cell_d) / sqrt(n_s)
    df_d <- n_s - 1
    ci_lower_d <- mean_d - qt(0.975, df_d) * se_d
    ci_upper_d <- mean_d + qt(0.975, df_d) * se_d
    
    cell_rows[[length(cell_rows) + 1]] <- data.frame(
      model = m,
      state = s,
      cell = paste(m, s, sep = "_"),
      n_sessions = n_s,
      mean_d = mean_d,
      se = se_d,
      df = df_d,
      ci_lower = ci_lower_d,
      ci_upper = ci_upper_d,
      exp_mean_d = exp(mean_d),
      exp_ci_lower = exp(ci_lower_d),
      exp_ci_upper = exp(ci_upper_d),
      stringsAsFactors = FALSE
    )
  }
}
cell_table <- do.call(rbind, cell_rows)

# Format output buffer
lines <- c()
lines <- c(lines, "=== Environment ===")
lines <- c(lines, paste("R version:", R.version.string))
lines <- c(lines, paste("lme4 version:", as.character(packageVersion("lme4"))))
lines <- c(lines, paste("lmerTest version:", as.character(packageVersion("lmerTest"))))
lines <- c(lines, "")

lines <- c(lines, "=== Primary H2 Test (fit_h2_primary) ===")
lines <- c(lines, paste("primary_method:", primary_res$primary_method))
lines <- c(lines, paste("complete_design:", primary_res$complete_design))
lines <- c(lines, paste("estimate (mean d):", sprintf("%.8f", primary_res$estimate)))
lines <- c(lines, paste("se:", sprintf("%.8f", primary_res$se)))
lines <- c(lines, paste("df:", sprintf("%.4f", primary_res$df)))
lines <- c(lines, paste("p_one_sided (H0: mean d <= 0 vs H1: mean d > 0):", sprintf("%.8e", primary_res$p_one_sided)))
lines <- c(lines, paste("ci_lower (95%):", sprintf("%.8f", primary_res$ci_lower)))
lines <- c(lines, paste("ci_upper (95%):", sprintf("%.8f", primary_res$ci_upper)))
lines <- c(lines, paste("exp_estimate:", sprintf("%.8f", primary_res$exp_estimate)))
lines <- c(lines, paste("exp_ci_lower:", sprintf("%.8f", primary_res$exp_ci_lower)))
lines <- c(lines, paste("exp_ci_upper:", sprintf("%.8f", primary_res$exp_ci_upper)))
lines <- c(lines, "")

lines <- c(lines, "=== Cross-Check Mixed Model (fit_h2_gap_model) ===")
lines <- c(lines, paste("converged:", mixed_res$converged))
lines <- c(lines, paste("singular:", mixed_res$singular))
lines <- c(lines, paste("complete_design:", mixed_res$complete_design))
lines <- c(lines, paste("mu_hat (equal-weight contrast):", sprintf("%.8f", mixed_res$mu_hat)))
lines <- c(lines, paste("se:", sprintf("%.8f", mixed_res$se)))
lines <- c(lines, paste("df:", sprintf("%.4f", mixed_res$df)))
lines <- c(lines, paste("p_one_sided:", sprintf("%.8e", mixed_res$p_one_sided)))
lines <- c(lines, paste("ci_lower (95%):", sprintf("%.8f", mixed_res$ci_lower)))
lines <- c(lines, paste("ci_upper (95%):", sprintf("%.8f", mixed_res$ci_upper)))
lines <- c(lines, paste("exp_mu_hat:", sprintf("%.8f", mixed_res$exp_mu_hat)))
lines <- c(lines, paste("exp_ci_lower:", sprintf("%.8f", mixed_res$exp_ci_lower)))
lines <- c(lines, paste("exp_ci_upper:", sprintf("%.8f", mixed_res$exp_ci_upper)))
lines <- c(lines, "")

lines <- c(lines, "=== Per-Cell Descriptive Statistics (95% t-interval over sessions) ===")
lines <- c(lines, sprintf("%-35s %-10s %-10s %-4s %-12s %-12s %-12s %-12s %-12s",
                          "cell", "mean_d", "se", "df", "ci_lower", "ci_upper", "exp_mean_d", "exp_ci_lower", "exp_ci_upper"))
for (i in seq_len(nrow(cell_table))) {
  r <- cell_table[i, ]
  lines <- c(lines, sprintf("%-35s %-10.6f %-10.6f %-4d %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f",
                            r$cell, r$mean_d, r$se, r$df, r$ci_lower, r$ci_upper, r$exp_mean_d, r$exp_ci_lower, r$exp_ci_upper))
}

writeLines(lines, output_path)
cat(paste(lines, collapse = "\n"), "\n")
