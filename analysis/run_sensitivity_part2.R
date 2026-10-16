# Registered sensitivity analyses part 2: A11 throttle sensitivity and specification curve
# Plan v1 sections 4, 8, amendment A11

suppressMessages(library(lme4))
suppressMessages(library(lmerTest))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "analysis"
repo_root <- normalizePath(file.path(script_dir, ".."))

source(file.path(repo_root, "analysis", "rq2_gap_model.R"))
source(file.path(repo_root, "analysis", "helpers.R"))

cond_specs_path <- file.path(repo_root, "analysis", "results", "condition_energy_specs.csv")
h2_a11_path <- file.path(repo_root, "analysis", "results", "h2_input_a11.csv")
output_path <- file.path(repo_root, "analysis", "results", "sensitivity_part2_output.txt")

specs_data <- read.csv(cond_specs_path, stringsAsFactors = FALSE)
h2_a11_data <- read.csv(h2_a11_path, stringsAsFactors = FALSE)

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
# Reconciliation Gate
# ==============================================================================
lines <- c(lines, "=== Step 3: Reconciliation Gate ===")
# Run reconciliation via python script
py_recon_cmd <- sprintf("%s scripts/repeat_level_energy.py --sessions-dir results_stage5/cpu",
                        file.path(repo_root, "..", "measurement_env", "bin", "python3"))
recon_out <- system(py_recon_cmd, intern = TRUE)
for (ro in recon_out) {
  lines <- c(lines, ro)
}
lines <- c(lines, "")

# ==============================================================================
# A11 Throttle Sensitivity Analysis
# ==============================================================================
lines <- c(lines, "=== Step 4: A11 Throttle Sensitivity Analysis ===")
lines <- c(lines, "Rule: Exclude warm repeats where either active window (a1 or a2) has thermal_pkg_throttle_count_delta > 0.")
lines <- c(lines, "")

# Compute throttle audit per cell
py_audit_code <- paste(
  "import json, sys",
  "from scripts.repeat_level_energy import generate_throttle_audit",
  "audit = generate_throttle_audit('results_stage5/cpu')",
  "print(json.dumps(audit))",
  sep = "; "
)
py_audit_cmd <- sprintf("%s -c \"%s\"",
                        file.path(repo_root, "..", "measurement_env", "bin", "python3"),
                        py_audit_code)
audit_json_str <- system(py_audit_cmd, intern = TRUE)
audit_list <- jsonlite::fromJSON(audit_json_str)

lines <- c(lines, "Per-cell throttle audit across all 6 sessions:")
lines <- c(lines, sprintf("%-35s %-12s %-14s %-12s %-14s %-14s",
                          "cell", "warm_repeats", "warm_excluded", "warm_active", "throttled_act", "throttle_share"))
for (i in seq_len(nrow(audit_list))) {
  r <- audit_list[i, ]
  lines <- c(lines, sprintf("%-35s %-12d %-14d %-12d %-14d %-14.4f",
                            r$cell, r$warm_repeats_total, r$warm_repeats_excluded,
                            r$warm_active_windows_total, r$warm_active_windows_throttled,
                            r$throttled_active_share))
}
lines <- c(lines, "")

# Run fit_h2_primary on A11 input
validate_h2_gap_input_v1(h2_a11_data)
h2_a11_res <- fit_h2_primary(h2_a11_data)

lines <- c(lines, "-- A11 fit_h2_primary Output --")
lines <- c(lines, paste("primary_method:", h2_a11_res$primary_method))
lines <- c(lines, paste("complete_design:", h2_a11_res$complete_design))
lines <- c(lines, paste("estimate (mean d):", sprintf("%.8f", h2_a11_res$estimate)))
lines <- c(lines, paste("se:", sprintf("%.8f", h2_a11_res$se)))
lines <- c(lines, paste("df:", sprintf("%.4f", h2_a11_res$df)))
lines <- c(lines, paste("p_one_sided (H0: mean d <= 0 vs H1: mean d > 0):", sprintf("%.8e", h2_a11_res$p_one_sided)))
lines <- c(lines, paste("ci_lower (95%):", sprintf("%.8f", h2_a11_res$ci_lower)))
lines <- c(lines, paste("ci_upper (95%):", sprintf("%.8f", h2_a11_res$ci_upper)))
lines <- c(lines, paste("exp_estimate:", sprintf("%.8f", h2_a11_res$exp_estimate)))
lines <- c(lines, paste("exp_ci_lower:", sprintf("%.8f", h2_a11_res$exp_ci_lower)))
lines <- c(lines, paste("exp_ci_upper:", sprintf("%.8f", h2_a11_res$exp_ci_upper)))
lines <- c(lines, "")

# ==============================================================================
# Specification Curve (Plan v1 section 8)
# ==============================================================================
lines <- c(lines, "=== Step 5: Specification Curve (x86-CPU, Plan v1 section 8) ===")
lines <- c(lines, "Axes: instrument {package-0, psys} x runtime {fp32_ts, fp32_eager} x quantity {gross, net} = up to 8 points.")
lines <- c(lines, "Unavailable points (2 of 8):")
lines <- c(lines, "  - (psys, fp32_ts, net): Unavailable -- no psys-specific idle quantity or net column exists in windows.csv.")
lines <- c(lines, "  - (psys, fp32_eager, net): Unavailable -- no psys-specific idle quantity or net column exists in windows.csv.")
lines <- c(lines, "Available points: 6 points.")
lines <- c(lines, "")

models <- c("resnet18", "mobilenet_v3_small", "efficientnet_b0")
pruned_states <- c("pruned30", "pruned50", "pruned70")
int8_states <- c("int8")

target_states <- c()
for (m in models) {
  for (s in pruned_states) {
    target_states <- c(target_states, paste(m, s, sep = "_"))
  }
  target_states <- c(target_states, paste(m, "int8", sep = "_"))
}

# Spec point definitions: list(name, inst, base_type, qty, col_name)
spec_defs <- list(
  list(name = "(package-0, fp32_ts, gross)", inst = "package-0", base_type = "fp32_ts", qty = "gross", col = "pkg_gross", is_primary = TRUE),
  list(name = "(package-0, fp32_ts, net)", inst = "package-0", base_type = "fp32_ts", qty = "net", col = "pkg_net", is_primary = FALSE),
  list(name = "(package-0, fp32_eager, gross)", inst = "package-0", base_type = "fp32_eager", qty = "gross", col = "pkg_gross", is_primary = FALSE),
  list(name = "(package-0, fp32_eager, net)", inst = "package-0", base_type = "fp32_eager", qty = "net", col = "pkg_net", is_primary = FALSE),
  list(name = "(psys, fp32_ts, gross)", inst = "psys", base_type = "fp32_ts", qty = "gross", col = "psys_gross", is_primary = FALSE),
  list(name = "(psys, fp32_eager, gross)", inst = "psys", base_type = "fp32_eager", qty = "gross", col = "psys_gross", is_primary = FALSE)
)

summary_rows <- list()

for (st in target_states) {
  # Determine arch
  arch <- if (startsWith(st, "resnet18")) "resnet18"
          else if (startsWith(st, "mobilenet_v3_small")) "mobilenet_v3_small"
          else if (startsWith(st, "efficientnet_b0")) "efficientnet_b0"
          else stop("Unknown arch: ", st)

  lines <- c(lines, paste("State:", st))
  lines <- c(lines, "  Available specification points (6):")

  spec_log_r_vals <- c()
  w_star <- NA
  log_r_star <- NA

  for (sp in spec_defs) {
    base_cond <- paste(arch, sp$base_type, sep = "_")

    # Merge state and baseline across sessions
    st_sub <- specs_data[specs_data$condition == st, ]
    base_sub <- specs_data[specs_data$condition == base_cond, ]
    m <- merge(st_sub, base_sub, by = "session", suffixes = c("_st", "_base"))

    st_e <- m[[paste0(sp$col, "_st")]]
    base_e <- m[[paste0(sp$col, "_base")]]
    session_log_r <- log(st_e / base_e)
    pt_mean <- mean(session_log_r)

    spec_log_r_vals <- c(spec_log_r_vals, pt_mean)

    tag <- if (sp$is_primary) " [PRIMARY]" else ""
    lines <- c(lines, sprintf("    %-35s: log_r = %11.6f%s", sp$name, pt_mean, tag))

    if (sp$is_primary) {
      log_r_star <- pt_mean
      se_star <- sd(session_log_r) / sqrt(length(session_log_r))
      df_star <- length(session_log_r) - 1
      w_star <- qt(0.975, df = df_star) * se_star
    }
  }

  sc_res <- spec_curve_metrics(spec_log_r_vals, w_star, log_r_star, margin = log(1.05))
  lines <- c(lines, sprintf("  sd_spec    : %.6f", sc_res$sd_spec))
  lines <- c(lines, sprintf("  w_star     : %.6f", w_star))
  lines <- c(lines, sprintf("  headline_1 : %.6f", sc_res$headline_1))
  lines <- c(lines, sprintf("  headline_2 : %.6f", sc_res$headline_2))
  lines <- c(lines, "")

  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    state = st,
    primary_log_r = log_r_star,
    w_star = w_star,
    sd_spec = sc_res$sd_spec,
    headline_1 = sc_res$headline_1,
    headline_2 = sc_res$headline_2,
    stringsAsFactors = FALSE
  )
}

summary_table <- do.call(rbind, summary_rows)

lines <- c(lines, "=== Specification Curve Summary Across States ===")
lines <- c(lines, sprintf("%-30s %-14s %-12s %-12s %-12s %-12s",
                          "state", "primary_log_r", "w_star", "sd_spec", "headline_1", "headline_2"))
for (i in seq_len(nrow(summary_table))) {
  r <- summary_table[i, ]
  lines <- c(lines, sprintf("%-30s %-14.6f %-12.6f %-12.6f %-12.6f %-12.6f",
                            r$state, r$primary_log_r, r$w_star, r$sd_spec, r$headline_1, r$headline_2))
}
lines <- c(lines, "")

# ==============================================================================
# Recorded Decision: _bnrecal Separate Fit
# ==============================================================================
lines <- c(lines, "=== Recorded Decision: _bnrecal Separate Fit ===")
lines <- c(lines, "Researcher decision recorded: The registered _bnrecal separate fit is not computed")
lines <- c(lines, "(only 6 of 9 cells exist by design under BNRECAL_SKIP; decision rule D3r1 answers its question).")
lines <- c(lines, "Per instructions, no attempt was made to fit fit_h2_gap_model on _bnrecal data.")
lines <- c(lines, "")

# Write to file
writeLines(lines, output_path)

# Print full output to stdout
cat(paste(lines, collapse = "\n"), "\n")
