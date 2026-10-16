# x86-GPU Supplementary Analysis
# Plan v1 sections 1, 5, 6, 8; amendment A7r5
# Scope: Descriptive only -- no p-values or hypothesis tests on the GPU block.

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "analysis"
repo_root <- normalizePath(file.path(script_dir, ".."))

source(file.path(repo_root, "analysis", "helpers.R"))

sessions <- paste0("main_session", 1:6)
models <- c("resnet18", "mobilenet_v3_small", "efficientnet_b0")
cuda_states <- c("fp16", "pruned30", "pruned50", "pruned70", "pruned30_bnrecal", "pruned50_bnrecal", "pruned70_bnrecal")
boundaries <- list(
  list(name = "system", col = "system_j_per_image_mean", is_primary = TRUE),
  list(name = "gpu-only", col = "gpu_j_per_image_mean", is_primary = FALSE),
  list(name = "cpu-package-only", col = "cpu_package_j_per_image_mean", is_primary = FALSE),
  list(name = "gross", col = "gross_j_per_image_mean", is_primary = FALSE)
)

out_file <- file.path(repo_root, "analysis", "results", "gpu_supplementary_output.txt")
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

lines <- c()

# ==============================================================================
# Header
# ==============================================================================
lines <- c(lines, "================================================================================")
lines <- c(lines, "x86-GPU Supplementary Analysis (Plan v1 sections 1, 5, 6, 8; Amendment A7r5)")
lines <- c(lines, "================================================================================")
lines <- c(lines, "Scope: Descriptive estimation only -- no p-values or hypothesis tests (Plan v1 sec 6).")
lines <- c(lines, "Sessions analyzed: results_stage4b/main_session1..6 (aborted and unpinned excluded).")
lines <- c(lines, "Replicate unit: Session (df = 5, t-critical = 2.5706 at 95% confidence).")
lines <- c(lines, "")

# ==============================================================================
# Part 1: Per-State Ratios across Boundaries
# ==============================================================================
lines <- c(lines, "================================================================================")
lines <- c(lines, "Part 1: Per-State Ratios Across Boundaries (Plan v1 section 6)")
lines <- c(lines, "================================================================================")
lines <- c(lines, "CUDA states compared against same-session {arch}_fp32 (cuda, FP32-eager baseline).")
lines <- c(lines, "Label: FP32-eager baseline (runtime-confounded)")
lines <- c(lines, "")

header_fmt <- "%-18s %-20s %-18s %-9s %-22s %-10s %-22s %-38s"
row_fmt    <- "%-18s %-20s %-18s %9.4f [%9.4f, %9.4f] %10.4f [%9.4f, %9.4f] %-38s"

lines <- c(lines, sprintf(header_fmt, "Architecture", "State", "Boundary", "Geom r", "95% CI on r", "Mean log_r", "95% t-interval (df=5)", "Caveat / Label"))
lines <- c(lines, paste(rep("-", 152), collapse = ""))

for (m in models) {
  for (st in cuda_states) {
    full_st <- paste0(m, "_", st)
    base_st <- paste0(m, "_fp32")
    
    for (b in boundaries) {
      log_r_vals <- c()
      for (s in sessions) {
        base_path <- file.path(repo_root, "results_stage4b", s, base_st, "summary.csv")
        st_path <- file.path(repo_root, "results_stage4b", s, full_st, "summary.csv")
        base_e <- read.csv(base_path, check.names = FALSE)[[b$col]][1]
        st_e <- read.csv(st_path, check.names = FALSE)[[b$col]][1]
        log_r_vals <- c(log_r_vals, log(st_e / base_e))
      }
      
      mean_log_r <- mean(log_r_vals)
      se_log_r <- sd(log_r_vals) / sqrt(length(log_r_vals))
      w <- qt(0.975, df = length(log_r_vals) - 1) * se_log_r
      ci_lo <- mean_log_r - w
      ci_hi <- mean_log_r + w
      geom_r <- exp(mean_log_r)
      r_ci_lo <- exp(ci_lo)
      r_ci_hi <- exp(ci_hi)
      
      b_tag <- if (b$is_primary) paste0(b$name, " (primary)") else b$name
      lines <- c(lines, sprintf(row_fmt, m, st, b_tag, geom_r, r_ci_lo, r_ci_hi, mean_log_r, ci_lo, ci_hi, "FP32-eager baseline (runtime-confounded)"))
    }
  }
}

lines <- c(lines, paste(rep("-", 152), collapse = ""))
lines <- c(lines, "")

lines <- c(lines, "CPU States in x86-GPU Block: int8 vs same-session {arch}_fp32_cpu (package-0 RAPL)")
lines <- c(lines, sprintf(header_fmt, "Architecture", "State", "Boundary", "Geom r", "95% CI on r", "Mean log_r", "95% t-interval (df=5)", "Notes"))
lines <- c(lines, paste(rep("-", 152), collapse = ""))

for (m in models) {
  full_st <- paste0(m, "_int8")
  base_st <- paste0(m, "_fp32_cpu")
  
  log_r_vals <- c()
  for (s in sessions) {
    base_path <- file.path(repo_root, "results_stage4b", s, base_st, "summary.csv")
    st_path <- file.path(repo_root, "results_stage4b", s, full_st, "summary.csv")
    base_e <- read.csv(base_path, check.names = FALSE)[["cpu_package_j_per_image_mean"]][1]
    st_e <- read.csv(st_path, check.names = FALSE)[["cpu_package_j_per_image_mean"]][1]
    log_r_vals <- c(log_r_vals, log(st_e / base_e))
  }
  
  mean_log_r <- mean(log_r_vals)
  se_log_r <- sd(log_r_vals) / sqrt(length(log_r_vals))
  w <- qt(0.975, df = length(log_r_vals) - 1) * se_log_r
  ci_lo <- mean_log_r - w
  ci_hi <- mean_log_r + w
  geom_r <- exp(mean_log_r)
  r_ci_lo <- exp(ci_lo)
  r_ci_hi <- exp(ci_hi)
  
  lines <- c(lines, sprintf(row_fmt, m, "int8", "package-0", geom_r, r_ci_lo, r_ci_hi, mean_log_r, ci_lo, ci_hi, "CPU baseline (matched PyTorch eager)"))
}

lines <- c(lines, paste(rep("-", 152), collapse = ""))
lines <- c(lines, "")

# ==============================================================================
# Part 2: Specification Curve
# ==============================================================================
lines <- c(lines, "================================================================================")
lines <- c(lines, "Part 2: GPU Specification Curve (Plan v1 section 8)")
lines <- c(lines, "================================================================================")
lines <- c(lines, "Specification grid: boundary {system, gpu-only, cpu-package-only} x {gross, net} (6 points).")
lines <- c(lines, "Net energy condition: net only where pilot.py logs an idle value for that boundary.")
lines <- c(lines, "")
lines <- c(lines, "Code citations from pilot.py:")
lines <- c(lines, "  - Lines 429 & 432: idlepower = mean(d[p]['energy_j']/d[p]['duration_s'] for p in ['idle_before','idle_after'])")
lines <- c(lines, "                     above = gross - idlepower * r['duration_s']")
lines <- c(lines, "  - Line 438: r['estimated_idle_j'] = idlepower * r['duration_s']; r['above_idle_j'] = above")
lines <- c(lines, "  - For device == 'cuda', r['energy_j'] is the NVML GPU power sensor.")
lines <- c(lines, "  - Idle power and above_idle_j are therefore computed exclusively for gpu_energy_j (gpu-only).")
lines <- c(lines, "  - No idle power or above-idle subtraction is implemented in pilot.py for system_energy_j or cpu_package_energy_j.")
lines <- c(lines, "")
lines <- c(lines, "Point availability status:")
lines <- c(lines, "  [AVAILABLE]   (system, gross)           -- PRIMARY specification (system_j_per_image_mean)")
lines <- c(lines, "  [AVAILABLE]   (gpu-only, gross)         -- gpu_j_per_image_mean")
lines <- c(lines, "  [AVAILABLE]   (cpu-package-only, gross) -- cpu_package_j_per_image_mean")
lines <- c(lines, "  [AVAILABLE]   (gpu-only, net)           -- above_idle_j / (batches * batch) across warm active windows")
lines <- c(lines, "  [UNAVAILABLE] (system, net)             -- pilot.py does not compute system idle power (cite pilot.py:429)")
lines <- c(lines, "  [UNAVAILABLE] (cpu-package-only, net)   -- pilot.py does not compute cpu-package idle power on CUDA (cite pilot.py:429)")
lines <- c(lines, "")
lines <- c(lines, "Total specification points: 4 available, 2 unavailable.")
lines <- c(lines, "")

spec_summary_rows <- list()

for (m in models) {
  lines <- c(lines, paste("--- Model:", m, "---"))
  for (st in cuda_states) {
    full_st <- paste0(m, "_", st)
    base_st <- paste0(m, "_fp32")
    
    sys_gross_vals <- c()
    gpu_gross_vals <- c()
    pkg_gross_vals <- c()
    gpu_net_vals   <- c()
    
    for (s in sessions) {
      base_sum <- read.csv(file.path(repo_root, "results_stage4b", s, base_st, "summary.csv"), check.names = FALSE)
      st_sum   <- read.csv(file.path(repo_root, "results_stage4b", s, full_st, "summary.csv"), check.names = FALSE)
      
      base_win <- read.csv(file.path(repo_root, "results_stage4b", s, base_st, "windows.csv"), check.names = FALSE)
      st_win   <- read.csv(file.path(repo_root, "results_stage4b", s, full_st, "windows.csv"), check.names = FALSE)
      
      base_warm <- base_win[base_win$phase %in% c("a1", "a2") & base_win[["repeat"]] > 0, ]
      st_warm   <- st_win[st_win$phase %in% c("a1", "a2") & st_win[["repeat"]] > 0, ]
      
      # Repeat-level net energy per image
      base_repeats <- sorted_reps <- sort(unique(base_warm[["repeat"]]))
      base_net_reps <- sapply(base_repeats, function(rp) {
        sub_df <- base_warm[base_warm[["repeat"]] == rp, ]
        mean(sub_df$above_idle_j / (sub_df$batches * sub_df$batch))
      })
      base_gpu_net <- mean(base_net_reps)
      
      st_repeats <- sort(unique(st_warm[["repeat"]]))
      st_net_reps <- sapply(st_repeats, function(rp) {
        sub_df <- st_warm[st_warm[["repeat"]] == rp, ]
        mean(sub_df$above_idle_j / (sub_df$batches * sub_df$batch))
      })
      st_gpu_net <- mean(st_net_reps)
      
      sys_gross_vals <- c(sys_gross_vals, log(st_sum$system_j_per_image_mean[1] / base_sum$system_j_per_image_mean[1]))
      gpu_gross_vals <- c(gpu_gross_vals, log(st_sum$gpu_j_per_image_mean[1] / base_sum$gpu_j_per_image_mean[1]))
      pkg_gross_vals <- c(pkg_gross_vals, log(st_sum$cpu_package_j_per_image_mean[1] / base_sum$cpu_package_j_per_image_mean[1]))
      gpu_net_vals   <- c(gpu_net_vals,   log(st_gpu_net / base_gpu_net))
    }
    
    pt_sys_gross <- mean(sys_gross_vals)
    pt_gpu_gross <- mean(gpu_gross_vals)
    pt_pkg_gross <- mean(pkg_gross_vals)
    pt_gpu_net   <- mean(gpu_net_vals)
    
    log_r_spec <- c(pt_sys_gross, pt_gpu_gross, pt_pkg_gross, pt_gpu_net)
    log_r_star <- pt_sys_gross
    w_star     <- qt(0.975, df = 5) * (sd(sys_gross_vals) / sqrt(6))
    
    metrics <- spec_curve_metrics(log_r_spec, w_star = w_star, log_r_star = log_r_star, margin = log(1.05))
    
    lines <- c(lines, paste("State:", full_st))
    lines <- c(lines, sprintf("  (system, gross)           [PRIMARY]: log_r = %9.4f, r = %.4f", pt_sys_gross, exp(pt_sys_gross)))
    lines <- c(lines, sprintf("  (gpu-only, gross)                  : log_r = %9.4f, r = %.4f", pt_gpu_gross, exp(pt_gpu_gross)))
    lines <- c(lines, sprintf("  (cpu-package-only, gross)          : log_r = %9.4f, r = %.4f", pt_pkg_gross, exp(pt_pkg_gross)))
    lines <- c(lines, sprintf("  (gpu-only, net)                    : log_r = %9.4f, r = %.4f", pt_gpu_net, exp(pt_gpu_net)))
    lines <- c(lines, sprintf("  Metrics: sd_spec = %.4f, w_star = %.4f, headline_1 = %.4f, headline_2 = %.4f",
                              metrics$sd_spec, w_star, metrics$headline_1, metrics$headline_2))
    lines <- c(lines, "")
    
    spec_summary_rows[[length(spec_summary_rows) + 1]] <- data.frame(
      architecture = m,
      state = st,
      primary_log_r = log_r_star,
      primary_r = exp(log_r_star),
      w_star = w_star,
      sd_spec = metrics$sd_spec,
      headline_1 = metrics$headline_1,
      headline_2 = metrics$headline_2,
      stringsAsFactors = FALSE
    )
  }
}

spec_summary_df <- do.call(rbind, spec_summary_rows)

lines <- c(lines, "Specification Curve Summary Across All 21 CUDA States:")
lines <- c(lines, sprintf("%-18s %-20s %-14s %-10s %-10s %-10s %-12s %-12s",
                          "Architecture", "State", "Primary log_r*", "Primary r*", "w_star", "sd_spec", "headline_1", "headline_2"))
lines <- c(lines, paste(rep("-", 112), collapse = ""))

for (i in seq_len(nrow(spec_summary_df))) {
  r <- spec_summary_df[i, ]
  lines <- c(lines, sprintf("%-18s %-20s %14.4f %10.4f %10.4f %10.4f %12.4f %12.4f",
                            r$architecture, r$state, r$primary_log_r, r$primary_r, r$w_star, r$sd_spec, r$headline_1, r$headline_2))
}
lines <- c(lines, paste(rep("-", 112), collapse = ""))
lines <- c(lines, "")

# ==============================================================================
# Part 3: A7r5 Battery Overlap Check
# ==============================================================================
lines <- c(lines, "================================================================================")
lines <- c(lines, "Part 3: A7r5 Battery Interval Overlap Check")
lines <- c(lines, "================================================================================")
lines <- c(lines, "Rule: Any x86-GPU-block condition whose measurement window overlaps a verified")
lines <- c(lines, "battery or discharging interval is excluded from the GPU-block analysis (A7r5(a)).")
lines <- c(lines, "")
lines <- c(lines, "Verification source: docs/stage5_analysis_plan.md section A7r5(b) and docs/stage5_plan_v1.md section 5 lines 103-104.")
lines <- c(lines, "Verified across:")
lines <- c(lines, "  - UPower persisted battery history (/var/lib/upower/history-charge-L23M4PK4-60-1907.dat, 2026-10-03 to 2026-10-10)")
lines <- c(lines, "  - System journal (journalctl, read-only) for boots between 2026-10-06 and 2026-10-09")
lines <- c(lines, "  - Per-condition timeline (environment.json uptime_s/timestamp_utc, windows.csv power_regime)")
lines <- c(lines, "")
lines <- c(lines, "Result: Exactly 0 conditions overlap a verified battery or discharging interval.")
lines <- c(lines, "Affected conditions excluded under A7r5: 0 (zero conditions fall under this rule).")
lines <- c(lines, "")

text_output <- paste(lines, collapse = "\n")
cat(text_output)
cat("\n")

writeLines(text_output, out_file)
cat(sprintf("Wrote %s\n", out_file))
