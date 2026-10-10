# Check script (not a simulation, not a test): computes the within-architecture Frisch-Waugh
# weights the registered RQ2 model's pooled log_mac slope implicitly puts on each architecture,
# and the resulting implied pooled slope for the two heterogeneous-true-slope scenarios already
# simulated in analysis/simulate_rq2_block_p.R/simulate_rq2_diagnostics.R. Registered in
# stage5_analysis_plan.md A7r9(b) as the explanation for those scenarios' earlier-reported
# "bias" figures -- the gap was the pooled estimator's own design weighting, not estimation
# noise or error. Synthetic/design-constant only -- no results_* file is read.
#
# Usage: Rscript analysis/check_pooled_slope_weights.R

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

# Frisch-Waugh-Lovell: the pooled slope on log_mac, controlling for the model fixed effect, is
# a within-model-sum-of-squares-weighted average of each architecture's own slope. Weight for
# architecture m = SSX_m / sum(SSX_m'), where SSX_m is the within-group sum of squared deviations
# of log_mac from its own group mean (equal weights over the 9 design points, as the simulation
# itself uses -- no session-count or other weighting enters this, since every session replicates
# the identical 9-point design).
ssx <- tapply(design$log_mac, design$model, function(x) sum((x - mean(x))^2))
weights <- ssx / sum(ssx)

cat("Within-architecture sum-of-squared-deviations (SSX) of log_mac:\n")
print(ssx)
cat("\nFrisch-Waugh weights (SSX_m / sum(SSX)):\n")
print(weights)
cat("Weights sum to:", sum(weights), "(should be 1)\n\n")

implied_pooled_slope <- function(beta_per_model) {
  sum(weights[names(beta_per_model)] * beta_per_model)
}

het_low <- c(resnet18 = 0.6, mobilenet_v3_small = 0.8, efficientnet_b0 = 1.0)
het_high <- c(resnet18 = 0.9, mobilenet_v3_small = 1.0, efficientnet_b0 = 1.1)

slope_low <- implied_pooled_slope(het_low)
slope_high <- implied_pooled_slope(het_high)

cat(sprintf("Scenario (0.6/0.8/1.0), simple pooled mean = %.4f, implied weighted slope = %.6f\n",
            mean(het_low), slope_low))
cat(sprintf("Scenario (0.9/1.0/1.1), simple pooled mean = %.4f, implied weighted slope = %.6f\n",
            mean(het_high), slope_high))

# Compare against the simulated mean beta_hat already recorded in analysis/sim_output (from
# simulate_rq2_block_p.R: bias = mean(beta_hats) - simple_pooled_mean, so
# mean(beta_hats) = simple_pooled_mean + bias).
sim_path <- "analysis/sim_output/summary_block_p.csv"
if (file.exists(sim_path)) {
  sim <- read.csv(sim_path)
  row_low <- sim[sim$label == "heterogeneous_pooled_0.8", ]
  row_high <- sim[sim$label == "heterogeneous_pooled_1.0", ]
  if (nrow(row_low) == 1 && nrow(row_high) == 1) {
    sim_mean_low <- row_low$beta_true + row_low$bias
    sim_mean_high <- row_high$beta_true + row_high$bias
    cat(sprintf("\nSimulated mean beta_hat (0.6/0.8/1.0 scenario): %.6f (implied weighted slope: %.6f, diff: %.6f)\n",
                sim_mean_low, slope_low, sim_mean_low - slope_low))
    cat(sprintf("Simulated mean beta_hat (0.9/1.0/1.1 scenario): %.6f (implied weighted slope: %.6f, diff: %.6f)\n",
                sim_mean_high, slope_high, sim_mean_high - slope_high))
  } else {
    cat("\n(Could not find both heterogeneous rows in", sim_path, "-- skipping simulated comparison.)\n")
  }
} else {
  cat("\n(", sim_path, "not found -- skipping simulated comparison.)\n")
}
