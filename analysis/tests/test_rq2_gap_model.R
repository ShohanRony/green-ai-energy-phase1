source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE))), "..", "rq2_gap_model.R"))

test_data <- data.frame(
  session = rep(1:6, each=9),
  model = rep(rep(c("resnet18", "mobilenet_v3_small", "efficientnet_b0"), each=3), times=6),
  state = rep(rep(c("pruned30", "pruned50", "pruned70"), times=3), times=6),
  log_ratio = rnorm(54),
  log_mac = rnorm(54)
)

# 1. Normal path
res <- fit_h2_gap_model(test_data)
stopifnot(!is.null(res$mu_hat))
stopifnot(!is.null(res$se))

# 2. Missing cell
test_data_missing <- test_data[-1, ]
res_missing <- fit_h2_gap_model(test_data_missing)
stopifnot(res_missing$complete_design == FALSE)
stopifnot(!is.null(res_missing$mu_hat))

# 3. Duplicate rows
test_data_dup <- rbind(test_data, test_data[1,])
res_dup <- tryCatch(fit_h2_gap_model(test_data_dup), error=function(e) e)
stopifnot(inherits(res_dup, "error"))

# 4. bnrecal rejection
test_data_bn <- test_data
test_data_bn$state[1] <- "pruned30_bnrecal"
res_bn <- tryCatch(fit_h2_gap_model(test_data_bn), error=function(e) e)
stopifnot(inherits(res_bn, "error"))

# 5. fallback triggered on error
# Pass a non-factor character to lmer via changing something inside fit_h2_gap_model? 
# We can force an error by sending bad data types
bad_data <- test_data
bad_data$log_ratio <- as.character(bad_data$log_ratio)
res_fall <- fit_h2_with_fallback(bad_data)
stopifnot(res_fall$model_used == "session_mean_t")
stopifnot(res_fall$fallback_triggered == TRUE)

# 6. session_mean_t equals manual calculation
test_data$d <- test_data$log_ratio - test_data$log_mac
sess_means <- tapply(test_data$d, test_data$session, mean)
tt_res <- session_mean_t_test(test_data)
stopifnot(abs(tt_res$mu_hat - mean(sess_means)) < 1e-10)

cat("test_rq2_gap_model.R: All checks passed\n")

test_fit_h2_primary <- function() {
  set.seed(42)
  # Complete design
  df_complete <- expand.grid(
    session = 1:6,
    model = c("resnet18", "mobilenet_v3_small", "efficientnet_b0"),
    state = c("pruned30", "pruned50", "pruned70"),
    stringsAsFactors = FALSE
  )
  df_complete$log_mac <- runif(nrow(df_complete), -1, 0)
  df_complete$log_ratio <- df_complete$log_mac + rnorm(nrow(df_complete), 0, 0.1)
  
  res_complete <- fit_h2_primary(df_complete)
  if (res_complete$primary_method != "session_mean_t") stop("Failed: complete design did not use session_mean_t")
  if (!res_complete$complete_design) stop("Failed: complete design flag wrong")
  
  # Estimates agree when complete
  t_res <- session_mean_t_test(df_complete)
  mixed_res <- fit_h2_gap_model(df_complete)
  if (abs(t_res$mu_hat - mixed_res$mu_hat) > 1e-10) stop("Failed: estimates don't agree")
  if (abs(res_complete$estimate - t_res$mu_hat) > 1e-10) stop("Failed: primary estimate doesn't match t_test")
  
  # One missing cell switches method
  df_missing <- df_complete[-1, ]
  res_missing <- fit_h2_primary(df_missing)
  if (res_missing$primary_method != "cell_fixed_mixed") stop("Failed: missing cell did not switch to cell_fixed_mixed")
  if (res_missing$complete_design) stop("Failed: complete design flag wrong for missing cell")
  if (abs(res_missing$estimate - fit_h2_gap_model(df_missing)$mu_hat) > 1e-10) stop("Failed: missing cell estimate doesn't match mixed model")
  
  cat("test_fit_h2_primary passed\n")
}

test_fit_h2_primary()
