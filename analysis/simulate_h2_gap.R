suppressMessages(library(lmerTest))
suppressMessages(library(parallel))
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "analysis"
source(file.path(script_dir, "rq2_gap_model.R"))
source(file.path(script_dir, "rq2_model.R"))

args <- commandArgs(trailingOnly = TRUE)
n_reps <- if (length(args) >= 1) as.integer(args[1]) else 1000

out_dir <- file.path(script_dir, "sim_output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

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

simulate_h2_gap <- function(type, param, session_sd, base_sd, state_sd, control_no_shared = FALSE) {
  if (type == "proportional") {
    c_val <- param
    true_log_ratio <- log(c_val) + design$log_mac
  } else if (type == "power_law") {
    beta <- param
    true_log_ratio <- beta * design$log_mac
  } else if (type == "energy_floor") {
    f <- param
    true_log_ratio <- log(f + (1 - f) * design$mac_ratio)
  } else if (type == "hetero_floor") {
    f_vals <- param[design$model]
    true_log_ratio <- log(f_vals + (1 - f_vals) * design$mac_ratio)
  } else if (type == "hetero_mean_zero") {
    d_vals <- param[design$key]
    true_log_ratio <- design$log_mac + d_vals
  }
  
  session_eff <- rnorm(n_sessions, 0, session_sd)
  
  rows <- do.call(rbind, lapply(seq_len(n_sessions), function(s) {
    base_noise_vals <- rnorm(3, 0, base_sd)
    names(base_noise_vals) <- models
    if (control_no_shared) {
      base_noise <- rep(0, nrow(design))
      state_noise <- rnorm(nrow(design), 0, sqrt(base_sd^2 + state_sd^2))
    } else {
      base_noise <- base_noise_vals[design$model]
      state_noise <- rnorm(nrow(design), 0, state_sd)
    }
    noise <- state_noise - base_noise
    log_ratio <- true_log_ratio + session_eff[s] + noise
    data.frame(session = as.factor(s), model = as.factor(design$model), 
               state = as.factor(design$state), log_ratio = log_ratio, 
               log_mac = design$log_mac, stringsAsFactors = FALSE)
  }))
  list(data = rows, true_d = true_log_ratio - design$log_mac, true_log_ratio = true_log_ratio)
}

scenarios <- list()
add_scen <- function(name, type, param, s_sd, b_sd, st_sd, ctrl=FALSE) {
  scenarios[[length(scenarios)+1]] <<- list(name=name, type=type, param=param, session_sd=s_sd, base_sd=b_sd, state_sd=st_sd, ctrl=ctrl)
}

for(c_val in c(0.8, 1.0, 1.1, 1.25)) add_scen(paste0("prop_c", c_val), "proportional", c_val, 0.04, 0.05, 0.02)
for(beta in c(0.8, 0.9, 1.0)) add_scen(paste0("pow_b", beta), "power_law", beta, 0.04, 0.05, 0.02)
for(f in c(0.1, 0.3, 0.5)) add_scen(paste0("floor_f", f), "energy_floor", f, 0.04, 0.05, 0.02)
add_scen("hetero_floor", "hetero_floor", c(resnet18=0.1, mobilenet_v3_small=0.3, efficientnet_b0=0.5), 0.04, 0.05, 0.02)
d_zero <- c(resnet18.pruned30=0.05, resnet18.pruned50=0.05, resnet18.pruned70=-0.1,
            mobilenet_v3_small.pruned30=-0.05, mobilenet_v3_small.pruned50=-0.05, mobilenet_v3_small.pruned70=0.1,
            efficientnet_b0.pruned30=0, efficientnet_b0.pruned50=0, efficientnet_b0.pruned70=0)
add_scen("hetero_mean0", "hetero_mean_zero", d_zero, 0.04, 0.05, 0.02)
add_scen("ctrl_prop1.0", "proportional", 1.0, 0.04, 0.05, 0.02, ctrl=TRUE)
add_scen("prop1.0_noise1", "proportional", 1.0, 0.01, 0.02, 0.05)
add_scen("prop1.0_noise2", "proportional", 1.0, 0.04, 0.05, 0.05)
add_scen("prop1.0_noise3", "proportional", 1.0, 0.01, 0.05, 0.02)

fit_intercept_only <- function(data) {
  fit <- tryCatch(lmerTest::lmer(d ~ 1 + (1|session), data=data, REML=TRUE), error=function(e) e)
  if (inherits(fit, "error")) return(list(error=fit$message))
  sum_fit <- summary(fit)
  list(mu_hat = sum_fit$coefficients[1, "Estimate"], se = sum_fit$coefficients[1, "Std. Error"],
       df = sum_fit$coefficients[1, "df"], t_stat = sum_fit$coefficients[1, "t value"],
       p_one_sided = pt(sum_fit$coefficients[1, "t value"], sum_fit$coefficients[1, "df"], lower.tail=FALSE))
}

fit_old_model_gap <- function(data) {
  fit <- tryCatch(lmerTest::lmer(d ~ 0 + cell + (1|session), data=data, REML=TRUE), error=function(e) e)
  if (inherits(fit, "error")) return(list(error=fit$message))
  L <- rep(1/9, 9)
  cres <- lmerTest::contest1D(fit, L, rhs=0)
  list(mu_hat = cres$Estimate, se = cres$`Std. Error`, df = cres$df,
       t_stat = cres$Estimate/cres$`Std. Error`, p_one_sided = pt(cres$Estimate/cres$`Std. Error`, cres$df, lower.tail=FALSE))
}

res_rows <- list()

for (i in seq_along(scenarios)) {
  sc <- scenarios[[i]]
  set.seed(9300 + i)
  
  sim_init <- simulate_h2_gap(sc$type, sc$param, sc$session_sd, sc$base_sd, sc$state_sd, sc$ctrl)
  true_d_vec <- sim_init$true_d
  true_log_ratio <- sim_init$true_log_ratio
  true_mean_d <- mean(true_d_vec)
  
  anc_slopes <- sapply(models, function(m) {
    idx <- which(design$model == m)
    sum(design$log_mac[idx] * true_log_ratio[idx]) / sum(design$log_mac[idx]^2)
  })
  true_anc_slope <- mean(anc_slopes)
  true_free_slope <- cov(design$log_mac, true_log_ratio) / var(design$log_mac)
  
  # pre-generate seeds for parallel runs
  rep_seeds <- sample.int(1e8, n_reps)
  
  results_list <- mclapply(seq_len(n_reps), function(r) {
    set.seed(rep_seeds[r])
    sim <- simulate_h2_gap(sc$type, sc$param, sc$session_sd, sc$base_sd, sc$state_sd, sc$ctrl)
    df_data <- sim$data
    df_data$d <- df_data$log_ratio - df_data$log_mac
    
    out <- list()
    f_pri <- suppressMessages(suppressWarnings(fit_h2_gap_model(df_data)))
    if (!is.null(f_pri$error) || length(f_pri) < 5) {
      out$pri_mu <- NA; out$pri_conv <- FALSE; out$pri_sing <- NA
    } else {
      out$pri_mu <- f_pri$mu_hat; out$pri_se <- f_pri$se; out$pri_p <- f_pri$p_one_sided
      out$pri_cov <- (true_mean_d >= f_pri$ci_lower) && (true_mean_d <= f_pri$ci_upper)
      out$pri_conv <- f_pri$converged; out$pri_sing <- f_pri$singular
    }
    
    f_tt <- suppressMessages(suppressWarnings(session_mean_t_test(df_data)))
    if (!is.null(f_tt$error)) out$tt_mu <- NA else {
      out$tt_mu <- f_tt$mu_hat; out$tt_se <- f_tt$se; out$tt_p <- f_tt$p_one_sided
      out$tt_cov <- (true_mean_d >= f_tt$ci_lower) && (true_mean_d <= f_tt$ci_upper)
    }
    
    f_int <- suppressMessages(suppressWarnings(fit_intercept_only(df_data)))
    if (!is.null(f_int$error)) out$int_mu <- NA else {
      out$int_mu <- f_int$mu_hat; out$int_se <- f_int$se; out$int_p <- f_int$p_one_sided
      out$int_cov <- (true_mean_d >= (f_int$mu_hat - qt(0.975, f_int$df)*f_int$se)) && 
                    (true_mean_d <= (f_int$mu_hat + qt(0.975, f_int$df)*f_int$se))
    }
    
    f_old <- suppressMessages(suppressWarnings(fit_old_model_gap(df_data)))
    if (!is.null(f_old$error)) out$old_mu <- NA else {
      out$old_mu <- f_old$mu_hat; out$old_se <- f_old$se; out$old_p <- f_old$p_one_sided
      out$old_cov <- (true_mean_d >= (f_old$mu_hat - qt(0.975, f_old$df)*f_old$se)) && 
                    (true_mean_d <= (f_old$mu_hat + qt(0.975, f_old$df)*f_old$se))
    }
    
    f_anc <- suppressMessages(suppressWarnings(fit_anchored_slopes(df_data)))
    if (!is.null(f_anc$error)) out$anc_mu <- NA else {
      out$anc_mu <- f_anc$mean_slope; out$anc_se <- f_anc$se; out$anc_p <- f_anc$p_one_sided
      out$anc_cov <- (true_anc_slope >= f_anc$ci_lower) && (true_anc_slope <= f_anc$ci_upper)
    }
    
    f_fre <- suppressMessages(suppressWarnings(fit_sensitivity_free_intercept(df_data)))
    if (!is.null(f_fre$error)) out$fre_mu <- NA else {
      out$fre_mu <- f_fre$beta_hat; out$fre_se <- f_fre$se; out$fre_p <- f_fre$p_one_sided
      out$fre_cov <- (true_free_slope >= (f_fre$beta_hat - qt(0.975, f_fre$df)*f_fre$se)) && 
                    (true_free_slope <= (f_fre$beta_hat + qt(0.975, f_fre$df)*f_fre$se))
    }
    out
  }, mc.cores = detectCores() - 1)
  
  # Aggregate
  agg <- function(name) sapply(results_list, function(x) x[[name]])
  
  pri_mu <- agg("pri_mu"); pri_se <- agg("pri_se"); pri_p <- agg("pri_p"); pri_cov <- agg("pri_cov")
  pri_conv <- agg("pri_conv"); pri_sing <- agg("pri_sing")
  tt_mu <- agg("tt_mu"); tt_se <- agg("tt_se"); tt_p <- agg("tt_p"); tt_cov <- agg("tt_cov")
  int_mu <- agg("int_mu"); int_se <- agg("int_se"); int_p <- agg("int_p"); int_cov <- agg("int_cov")
  old_mu <- agg("old_mu"); old_se <- agg("old_se"); old_p <- agg("old_p"); old_cov <- agg("old_cov")
  anc_mu <- agg("anc_mu"); anc_se <- agg("anc_se"); anc_p <- agg("anc_p"); anc_cov <- agg("anc_cov")
  fre_mu <- agg("fre_mu"); fre_se <- agg("fre_se"); fre_p <- agg("fre_p"); fre_cov <- agg("fre_cov")
  
  sum_mod <- function(prefix, mu, se, p, cov, tr_val) {
    res <- list()
    res[[paste0(prefix, "_bias")]] <- mean(unlist(mu), na.rm=TRUE) - tr_val
    res[[paste0(prefix, "_sd")]] <- sd(unlist(mu), na.rm=TRUE)
    res[[paste0(prefix, "_mean_se")]] <- mean(unlist(se), na.rm=TRUE)
    res[[paste0(prefix, "_se_sd_ratio")]] <- res[[paste0(prefix, "_mean_se")]] / res[[paste0(prefix, "_sd")]]
    res[[paste0(prefix, "_coverage")]] <- mean(unlist(cov), na.rm=TRUE)
    res[[paste0(prefix, "_rej_rate")]] <- mean(unlist(p) < 0.05, na.rm=TRUE)
    res
  }
  
  rlist <- list(
    scenario = sc$name,
    true_mean_d = true_mean_d,
    true_anc_slope = true_anc_slope,
    true_free_slope = true_free_slope,
    pri_n_nonconv = sum(!unlist(pri_conv), na.rm=TRUE),
    pri_n_sing = sum(unlist(pri_sing), na.rm=TRUE)
  )
  rlist <- c(rlist, sum_mod("pri", pri_mu, pri_se, pri_p, pri_cov, true_mean_d))
  rlist <- c(rlist, sum_mod("tt", tt_mu, tt_se, tt_p, tt_cov, true_mean_d))
  rlist <- c(rlist, sum_mod("int", int_mu, int_se, int_p, int_cov, true_mean_d))
  rlist <- c(rlist, sum_mod("old", old_mu, old_se, old_p, old_cov, true_mean_d))
  rlist <- c(rlist, sum_mod("anc", anc_mu, anc_se, anc_p, anc_cov, true_anc_slope))
  rlist <- c(rlist, sum_mod("fre", fre_mu, fre_se, fre_p, fre_cov, true_free_slope))
  
  res_rows[[i]] <- as.data.frame(rlist)
  cat(sprintf("Done %s\n", sc$name))
}

final_df <- do.call(rbind, res_rows)
write.csv(final_df, file.path(out_dir, "summary_h2_gap.csv"), row.names=FALSE)
cat("Written summary_h2_gap.csv\n")
