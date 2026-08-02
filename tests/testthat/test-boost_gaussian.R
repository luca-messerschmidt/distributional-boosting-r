# tests/testthat/test-boost_gaussian.R
#
# Core boost_gaussian() behaviour: one-step algorithm correctness, fitted
# object structure, input validation, the formula interface, update
# schedules (cyclic/noncyclic, separate mstop_mu/mstop_sigma), and the risk
# trace / internal early stopping.

test_that("one boosting step matches the manual intercept-corrected base learner", {
  dat <- make_boost_data()
  fit <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 1, nu_mu = 0.1, nu_sigma = 0.1)

  X_std  <- scale(dat$X_mat)
  sigma0 <- sd(dat$y)
  u_mu   <- (dat$y - mean(dat$y)) / sigma0^2
  a_mu   <- mean(u_mu)
  rss <- vapply(seq_len(ncol(X_std)), function(j) {
    xj <- X_std[, j]; b <- sum(xj * u_mu) / sum(xj^2)
    sum((u_mu - (a_mu + b * xj))^2)
  }, numeric(1))
  j  <- which.min(rss)
  xj <- X_std[, j]
  b  <- sum(xj * u_mu) / sum(xj^2)

  expect_equal(fit$selected_mu[1], colnames(X_std)[j])
  expect_equal(fit$coef_mu[1], unname(0.1 * b), tolerance = 1e-10)
  expect_equal(fit$fitted_mu, as.numeric(mean(dat$y) + 0.1 * (a_mu + b * xj)),
               tolerance = 1e-10)
})

test_that("nu_mu and nu_sigma each scale their own step exactly, independently", {
  # Vary one learning rate at a time so the other submodel's gradient target
  # (which depends on the OTHER submodel's current fit) stays identical.
  dat <- make_boost_data()
  base     <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 1, nu_mu = 0.1, nu_sigma = 0.1)
  fast_mu  <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 1, nu_mu = 1.0, nu_sigma = 0.1)
  fast_sig <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 1, nu_mu = 0.1, nu_sigma = 1.0)

  expect_equal(fast_mu$coef_mu[1],    10 * base$coef_mu[1],    tolerance = 1e-10)
  expect_equal(fast_sig$coef_sigma[1], 10 * base$coef_sigma[1], tolerance = 1e-10)
})

test_that("linear rescaling of a predictor leaves fitted values unchanged", {
  dat <- make_boost_data()
  X2 <- dat$X_df; X2$x1 <- 100 * X2$x1 + 50
  Z2 <- dat$Z_df; Z2$z1 <- 50  * Z2$z1  - 10

  fit_orig <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 30)
  fit_X    <- boost_gaussian(X2,       dat$Z_df, dat$y, mstop = 30)
  fit_Z    <- boost_gaussian(dat$X_df, Z2,       dat$y, mstop = 30)

  expect_equal(fit_orig$fitted_mu,    fit_X$fitted_mu,    tolerance = 1e-8)
  expect_equal(fit_orig$fitted_sigma, fit_Z$fitted_sigma, tolerance = 1e-8)
})

test_that("negative log-likelihood is non-increasing across boosting iterations", {
  dat <- make_boost_data()
  fit <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 100)

  full_trace <- c(fit$risk0, fit$risk)
  expect_length(fit$risk, 100)
  expect_true(all(diff(full_trace) <= 1e-6))
  expect_lt(fit$risk[100], fit$risk0)
  expect_equal(fit$risk[100],
               -sum(dnorm(dat$y, fit$fitted_mu, fit$fitted_sigma, log = TRUE)),
               tolerance = 1e-10)
})

test_that("fitted object has the expected fields, dimensions, and derived quantities", {
  obj <- make_boost_fit(n = 80, pX = 3, pZ = 3, mstop = 50)
  fit <- obj$fit

  required <- c(
    "initial_mu", "selected_mu", "coef_mu", "intercept_step_mu", "round_mu",
    "net_coef_mu_std", "net_coef_mu_orig", "intercept_mu", "fitted_mu", "smooth_terms_mu",
    "initial_log_sigma", "selected_sigma", "coef_sigma", "intercept_step_sigma",
    "round_sigma", "net_coef_sigma_std", "net_coef_sigma_orig", "intercept_sigma",
    "fitted_log_sigma", "fitted_sigma", "smooth_terms_sigma",
    "residuals", "r_squared", "method", "learner", "mstop", "mstop_mu", "mstop_sigma",
    "step_log", "risk", "risk0", "early_stopping", "nu_mu", "nu_sigma", "n", "pX", "pZ",
    "X_names", "Z_names", "X_center", "X_scale", "Z_center", "Z_scale", "X_std", "Z_std",
    "terms_mu", "terms_sigma", "call"
  )
  expect_named(fit, required, ignore.order = TRUE)

  expect_length(fit$selected_mu, 50)
  expect_length(fit$fitted_mu, obj$n)
  expect_length(fit$net_coef_mu_orig, obj$pX)
  expect_equal(fit$X_center, colMeans(obj$X_mat), tolerance = 1e-10)
  expect_equal(fit$initial_mu, mean(obj$y), tolerance = 1e-10)
  expect_equal(fit$initial_log_sigma, log(sd(obj$y)), tolerance = 1e-10)
  expect_equal(fit$residuals, obj$y - fit$fitted_mu, tolerance = 1e-10)
  expect_equal(fit$fitted_sigma, exp(fit$fitted_log_sigma), tolerance = 1e-10)
  expect_true(all(fit$fitted_sigma > 0))
  expect_equal(fit$r_squared,
               1 - sum(fit$residuals^2) / sum((obj$y - mean(obj$y))^2), tolerance = 1e-10)
  expect_true(all(fit$selected_mu %in% names(obj$X_df)))
  expect_true(all(is.finite(fit$fitted_mu)) && all(is.finite(fit$coef_mu)))
})

test_that("matrix inputs, p > n, and a constant response are all handled", {
  dat <- make_boost_data()
  fit_mat <- boost_gaussian(dat$X_mat, dat$Z_mat, dat$y, mstop = 10)
  expect_s3_class(fit_mat, "boost_gaussian")
  expect_equal(fit_mat$n, dat$n)

  X_hd <- as.data.frame(matrix(rnorm(20 * 30), nrow = 20))
  fit_hd <- boost_gaussian(X_hd, X_hd, rnorm(20), mstop = 10)
  expect_equal(fit_hd$pX, 30)
  expect_length(fit_hd$fitted_mu, 20)

  fit_const <- boost_gaussian(dat$X_df, dat$Z_df, rep(3, dat$n), mstop = 5)
  expect_true(is.na(fit_const$r_squared))
})

test_that("malformed inputs are rejected with informative errors", {
  dat <- make_boost_data()

  expect_error(boost_gaussian(dat$X_df$x1, dat$Z_df, dat$y), "matrix or data.frame")
  expect_error(boost_gaussian(dat$X_df, dat$Z_df, dat$y[-1]), "match length")
  expect_error(boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = -1), "mstop")
  expect_error(boost_gaussian(dat$X_df, dat$Z_df, dat$y, nu_mu = 1.5), "nu_mu")
  expect_error(boost_gaussian(dat$X_df, dat$Z_df, dat$y, nu_sigma = 0), "nu_sigma")

  x_na <- dat$X_df; x_na[1, 1] <- NA
  expect_error(boost_gaussian(x_na, dat$Z_df, dat$y), "missing")
  x_inf <- dat$X_df; x_inf[1, 1] <- Inf
  expect_error(boost_gaussian(x_inf, dat$Z_df, dat$y), "finite")

  x_factor <- dat$X_df; x_factor$group <- factor("a")[1]
  expect_error(boost_gaussian(x_factor, dat$Z_df, dat$y), "numeric")
})

test_that("method = 'noncyclic' updates exactly one submodel per round", {
  dat <- make_boost_data()
  fit <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 50, method = "noncyclic")

  expect_equal(fit$mstop_mu + fit$mstop_sigma, 50)
  expect_gt(fit$mstop_mu, 0)
  expect_gt(fit$mstop_sigma, 0)
  expect_equal(sort(c(fit$round_mu, fit$round_sigma)), seq_len(50))
  expect_error(
    boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = c(mu = 10, sigma = 20),
                   method = "noncyclic"),
    "cyclic"
  )
})

test_that("separate mstop_mu/mstop_sigma freezes the smaller submodel", {
  dat <- make_boost_data()
  fit <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = c(mu = 50, sigma = 20))

  expect_equal(fit$mstop_mu, 50)
  expect_equal(fit$mstop_sigma, 20)
  expect_equal(max(fit$round_sigma), 20)

  sigma_20 <- predict(fit, mstop = 20, what = "sigma")
  sigma_50 <- predict(fit, mstop = 50, what = "sigma")
  expect_equal(sigma_20, sigma_50, tolerance = 1e-10)

  fit_equal <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = c(mu = 40, sigma = 40))
  fit_scalar <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 40)
  expect_equal(fit_equal$fitted_mu, fit_scalar$fitted_mu, tolerance = 1e-10)
})

test_that("formula interface reproduces the matrix-interface fit", {
  dat <- make_boost_data()
  df  <- cbind(dat$X_df, dat$Z_df, y = dat$y)

  fit_matrix  <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 40)
  fit_formula <- boost_gaussian(
    formula = list(mu = y ~ x1 + x2 + x3, sigma = ~ z1 + z2 + z3),
    data = df, mstop = 40
  )

  expect_equal(fit_formula$fitted_mu, fit_matrix$fitted_mu, tolerance = 1e-10)
  expect_false(is.null(fit_formula$terms_mu))
  expect_true(is.null(fit_matrix$terms_mu))

  pred_formula <- predict(fit_formula, newdata = df[1:5, ], what = "mu")
  pred_matrix  <- predict(fit_matrix, newdata_X = df[1:5, names(dat$X_df)],
                          newdata_Z = df[1:5, names(dat$Z_df)], what = "mu")
  expect_equal(pred_formula, pred_matrix, tolerance = 1e-10)

  expect_error(boost_gaussian(formula = list(mu = y ~ x1), data = df, mstop = 10), "sigma")
})

test_that("internal early stopping (patience) is reproducible and validated", {
  dat <- make_boost_data(n = 150)
  fit <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 300, patience = 10,
                        validation_split = 0.2, seed = 42)

  expect_true(fit$early_stopping$used)
  expect_equal(fit$mstop, fit$early_stopping$best_round)
  expect_equal(fit$n, dat$n)

  fit2 <- boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 300, patience = 10,
                         validation_split = 0.2, seed = 42)
  expect_equal(fit$early_stopping$best_round, fit2$early_stopping$best_round)

  expect_error(
    boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = c(mu = 50, sigma = 20), patience = 5),
    "early stopping"
  )
})

test_that("a global variable named n does not leak into the fit", {
  # Regression test: boost_gaussian() must compute n from its own arguments,
  # not read a same-named object from the calling environment.
  old <- if (exists("n", envir = .GlobalEnv, inherits = FALSE)) get("n", envir = .GlobalEnv) else NULL
  assign("n", 1L, envir = .GlobalEnv)
  on.exit({
    if (is.null(old)) rm("n", envir = .GlobalEnv) else assign("n", old, envir = .GlobalEnv)
  }, add = TRUE)

  dat <- make_boost_data(n = 20, pX = 2, pZ = 2)
  expect_no_error(boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 5))
})
