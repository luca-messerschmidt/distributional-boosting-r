# tests/testthat/test-boost_gaussian-methods.R
#
# predict()/print()/summary()/plot() for boost_gaussian().

test_that("predict returns in-sample fitted values for mu, sigma, and both", {
  obj <- make_boost_fit(n = 60, pX = 3, pZ = 3, mstop = 80)
  fit <- obj$fit

  expect_equal(predict(fit), as.numeric(fit$fitted_mu), tolerance = 1e-10)
  expect_equal(predict(fit, what = "sigma"), as.numeric(fit$fitted_sigma), tolerance = 1e-10)

  both <- predict(fit, what = "both")
  expect_named(both, c("mu", "sigma"))
  expect_equal(both$mu, as.numeric(fit$fitted_mu), tolerance = 1e-10)
})

test_that("predict reconstructs partial mstop the same way in- and out-of-sample", {
  obj <- make_boost_fit(n = 60, pX = 3, pZ = 3, mstop = 80)
  fit <- obj$fit

  expect_equal(predict(fit, mstop = 0, what = "mu"), rep(fit$initial_mu, obj$n), tolerance = 1e-10)
  expect_equal(predict(fit, mstop = fit$mstop, what = "both")$mu, as.numeric(fit$fitted_mu),
               tolerance = 1e-10)

  pred_in  <- predict(fit, mstop = 30, what = "both")
  pred_out <- predict(fit, newdata_X = obj$X_df, newdata_Z = obj$Z_df, mstop = 30, what = "both")
  expect_equal(pred_in, pred_out, tolerance = 1e-10)

  # monotone approach toward the full-model fit
  d10 <- mean(abs(predict(fit, mstop = 10) - fit$fitted_mu))
  d80 <- mean(abs(predict(fit, mstop = 80) - fit$fitted_mu))
  expect_gt(d10, d80)
})

test_that("out-of-sample predict tolerates column order and extra columns", {
  obj <- make_boost_fit(n = 60, pX = 3, pZ = 3, mstop = 80)
  fit <- obj$fit

  reordered <- predict(fit, newdata_X = obj$X_df[1:10, c(3, 1, 2)],
                       newdata_Z = obj$Z_df[1:10, ], what = "mu")
  extra_col <- obj$X_df[1:10, ]; extra_col$junk <- 1
  with_extra <- predict(fit, newdata_X = extra_col, newdata_Z = obj$Z_df[1:10, ], what = "mu")
  baseline <- predict(fit, newdata_X = obj$X_df[1:10, ], newdata_Z = obj$Z_df[1:10, ], what = "mu")

  expect_equal(reordered, baseline, tolerance = 1e-10)
  expect_equal(with_extra, baseline, tolerance = 1e-10)
})

test_that("predict validates its arguments", {
  obj <- make_boost_fit(n = 60, pX = 3, pZ = 3, mstop = 80)
  fit <- obj$fit

  expect_error(predict(fit, newdata_X = obj$X_df[1:3, ]), "together")
  expect_error(predict(fit, newdata_X = obj$X_df[1:5, ], newdata_Z = obj$Z_df[1:3, ]),
               "same number of rows")
  expect_error(predict(fit, mstop = fit$mstop + 1), "exceeds")
  expect_error(predict(fit, newdata_X = obj$X_df[1:3, "x1", drop = FALSE],
                       newdata_Z = obj$Z_df[1:3, ]), "missing columns")

  x_na <- obj$X_df[1:3, ]; x_na[1, 1] <- NA
  expect_error(predict(fit, newdata_X = x_na, newdata_Z = obj$Z_df[1:3, ]), "missing")
})

test_that("print prints the key sections and returns the object invisibly", {
  obj <- make_boost_fit(n = 45, pX = 3, pZ = 3, mstop = 25)
  out <- capture.output(ret <- print(obj$fit))

  expect_identical(ret, obj$fit)
  expect_true(any(grepl("Distributional Gaussian Gradient Boosting", out)))
  expect_true(any(grepl("Location predictors", out)))
  expect_true(any(grepl("Scale predictors", out)))
  expect_true(any(grepl("RSS", out)))
})

test_that("print truncates long selection lists and handles NA R-squared", {
  obj <- make_boost_fit(n = 120, pX = 20, pZ = 3, mstop = 20)
  obj$fit$selected_mu <- paste0("x", seq_len(20))
  out <- capture.output(print(obj$fit))
  expect_true(any(grepl("Top 10 selected variables", out)))

  fit_const <- boost_gaussian(obj$X_df, obj$Z_df, rep(3, obj$n), mstop = 5)
  out_const <- capture.output(print(fit_const))
  expect_true(any(grepl("R-squared", out_const) & grepl("NA", out_const)))
})

test_that("summary is consistent with the underlying fit", {
  obj <- make_boost_fit(n = 70, pX = 4, pZ = 3, mstop = 60)
  fit <- obj$fit
  s <- summary(fit)

  expect_s3_class(s, "summary.boost_gaussian")
  expect_equal(s$net_coef_mu_all, fit$net_coef_mu_orig, tolerance = 1e-10)
  expect_equal(s$intercept_mu, fit$intercept_mu, tolerance = 1e-10)
  expect_equal(s$freq_mu, sort(table(fit$selected_mu), decreasing = TRUE))
  expect_equal(sort(s$mu_zero_vars), sort(fit$X_names[fit$net_coef_mu_orig == 0]))
  expect_equal(s$res_summary$rss, sum(fit$residuals^2), tolerance = 1e-10)

  out <- capture.output(print(s))
  expect_true(any(grepl("Model Summary", out)))
  expect_true(any(grepl("Location submodel", out)))
  expect_true(any(grepl("Scale submodel", out)))
})

test_that("plot runs for every type/submodel combination and rejects bad arguments", {
  obj <- make_boost_fit(n = 50, pX = 4, pZ = 3, mstop = 40)
  fit <- obj$fit

  pdf(NULL); on.exit(dev.off(), add = TRUE)
  ret <- plot(fit, type = "path", submodel = "mu")
  expect_identical(ret, fit)
  expect_no_error(plot(fit, type = "frequency", submodel = "sigma"))
  expect_no_error(plot(fit, type = "risk"))
  expect_no_error(plot(fit, type = "path", submodel = "mu", scale = "standardized"))

  expect_error(plot(fit, type = "bad_type"), "arg")
  expect_error(plot(fit, top_n = -1), "top_n")
})
