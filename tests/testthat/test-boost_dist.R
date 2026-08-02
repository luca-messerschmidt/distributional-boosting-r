# tests/testthat/test-boost_dist.R
#
# boost_gamma()/boost_poisson()/boost_binomial(): fitting, validation, the
# formula interface, and the shared predict/print/summary/plot.boost_dist
# methods.

.dist_cases <- function(n = 200) {
  X <- data.frame(x1 = rnorm(n))
  Z <- data.frame(z1 = rnorm(n))
  y_gamma <- rgamma(n, shape = exp(1 + 0.3 * Z$z1), rate = exp(1 + 0.3 * Z$z1) / exp(1 + 0.5 * X$x1))
  y_pois  <- rpois(n, lambda = exp(0.5 + 0.4 * X$x1))
  y_bin   <- rbinom(n, 1, plogis(0.3 + 0.8 * X$x1))

  list(
    gamma    = list(fitter = function(...) boost_gamma(X, Z, y_gamma, ...), y = y_gamma),
    poisson  = list(fitter = function(...) boost_poisson(X, y_pois, ...), y = y_pois),
    binomial = list(fitter = function(...) boost_binomial(X, y_bin, ...), y = y_bin)
  )
}

test_that("each new family fits, decreases risk, and returns in-domain fitted values", {
  set.seed(1)
  cases <- .dist_cases()

  for (nm in names(cases)) {
    y <- cases[[nm]]$y
    fit <- cases[[nm]]$fitter(mstop = 80)

    expect_s3_class(fit, "boost_dist")
    expect_true(all(diff(fit$risk) <= 1e-8), info = nm)
    expect_lt(tail(fit$risk, 1), fit$risk0)
    expect_true(all(is.finite(fit$fitted$mu)), info = nm)
  }
})

test_that("each new family rejects out-of-domain responses", {
  set.seed(2)
  X <- data.frame(x1 = rnorm(30))
  expect_error(boost_gamma(X, X, y = c(-1, abs(rnorm(29)) + 0.1), mstop = 5), "strictly positive")
  expect_error(boost_poisson(X, y = c(-1, rpois(29, 3)), mstop = 5), "non-negative")
  expect_error(boost_binomial(X, y = c(2, rbinom(29, 1, 0.5)), mstop = 5), "binary")
})

test_that("formula interface and learner = 'auto' work for the new families", {
  set.seed(3)
  n <- 100
  x1 <- rnorm(n)
  df_pois <- data.frame(x1 = x1, y = rpois(n, exp(0.5 + 0.4 * x1)))
  fit_f <- boost_poisson(formula = y ~ x1, data = df_pois, mstop = 30)
  expect_s3_class(fit_f, "boost_poisson")

  fit_auto <- boost_poisson(data.frame(x1 = x1), df_pois$y, mstop = 40, learner = "auto")
  expect_true(all(diff(fit_auto$risk) <= 1e-8))
})

test_that("internal early stopping (patience) works for the new families", {
  set.seed(4)
  X <- data.frame(x1 = rnorm(150))
  y <- rpois(150, lambda = exp(0.3 + 0.5 * X$x1))
  fit <- boost_poisson(X, y, mstop = 100, patience = 5, seed = 1)
  expect_true(isTRUE(fit$early_stopping$used))
})

test_that("predict.boost_dist reconstructs fitted values in- and out-of-sample", {
  set.seed(5)
  n <- 150
  X <- data.frame(x1 = rnorm(n))
  Z <- data.frame(z1 = rnorm(n))
  y <- rgamma(n, shape = exp(1 + 0.3 * Z$z1), rate = exp(1 + 0.3 * Z$z1) / exp(1 + 0.5 * X$x1))
  fit <- boost_gamma(X, Z, y, mstop = 60)

  expect_equal(predict(fit, what = "mu"), fit$fitted$mu, tolerance = 0)
  all_df <- predict(fit, what = "all")
  expect_named(all_df, c("mu", "shape"))

  pred_oos <- predict(fit, newdata_X = X, newdata_Z = Z, what = "mu")
  expect_equal(pred_oos, fit$fitted$mu, tolerance = 1e-8)

  pred_partial <- predict(fit, mstop = 20, what = "mu")
  expect_length(pred_partial, n)
  expect_true(all(pred_partial > 0))

  fit_pois <- boost_poisson(X, rpois(n, 3), mstop = 20)
  expect_error(predict(fit_pois, newdata_Z = Z), "single-parameter family")
})

test_that("print/summary/plot.boost_dist run without error", {
  set.seed(6)
  X <- data.frame(x1 = rnorm(100))
  y <- rpois(100, lambda = exp(0.3 + 0.4 * X$x1))
  fit <- boost_poisson(X, y, mstop = 40)

  expect_output(print(fit), "Distributional Poisson Gradient Boosting")
  expect_output(print(summary(fit)), "Model Summary")

  pdf(NULL); on.exit(dev.off(), add = TRUE)
  expect_no_error(plot(fit, type = "path"))
  expect_no_error(plot(fit, type = "frequency"))
  expect_no_error(plot(fit, type = "risk"))
  expect_no_error(plot(fit, type = "partial", variable = "x1"))
})
