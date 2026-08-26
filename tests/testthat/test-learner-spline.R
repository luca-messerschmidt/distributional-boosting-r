# tests/testthat/test-learner-spline.R
#
# The P-spline base learner and its integration into boost_gaussian()
# (learner = "spline"/"auto"): df-equalization, exact basis reconstruction
# in predict(), and the smooth-terms/partial-effect reporting that replaces
# a single scalar coefficient when a variable is fit by a spline.

test_that("df-equalization hits the target effective df", {
  set.seed(1)
  x <- rnorm(150)
  basis <- .make_bspline_basis(x)
  K   <- .difference_penalty(ncol(basis$B))
  BtB <- crossprod(basis$B)

  for (target in c(3, 4, 6)) {
    lambda <- .find_lambda_for_edf(BtB, K, target_df = target)
    expect_equal(.pspline_edf(BtB, K, lambda), target, tolerance = 0.05)
  }
})

test_that("learner = 'auto' matches linear on linear truth and beats it on nonlinear truth", {
  set.seed(2)
  n <- 250
  X <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  Z <- data.frame(z1 = rnorm(n))

  y_lin <- 2 + 3 * X$x1 - 1.5 * X$x2 + exp(0.3 * Z$z1) * rnorm(n)
  fit_lin_linear <- boost_gaussian(X, Z, y_lin, mstop = 100, learner = "linear")
  fit_lin_auto   <- boost_gaussian(X, Z, y_lin, mstop = 100, learner = "auto")
  expect_true(tail(fit_lin_auto$risk, 1) <= tail(fit_lin_linear$risk, 1) * 1.05)

  y_nl <- 2 + 3 * sin(2 * X$x1) - X$x2 + exp(0.3 * Z$z1) * rnorm(n)
  fit_nl_linear <- boost_gaussian(X, Z, y_nl, mstop = 100, learner = "linear")
  fit_nl_auto   <- boost_gaussian(X, Z, y_nl, mstop = 100, learner = "auto")
  expect_true(tail(fit_nl_auto$risk, 1) <= tail(fit_nl_linear$risk, 1) * 0.95)
})

test_that("predict() exactly reconstructs spline steps in- and out-of-sample", {
  set.seed(3)
  n <- 200
  X <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  Z <- data.frame(z1 = rnorm(n))
  y <- 2 + 3 * sin(2 * X$x1) - X$x2 + exp(0.3 * Z$z1) * rnorm(n)
  fit <- boost_gaussian(X, Z, y, mstop = 60, learner = "spline")

  expect_true(is.list(fit$coef_mu))
  step <- Filter(function(s) s$type == "spline", fit$coef_mu)[[1]]
  x_new <- seq(step$boundary[1], step$boundary[2], length.out = 25)
  manual_B <- splines::bs(x_new, knots = step$knots, degree = step$degree,
                          Boundary.knots = step$boundary, intercept = TRUE)
  expect_equal(.evaluate_step(step, x_new), as.numeric(manual_B %*% step$coefs),
               tolerance = 1e-10)

  expect_equal(predict(fit, newdata_X = X, newdata_Z = Z, what = "mu"), fit$fitted_mu,
               tolerance = 1e-6)
})

test_that("boost_gaussian() rejects an invalid learner argument", {
  dat <- make_boost_data(n = 40)
  expect_error(boost_gaussian(dat$X_df, dat$Z_df, dat$y, mstop = 5, learner = "bad"),
               "should be one of")
})

test_that("smooth terms are reported additively and only for spline-fit variables", {
  set.seed(4)
  n <- 250
  X <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  Z <- data.frame(z1 = rnorm(n))
  y <- 2 + 3 * sin(2 * X$x1) - X$x2 + exp(0.3 * Z$z1) * rnorm(n)

  fit_linear <- boost_gaussian(X, Z, y, mstop = 60, learner = "linear")
  expect_null(fit_linear$smooth_terms_mu)
  expect_false(grepl("Smooth terms", paste(capture.output(print(fit_linear)), collapse = "\n")))

  fit_auto <- boost_gaussian(X, Z, y, mstop = 60, learner = "auto")
  s <- summary(fit_auto)
  expect_false(anyNA(names(s$net_coef_mu)))       # NA net coefs excluded, not left dangling
  expect_false(any(is.na(s$mu_zero_vars)))
  expect_false(is.null(fit_auto$smooth_terms_mu))
  expect_true(grepl("Smooth terms", paste(capture.output(print(fit_auto)), collapse = "\n")))
})

test_that("plot(type = 'partial') works for both spline and linear-only submodels", {
  set.seed(5)
  n <- 200
  X <- data.frame(x1 = rnorm(n))
  Z <- data.frame(z1 = rnorm(n))
  y <- 2 + 3 * sin(2 * X$x1) + exp(0.3 * Z$z1) * rnorm(n)
  fit <- boost_gaussian(X, Z, y, mstop = 60, learner = "spline")

  pdf(NULL); on.exit(dev.off(), add = TRUE)
  expect_no_error(plot(fit, type = "partial", submodel = "mu", variable = "x1"))
  expect_error(plot(fit, type = "partial", submodel = "mu"), "variable must be supplied")
})
