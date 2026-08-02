test_that("correct output structure",{
  set.seed(42)
  X <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
  Z <- data.frame(z1 = rnorm(50))
  y <- 1 + 2 * X$x1 + exp(0.3 * Z$z1) * rnorm(50)

  res <- cv_boost_gaussian(X, Z, y, k = 3, mstop = 10, nu_mu = 0.1,
                           nu_sigma = 0.1)
  expect_s3_class(res, "cv_boost_gaussian")
  expect_equal(nrow(res$cv_error_matrix), 3)
  expect_equal(ncol(res$cv_error_matrix), 10)
  expect_true(res$optimal_stop >= 1 && res$optimal_stop <= 10)
  }
)

test_that("predict.cv_boost_gaussian",{
  X <- data.frame(x1 = rnorm(20))
  Z <- data.frame(z1 = rnorm(20))
  y <- 5 * X$x1 + exp(0.3 * Z$z1) * rnorm(20)

  res <- cv_boost_gaussian(X, Z, y, k = 3, mstop = 5)

  preds <- predict(res, newdata_X = X[1:5, , drop = F],
                   newdata_Z = Z[1:5, ,drop = F])
  expect_length(preds, 5)
  expect_type(preds, "double")
})

test_that("summary.cv_boost_gaussian", {
  X <- data.frame(x1 = rnorm(20))
  Z <- data.frame(z1 = rnorm(20))
  y <- 2 * X$x1 + exp(0.3 * Z$z1) * rnorm(20)

  res <- cv_boost_gaussian(X, Z, y, k = 2, mstop = 5)
  sum_obj <- summary(res)

  expect_s3_class(sum_obj, "summary.cv_boost_gaussian")
  expect_true(!is.null(sum_obj$residual_summary))
  expect_equal(sum_obj$optimal_stop, res$optimal_stop)
})

test_that("reproducibility", {
  X <- data.frame(x1 = rnorm(20))
  Z <- data.frame(z1 = rnorm(20))
  y <- 2 * X$x1 + exp(0.3 * Z$z1) * rnorm(20)

  res1 <- cv_boost_gaussian(X, Z, y, k = 2, mstop = 5, seed = 123)
  res2 <- cv_boost_gaussian(X, Z, y, k = 2, mstop = 5, seed = 123)

  expect_equal(res1$optimal_stop, res2$optimal_stop)
  expect_equal(res1$mean_error_cv, res2$mean_error_cv)
})

test_that("cv_boost_gaussian dimension mismatch", {
  X <- data.frame(x1 = rnorm(5))
  X_false <- data.frame(x1 = rnorm(10))
  Z <- data.frame(z1 = rnorm(5))
  Z_false <- data.frame(z1 = rnorm(10))
  y <- rnorm(5)

  expect_error(cv_boost_gaussian(X_false, Z, y),
               "Number of rows in X must match length of y")
  expect_error(cv_boost_gaussian(X, Z_false, y),
               "Number of rows in Z must match length of y")
})

test_that("cv_boost_gaussian invalid number of folds", {
  X <- data.frame(x1 = rnorm(10))
  Z <- data.frame(z1 = rnorm(10))
  y <- rnorm(10)

  expect_error(cv_boost_gaussian(X, Z, y, k = 1), "k must be greater than 1")
  expect_error(cv_boost_gaussian(X, Z, y, k = 11))
})

test_that("cv_boost_grid invalid inputs", {
  X <- data.frame(x1 = rnorm(20))
  Z <- data.frame(z1 = rnorm(20))
  y <- rnorm(20)

  expect_error(cv_boost_grid(X, Z, y, mstop_grid = numeric(0)))
  expect_error(cv_boost_grid(X, Z, y, nu_mu_grid = numeric(0)))
  expect_error(cv_boost_grid(X, Z, y, nu_sigma_grid = numeric(0)))
  expect_error(cv_boost_grid(X, Z, y, mstop_grid = c(-1, 10)))
  expect_error(cv_boost_grid(X, Z, y, nu_mu_grid = c(0, 0.5)))
  expect_error(cv_boost_grid(X, Z, y, nu_sigma_grid = c(0, 0.5)))
  expect_error(cv_boost_grid(X, Z, y, k = 1))
  expect_error(cv_boost_grid(data.frame(x1 = rnorm(10)),
                             data.frame(z1 = rnorm(50)), rnorm(5)))
})

test_that("cv_boost_grid risk-minimization test", {
  set.seed(1)
  X <- data.frame(x1 = rnorm(30))
  Z <- data.frame(z1 = rnorm(30))
  y <- 5 * X$x1 + exp(0.3 * Z$z1) * rnorm(30)

  grid_res <- cv_boost_grid(X, Z, y, k = 2, mstop_grid = c(5, 10),
                            nu_mu_grid = c(0.1, 0.5),
                            nu_sigma_grid = c(0.1, 0.5))

  expect_equal(grid_res$final_model$nu_mu, grid_res$best_combination$nu_mu)
  expect_equal(grid_res$final_model$nu_sigma, grid_res$best_combination$nu_sigma)
  expect_equal(grid_res$final_model$mstop, grid_res$best_combination$mstop)
})

test_that("cv_boost_grid reproducibility", {
  set.seed(1)
  X <- data.frame(x1 = rnorm(30))
  Z <- data.frame(z1 = rnorm(30))
  y <- 5 * X$x1 + exp(0.3 * Z$z1) * rnorm(30)

  res1 <- cv_boost_grid(X, Z, y, k = 2, mstop_grid = c(5, 10),
                        nu_mu_grid = c(0.1, 0.5), nu_sigma_grid = c(0.1, 0.5),
                        seed = 42)
  res2 <- cv_boost_grid(X, Z, y, k = 2, mstop_grid = c(5, 10),
                        nu_mu_grid = c(0.1, 0.5), nu_sigma_grid = c(0.1, 0.5),
                        seed = 42)

  expect_equal(res1$all_results$cv_error, res2$all_results$cv_error)
})

test_that("cv_boost_grid returns correct class", {
  set.seed(1)
  X <- data.frame(x1 = rnorm(30))
  Z <- data.frame(z1 = rnorm(30))
  y <- 5 * X$x1 + exp(0.3 * Z$z1) * rnorm(30)

  res <- cv_boost_grid(X, Z, y, k = 2, mstop_grid = c(5, 10),
                       nu_mu_grid = c(0.1, 0.5), nu_sigma_grid = c(0.1, 0.5))

  expect_s3_class(res, "cv_boost_grid")
})

test_that("cv_boost_gaussian invalid folds", {
  X <- data.frame(x1 = rnorm(10))
  Z <- data.frame(z1 = rnorm(10))
  y <- rnorm(10)

  expect_error(cv_boost_gaussian(X, Z, y, k = 2, folds = rep(1, 10)))
  expect_error(cv_boost_gaussian(X, Z, y, k = 2, folds = 1:9))
  expect_error(cv_boost_gaussian(X, Z, y, k = 2, folds = c(1:5, rep(3, 5))))
})

test_that("cv_boost_gaussian forwards learner to the fitted model", {
  set.seed(1)
  X <- data.frame(x1 = rnorm(60))
  Z <- data.frame(z1 = rnorm(60))
  y <- 2 + 3 * sin(2 * X$x1) + exp(0.3 * Z$z1) * rnorm(60)

  # Spline basis boundary knots are fit per-CV-fold, so held-out rows
  # occasionally fall outside them; splines::bs() warns (harmlessly) about
  # extrapolation in that case, which is expected here and not asserted on.
  res <- suppressWarnings(
    cv_boost_gaussian(X, Z, y, k = 2, mstop = 20, learner = "spline")
  )
  expect_equal(res$model$learner, "spline")
})

test_that("cv_boost_grid compares learners and records the winner", {
  set.seed(1)
  X <- data.frame(x1 = rnorm(60))
  Z <- data.frame(z1 = rnorm(60))
  y <- 2 + 3 * sin(2 * X$x1) + exp(0.3 * Z$z1) * rnorm(60)

  res <- suppressWarnings(
    cv_boost_grid(X, Z, y, k = 2, mstop_grid = 20,
                 learner_grid = c("linear", "auto"))
  )

  expect_true(all(c("linear", "auto") %in% res$all_results$learner))
  expect_equal(res$final_model$learner, res$best_combination$learner)
  expect_error(cv_boost_grid(X, Z, y, learner_grid = "bogus"))
})



