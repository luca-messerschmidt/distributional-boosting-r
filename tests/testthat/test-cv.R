test_that("correct output structure",{
  set.seed(42)
  x <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
  y <- 1 + 2 * x$x1 + rnorm(50)

  res <- cv_boost_gaussian(x, y, k = 3, mstop = 10, nu = 0.1)
  expect_s3_class(res, "cv_boost_gaussian")
  expect_equal(nrow(res$cv_error_matrix), 3)
  expect_equal(ncol(res$cv_error_matrix), 10)
  expect_true(res$optimal_stop >= 1 && res$optimal_stop <= 10)
  }
)

test_that("predict.cv_boost_gaussian",{
  x <- data.frame(x1 = rnorm(20))
  y <- 5 * x$x1 + rnorm(20)

  res <- cv_boost_gaussian(x, y, k = 3, mstop = 5)

  preds <- predict(res, newdata = x[1:5, , drop = F])
  expect_length(preds, 5)
  expect_type(preds, "double")
})

test_that("summary.cv_boost_gaussian", {
  x <- data.frame(x1 = rnorm(20))
  y <- 2 * x$x1 + rnorm(20)
  res <- cv_boost_gaussian(x, y, k = 2, mstop = 5)
  sum_obj <- summary(res)

  expect_s3_class(sum_obj, "summary.cv_boost_gaussian")
  expect_true(!is.null(sum_obj$residual_summary))
  expect_equal(sum_obj$optimal_stop, res$optimal_stop)
})
