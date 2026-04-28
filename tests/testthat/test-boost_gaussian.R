test_that("boost_gaussian returns correct object", {
  set.seed(123)
  
  x <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50)
  )
  
  y <- 1 + 2 * x$x1 + rnorm(50)
  
  fit <- boost_gaussian(x, y, mstop = 10, nu = 0.1)
  
  expect_s3_class(fit, "boost_gaussian")
  expect_equal(length(fit$fitted_values), length(y))
  expect_equal(length(fit$selected_variables), 10)
  expect_equal(length(fit$coefficients), 10)
})

test_that("predict returns numeric vector", {
  set.seed(123)
  
  x <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50)
  )
  
  y <- 1 + 2 * x$x1 + rnorm(50)
  
  fit <- boost_gaussian(x, y, mstop = 10, nu = 0.1)
  
  pred <- predict(fit, x)
  
  expect_type(pred, "double")
  expect_equal(length(pred), nrow(x))
})

test_that("boost_gaussian stops for invalid input", {
  x <- data.frame(x1 = rnorm(10))
  y <- rnorm(9)
  
  expect_error(
    boost_gaussian(x, y),
    "Number of rows in x must match length of y."
  )
  
  expect_error(
    boost_gaussian(x, rnorm(10), mstop = 0),
    "mstop must be at least 1."
  )
  
  expect_error(
    boost_gaussian(x, rnorm(10), nu = 2),
    "nu must be between 0 and 1."
  )
})