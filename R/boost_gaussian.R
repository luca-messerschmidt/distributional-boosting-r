#' Gaussian Gradient Boosting
#'
#' Fits a simple component-wise gradient boosting model for Gaussian responses.
#'
#' @param x A data frame or matrix of predictors.
#' @param y A numeric response vector.
#' @param mstop Number of boosting iterations.
#' @param nu Learning rate.
#'
#' @returns A list containing the fitted model.
#' @importFrom stats lm fitted coef
#' @export
#'
#' @examples
#' set.seed(123)
#' x <- data.frame(
#'   x1 = rnorm(100),
#'   x2 = rnorm(100)
#' )
#' y <- 2 + 3 * x$x1 + rnorm(100)
#'
#' fit <- boost_gaussian(x, y, mstop = 100, nu = 0.1)
#' predict(fit, x[1:5, ])

boost_gaussian <- function(x, y, mstop = 100, nu = 0.1) {

  x <- as.data.frame(x)
  y <- as.numeric(y)

  if (nrow(x) != length(y)) {
    stop("Number of rows in x must match length of y.")
  }

  if (mstop < 1) {
    stop("mstop must be at least 1.")
  }

  if (nu <= 0 || nu > 1) {
    stop("nu must be between 0 and 1.")
  }

  n <- length(y)
  p <- ncol(x)

  initial_value <- mean(y)
  fitted_values <- rep(initial_value, n)

  selected_variables <- character(mstop)
  coefficients <- numeric(mstop)

  for (m in seq_len(mstop)) {

    residuals <- y - fitted_values

    best_variable <- NULL
    best_coefficient <- NULL
    best_rss <- Inf
    best_prediction <- NULL

    for (j in seq_len(p)) {

      xj <- x[[j]]

      model <- lm(residuals ~ xj)
      prediction <- fitted(model)

      rss <- sum((residuals - prediction)^2)

      if (rss < best_rss) {
        best_rss <- rss
        best_variable <- names(x)[j]
        best_coefficient <- coef(model)[2]
        best_prediction <- prediction
      }
    }

    fitted_values <- fitted_values + nu * best_prediction

    selected_variables[m] <- best_variable
    coefficients[m] <- nu * best_coefficient
  }

  result <- list(
    initial_value = initial_value,
    selected_variables = selected_variables,
    coefficients = coefficients,
    fitted_values = fitted_values,
    mstop = mstop,
    nu = nu,
    predictor_names = names(x)
  )

  class(result) <- "boost_gaussian"

  return(result)
}