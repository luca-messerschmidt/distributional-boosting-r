
#' k-fold Cross-validation for Hyperparameter Tuning for Boosting Algorithm
#'
#' @param x Datamatrix or Dataframe
#' @param y numeric response-vector
#' @param k number of folds
#' @param mstop limit to iterations
#' @param nu learning rate for boosting
#'
#' @returns object of class "cv_boost_gaussian"
#' @export

cv_boost_gaussian <- function(x, y, k = 5, mstop = 100, nu = 0.1){
  n <- length(y)

  # set a seed for reproducibility
  set.seed(42)

  # create random folds
  folds <- sample(rep(1:k, length.out = n))

  # create error matrix
  cv_errors <- matrix(0, nrow = k, ncol = mstop)

  for (i in 1:k) {
    # split into train and test
    train_x <- x[folds != i, drop = F]
    train_y <- y[folds != i]
    test_x <- x[folds == i, drop = F]
    test_y <- y[folds == i]

    fit <- boost_gaussian(train_x, train_y, mstop = mstop, nu = nu)

    prediction <- rep(fit$initial_value, nrow(test_x))

    for (m in 1:mstop){
      var_m <- fit$selected_variables[m]
      coef_m <- fit$coefficients[m]

      prediction <- prediction + coef_m *test_x[[var_m]]
      # Compute and store the MSE
      cv_errors[i, m] <- mean((test_y - prediction)^2)
      }
  }
  mean_error_cv <- colMeans(cv_errors)
  optimal_stop <- which.min(mean_error_cv)

  result <- list(
    optimal_stop = optimal_stop,
    mean_error_cv = mean_error_cv,
    cv_error_matrix = cv_errors,
    mstop_max = mstop,
    nu = nu
  )

  class(result) <- "cv_boost_gaussian"
  return(result)
}


#' Plot-method for CV Function "cv_boost_gaussian"
#' @param x object of class "cv_boost_gaussian"
#' @param ... other parameters for the plot() function
#'
#' @importFrom graphics abline legend
#'
#' @export
#'
plot.cv_boost_gaussian <- function(x, ...){
  m <- 1:x$mstop_max
  plot(m, x$mean_error_cv, type = "l", col = "blue", lwd = 2,
       xlab = "Boosting Iterations",
       ylab = "Mean Squared Error",
       main = "Hyperparameter Tuning of mstop")
  abline(v = x$optimal_stop, col = "red", lty = 2)
  legend("topright",
         legend = (c("CV Error",paste("Optimal mstop:", x$optimal_stop))),
         col = c("blue", "red"),
         lty = 1:2)
}





