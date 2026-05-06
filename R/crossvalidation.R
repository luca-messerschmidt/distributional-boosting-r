
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

cv_boost_gaussian <- function(x, y, k = 5, mstop = 100, nu = 0.1, seed = NULL){
  n <- length(y)

  # set a seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }

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

  final_model <- boost_gaussian(x, y, mstop = optimal_stop, nu = nu)

  final_residuals <- y - final_model$fitted_values

  result <- list(
    optimal_stop = optimal_stop,
    mean_error_cv = mean_error_cv,
    cv_error_matrix = cv_errors,
    mstop_max = mstop,
    call = match.call(),
    model = final_model,
    residuals = final_residuals
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
         legend = c("CV Error",paste("Optimal mstop:", x$optimal_stop)),
         col = c("blue", "red"),
         lty = 1:2)
}


#' print-method for CV Function "cv_boost_gaussian"
#'
#' @param x object of class "cv_boost_gaussian"
#' @param ... other parameters for the print() function
#'
#' @export
print.cv_boost_gaussian <- function(x, ...){
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"),
      "\n\n", sep = "")

  cat("Optimal mstop:", x$optimal_stop, "\n")
  cat("Selected variables in new model:\n")
  print(unique(x$model$selected_variables))
  invisible(x)
}

#' summary-method for CV Function "cv_boost_gaussian"
#'
#' @param object of class "cv_boost_gaussian"
#' @param ... other parameters for the summary() function
#'
#' @returns list of values for model summary
#' @export
summary.cv_boost_gaussian <- function(object, ...){

  res <- list(
    call = object$call,
    optimal_stop = object$optimal_stop,
    coefficients = object$model$coefficients,
    selected_variables = object$model$selected_variables,
    residual_summary = summary(object$residuals)
  )

  class(res) <- "summary.cv_boost_gaussian"
  return(res)
}

#' extra print-method for summary-method of "cv_boost_gaussian"
#'
#' @param x object of class "summary.cv_boost_gaussian"
#' @param ... other parameters passed
#'
#' @export
print.summary.cv_boost_gaussian <- function(x, ...){
  cat("Model Summary:\n")
  cat("Residuals:\n")
  print(x$residual_summary)
  cat("\nCoefficients:\n")
  print(x$coefficients)
}

#' predict-method for CV Function "cv_boost_gaussian"
#'
#' @param object object of class "cv_boost_gaussian"
#' @param newdata new data for the prediction
#' @param ... further parameters for the prediction
#'
#' @importFrom stats predict
#' @returns numeric prediction-vector
#' @export
predict.cv_boost_gaussian <- function(object, newdata, ...){
  if(missing(newdata)){
    return(object$model$fitted_values)
  }

  return(predict(object$model, newdata = newdata, ...))
}

