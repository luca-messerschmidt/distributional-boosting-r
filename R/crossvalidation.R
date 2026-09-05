#' k-fold Cross-validation for Hyperparameter Tuning for Boosting Algorithm
#'
#' @param X data matrix or data frame for location submodel
#' @param Z data matrix or data frame for scale submodel
#' @param y numeric response-vector
#' @param k number of folds
#' @param mstop limit to iterations
#' @param nu_mu learning rate for location submodel
#' @param nu_sigma learning rate for scale submodel
#' @param method boosting update rule ("cyclic", "noncyclic")
#' @param seed random seed for reproducibility
#' @param folds optional fold vector, if random generation is not wanted
#' @param patience_cv threshold for early stopping
#' @param learner base learner type forwarded to \code{boost_gaussian()}:
#'   "linear" (default), "spline", or "auto"
#'
#' @returns object of class "cv_boost_gaussian"
#' @export

cv_boost_gaussian <- function(X, Z, y, k = 5, mstop = 100, nu_mu = 0.1,
                              nu_sigma = 0.1, method = "cyclic", seed = NULL,
                              folds = NULL, patience_cv = NULL,
                              learner = "linear") {
  # check input dimensions
  n <- length(y)
  if (nrow(X) != n) stop("Number of rows in X must match length of y", call. = FALSE)
  if (nrow(Z) != n) stop("Number of rows in Z must match length of y", call. = FALSE)
  if (!is.numeric(k) || length(k) != 1L || k != as.integer(k)) {
    stop("k must be a single integer.", call. = FALSE)
  }
  if (k <= 1) stop("k must be greater than 1", call. = FALSE)
  if (k > n) stop("k must be smaller than sample size n", call. = FALSE)
  if (!is.numeric(mstop) || length(mstop) != 1L || mstop < 1 ||
      mstop != as.integer(mstop)) {
    stop("mstop must be a single positive integer, separate mu/sigma budgets are not supported in cross-validation",
         call. = FALSE)
  }
  if (!is.numeric(nu_mu) || length(nu_mu) != 1L || nu_mu <= 0 || nu_mu > 1) {
    stop("nu_mu must be a single value in (0, 1].", call. = FALSE)
  }
  if (!is.numeric(nu_sigma) || length(nu_sigma) != 1L || nu_sigma <= 0 ||
      nu_sigma > 1) {
    stop("nu_sigma must be a single value in (0, 1].", call. = FALSE)
  }
  if (!is.null(patience_cv) && (!is.numeric(patience_cv) ||
      length(patience_cv) != 1L || patience_cv < 1 ||
      patience_cv != as.integer(patience_cv))) {
    stop("patience_cv must be a single positive integer.", call. = FALSE)
  }

  if (is.null(folds)) {
    # set a seed if provided
    if (!is.null(seed)) set.seed(seed)
    # create random folds
    folds <- generate_folds(n, k)
  } else {
    if (length(folds) != n) {
      stop("folds must be same length as y", call. = FALSE)
    }
    if (!all(folds %in% seq_len(k))) {
      stop("folds must only contain integer values from 1 to k", call. = FALSE)
    }
    if (length(unique(folds)) != k) {
      stop("folds must contain all fold-indices from 1 to k", call. = FALSE)
    }
  }

  # create error matrix
  cv_errors <- matrix(NA_real_, nrow = k, ncol = mstop)

  for (i in 1:k) {
    # split into train and test
    train_X <- X[folds != i, , drop = FALSE]
    train_Z <- Z[folds != i, , drop = FALSE]
    train_y <- y[folds != i]
    test_X <- X[folds == i, , drop = FALSE]
    test_Z <- Z[folds == i, , drop = FALSE]
    test_y <- y[folds == i]

    fit <- boost_gaussian(train_X, train_Z, train_y, mstop = mstop,
                          nu_mu = nu_mu, nu_sigma = nu_sigma, method = method,
                          patience = NULL, learner = learner)

    # early-stopping closure variables
    best_error <- Inf
    no_improve <- 0

    for (m in 1:mstop) {
      prediction <- predict(fit, newdata_X = test_X, newdata_Z = test_Z,
                            mstop = m, what = "both")
      cv_errors[i, m] <- mean(
        log(prediction$sigma) + (test_y - prediction$mu)^2 /
          (2 * prediction$sigma^2)
        )

      # early stopping algorithm
      if (cv_errors[i, m] < best_error) {
        best_error <- cv_errors[i, m]
        no_improve <- 0
      }
      else {
        no_improve <- no_improve + 1
      }

      if (!is.null(patience_cv) && no_improve >= patience_cv) break
      }
  }

  complete_cols <- which(colSums(!is.na(cv_errors)) == k)
  mean_error_cv <- colMeans(cv_errors[, complete_cols, drop = FALSE],
                            na.rm = TRUE)
  optimal_stop <- complete_cols[which.min(mean_error_cv)]

  final_model <- boost_gaussian(X, Z, y, mstop = optimal_stop, nu_mu = nu_mu,
                                nu_sigma = nu_sigma, method = method,
                                patience = NULL, learner = learner)

  final_residuals <- y - final_model$fitted_mu

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
#' @export
plot.cv_boost_gaussian <- function(x, ...) {
  m <- seq_along(x$mean_error_cv)
  graphics::plot(m, x$mean_error_cv, type = "l", col = "blue", lwd = 2,
                 xlab = "Boosting Iterations",
                 ylab = "Negative Log-Likelihood",
                 main = "Hyperparameter Tuning of mstop")
  graphics::abline(v = x$optimal_stop, col = "red", lty = 2)
  graphics::legend("topright",
                   legend = c("CV Error", paste("Optimal mstop:", x$optimal_stop)),
                   col = c("blue", "red"),
                   lty = 1:2)
  invisible(x)
}


#' print-method for CV Function "cv_boost_gaussian"
#'
#' @param x object of class "cv_boost_gaussian"
#' @param ... other parameters for the print() function
#'
#' @export
print.cv_boost_gaussian <- function(x, ...) {
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"),
      "\n\n", sep = "")

  cat("Optimal mstop:", x$optimal_stop, "\n")
  cat("Selected variables (location):\n")
  print(unique(x$model$selected_mu))
  cat("Selected variables (scale):\n")
  print(unique(x$model$selected_sigma))
  invisible(x)
}

#' summary-method for CV Function "cv_boost_gaussian"
#'
#' @param object An object of class "cv_boost_gaussian"
#' @param ... other parameters for the summary() function
#'
#' @returns list of values for model summary
#' @export
summary.cv_boost_gaussian <- function(object, ...) {

  res <- list(
    call = object$call,
    optimal_stop = object$optimal_stop,
    coefficients_mu = object$model$net_coef_mu_orig,
    coefficients_sigma = object$model$net_coef_sigma_orig,
    selected_mu = object$model$selected_mu,
    selected_sigma = object$model$selected_sigma,
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
print.summary.cv_boost_gaussian <- function(x, ...) {
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"),
      "\n\n", sep = "")
  cat("Optimal mstop:", x$optimal_stop, "\n\n")
  cat("Residuals:\n")
  print(x$residual_summary)
  cat("\nCoefficients (location):\n")
  print(x$coefficients_mu)
  cat("\nCoefficients (scale):\n")
  print(x$coefficients_sigma)
  invisible(x)
}

#' predict-method for CV Function "cv_boost_gaussian"
#'
#' @param object object of class "cv_boost_gaussian"
#' @param newdata_X new location predictor data
#' @param newdata_Z new scale predictor data
#' @param ... further parameters for the prediction
#'
#' @importFrom stats predict
#' @returns numeric prediction-vector
#' @export
predict.cv_boost_gaussian <- function(object, newdata_X = NULL,
                                      newdata_Z = NULL, ...) {
  if (is.null(newdata_X) && is.null(newdata_Z)) {
    return(object$model$fitted_mu)
  }
  return(predict(object$model, newdata_X = newdata_X, newdata_Z = newdata_Z,
                 ...))
}



#' CV with Parameter Grid
#'
#' @param X data matrix or dataframe for the location submodel
#' @param Z data matrix or dataframe for the scale submodel
#' @param y numeric response vector
#' @param k number of folds
#' @param mstop_grid vector of mstop-values
#' @param nu_mu_grid vector of nu-values for the location submodel
#' @param nu_sigma_grid vector of nu-values for the scale submodel
#' @param method boosting update rule ("cyclic", "noncyclic")
#' @param learner_grid vector of base-learner types to compare, subset of
#'   c("linear", "spline", "auto")
#' @param seed optional seed
#'
#' @returns object of class \code{"cv_boost_grid"} containing:
#' \describe{
#'    \item{call}{the matched function call}
#'    \item{best_combination}{data frame row with the best mstop, nu, and learner values}
#'    \item{all_results}{data frame with CV errors for all parameter combinations}
#'    \item{final_model}{fitted \code{boost_gaussian} model using the best parameters}
#'}
#' @export
cv_boost_grid <- function(X, Z, y, k = 5, mstop_grid = 200,
                          nu_mu_grid = c(0.01, 0.1, 0.3),
                          nu_sigma_grid = c(0.01, 0.1, 0.3),
                          method = "cyclic",
                          learner_grid = "linear",
                          seed = NULL) {

  n <- length(y)
  if (nrow(X) != n) stop("Number of rows in X must match length of y", call. = FALSE)
  if (nrow(Z) != n) stop("Number of rows in Z must match length of y", call. = FALSE)
  if (!is.numeric(k) || length(k) != 1L || k != as.integer(k)) {
    stop("k must be a single integer.", call. = FALSE)
  }
  if (k<=1) stop("number of folds must be greater than 1", call. = FALSE)
  if (k > n) stop("number of folds must be smaller than sample size", call. = FALSE)

  if (!is.numeric(mstop_grid) || length(mstop_grid) == 0L ||
      any(mstop_grid < 1) || any(mstop_grid != as.integer(mstop_grid))) {
    stop("mstop_grid must be a non_empty vector of positive integers", call. = FALSE)
  }
  if (!is.numeric(nu_mu_grid) || any(nu_mu_grid > 1) || any(nu_mu_grid <= 0) ||
      length(nu_mu_grid) == 0) {
    stop("nu_mu_grid must be non-empty vector of values in (0,1]", call. = FALSE)
  }
  if (!is.numeric(nu_sigma_grid) || any(nu_sigma_grid > 1) ||
      any(nu_sigma_grid <= 0) || length(nu_sigma_grid) == 0) {
    stop("nu_sigma_grid must be non-empty vector of values in (0,1]", call. = FALSE)
  }
  if (!is.character(learner_grid) || length(learner_grid) == 0L ||
      !all(learner_grid %in% c("linear", "spline", "auto"))) {
    stop("learner_grid must be a non-empty vector with values in ",
         "c('linear', 'spline', 'auto')", call. = FALSE)
  }

  if (!is.null(seed)) set.seed(seed)
  folds <- generate_folds(n, k)

  results <- expand.grid(mstop = mstop_grid, nu_mu = nu_mu_grid,
                         nu_sigma = nu_sigma_grid, learner = learner_grid,
                         stringsAsFactors = FALSE)
  results$cv_error <- NA
  results$optimal_stop <- NA

  for (i in seq_len(nrow(results))) {
    cv_fit <- cv_boost_gaussian(X, Z, y, k = k,
                                mstop = results$mstop[i],
                                nu_mu = results$nu_mu[i],
                                nu_sigma = results$nu_sigma[i],
                                method = method,
                                learner = results$learner[i],
                                folds = folds)
    results$cv_error[i] <- min(cv_fit$mean_error_cv)
    results$optimal_stop[i] <- cv_fit$optimal_stop
  }

  best_combination <- which.min(results$cv_error)

  final_model <- boost_gaussian(X, Z, y,
                                mstop = results$optimal_stop[best_combination],
                                nu_mu = results$nu_mu[best_combination],
                                nu_sigma = results$nu_sigma[best_combination],
                                method = method,
                                learner = results$learner[best_combination],
                                patience = NULL)

  result <- list(call = match.call(),
                 best_combination = results[best_combination,],
                 all_results = results,
                 final_model = final_model)

  class(result) <- "cv_boost_grid"
  return(result)
}


#' print-method for CV Function "cv_boost_grid"
#'
#' @param x object of class "cv_boost_grid"
#' @param ... other parameters for the print() function
#'
#' @export
print.cv_boost_grid <- function(x, ...) {
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"),
      "\n\n", sep = "")

  cat("Best combination:\n")
  print(x$best_combination)
  invisible(x)
}

#' summary-method for CV Function "cv_boost_grid"
#'
#' @param object object of class "cv_boost_grid"
#' @param ... other parameters for the summary() function
#'
#' @returns list with sorted grid results and best combination
#' @export
summary.cv_boost_grid <- function(object, ...) {
  res <- list(
    call = object$call,
    best_combination = object$best_combination,
    sorted_results = object$all_results[order(object$all_results$cv_error),]
  )

  class(res) <- "summary.cv_boost_grid"
  return(res)
}

#' extra print-method of summary-method of "cv_boost_grid"
#'
#' @param x object of class "summary.cv_boost_grid"
#' @param ... other parameters passed
#'
#' @export
print.summary.cv_boost_grid <- function(x, ...) {
  cat("All combinations: \n")
  print(x$sorted_results)
  cat("\nBest combination:\n")
  print(x$best_combination)
  invisible(x)
}

#' predict-method for CV Function "cv_boost_grid"
#'
#' @param object object of class "cv_boost_grid"
#' @param newdata_X new location predictor data
#' @param newdata_Z new scale predictor data
#' @param ... further parameters for the prediction
#'
#' @importFrom stats predict
#' @returns numeric prediction-vector
#' @export
predict.cv_boost_grid <- function(object, newdata_X = NULL,
                                  newdata_Z = NULL, ...) {
  if (is.null(newdata_X) && is.null(newdata_Z)) {
    return(object$final_model$fitted_mu)
  }

  return(predict(object$final_model, newdata_X = newdata_X,
                 newdata_Z = newdata_Z, ...))
}



