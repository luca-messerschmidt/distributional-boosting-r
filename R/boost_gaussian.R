#' Fit a Gaussian Component-wise Boosting Model
#'
#' Fits a component-wise gradient boosting model for a numeric response using
#' squared-error loss and univariate linear base learners.
#'
#' @param x Numeric matrix or data frame of predictors. Columns must be numeric,
#'   finite, non-missing, non-constant, and uniquely named.
#' @param y Numeric response vector. Must have the same length as the number of
#'   rows in \code{x}, with finite and non-missing values.
#' @param mstop Single positive integer giving the number of boosting iterations.
#' @param nu Single numeric value in \eqn{(0, 1]} giving the learning rate.
#' @param keep_path Logical. If \code{TRUE}, stores the full in-sample
#'   fitted-value path. This enables early-stopped in-sample prediction but uses
#'   more memory.
#'
#' @details
#' The model is initialized with the mean of \code{y}. At each iteration, the
#' current residuals are regressed separately on each standardized predictor.
#' The predictor giving the largest squared-error improvement is selected, and
#' the fitted values are updated by a shrunken step of size \code{nu}.
#'
#' Predictors are centered and scaled before fitting. Coefficients are therefore
#' accumulated internally on the standardized scale and converted back to the
#' original predictor scale in the fitted object.
#'
#' @returns An object of class \code{"boost_gaussian"}, a list containing fitted
#'   values, residuals, selected variables, coefficient updates, coefficient
#'   summaries on the standardized and original scales, residual-sum-of-squares
#'   path, training \eqn{R^2}, preprocessing information, and the matched call.
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' x <- data.frame(
#'   x1 = rnorm(100),
#'   x2 = rnorm(100),
#'   x3 = rnorm(100)
#' )
#' y <- 2 + 3 * x$x1 - 1.5 * x$x3 + rnorm(100)
#'
#' fit <- boost_gaussian(x, y, mstop = 100, nu = 0.1)
#' fit
#' summary(fit)
#' predict(fit, newdata = x[1:5, ])

boost_gaussian <- function(x, y, mstop = 100, nu = 0.1, keep_path = TRUE) {
  # Basic input checks
  if (!is.matrix(x) && !is.data.frame(x)) {
    stop("x must be a matrix or data.frame.", call. = FALSE)
  }
  
  if (!is.numeric(y) || length(y) == 0L) {
    stop("y must be a non-empty numeric vector.", call. = FALSE)
  }
  if (!is.numeric(mstop) || length(mstop) != 1L || is.na(mstop) ||
      !is.finite(mstop) || mstop < 1L || mstop != as.integer(mstop)) {
    stop("mstop must be a single positive integer.", call. = FALSE)
  }
  if (!is.numeric(nu) || length(nu) != 1L || is.na(nu) ||
      !is.finite(nu) || nu <= 0 || nu > 1) {
    stop("nu must be a single numeric value in (0, 1].", call. = FALSE)
  }
  if (!is.logical(keep_path) || length(keep_path) != 1L || is.na(keep_path)) {
    stop("keep_path must be a single logical value (TRUE or FALSE).", call. = FALSE)
  }
  
  mstop <- as.integer(mstop)
  y <- as.numeric(y)
  
  # Prepare and standardize predictors
  x_df <- as.data.frame(x, stringsAsFactors = FALSE)
  
  if (nrow(x_df) != length(y)) {
    stop("Number of rows in x must match length of y.", call. = FALSE)
  }
  if (ncol(x_df) < 1L) {
    stop("x must contain at least one predictor.", call. = FALSE)
  }
  if (!all(vapply(x_df, is.numeric, logical(1)))) {
    stop("All predictors in x must be numeric.", call. = FALSE)
  }
  if (anyNA(x_df) || anyNA(y)) {
    stop("x and y must not contain missing values.", call. = FALSE)
  }
  
  x_mat <- as.matrix(x_df)
  storage.mode(x_mat) <- "double"
  
  if (ncol(x_mat) < 1L) {
    stop("x must contain at least one predictor.", call. = FALSE)
  }
  
  if (!all(is.finite(x_mat)) || !all(is.finite(y))) {
    stop("x and y must contain only finite values.", call. = FALSE)
  }
  
  predictor_names <- colnames(x_mat)
  if (is.null(predictor_names)) {
    predictor_names <- paste0("X", seq_len(ncol(x_mat)))
    colnames(x_mat) <- predictor_names
  }
  if (anyNA(predictor_names) || any(predictor_names == "")) {
    predictor_names[predictor_names == "" | is.na(predictor_names)] <-
      paste0("X", which(predictor_names == "" | is.na(predictor_names)))
    colnames(x_mat) <- predictor_names
  }
  if (anyDuplicated(predictor_names)) {
    stop("Predictor names must be unique.", call. = FALSE)
  }
  
  n <- length(y)
  p <- ncol(x_mat)
  
  if (n < 2L) {
    stop("At least two observations are required.", call. = FALSE)
  }
  
  x_center <- colMeans(x_mat)
  x_centered <- sweep(x_mat, 2L, x_center, FUN = "-")
  x_scale <- sqrt(colSums(x_centered^2) / (n - 1L))
  
  if (any(!is.finite(x_scale)) || any(x_scale <= 0)) {
    stop("One or more predictors have zero variance.", call. = FALSE)
  }
  
  x_std <- sweep(x_centered, 2L, x_scale, FUN = "/")
  xj_sum_sq <- colSums(x_std^2)   # = n - 1 for all j
  
  initial_value <- mean(y)
  fitted_values <- rep(initial_value, n)
  
  # Main boosting loop
  selected_variables <- character(mstop)
  coefficients       <- numeric(mstop)
  rss_path           <- numeric(mstop)
  fitted_path        <- if (keep_path) matrix(NA_real_, nrow = n, ncol = mstop) else NULL
  
  for (m in seq_len(mstop)) {
    residuals <- y - fitted_values
    inner_products   <- as.vector(crossprod(x_std, residuals))
    scaled_criterion <- (inner_products^2) / xj_sum_sq
    best_j           <- which.max(scaled_criterion)
    
    beta_j   <- inner_products[best_j] / xj_sum_sq[best_j]
    update_j <- nu * beta_j
    
    fitted_values <- fitted_values + update_j * x_std[, best_j]
    
    selected_variables[m] <- predictor_names[best_j]
    coefficients[m]        <- update_j
    if (keep_path) fitted_path[, m] <- fitted_values
    rss_path[m] <- sum((y - fitted_values)^2)
  }
  
  residuals <- y - fitted_values
  
  net_coef_standardized <- tapply(coefficients, selected_variables, sum)
  net_coef_standardized <- net_coef_standardized[
    predictor_names[predictor_names %in% names(net_coef_standardized)]
  ]
  net_coef_original <- numeric(p)
  names(net_coef_original) <- predictor_names
  net_coef_original[names(net_coef_standardized)] <-
    net_coef_standardized / x_scale[names(net_coef_standardized)]
  intercept_original <- initial_value - sum(net_coef_original * x_center)
  
  tss   <- sum((y - mean(y))^2)
  r_squared <- if (isTRUE(all.equal(tss, 0))) NA_real_ else 1 - sum(residuals^2) / tss
  
  result <- list(
    initial_value         = initial_value,
    intercept             = unname(intercept_original),
    selected_variables    = selected_variables,
    coefficients          = coefficients,
    fitted_values         = fitted_values,
    fitted_path           = fitted_path,
    residuals             = residuals,
    rss_path              = rss_path,
    r_squared             = r_squared,
    mstop                 = mstop,
    nu                    = nu,
    keep_path             = keep_path,
    predictor_names       = predictor_names,
    x_center              = x_center,
    x_scale               = x_scale,
    xj_sum_sq             = xj_sum_sq,
    net_coef_standardized = net_coef_standardized,
    net_coef_original     = net_coef_original,
    n                     = n,
    p                     = p,
    call                  = match.call()
  )
  
  class(result) <- "boost_gaussian"
  result
}
