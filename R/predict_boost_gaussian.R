#' Predict from a Gaussian Boosting Model
#'
#' Computes fitted or predicted values from a \code{boost_gaussian} object.
#'
#' @param object A fitted object of class \code{"boost_gaussian"}.
#' @param newdata Optional numeric matrix or data frame of predictor values. If
#'   omitted, in-sample fitted values are returned. If supplied, \code{newdata}
#'   must contain all predictors used during fitting; columns are matched by
#'   name and additional columns are ignored.
#' @param mstop Optional non-negative integer giving the number of boosting
#'   iterations to use. If omitted, the full fitted model is used.
#'   \code{mstop = 0} returns the intercept-only prediction.
#' @param ... Further arguments, currently ignored.
#'
#' @details
#' Early-stopped in-sample prediction requires the model to have been fitted with
#' \code{keep_path = TRUE}. Prediction for \code{newdata} is reconstructed from
#' the stored coefficient updates and does not require the fitted path.
#'
#' @returns A numeric vector of predictions.
#'
#' @export
#' @method predict boost_gaussian
#'
#' @examples
#' set.seed(123)
#' x <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' y <- 1 + 2 * x$x1 + rnorm(50)
#'
#' fit <- boost_gaussian(x, y, mstop = 50)
#'
#' predict(fit)
#' predict(fit, mstop = 10)
#' predict(fit, newdata = x[1:3, ])

predict.boost_gaussian <- function(object, newdata = NULL, mstop = NULL, ...) {
  # Validate model and stopping iteration
  if (!inherits(object, "boost_gaussian")) {
    stop("object must inherit from class 'boost_gaussian'.", call. = FALSE)
  }
  
  if (is.null(mstop)) {
    m_use <- object$mstop
  } else {
    if (!is.numeric(mstop) || length(mstop) != 1L || is.na(mstop) ||
        !is.finite(mstop) || mstop < 0L || mstop != as.integer(mstop)) {
      stop("mstop must be a single non-negative integer.", call. = FALSE)
    }
    if (mstop > object$mstop) {
      stop(sprintf(
        "mstop (%d) exceeds the number of iterations the model was trained with (%d).",
        as.integer(mstop), object$mstop
      ), call. = FALSE)
    }
    m_use <- as.integer(mstop)
  }
  
  # In-sample prediction: use the stored fitted path when available
  if (is.null(newdata)) {
    if (m_use == 0L) {
      return(rep(object$initial_value, object$n))
    }
    if (m_use == object$mstop) {
      return(as.numeric(object$fitted_values))
    }
    # Early-stopped in-sample prediction requires the path
    if (!is.null(object$fitted_path)) {
      return(as.numeric(object$fitted_path[, m_use]))
    }
    stop(
      paste0(
        "In-sample early-stopped prediction requires keep_path = TRUE at fit time. ",
        "Either refit with keep_path = TRUE or supply newdata."
      ),
      call. = FALSE
    )
  }
  
  # Check and align new predictor data
  if (!is.matrix(newdata) && !is.data.frame(newdata)) {
    stop("newdata must be a matrix or data.frame.", call. = FALSE)
  }
  
  nd <- as.data.frame(newdata, stringsAsFactors = FALSE)
  
  missing_vars <- setdiff(object$predictor_names, names(nd))
  if (length(missing_vars) > 0L) {
    stop(paste("newdata is missing columns:", paste(missing_vars, collapse = ", ")),
         call. = FALSE)
  }
  if (!all(vapply(nd[object$predictor_names], is.numeric, logical(1)))) {
    stop("All required predictors in newdata must be numeric.", call. = FALSE)
  }
  
  x_new <- as.matrix(nd[, object$predictor_names, drop = FALSE])
  storage.mode(x_new) <- "double"
  
  if (anyNA(x_new)) {
    stop("newdata must not contain missing, NaN, or infinite values.", call. = FALSE)
  }
  if (!all(is.finite(x_new))) {
    stop("newdata must not contain missing, NaN, or infinite values.", call. = FALSE)
  }
  
  if (m_use == 0L) {
    return(rep(object$initial_value, nrow(x_new)))
  }
  
  x_std <- sweep(x_new, 2L, object$x_center, FUN = "-")
  x_std <- sweep(x_std, 2L, object$x_scale,  FUN = "/")
  
  prediction <- rep(object$initial_value, nrow(x_std))
  
  for (m in seq_len(m_use)) {
    variable   <- object$selected_variables[m]
    prediction <- prediction + object$coefficients[m] * x_std[, variable]
  }
  
  as.numeric(prediction)
}
