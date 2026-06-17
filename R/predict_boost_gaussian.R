#' Predict from a Distributional Gaussian Boosting Model
#'
#' Computes fitted or predicted values from a \code{boost_gaussian} object.
#' Both the location (\eqn{\hat\mu}) and the scale (\eqn{\hat\sigma}) can be
#' returned.
#'
#' @param object A fitted object of class \code{"boost_gaussian"}.
#' @param newdata_X Optional data frame or matrix of location predictors.
#'   Must contain all columns used in \code{X} at fit time, matched by name.
#'   If omitted, in-sample fitted values are returned.
#' @param newdata_Z Optional data frame or matrix of scale predictors.
#'   Must contain all columns used in \code{Z} at fit time, matched by name.
#'   If omitted, in-sample fitted values are returned.
#' @param mstop Optional non-negative integer: number of boosting iterations
#'   to use. Defaults to the full fitted model. \code{mstop = 0} returns the
#'   intercept-only predictions. Any value in \code{0:object$mstop} is allowed,
#'   so predictions can be obtained at an earlier stopping iteration both
#'   in-sample and out-of-sample.
#' @param what Character string specifying what to return:
#'   \code{"mu"} (default) for location predictions,
#'   \code{"sigma"} for scale predictions (on the original sigma scale), or
#'   \code{"both"} for a data frame with columns \code{mu} and \code{sigma}.
#' @param ... Further arguments, currently ignored.
#'
#' @details
#' Predictions are reconstructed by accumulating the stored per-iteration
#' updates (intercept plus slope) on the standardised predictor scale.  For
#' out-of-sample data the supplied \code{newdata_X} / \code{newdata_Z} are
#' standardised with the training means and standard deviations; for in-sample
#' prediction the stored standardised training design is used.  Scale
#' predictions are returned on the original \eqn{\sigma} scale (i.e. \code{exp}
#' of the linear predictor on the log scale).
#'
#' @returns A numeric vector (for \code{what = "mu"} or \code{"sigma"}) or a
#'   data frame with columns \code{mu} and \code{sigma}
#'   (for \code{what = "both"}).
#'
#' @export
#' @method predict boost_gaussian
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' Z <- data.frame(z1 = rnorm(50))
#' y <- 1 + 2 * X$x1 + exp(0.3 * Z$z1) * rnorm(50)
#'
#' fit <- boost_gaussian(X, Z, y, mstop = 50)
#'
#' # In-sample predictions (full model)
#' predict(fit)
#' predict(fit, what = "sigma")
#' predict(fit, what = "both")
#'
#' # Early stopping via mstop, in-sample and out-of-sample
#' predict(fit, mstop = 10)
#' predict(fit, newdata_X = X[1:3, ], newdata_Z = Z[1:3, , drop = FALSE])
#' predict(fit, newdata_X = X[1:3, ], newdata_Z = Z[1:3, , drop = FALSE],
#'         mstop = 10)
#' predict(fit, newdata_X = X[1:3, ], newdata_Z = Z[1:3, , drop = FALSE],
#'         what = "both")

predict.boost_gaussian <- function(object,
                                   newdata_X = NULL,
                                   newdata_Z = NULL,
                                   mstop     = NULL,
                                   what      = c("mu", "sigma", "both"),
                                   ...) {
  
  if (!inherits(object, "boost_gaussian")) {
    stop("object must inherit from class 'boost_gaussian'.", call. = FALSE)
  }
  
  what <- match.arg(what)
  
  # ── Validate mstop ────────────────────────────────────────────────────────
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
  
  # ── Resolve the standardised design to predict on ─────────────────────────
  # In-sample (no newdata): use the stored standardised training design.
  # Out-of-sample: validate, align, and standardise the supplied newdata with
  # the training centering/scaling constants.  A single reconstruction path is
  # then used for both, so in-sample early stopping (0 < mstop < object$mstop)
  # works exactly like out-of-sample.
  in_sample <- is.null(newdata_X) && is.null(newdata_Z)
  
  if (in_sample) {
    x_std <- object$X_std
    z_std <- object$Z_std
  } else {
    if (is.null(newdata_X) || is.null(newdata_Z)) {
      stop("Both newdata_X and newdata_Z must be supplied together.", call. = FALSE)
    }
    x_new <- .validate_newdata(newdata_X, object$X_names, "newdata_X")
    z_new <- .validate_newdata(newdata_Z, object$Z_names, "newdata_Z")
    if (nrow(x_new) != nrow(z_new)) {
      stop("newdata_X and newdata_Z must have the same number of rows.", call. = FALSE)
    }
    x_std <- sweep(sweep(x_new, 2L, object$X_center, "-"), 2L, object$X_scale, "/")
    z_std <- sweep(sweep(z_new, 2L, object$Z_center, "-"), 2L, object$Z_scale, "/")
  }
  
  n_pred <- nrow(x_std)
  
  # Fast, exact path: full model in-sample returns the stored fitted values.
  if (in_sample && m_use == object$mstop) {
    return(.format_predict_output(object$fitted_mu, object$fitted_sigma, what))
  }
  
  # Intercept-only prediction.
  if (m_use == 0L) {
    mu_pred    <- rep(object$initial_mu,             n_pred)
    sigma_pred <- rep(exp(object$initial_log_sigma), n_pred)
    return(.format_predict_output(mu_pred, sigma_pred, what))
  }
  
  # General reconstruction (used for early stopping in- and out-of-sample, and
  # for the full out-of-sample model): accumulate per-step intercept + slope.
  mu_pred <- rep(object$initial_mu, n_pred)
  for (m in seq_len(m_use)) {
    mu_pred <- mu_pred +
      object$intercept_step_mu[m] +
      object$coef_mu[m] * x_std[, object$selected_mu[m]]
  }
  
  log_sigma_pred <- rep(object$initial_log_sigma, n_pred)
  for (m in seq_len(m_use)) {
    log_sigma_pred <- log_sigma_pred +
      object$intercept_step_sigma[m] +
      object$coef_sigma[m] * z_std[, object$selected_sigma[m]]
  }
  sigma_pred <- exp(log_sigma_pred)
  
  .format_predict_output(as.numeric(mu_pred), as.numeric(sigma_pred), what)
}

# ── Internal helpers ───────────────────────────────────────────────────────────

# Validate a newdata matrix/data.frame and return it as a named numeric matrix
# with columns aligned to `required_names`.
.validate_newdata <- function(newdata, required_names, arg_name) {
  if (!is.matrix(newdata) && !is.data.frame(newdata)) {
    stop(sprintf("%s must be a matrix or data.frame.", arg_name), call. = FALSE)
  }
  nd <- as.data.frame(newdata, stringsAsFactors = FALSE)
  missing_vars <- setdiff(required_names, names(nd))
  if (length(missing_vars) > 0L) {
    stop(sprintf("%s is missing columns: %s",
                 arg_name, paste(missing_vars, collapse = ", ")),
         call. = FALSE)
  }
  if (!all(vapply(nd[required_names], is.numeric, logical(1)))) {
    stop(sprintf("All required predictors in %s must be numeric.", arg_name),
         call. = FALSE)
  }
  mat <- as.matrix(nd[, required_names, drop = FALSE])
  storage.mode(mat) <- "double"
  if (!all(is.finite(mat))) {
    stop(sprintf("%s must not contain missing, NaN, or infinite values.", arg_name),
         call. = FALSE)
  }
  mat
}

# Return mu, sigma, or both depending on `what`.
.format_predict_output <- function(mu_pred, sigma_pred, what) {
  if (what == "mu")    return(as.numeric(mu_pred))
  if (what == "sigma") return(as.numeric(sigma_pred))
  data.frame(mu = as.numeric(mu_pred), sigma = as.numeric(sigma_pred))
}