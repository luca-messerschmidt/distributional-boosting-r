#' Predict from a Distributional Boosting Model
#'
#' Computes fitted or predicted values from a \code{boost_gamma},
#' \code{boost_poisson}, or \code{boost_binomial} object (all of class
#' \code{"boost_dist"}). Generalizes \code{\link{predict.boost_gaussian}}'s
#' reconstruction logic to any number of distribution parameters.
#'
#' @param object A fitted object inheriting from class \code{"boost_dist"}.
#' @param newdata_X Optional data frame or matrix of predictors for the
#'   family's first parameter (e.g. the mean). If omitted (together with
#'   \code{newdata_Z} and \code{newdata}), in-sample fitted values are
#'   returned.
#' @param newdata_Z Optional data frame or matrix of predictors for the
#'   family's second parameter, if it has one (e.g. Gamma's shape). Must be
#'   \code{NULL} for single-parameter families (Poisson, Binomial).
#' @param newdata Optional single data frame with the original variable
#'   names, used when \code{object} was fit via the formula interface.
#' @param mstop Optional non-negative integer number of boosting rounds to
#'   use; defaults to the full fitted model.
#' @param what Which parameter to return predictions for (one of
#'   \code{object$parameters}), or \code{"all"} for a data frame with one
#'   column per parameter. Defaults to the family's only parameter for
#'   single-parameter families, or \code{"all"} otherwise.
#' @param ... Further arguments, currently ignored.
#'
#' @returns A numeric vector (single parameter) or a data frame with one
#'   column per parameter (\code{what = "all"}).
#'
#' @export
#' @method predict boost_dist
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(80))
#' y <- rpois(80, lambda = exp(0.3 + 0.5 * X$x1))
#' fit <- boost_poisson(X, y, mstop = 60)
#' predict(fit)
#' predict(fit, mstop = 20)
predict.boost_dist <- function(object, newdata_X = NULL, newdata_Z = NULL,
                               newdata = NULL, mstop = NULL, what = NULL, ...) {
  if (!inherits(object, "boost_dist")) {
    stop("object must inherit from class 'boost_dist'.", call. = FALSE)
  }
  P <- object$parameters

  if (is.null(what)) what <- if (length(P) == 1L) P[1L] else "all"
  if (!(what %in% c(P, "all"))) {
    stop(sprintf("what must be one of: %s, or \"all\".",
                 paste(sprintf("\"%s\"", P), collapse = ", ")), call. = FALSE)
  }
  if (length(P) == 1L && !is.null(newdata_Z)) {
    stop(sprintf("'%s' is a single-parameter family; newdata_Z is not used.",
                 object$family_name), call. = FALSE)
  }

  if (!is.null(newdata)) {
    if (!is.null(newdata_X) || !is.null(newdata_Z)) {
      stop("Supply either newdata, or newdata_X/newdata_Z, not both.", call. = FALSE)
    }
    if (is.null(object$terms)) {
      stop("newdata can only be used when object was fit via the formula ",
           "interface; use newdata_X/newdata_Z instead.", call. = FALSE)
    }
    terms1 <- object$terms[[P[1L]]]
    newdata_X <- stats::model.matrix(stats::delete.response(terms1), data = newdata)
    newdata_X <- newdata_X[, colnames(newdata_X) != "(Intercept)", drop = FALSE]
    if (length(P) == 2L) {
      terms2    <- object$terms[[P[2L]]]
      newdata_Z <- stats::model.matrix(terms2, data = newdata)
      newdata_Z <- newdata_Z[, colnames(newdata_Z) != "(Intercept)", drop = FALSE]
    }
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

  in_sample <- is.null(newdata_X) && is.null(newdata_Z)

  if (in_sample) {
    x_std <- object$designs_std
  } else {
    if (length(P) == 2L && (is.null(newdata_X) || is.null(newdata_Z))) {
      stop("Both newdata_X and newdata_Z must be supplied together.", call. = FALSE)
    }
    x_std <- stats::setNames(vector("list", length(P)), P)
    x_new_1 <- .validate_newdata(newdata_X, object$design_names[[P[1L]]], "newdata_X")
    x_std[[P[1L]]] <- sweep(sweep(x_new_1, 2L, object$centers[[P[1L]]], "-"),
                            2L, object$scales[[P[1L]]], "/")
    if (length(P) == 2L) {
      x_new_2 <- .validate_newdata(newdata_Z, object$design_names[[P[2L]]], "newdata_Z")
      if (nrow(x_new_1) != nrow(x_new_2)) {
        stop("newdata_X and newdata_Z must have the same number of rows.", call. = FALSE)
      }
      x_std[[P[2L]]] <- sweep(sweep(x_new_2, 2L, object$centers[[P[2L]]], "-"),
                              2L, object$scales[[P[2L]]], "/")
    }
  }

  n_pred <- nrow(x_std[[P[1L]]])

  if (in_sample && m_use == object$mstop) {
    return(.format_dist_predict_output(object$fitted, what, P))
  }

  fitted_list <- stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    eta_pred <- rep(object$init_eta[[k]], n_pred)
    if (m_use > 0L) {
      k_use  <- sum(object$round[[k]] <= m_use)
      coef_k <- object$coef[[k]]
      if (is.list(coef_k)) {
        for (m in seq_len(k_use)) {
          step <- coef_k[[m]]
          x_m  <- x_std[[k]][, object$selected[[k]][m]]
          eta_pred <- eta_pred + .evaluate_step(step, x_m)
        }
      } else {
        for (m in seq_len(k_use)) {
          eta_pred <- eta_pred +
            object$intercept_step[[k]][m] +
            coef_k[m] * x_std[[k]][, object$selected[[k]][m]]
        }
      }
    }
    fitted_list[[k]] <- as.numeric(object$family$invlink[[k]](eta_pred))
  }

  .format_dist_predict_output(fitted_list, what, P)
}

# Returns a single parameter's fitted vector, or (what = "all") a data frame
# with one column per parameter, in family-parameter order.
.format_dist_predict_output <- function(fitted_list, what, P) {
  if (what == "all") {
    df <- as.data.frame(fitted_list[P])
    names(df) <- P
    return(df)
  }
  as.numeric(fitted_list[[what]])
}
