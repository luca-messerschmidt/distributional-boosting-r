#' Distributional Gamma Gradient Boosting
#'
#' Fits a component-wise gradient boosting model for strictly positive,
#' right-skewed responses, jointly estimating the Gamma mean (\eqn{\mu}) and
#' shape (\eqn{\nu}) parameters (both log-linked). Built on the same
#' family-driven boosting engine as \code{\link{boost_gaussian}}.
#'
#' @param X A data frame or matrix of predictors for the mean submodel.
#'   Ignored if \code{formula} is supplied.
#' @param Z A data frame or matrix of predictors for the shape submodel.
#'   Ignored if \code{formula} is supplied.
#' @param y A numeric response vector; every value must be strictly positive.
#'   Ignored if \code{formula} is supplied.
#' @param formula A list with components \code{mu} (a two-sided formula) and
#'   \code{shape} (a one-sided formula). When supplied, \code{X}, \code{Z},
#'   and \code{y} are constructed from \code{data} instead.
#' @param data A data frame containing the variables referenced in
#'   \code{formula}. Required (and only used) when \code{formula} is supplied.
#' @param mstop Number of boosting iterations. Either a single positive
#'   integer or, for \code{method = "cyclic"} only, \code{c(mu = ..., shape = ...)}.
#' @param nu_mu Learning rate for the mean submodel.
#' @param nu_shape Learning rate for the shape submodel.
#' @param method Boosting update schedule, \code{"cyclic"} (default) or
#'   \code{"noncyclic"} -- see \code{\link{boost_gaussian}} for details.
#' @param patience Optional internal early-stopping patience (see
#'   \code{\link{boost_gaussian}}).
#' @param validation_split Fraction held out for early-stopping search.
#' @param seed Optional seed for the early-stopping train/validation split.
#' @param learner Base learner type: \code{"linear"} (default), \code{"spline"}
#'   (df-equalized P-splines), or \code{"auto"} (best of both per column).
#'
#' @returns A list of class \code{c("boost_gamma", "boost_dist")}.
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(150))
#' Z <- data.frame(z1 = rnorm(150))
#' mu    <- exp(1 + 0.5 * X$x1)
#' shape <- exp(0.3 * Z$z1 + 1)
#' y <- rgamma(150, shape = shape, rate = shape / mu)
#'
#' fit <- boost_gamma(X, Z, y, mstop = 100)
boost_gamma <- function(X = NULL, Z = NULL, y = NULL, formula = NULL,
                        data = NULL, mstop = 100, nu_mu = 0.1, nu_shape = 0.1,
                        method = c("cyclic", "noncyclic"), patience = NULL,
                        validation_split = 0.2, seed = NULL,
                        learner = c("linear", "spline", "auto")) {
  method  <- match.arg(method)
  learner <- match.arg(learner)
  family  <- .family_gamma()

  if (!is.null(formula)) {
    built <- .build_design_from_formula_general(formula, data, family$parameters)
    X <- built$designs$mu
    Z <- built$designs$shape
    y <- built$y
    terms_list <- built$terms
  } else {
    terms_list <- NULL
  }

  y <- as.numeric(y)
  if (anyNA(y) || !all(is.finite(y)) || !all(y > 0)) {
    stop("y must be strictly positive (Gamma response), with no missing or ",
         "infinite values.", call. = FALSE)
  }

  core <- .boost_dist_fit(
    designs = list(mu = X, shape = Z), y = y, family = family,
    nus = list(mu = nu_mu, shape = nu_shape), mstop = mstop, method = method,
    patience = patience, validation_split = validation_split, seed = seed,
    learner = learner
  )

  result <- .assemble_boost_dist_result(core, terms_list = terms_list,
                                        call = match.call())
  class(result) <- c("boost_gamma", "boost_dist")
  result
}
