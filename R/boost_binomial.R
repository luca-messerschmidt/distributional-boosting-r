#' Binomial/Bernoulli Gradient Boosting
#'
#' Fits a component-wise gradient boosting model for binary responses,
#' estimating \eqn{P(y = 1)} (logit-linked). A single-parameter special case
#' of the same family-driven boosting engine used by
#' \code{\link{boost_gaussian}}: \code{method} is accepted for interface
#' consistency but has no effect, since cyclic and non-cyclic updates
#' coincide when there is only one parameter to boost.
#'
#' @param X A data frame or matrix of predictors. Ignored if \code{formula}
#'   is supplied.
#' @param y A numeric response vector; every value must be exactly 0 or 1.
#'   Ignored if \code{formula} is supplied.
#' @param formula A two-sided formula, e.g. \code{y ~ x1 + x2}. When
#'   supplied, \code{X} and \code{y} are constructed from \code{data} instead.
#' @param data A data frame containing the variables referenced in
#'   \code{formula}. Required (and only used) when \code{formula} is supplied.
#' @param mstop Number of boosting iterations (single positive integer).
#' @param nu Learning rate.
#' @param method Accepted for interface consistency with
#'   \code{\link{boost_gaussian}}; has no effect for this single-parameter
#'   family.
#' @param patience Optional internal early-stopping patience (see
#'   \code{\link{boost_gaussian}}).
#' @param validation_split Fraction held out for early-stopping search.
#' @param seed Optional seed for the early-stopping train/validation split.
#' @param learner Base learner type: \code{"linear"} (default), \code{"spline"}
#'   (df-equalized P-splines), or \code{"auto"} (best of both per column).
#'
#' @returns A list of class \code{c("boost_binomial", "boost_dist")}.
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(150))
#' p  <- plogis(0.3 + 0.8 * X$x1)
#' y  <- rbinom(150, size = 1, prob = p)
#'
#' fit <- boost_binomial(X, y, mstop = 100)
boost_binomial <- function(X = NULL, y = NULL, formula = NULL, data = NULL,
                           mstop = 100, nu = 0.1,
                           method = c("cyclic", "noncyclic"), patience = NULL,
                           validation_split = 0.2, seed = NULL,
                           learner = c("linear", "spline", "auto")) {
  method  <- match.arg(method)
  learner <- match.arg(learner)
  family  <- .family_binomial()

  if (!is.null(formula)) {
    built <- .build_design_from_formula_general(formula, data, family$parameters)
    X <- built$designs$mu
    y <- built$y
    terms_list <- built$terms
  } else {
    terms_list <- NULL
  }

  y <- as.numeric(y)
  if (anyNA(y) || !all(y %in% c(0, 1))) {
    stop("y must be binary (exactly 0 or 1), with no missing values.",
         call. = FALSE)
  }

  core <- .boost_dist_fit(
    designs = list(mu = X), y = y, family = family, nus = list(mu = nu),
    mstop = mstop, method = method, patience = patience,
    validation_split = validation_split, seed = seed, learner = learner
  )

  result <- .assemble_boost_dist_result(core, terms_list = terms_list,
                                        call = match.call())
  class(result) <- c("boost_binomial", "boost_dist")
  result
}
