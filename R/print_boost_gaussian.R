#' Print a Distributional Gaussian Boosting Model
#'
#' Prints a compact overview of a fitted \code{boost_gaussian} object,
#' including tuning parameters, predictor selection counts for both the
#' location and scale submodels, and basic training-fit statistics.
#'
#' @param x A fitted object of class \code{"boost_gaussian"}.
#' @param digits Number of digits used when printing numeric values.
#' @param ... Further arguments, currently ignored.
#'
#' @returns The input object \code{x}, invisibly.
#'
#' @export
#' @method print boost_gaussian
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' Z <- data.frame(z1 = rnorm(50))
#' y <- 1 + 2 * X$x1 + exp(0.3 * Z$z1) * rnorm(50)
#'
#' fit <- boost_gaussian(X, Z, y, mstop = 30)
#' print(fit)

print.boost_gaussian <- function(x, digits = 4, ...) {
  if (!inherits(x, "boost_gaussian")) {
    stop("x must inherit from class 'boost_gaussian'.", call. = FALSE)
  }
  
  cat("\nDistributional Gaussian Gradient Boosting\n")
  cat(rep("-", 45), "\n", sep = "")
  cat("Call:\n  ", deparse(x$call), "\n\n", sep = "")
  
  method_label <- if (is.null(x$method)) "cyclic" else x$method

  cat("Tuning parameters:\n")
  cat("  method        :", method_label, "\n")
  if (!is.null(x$mstop_mu) && !is.null(x$mstop_sigma) &&
      x$mstop_mu != x$mstop_sigma) {
    cat("  mstop         :", x$mstop, "rounds  (mstop_mu =", x$mstop_mu,
        ", mstop_sigma =", x$mstop_sigma, ")\n")
  } else {
    cat("  mstop         :", x$mstop, "\n")
  }
  cat("  nu_mu         :", x$nu_mu, "\n")
  cat("  nu_sigma      :", x$nu_sigma, "\n")

  if (!is.null(x$early_stopping) && isTRUE(x$early_stopping$used)) {
    cat("\nInternal early stopping:\n")
    cat("  best_round       :", x$early_stopping$best_round,
        "(searched up to", x$early_stopping$mstop_max, "rounds)\n")
    cat("  patience         :", x$early_stopping$patience, "\n")
    cat("  validation_split :", x$early_stopping$validation_split,
        " (n_train =", x$early_stopping$n_train,
        ", n_val =", x$early_stopping$n_val, ")\n")
  }
  cat("\n")
  
  # ── Location submodel ──────────────────────────────────────────────────────
  freq_mu    <- sort(table(x$selected_mu), decreasing = TRUE)
  n_sel_mu   <- length(freq_mu)
  n_zero_mu  <- x$pX - n_sel_mu
  
  cat("Location predictors (X):\n")
  cat("  Total                  :", x$pX, "\n")
  cat("  Selected (>= 1 time)   :", n_sel_mu, "\n")
  cat("  Never selected (coef=0):", n_zero_mu, "\n\n")
  
  if (n_sel_mu <= 10L) {
    cat("  Selected variables and selection counts:\n")
    print(freq_mu)
  } else {
    cat("  Top 10 selected variables by frequency:\n")
    print(freq_mu[seq_len(10L)])
    cat("  ... and", n_sel_mu - 10L, "more.\n")
  }
  
  # ── Scale submodel ─────────────────────────────────────────────────────────
  freq_sigma   <- sort(table(x$selected_sigma), decreasing = TRUE)
  n_sel_sigma  <- length(freq_sigma)
  n_zero_sigma <- x$pZ - n_sel_sigma
  
  cat("\nScale predictors (Z):\n")
  cat("  Total                  :", x$pZ, "\n")
  cat("  Selected (>= 1 time)   :", n_sel_sigma, "\n")
  cat("  Never selected (coef=0):", n_zero_sigma, "\n\n")
  
  if (n_sel_sigma <= 10L) {
    cat("  Selected variables and selection counts:\n")
    print(freq_sigma)
  } else {
    cat("  Top 10 selected variables by frequency:\n")
    print(freq_sigma[seq_len(10L)])
    cat("  ... and", n_sel_sigma - 10L, "more.\n")
  }
  
  # ── Training fit ───────────────────────────────────────────────────────────
  r2_text <- if (is.null(x$r_squared) || is.na(x$r_squared)) {
    "NA"
  } else {
    format(round(x$r_squared, digits), nsmall = 0)
  }
  
  cat("\nTraining fit (location):\n")
  cat("  RSS       :", round(sum(x$residuals^2), digits), "\n")
  cat("  MSE       :", round(mean(x$residuals^2), digits), "\n")
  cat("  R-squared :", r2_text, "\n\n")
  
  invisible(x)
}