#' Print a Gaussian Boosting Model
#'
#' Prints a compact summary of a fitted \code{boost_gaussian} object.
#'
#' @param x A fitted object of class \code{"boost_gaussian"}.
#' @param digits Number of digits used when printing numeric values.
#' @param ... Further arguments, currently ignored.
#'
#' @details
#' The output includes the model call, tuning parameters, predictor selection
#' counts, and basic training-fit statistics.
#'
#' @returns The input object \code{x}, invisibly.
#'
#' @export
#' @method print boost_gaussian
#'
#' @examples
#' set.seed(123)
#' x <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' y <- 1 + 2 * x$x1 + rnorm(50)
#'
#' fit <- boost_gaussian(x, y, mstop = 30)
#' print(fit)

print.boost_gaussian <- function(x, digits = 4, ...) {
  if (!inherits(x, "boost_gaussian")) {
    stop("x must inherit from class 'boost_gaussian'.", call. = FALSE)
  }
  
  cat("\nGaussian Gradient Boosting\n")
  cat(rep("-", 40), "\n", sep = "")
  cat("Call:\n  ", deparse(x$call), "\n\n", sep = "")
  
  cat("Tuning parameters:\n")
  cat("  mstop :", x$mstop, "\n")
  cat("  nu    :", x$nu, "\n\n")
  
  # Show the most frequently selected predictors
  freq       <- sort(table(x$selected_variables), decreasing = TRUE)
  n_selected <- length(freq)
  n_zero     <- x$p - n_selected
  
  cat("Predictors:\n")
  cat("  Total                  :", x$p, "\n")
  cat("  Selected (>= 1 time)   :", n_selected, "\n")
  cat("  Never selected (coef=0):", n_zero, "\n\n")
  
  r2_text <- if (is.null(x$r_squared) || is.na(x$r_squared)) {
    "NA"
  } else {
    format(round(x$r_squared, digits), nsmall = 0)
  }
  
  cat("Training fit:\n")
  cat("  RSS       :", round(sum(x$residuals^2), digits), "\n")
  cat("  MSE       :", round(mean(x$residuals^2), digits), "\n")
  cat("  R-squared :", r2_text, "\n\n")
  
  if (n_selected <= 10L) {
    cat("Selected variables and selection counts:\n")
    print(freq)
  } else {
    cat("Top 10 selected variables by frequency:\n")
    print(freq[seq_len(10L)])
    cat("  ... and", n_selected - 10L, "more.\n")
  }
  
  invisible(x)
}