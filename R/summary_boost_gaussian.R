#' Summarize a Distributional Gaussian Boosting Model
#'
#' Computes a structured summary of a fitted \code{boost_gaussian} object,
#' covering both the location (mean) and scale (standard deviation) submodels.
#'
#' @param object A fitted object of class \code{"boost_gaussian"}.
#' @param ... Further arguments, currently ignored.
#'
#' @details
#' Both location and scale coefficients are reported on the
#' \emph{original predictor scale}:
#' \itemize{
#'   \item \strong{Location}: a coefficient \eqn{\beta_j} means one unit
#'     increase in the original \eqn{x_j} increases \eqn{\hat\mu} by
#'     \eqn{\beta_j}.
#'   \item \strong{Scale}: a coefficient \eqn{\gamma_j} means one unit
#'     increase in the original \eqn{z_j} changes \eqn{\log\hat\sigma} by
#'     \eqn{\gamma_j}, i.e. multiplies \eqn{\hat\sigma} by
#'     \eqn{\exp(\gamma_j)}.  The linear predictor is always on the
#'     \eqn{\log\sigma} scale to ensure \eqn{\hat\sigma > 0}.
#' }
#' Residual diagnostics are based on the raw residuals \eqn{y - \hat\mu}.
#'
#' @returns An object of class \code{"summary.boost_gaussian"}.
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(60), x2 = rnorm(60), x3 = rnorm(60))
#' Z <- data.frame(z1 = rnorm(60), z2 = rnorm(60))
#' y <- 1 + 2 * X$x1 - X$x3 + exp(0.4 * Z$z1) * rnorm(60)
#'
#' fit <- boost_gaussian(X, Z, y, mstop = 50)
#' summary(fit)

summary.boost_gaussian <- function(object, ...) {
  if (!inherits(object, "boost_gaussian")) {
    stop("object must inherit from class 'boost_gaussian'.", call. = FALSE)
  }
  
  # ── Location submodel ──────────────────────────────────────────────────────
  freq_mu     <- sort(table(object$selected_mu), decreasing = TRUE)
  
  net_mu_orig <- object$net_coef_mu_orig
  mu_selected <- net_mu_orig[net_mu_orig != 0]
  mu_selected <- mu_selected[order(abs(mu_selected), decreasing = TRUE)]
  mu_zero_vars <- object$X_names[net_mu_orig == 0]
  
  # ── Scale submodel — reported on original predictor scale ─────────────────
  freq_sigma        <- sort(table(object$selected_sigma), decreasing = TRUE)
  
  net_sigma_orig    <- object$net_coef_sigma_orig
  sigma_selected    <- net_sigma_orig[net_sigma_orig != 0]
  sigma_selected    <- sigma_selected[order(abs(sigma_selected), decreasing = TRUE)]
  sigma_zero_vars   <- object$Z_names[net_sigma_orig == 0]
  
  # ── Residual diagnostics ───────────────────────────────────────────────────
  res <- object$residuals
  rss <- sum(res^2)
  
  res_summary <- list(
    min    = min(res),
    q1     = unname(stats::quantile(res, 0.25)),
    median = stats::median(res),
    mean   = mean(res),
    q3     = unname(stats::quantile(res, 0.75)),
    max    = max(res),
    rss    = rss,
    mse    = mean(res^2),
    r_sq   = object$r_squared
  )
  
  out <- list(
    call                 = object$call,
    mstop                = object$mstop,
    nu_mu                = object$nu_mu,
    nu_sigma             = object$nu_sigma,
    n                    = object$n,
    pX                   = object$pX,
    pZ                   = object$pZ,
    # Location
    intercept_mu         = object$intercept_mu,
    net_coef_mu          = mu_selected,
    net_coef_mu_all      = net_mu_orig,
    mu_zero_vars         = mu_zero_vars,
    freq_mu              = freq_mu,
    # Scale (original predictor scale, log-sigma linear predictor)
    intercept_sigma      = object$intercept_sigma,
    net_coef_sigma       = sigma_selected,
    net_coef_sigma_all   = net_sigma_orig,
    sigma_zero_vars      = sigma_zero_vars,
    freq_sigma           = freq_sigma,
    # Residuals
    res_summary          = res_summary
  )
  
  class(out) <- "summary.boost_gaussian"
  out
}


#' Print a Distributional Gaussian Boosting Model Summary
#'
#' Prints a formatted summary produced by \code{\link{summary.boost_gaussian}}.
#'
#' @param x An object of class \code{"summary.boost_gaussian"}.
#' @param digits Number of digits used when printing numeric values.
#' @param ... Further arguments, currently ignored.
#'
#' @returns The input object \code{x}, invisibly.
#'
#' @export
#' @method print summary.boost_gaussian
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(60), x2 = rnorm(60), x3 = rnorm(60))
#' Z <- data.frame(z1 = rnorm(60), z2 = rnorm(60))
#' y <- 1 + 2 * X$x1 - X$x3 + exp(0.4 * Z$z1) * rnorm(60)
#'
#' fit <- boost_gaussian(X, Z, y, mstop = 50)
#' print(summary(fit))

print.summary.boost_gaussian <- function(x, digits = 4, ...) {
  if (!inherits(x, "summary.boost_gaussian")) {
    stop("x must inherit from class 'summary.boost_gaussian'.", call. = FALSE)
  }
  
  cat("\nDistributional Gaussian Gradient Boosting - Model Summary\n")
  cat(rep("-", 55), "\n", sep = "")
  cat("Call:\n  ", deparse(x$call), "\n\n", sep = "")
  
  cat("Dimensions : n =", x$n, "\n")
  cat("             pX =", x$pX, "(location predictors)",
      x$pX - length(x$mu_zero_vars), "selected,",
      length(x$mu_zero_vars), "zero\n")
  cat("             pZ =", x$pZ, "(scale predictors)   ",
      x$pZ - length(x$sigma_zero_vars), "selected,",
      length(x$sigma_zero_vars), "zero\n\n")
  
  cat("Iterations : mstop =", x$mstop,
      " nu_mu =", x$nu_mu,
      " nu_sigma =", x$nu_sigma, "\n\n")
  
  # ── Residuals ──────────────────────────────────────────────────────────────
  cat("Residuals (y - mu_hat):\n")
  res_mat <- matrix(
    round(c(x$res_summary$min, x$res_summary$q1, x$res_summary$median,
            x$res_summary$mean, x$res_summary$q3, x$res_summary$max), digits),
    nrow = 1,
    dimnames = list("", c("Min", "1Q", "Median", "Mean", "3Q", "Max"))
  )
  print(res_mat)
  
  r2_text <- if (is.na(x$res_summary$r_sq)) {
    "NA"
  } else {
    format(round(x$res_summary$r_sq, digits), nsmall = 0)
  }
  cat(sprintf("\n  RSS : %.4f    MSE : %.4f    R-squared : %s\n\n",
              x$res_summary$rss, x$res_summary$mse, r2_text))
  
  # ── Location coefficients — original predictor scale ──────────────────────
  cat("Location submodel coefficients (original predictor scale):\n")
  cat("  Variable                   Coefficient\n")
  cat("  -------------------------  -----------\n")
  cat(sprintf("  %-25s  %s\n", "(Intercept)", round(x$intercept_mu, digits)))
  
  if (length(x$net_coef_mu) > 0L) {
    for (nm in names(x$net_coef_mu)) {
      cat(sprintf("  %-25s  %s\n", nm, round(x$net_coef_mu[[nm]], digits)))
    }
  } else {
    cat("  No predictors selected.\n")
  }
  
  if (length(x$mu_zero_vars) > 0L) {
    cat("\n  Variables never selected (zero coefficient):\n")
    if (length(x$mu_zero_vars) <= 20L) {
      cat("   ", paste(x$mu_zero_vars, collapse = ", "), "\n", sep = "")
    } else {
      cat("   ", paste(x$mu_zero_vars[1:20], collapse = ", "),
          " ... and ", length(x$mu_zero_vars) - 20L, " more.\n", sep = "")
    }
  }
  
  # ── Scale coefficients — original predictor scale, log-sigma LP ───────────
  cat("\nScale submodel coefficients\n")
  cat("  Linear predictor : log(sigma)  [original predictor scale]\n")
  cat("  Interpretation   : one unit increase in z_j multiplies sigma\n")
  cat("                     by exp(coefficient)\n")
  cat("  Variable                   Coefficient\n")
  cat("  -------------------------  -----------\n")
  cat(sprintf("  %-25s  %s\n", "(Intercept log-sigma)",
              round(x$intercept_sigma, digits)))
  
  if (length(x$net_coef_sigma) > 0L) {
    for (nm in names(x$net_coef_sigma)) {
      cat(sprintf("  %-25s  %s\n", nm, round(x$net_coef_sigma[[nm]], digits)))
    }
  } else {
    cat("  No predictors selected.\n")
  }
  
  if (length(x$sigma_zero_vars) > 0L) {
    cat("\n  Variables never selected (zero coefficient):\n")
    if (length(x$sigma_zero_vars) <= 20L) {
      cat("   ", paste(x$sigma_zero_vars, collapse = ", "), "\n", sep = "")
    } else {
      cat("   ", paste(x$sigma_zero_vars[1:20], collapse = ", "),
          " ... and ", length(x$sigma_zero_vars) - 20L, " more.\n", sep = "")
    }
  }
  
  invisible(x)
}