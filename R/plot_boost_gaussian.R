#' Plot a Distributional Gaussian Boosting Model
#'
#' Produces diagnostic plots for a fitted \code{boost_gaussian} object.
#' Both the location and the scale submodel can be visualised.
#'
#' @param x A fitted object of class \code{"boost_gaussian"}.
#' @param type Plot type: \code{"path"} for cumulative coefficient paths or
#'   \code{"frequency"} for variable selection frequencies.
#' @param submodel Which submodel to plot: \code{"mu"} (location, default) or
#'   \code{"sigma"} (scale).
#' @param top_n Maximum number of variables shown in the coefficient path plot.
#'   Variables are chosen by selection frequency.
#' @param scale Coefficient scale for \code{type = "path"}.
#'   \code{"original"} (default) shows coefficients on the original predictor
#'   scale for both submodels.  For the scale submodel \code{"original"} means
#'   the additive effect on \eqn{\log\sigma} per unit change in the original
#'   \eqn{z_j}.  \code{"standardized"} shows coefficients on the standardised
#'   predictor scale.
#' @param ... Additional graphical arguments passed to base plotting functions.
#'
#' @details
#' The path plot shows cumulative coefficient updates over boosting iterations.
#' The frequency plot shows how often each predictor was selected.
#'
#' @returns The input object \code{x}, invisibly.
#'
#' @export
#' @method plot boost_gaussian
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(80), x2 = rnorm(80), x3 = rnorm(80))
#' Z <- data.frame(z1 = rnorm(80), z2 = rnorm(80))
#' y <- 1 + 2 * X$x1 - X$x3 + exp(0.4 * Z$z1) * rnorm(80)
#'
#' fit <- boost_gaussian(X, Z, y, mstop = 80)
#'
#' plot(fit, type = "path",      submodel = "mu")
#' plot(fit, type = "frequency", submodel = "mu")
#' plot(fit, type = "path",      submodel = "sigma")
#' plot(fit, type = "frequency", submodel = "sigma")

plot.boost_gaussian <- function(x,
                                type     = c("path", "frequency"),
                                submodel = c("mu", "sigma"),
                                top_n    = 8,
                                scale    = c("original", "standardized"),
                                ...) {
  if (!inherits(x, "boost_gaussian")) {
    stop("x must inherit from class 'boost_gaussian'.", call. = FALSE)
  }
  type     <- match.arg(type)
  submodel <- match.arg(submodel)
  scale    <- match.arg(scale)
  
  if (!is.numeric(top_n) || length(top_n) != 1L || is.na(top_n) ||
      !is.finite(top_n) || top_n < 1L || top_n != as.integer(top_n)) {
    stop("top_n must be a single positive integer.", call. = FALSE)
  }
  top_n <- as.integer(top_n)
  
  if (type == "frequency") {
    .plot_boost_frequency(x, submodel = submodel, ...)
  } else {
    .plot_boost_path(x, submodel = submodel, top_n = top_n, scale = scale, ...)
  }
  
  invisible(x)
}

# ── Internal: coefficient path plot ───────────────────────────────────────────

.plot_boost_path <- function(x, submodel = "mu", top_n = 8,
                             scale = "original", ...) {
  mstop <- x$mstop
  
  if (submodel == "mu") {
    vars          <- x$X_names
    selected      <- x$selected_mu
    coefs         <- x$coef_mu
    pred_scale    <- x$X_scale   # divide std coef by this to get original scale
    default_title <- "Location Coefficient Path"
    default_ylab  <- if (identical(scale, "original")) {
      "Coefficient (original predictor scale, effect on mu)"
    } else {
      "Coefficient (standardized predictor scale)"
    }
  } else {
    vars          <- x$Z_names
    selected      <- x$selected_sigma
    coefs         <- x$coef_sigma
    pred_scale    <- x$Z_scale   # divide std coef by this to get original scale
    default_title <- "Scale Coefficient Path"
    default_ylab  <- if (identical(scale, "original")) {
      "Coefficient (original predictor scale, effect on log-sigma)"
    } else {
      "Coefficient (standardized predictor scale, effect on log-sigma)"
    }
  }
  
  p <- length(vars)
  
  # Build cumulative path matrix on the STANDARDISED scale first
  path_std <- matrix(0, nrow = p, ncol = mstop,
                     dimnames = list(vars, paste0("m", seq_len(mstop))))
  
  for (m in seq_len(mstop)) {
    if (m > 1L) path_std[, m] <- path_std[, m - 1L]
    sel              <- selected[m]
    path_std[sel, m] <- path_std[sel, m] + coefs[m]
  }
  
  freq     <- sort(table(selected), decreasing = TRUE)
  sel_vars <- names(freq)[seq_len(min(top_n, length(freq)))]
  
  path_plot <- path_std[sel_vars, , drop = FALSE]
  
  # Convert to original predictor scale if requested (both mu and sigma)
  if (identical(scale, "original")) {
    path_plot <- sweep(path_plot, 1L, pred_scale[sel_vars], FUN = "/")
  }
  
  y_range <- range(path_plot, finite = TRUE)
  if (!all(is.finite(y_range))) y_range <- c(-1, 1)
  y_pad <- diff(y_range) * 0.1
  if (!is.finite(y_pad) || y_pad == 0) y_pad <- 0.1
  
  dots       <- list(...)
  main_title <- if (!is.null(dots$main)) dots$main else default_title
  line_lwd   <- if (!is.null(dots$lwd))  dots$lwd  else 2
  cols       <- if (!is.null(dots$col) && length(dots$col) >= length(sel_vars)) {
    dots$col
  } else {
    seq_along(sel_vars)
  }
  plot_args <- dots[setdiff(names(dots),
                            c("main", "xlab", "ylab", "xlim", "ylim",
                              "type", "lwd", "col"))]
  
  do.call(graphics::plot, c(list(
    x    = seq_len(mstop),
    y    = path_plot[1L, ],
    type = "l",
    lwd  = line_lwd,
    col  = cols[1L],
    xlim = c(-1, mstop),
    ylim = c(y_range[1L] - y_pad, y_range[2L] + y_pad),
    xlab = if (!is.null(dots$xlab)) dots$xlab else "Boosting iteration",
    ylab = if (!is.null(dots$ylab)) dots$ylab else default_ylab,
    main = main_title
  ), plot_args))
  
  graphics::grid(nx = NA, ny = NULL, lty = 2, col = "grey85")
  graphics::abline(h = 0, lty = 3, col = "grey70")
  
  if (length(sel_vars) > 1L) {
    for (k in seq_along(sel_vars)[-1L]) {
      graphics::lines(seq_len(mstop), path_plot[k, ], lwd = line_lwd, col = cols[k])
    }
  }
  
  graphics::legend("topleft", legend = sel_vars, col = cols,
                   lwd = line_lwd, bty = "n", cex = 0.85)
  
  n_hidden <- length(freq) - length(sel_vars)
  if (n_hidden > 0L) {
    graphics::mtext(
      sprintf("(%d further selected variable(s) not shown)", n_hidden),
      side = 1, line = 4, cex = 0.75, col = "grey50"
    )
  }
}

# ── Internal: selection frequency bar plot ─────────────────────────────────────

.plot_boost_frequency <- function(x, submodel = "mu", ...) {
  
  if (submodel == "mu") {
    freq          <- sort(table(x$selected_mu), decreasing = TRUE)
    default_title <- "Location Variable Selection Frequency"
  } else {
    freq          <- sort(table(x$selected_sigma), decreasing = TRUE)
    default_title <- "Scale Variable Selection Frequency"
  }
  
  dots       <- list(...)
  main_title <- if (!is.null(dots$main)) dots$main else default_title
  bar_args   <- dots[setdiff(names(dots), c("main", "xlab", "ylab"))]
  
  bp <- do.call(graphics::barplot, c(list(
    height    = freq,
    main      = main_title,
    xlab      = if (!is.null(dots$xlab)) dots$xlab else "Predictor",
    ylab      = if (!is.null(dots$ylab)) dots$ylab else
      paste0("Selection count (total = ", x$mstop, " iterations)"),
    las       = 2,
    cex.names = 0.85
  ), bar_args))
  
  graphics::text(bp, freq + max(freq) * 0.02,
                 labels = paste0(round(100 * freq / x$mstop, 1), "%"),
                 cex = 0.75, col = "grey30")
}