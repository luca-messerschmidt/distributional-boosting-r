#' Plot a Distributional Gaussian Boosting Model
#'
#' Produces diagnostic plots for a fitted \code{boost_gaussian} object.
#' Both the location and the scale submodel can be visualised.
#'
#' @param x A fitted object of class \code{"boost_gaussian"}.
#' @param type Plot type: \code{"path"} for cumulative coefficient paths,
#'   \code{"frequency"} for variable selection frequencies, \code{"risk"}
#'   for the in-sample (and, if internal early stopping was used, validation)
#'   negative log-likelihood trace over boosting rounds, or \code{"partial"}
#'   for a variable's fitted partial effect (needed when \code{learner =
#'   "spline"}/\code{"auto"} was used, since a smooth term can't be
#'   summarised by a single coefficient).
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
#' @param variable Required when \code{type = "partial"}: the name of the
#'   predictor (in \code{submodel}) whose fitted partial effect to plot.
#' @param ... Additional graphical arguments passed to base plotting functions.
#'
#' @details
#' The path plot shows cumulative coefficient updates over boosting iterations.
#' The frequency plot shows how often each predictor was selected. The
#' partial-effect plot shows a single variable's fitted contribution to the
#' (standardised) linear predictor across its observed range -- a straight
#' line for a variable fit entirely by the linear base learner, or a curve
#' for one that received spline steps.
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
                                type     = c("path", "frequency", "risk", "partial"),
                                submodel = c("mu", "sigma"),
                                top_n    = 8,
                                scale    = c("original", "standardized"),
                                variable = NULL,
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
  } else if (type == "risk") {
    .plot_boost_risk(x, ...)
  } else if (type == "partial") {
    if (is.null(variable)) {
      stop("variable must be supplied when type = \"partial\".", call. = FALSE)
    }
    if (submodel == "mu") {
      .plot_boost_partial(selected_k = x$selected_mu, coef_k = x$coef_mu,
                          intercept_step_k = x$intercept_step_mu,
                          x_std = x$X_std, center_k = x$X_center,
                          scale_k = x$X_scale, variable = variable,
                          param_label = "Location", ...)
    } else {
      .plot_boost_partial(selected_k = x$selected_sigma, coef_k = x$coef_sigma,
                          intercept_step_k = x$intercept_step_sigma,
                          x_std = x$Z_std, center_k = x$Z_center,
                          scale_k = x$Z_scale, variable = variable,
                          param_label = "Scale (log-sigma)", ...)
    }
  } else {
    .plot_boost_path(x, submodel = submodel, top_n = top_n, scale = scale, ...)
  }

  invisible(x)
}

# ── Internal: coefficient path plot ───────────────────────────────────────────

.plot_boost_path <- function(x, submodel = "mu", top_n = 8,
                             scale = "original", ...) {
  mstop <- x$mstop   # total rounds executed (x-axis range)

  if (submodel == "mu") {
    vars          <- x$X_names
    selected      <- x$selected_mu
    coefs         <- x$coef_mu
    rounds        <- x$round_mu
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
    rounds        <- x$round_sigma
    pred_scale    <- x$Z_scale   # divide std coef by this to get original scale
    default_title <- "Scale Coefficient Path"
    default_ylab  <- if (identical(scale, "original")) {
      "Coefficient (original predictor scale, effect on log-sigma)"
    } else {
      "Coefficient (standardized predictor scale, effect on log-sigma)"
    }
  }

  p       <- length(vars)
  n_steps <- length(selected)

  # Build cumulative path matrix on the STANDARDISED scale first. Columns
  # correspond to this submodel's own steps (n_steps, not the shared
  # `mstop`), each tagged with the global round it occurred at (`rounds`) so
  # the path can be plotted against the shared boosting-iteration axis even
  # when a submodel was updated less often or irregularly (two-mstop cyclic,
  # noncyclic).
  path_std <- matrix(0, nrow = p, ncol = max(n_steps, 1L),
                     dimnames = list(vars, paste0("m", seq_len(max(n_steps, 1L)))))

  for (m in seq_len(n_steps)) {
    if (m > 1L) path_std[, m] <- path_std[, m - 1L]
    sel              <- selected[m]
    path_std[sel, m] <- path_std[sel, m] + coefs[m]
  }

  freq     <- sort(table(selected), decreasing = TRUE)
  sel_vars <- names(freq)[seq_len(min(top_n, length(freq)))]

  path_plot <- path_std[sel_vars, seq_len(n_steps), drop = FALSE]
  plot_x    <- rounds
  
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
    x    = plot_x,
    y    = path_plot[1L, ],
    type = "s",
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
      graphics::lines(plot_x, path_plot[k, ], type = "s", lwd = line_lwd, col = cols[k])
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
    n_steps       <- length(x$selected_mu)
    default_title <- "Location Variable Selection Frequency"
  } else {
    freq          <- sort(table(x$selected_sigma), decreasing = TRUE)
    n_steps       <- length(x$selected_sigma)
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
      paste0("Selection count (total = ", n_steps, " updates)"),
    las       = 2,
    cex.names = 0.85
  ), bar_args))

  graphics::text(bp, freq + max(freq) * 0.02,
                 labels = paste0(round(100 * freq / n_steps, 1), "%"),
                 cex = 0.75, col = "grey30")
}

# ── Internal: partial-effect plot ─────────────────────────────────────────────
#
# Plots one variable's accumulated fitted effect (linear and/or spline steps
# that ever selected it) across its observed range -- a straight line if
# every step for that variable was linear, a curve if any were spline steps.
# Uses .compute_partial_effect() (R/learner_spline.R), which reconstructs the
# effect via .evaluate_step() from the stored per-step coefficients/basis
# information, so this works identically for boost_gaussian() and the
# boost_dist family (Gamma/Poisson/Binomial) submodels.
.plot_boost_partial <- function(selected_k, coef_k, intercept_step_k, x_std,
                                center_k, scale_k, variable, param_label,
                                n_grid = 100, ...) {
  if (!(variable %in% colnames(x_std))) {
    stop(sprintf("variable '%s' not found among this submodel's predictors.",
                 variable), call. = FALSE)
  }

  pe <- .compute_partial_effect(selected_k, coef_k, intercept_step_k,
                                x_std[, variable], variable, n_grid = n_grid)
  grid_orig <- pe$grid_std * scale_k[[variable]] + center_k[[variable]]

  dots       <- list(...)
  main_title <- if (!is.null(dots$main)) dots$main else
    sprintf("%s: partial effect of %s", param_label, variable)
  plot_args  <- dots[setdiff(names(dots), c("main", "xlab", "ylab", "type", "col", "lwd"))]

  do.call(graphics::plot, c(list(
    x = grid_orig, y = pe$effect, type = "l", lwd = 2, col = "steelblue",
    xlab = if (!is.null(dots$xlab)) dots$xlab else variable,
    ylab = if (!is.null(dots$ylab)) dots$ylab else
      "Partial effect (standardised linear predictor scale)",
    main = main_title
  ), plot_args))

  graphics::abline(h = 0, lty = 3, col = "grey70")
  invisible(NULL)
}

# ── Internal: risk (negative log-likelihood) trace plot ────────────────────────

.plot_boost_risk <- function(x, ...) {
  if (is.null(x$risk)) {
    stop("x does not contain a risk trace (fit with an older version of ",
         "boost_gaussian()?).", call. = FALSE)
  }

  train_trace  <- c(x$risk0, x$risk)
  train_rounds <- seq(0L, length(train_trace) - 1L)

  has_val <- !is.null(x$early_stopping) && isTRUE(x$early_stopping$used)
  if (has_val) {
    val_trace  <- c(x$early_stopping$val_risk0, x$early_stopping$val_risk)
    val_rounds <- seq(0L, length(val_trace) - 1L)
    y_range    <- range(c(train_trace, val_trace), finite = TRUE)
  } else {
    y_range <- range(train_trace, finite = TRUE)
  }

  dots       <- list(...)
  main_title <- if (!is.null(dots$main)) dots$main else
    "Boosting Risk (Negative Log-Likelihood)"
  plot_args  <- dots[setdiff(names(dots),
                            c("main", "xlab", "ylab", "type", "col", "lwd"))]

  do.call(graphics::plot, c(list(
    x    = train_rounds,
    y    = train_trace,
    type = "l",
    lwd  = 2,
    col  = "steelblue",
    ylim = y_range,
    xlab = if (!is.null(dots$xlab)) dots$xlab else "Boosting iteration",
    ylab = if (!is.null(dots$ylab)) dots$ylab else "Negative log-likelihood",
    main = main_title
  ), plot_args))

  if (has_val) {
    graphics::lines(val_rounds, val_trace, lwd = 2, col = "firebrick", lty = 2)
    graphics::abline(v = x$early_stopping$best_round, lty = 3, col = "grey40")
    graphics::legend("topright",
                     legend = c("Training risk", "Validation risk",
                                paste("best_round =", x$early_stopping$best_round)),
                     col = c("steelblue", "firebrick", "grey40"),
                     lty = c(1, 2, 3), lwd = c(2, 2, 1), bty = "n", cex = 0.85)
  } else {
    graphics::legend("topright", legend = "Training risk", col = "steelblue",
                     lty = 1, lwd = 2, bty = "n", cex = 0.85)
  }
}