#' Print a Distributional Boosting Model
#'
#' Prints a compact overview of a fitted \code{boost_gamma},
#' \code{boost_poisson}, or \code{boost_binomial} object (all of class
#' \code{"boost_dist"}), generalizing \code{\link{print.boost_gaussian}} to
#' any number of distribution parameters.
#'
#' @param x A fitted object inheriting from class \code{"boost_dist"}.
#' @param digits Number of digits used when printing numeric values.
#' @param ... Further arguments, currently ignored.
#'
#' @returns The input object \code{x}, invisibly.
#'
#' @importFrom utils tail
#' @export
#' @method print boost_dist
print.boost_dist <- function(x, digits = 4, ...) {
  if (!inherits(x, "boost_dist")) {
    stop("x must inherit from class 'boost_dist'.", call. = FALSE)
  }
  P <- x$parameters

  cat(sprintf("\nDistributional %s Gradient Boosting\n",
              .capitalize(x$family_name)))
  cat(rep("-", 45), "\n", sep = "")
  cat("Call:\n  ", deparse(x$call), "\n\n", sep = "")

  cat("Tuning parameters:\n")
  cat("  method  :", x$method, "\n")
  cat("  learner :", x$learner, "\n")
  cat("  mstop   :", x$mstop, "\n")
  for (k in P) cat(sprintf("  nu_%s%s: %s\n", k, strrep(" ", max(0, 5 - nchar(k))), x$nus[[k]]))

  if (!is.null(x$early_stopping) && isTRUE(x$early_stopping$used)) {
    cat("\nInternal early stopping:\n")
    cat("  best_round :", x$early_stopping$best_round,
        "(searched up to", x$early_stopping$mstop_max, "rounds)\n")
  }
  cat("\n")

  for (k in P) {
    freq   <- sort(table(x$selected[[k]]), decreasing = TRUE)
    n_sel  <- length(freq)
    n_zero <- x$p[[k]] - n_sel
    cat(sprintf("%s predictors:\n", k))
    cat("  Total                  :", x$p[[k]], "\n")
    cat("  Selected (>= 1 time)   :", n_sel, "\n")
    cat("  Never selected (coef=0):", n_zero, "\n\n")
    if (n_sel > 0L) {
      if (n_sel <= 10L) {
        print(freq)
      } else {
        print(freq[seq_len(10L)])
        cat("  ... and", n_sel - 10L, "more.\n")
      }
    }
    smooth_tbl <- .smooth_terms_table(x$coef[[k]])
    if (!is.null(smooth_tbl)) {
      cat("\n  Smooth terms:\n")
      print(smooth_tbl, row.names = FALSE)
    }
    cat("\n")
  }

  cat("Risk (negative log-likelihood):", round(tail(x$risk, 1), digits),
      " (initial:", round(x$risk0, digits), ")\n\n")

  invisible(x)
}

#' Summarize a Distributional Boosting Model
#'
#' @param object A fitted object inheriting from class \code{"boost_dist"}.
#' @param ... Further arguments, currently ignored.
#'
#' @returns An object of class \code{"summary.boost_dist"}.
#'
#' @export
summary.boost_dist <- function(object, ...) {
  if (!inherits(object, "boost_dist")) {
    stop("object must inherit from class 'boost_dist'.", call. = FALSE)
  }
  P <- object$parameters

  per_param <- stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    net_k     <- object$net_coef_orig[[k]]
    net_k_lin <- net_k[!is.na(net_k)]
    selected_k <- net_k_lin[net_k_lin != 0]
    selected_k <- selected_k[order(abs(selected_k), decreasing = TRUE)]
    zero_k     <- names(net_k_lin)[net_k_lin == 0]

    per_param[[k]] <- list(
      intercept    = object$intercept[[k]],
      net_coef     = selected_k,
      net_coef_all = net_k,
      zero_vars    = zero_k,
      freq         = sort(table(object$selected[[k]]), decreasing = TRUE),
      smooth_terms = .smooth_terms_table(object$coef[[k]])
    )
  }

  out <- list(
    call = object$call, family_name = object$family_name, parameters = P,
    method = object$method, learner = object$learner, mstop = object$mstop,
    early_stopping = object$early_stopping, nus = object$nus, n = object$n,
    p = object$p, per_param = per_param
  )
  class(out) <- "summary.boost_dist"
  out
}

#' Print a Distributional Boosting Model Summary
#'
#' @param x An object of class \code{"summary.boost_dist"}.
#' @param digits Number of digits used when printing numeric values.
#' @param ... Further arguments, currently ignored.
#'
#' @returns The input object \code{x}, invisibly.
#'
#' @export
#' @method print summary.boost_dist
print.summary.boost_dist <- function(x, digits = 4, ...) {
  cat(sprintf("\nDistributional %s Gradient Boosting - Model Summary\n",
              .capitalize(x$family_name)))
  cat(rep("-", 55), "\n", sep = "")
  cat("Call:\n  ", deparse(x$call), "\n\n", sep = "")
  cat("Dimensions : n =", x$n, "\n")
  cat("Iterations : method =", x$method, " learner =", x$learner,
      " mstop =", x$mstop, "\n\n")

  for (k in x$parameters) {
    pp <- x$per_param[[k]]
    cat(sprintf("%s submodel coefficients (original predictor scale):\n", k))
    cat(sprintf("  %-25s  %s\n", "(Intercept)", round(pp$intercept, digits)))
    if (length(pp$net_coef) > 0L) {
      for (nm in names(pp$net_coef)) {
        cat(sprintf("  %-25s  %s\n", nm, round(pp$net_coef[[nm]], digits)))
      }
    } else {
      cat("  No predictors selected.\n")
    }
    if (length(pp$zero_vars) > 0L) {
      cat("\n  Variables never selected (zero coefficient):\n")
      cat("   ", paste(pp$zero_vars, collapse = ", "), "\n", sep = "")
    }
    if (!is.null(pp$smooth_terms)) {
      cat("\n  Smooth terms:\n")
      print(pp$smooth_terms, row.names = FALSE)
    }
    cat("\n")
  }

  invisible(x)
}

#' Plot a Distributional Boosting Model
#'
#' Produces diagnostic plots for a fitted \code{boost_gamma},
#' \code{boost_poisson}, or \code{boost_binomial} object, generalizing
#' \code{\link{plot.boost_gaussian}} to any number of distribution
#' parameters.
#'
#' @param x A fitted object inheriting from class \code{"boost_dist"}.
#' @param type Plot type: \code{"path"}, \code{"frequency"}, \code{"risk"}, or
#'   \code{"partial"} -- see \code{\link{plot.boost_gaussian}} for details.
#' @param param Which parameter's submodel to plot (one of
#'   \code{x$parameters}); defaults to the first parameter.
#' @param top_n Maximum number of variables shown in the coefficient path plot.
#' @param scale Coefficient scale for \code{type = "path"}: \code{"original"}
#'   or \code{"standardized"}.
#' @param variable Required when \code{type = "partial"}.
#' @param ... Additional graphical arguments passed to base plotting functions.
#'
#' @returns The input object \code{x}, invisibly.
#'
#' @export
#' @method plot boost_dist
plot.boost_dist <- function(x, type = c("path", "frequency", "risk", "partial"),
                            param = NULL, top_n = 8,
                            scale = c("original", "standardized"),
                            variable = NULL, ...) {
  if (!inherits(x, "boost_dist")) {
    stop("x must inherit from class 'boost_dist'.", call. = FALSE)
  }
  type  <- match.arg(type)
  scale <- match.arg(scale)
  P <- x$parameters
  if (is.null(param)) param <- P[1L]
  if (!(param %in% P)) {
    stop(sprintf("param must be one of: %s.", paste(P, collapse = ", ")), call. = FALSE)
  }

  if (!is.numeric(top_n) || length(top_n) != 1L || is.na(top_n) ||
      !is.finite(top_n) || top_n < 1L || top_n != as.integer(top_n)) {
    stop("top_n must be a single positive integer.", call. = FALSE)
  }
  top_n <- as.integer(top_n)

  if (type == "risk") {
    .plot_boost_risk(x, ...)
  } else if (type == "frequency") {
    freq    <- sort(table(x$selected[[param]]), decreasing = TRUE)
    n_steps <- length(x$selected[[param]])
    dots       <- list(...)
    main_title <- if (!is.null(dots$main)) dots$main else
      sprintf("%s Variable Selection Frequency", param)
    bar_args <- dots[setdiff(names(dots), c("main", "xlab", "ylab"))]
    bp <- do.call(graphics::barplot, c(list(
      height = freq, main = main_title,
      xlab = if (!is.null(dots$xlab)) dots$xlab else "Predictor",
      ylab = if (!is.null(dots$ylab)) dots$ylab else
        paste0("Selection count (total = ", n_steps, " updates)"),
      las = 2, cex.names = 0.85
    ), bar_args))
    graphics::text(bp, freq + max(freq) * 0.02,
                   labels = paste0(round(100 * freq / n_steps, 1), "%"),
                   cex = 0.75, col = "grey30")
  } else if (type == "partial") {
    if (is.null(variable)) {
      stop("variable must be supplied when type = \"partial\".", call. = FALSE)
    }
    .plot_boost_partial(
      selected_k = x$selected[[param]], coef_k = x$coef[[param]],
      intercept_step_k = x$intercept_step[[param]],
      x_std = x$designs_std[[param]], center_k = x$centers[[param]],
      scale_k = x$scales[[param]], variable = variable,
      param_label = param, ...
    )
  } else {
    .plot_boost_dist_path(x, param = param, top_n = top_n, scale = scale, ...)
  }

  invisible(x)
}

# ── Internal: coefficient path plot, generalized over an arbitrary parameter ──
.plot_boost_dist_path <- function(x, param, top_n = 8,
                                  scale = "original", ...) {
  vars       <- x$design_names[[param]]
  selected   <- x$selected[[param]]
  coefs      <- x$coef[[param]]
  rounds     <- x$round[[param]]
  pred_scale <- x$scales[[param]]

  if (is.list(coefs)) {
    stop("type = \"path\" only supports linear-only submodels; use ",
         "type = \"partial\" to visualise a submodel containing spline steps.",
         call. = FALSE)
  }

  p       <- length(vars)
  n_steps <- length(selected)
  path_std <- matrix(0, nrow = p, ncol = max(n_steps, 1L),
                     dimnames = list(vars, paste0("m", seq_len(max(n_steps, 1L)))))
  for (m in seq_len(n_steps)) {
    if (m > 1L) path_std[, m] <- path_std[, m - 1L]
    sel <- selected[m]
    path_std[sel, m] <- path_std[sel, m] + coefs[m]
  }

  freq     <- sort(table(selected), decreasing = TRUE)
  sel_vars <- names(freq)[seq_len(min(top_n, length(freq)))]
  path_plot <- path_std[sel_vars, seq_len(n_steps), drop = FALSE]
  plot_x    <- rounds

  if (identical(scale, "original")) {
    path_plot <- sweep(path_plot, 1L, pred_scale[sel_vars], FUN = "/")
  }

  y_range <- range(path_plot, finite = TRUE)
  if (!all(is.finite(y_range))) y_range <- c(-1, 1)
  y_pad <- diff(y_range) * 0.1
  if (!is.finite(y_pad) || y_pad == 0) y_pad <- 0.1

  dots       <- list(...)
  main_title <- if (!is.null(dots$main)) dots$main else
    sprintf("%s Coefficient Path", param)
  line_lwd   <- if (!is.null(dots$lwd)) dots$lwd else 2
  cols       <- if (!is.null(dots$col) && length(dots$col) >= length(sel_vars)) {
    dots$col
  } else {
    seq_along(sel_vars)
  }
  plot_args <- dots[setdiff(names(dots),
                           c("main", "xlab", "ylab", "xlim", "ylim", "type", "lwd", "col"))]

  do.call(graphics::plot, c(list(
    x = plot_x, y = path_plot[1L, ], type = "s", lwd = line_lwd, col = cols[1L],
    xlim = c(-1, x$mstop), ylim = c(y_range[1L] - y_pad, y_range[2L] + y_pad),
    xlab = if (!is.null(dots$xlab)) dots$xlab else "Boosting iteration",
    ylab = if (!is.null(dots$ylab)) dots$ylab else "Coefficient",
    main = main_title
  ), plot_args))

  graphics::grid(nx = NA, ny = NULL, lty = 2, col = "grey85")
  graphics::abline(h = 0, lty = 3, col = "grey70")
  if (length(sel_vars) > 1L) {
    for (kk in seq_along(sel_vars)[-1L]) {
      graphics::lines(plot_x, path_plot[kk, ], type = "s", lwd = line_lwd, col = cols[kk])
    }
  }
  graphics::legend("topleft", legend = sel_vars, col = cols, lwd = line_lwd,
                   bty = "n", cex = 0.85)
  invisible(NULL)
}
