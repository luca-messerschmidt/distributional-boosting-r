# ── P-spline base learner (Eilers & Marx penalized B-splines) ───────────────
# Analogous to mboost's bbs(). Basis size is fixed; the ridge penalty
# (2nd-order difference matrix) is tuned per column so the effective df
# matches target_df, keeping RSS comparable to the linear learner's df
# (intercept + slope = 2). edf only depends on the basis/penalty matrices,
# not on the working response, so the penalty search runs once per column
# before the round loop starts.

# B-spline basis for one predictor column: df_basis basis functions,
# interior knots at quantiles of xj, boundary knots at range(xj).
.make_bspline_basis <- function(xj, df_basis = 10, degree = 3) {
  n_interior <- df_basis - degree - 1L
  if (n_interior < 0L) {
    stop("df_basis must be at least degree + 1.", call. = FALSE)
  }

  boundary <- range(xj)
  if (diff(boundary) == 0) {
    # constant column: widen boundary so bs() doesn't error
    boundary <- boundary + c(-1e-6, 1e-6)
  }

  knots <- numeric(0)
  if (n_interior > 0L) {
    probs <- seq(0, 1, length.out = n_interior + 2L)
    probs <- probs[-c(1L, length(probs))]
    knots <- unique(stats::quantile(xj, probs = probs, names = FALSE))
  }

  B <- splines::bs(xj, knots = knots, degree = degree,
                    Boundary.knots = boundary, intercept = TRUE)

  list(B = B, knots = knots, boundary = boundary, degree = degree)
}

# 2nd-order (by default) difference penalty matrix, the standard P-spline
# roughness penalty (Eilers & Marx 1996).
.difference_penalty <- function(ncol_B, order = 2) {
  D <- diff(diag(ncol_B), differences = order)
  crossprod(D)
}

# Effective df of the penalized-ridge smoother S = B (B'B + lambda*K)^-1 B',
# via tr(S) = tr((B'B + lambda*K)^-1 B'B).
.pspline_edf <- function(BtB, K, lambda) {
  sum(diag(solve(BtB + lambda * K, BtB)))
}

# Root-finds (on the log scale) the penalty lambda that makes the spline's
# effective df equal target_df. edf(lambda) is monotonically decreasing in
# lambda, so a bracketed uniroot search is well posed.
.find_lambda_for_edf <- function(BtB, K, target_df, lower = 1e-6, upper = 1e8) {
  ncol_B <- ncol(BtB)
  if (target_df >= ncol_B) {
    return(lower)  # target flexibility exceeds the raw basis; use minimal penalty
  }

  edf_gap <- function(log_lambda) .pspline_edf(BtB, K, exp(log_lambda)) - target_df

  root <- tryCatch(
    stats::uniroot(edf_gap, interval = log(c(lower, upper)), extendInt = "yes")$root,
    error = function(e) {
      if (abs(edf_gap(log(lower))) < abs(edf_gap(log(upper)))) log(lower) else log(upper)
    }
  )
  exp(root)
}

# Precomputes the basis/penalty/df-equalized lambda for each column of
# `design`, so every boosting round only needs a ridge solve, not a fresh
# root-find.
.build_spline_cache <- function(design, df_basis = 10, degree = 3, order = 2,
                                 target_df = 4) {
  cache <- vector("list", ncol(design))
  names(cache) <- names(design)
  for (j in seq_len(ncol(design))) {
    xj     <- design[[j]]
    basis  <- .make_bspline_basis(xj, df_basis = df_basis, degree = degree)
    K      <- .difference_penalty(ncol(basis$B), order = order)
    BtB    <- crossprod(basis$B)
    lambda <- .find_lambda_for_edf(BtB, K, target_df = target_df)
    edf    <- .pspline_edf(BtB, K, lambda)
    cache[[j]] <- list(basis = basis, K = K, BtB = BtB, lambda = lambda, edf = edf)
  }
  cache
}

# Builds a (variable, n_steps, edf) table of a parameter's spline steps for
# print()/summary(); NULL if no spline steps were taken.
.smooth_terms_table <- function(coef_k) {
  if (!is.list(coef_k)) return(NULL)
  spline_steps <- Filter(function(s) s$type == "spline", coef_k)
  if (length(spline_steps) == 0L) return(NULL)

  vars <- vapply(spline_steps, function(s) s$var, character(1))
  edfs <- vapply(spline_steps, function(s) s$edf, numeric(1))
  n_steps    <- table(vars)
  edf_by_var <- tapply(edfs, vars, function(x) x[1])

  data.frame(
    variable = names(n_steps),
    n_steps  = as.integer(n_steps),
    edf      = as.numeric(edf_by_var[names(n_steps)]),
    stringsAsFactors = FALSE, row.names = NULL
  )
}

# Accumulates one variable's partial effect on the standardised linear
# predictor over a grid of its range, by replaying every step (linear or
# spline) that ever selected it through .evaluate_step().
.compute_partial_effect <- function(selected_k, coef_k, intercept_step_k,
                                     x_std_col, variable, n_grid = 100) {
  grid_std <- seq(min(x_std_col), max(x_std_col), length.out = n_grid)
  effect   <- rep(0, n_grid)
  is_list_coef <- is.list(coef_k)

  for (m in seq_along(selected_k)) {
    if (selected_k[m] != variable) next
    step <- if (is_list_coef) {
      coef_k[[m]]
    } else {
      list(type = "linear", slope = coef_k[m], intercept = intercept_step_k[m])
    }
    effect <- effect + .evaluate_step(step, grid_std)
  }

  list(grid_std = grid_std, effect = effect)
}

# Evaluates a stored step at new x-values. Spline steps rebuild splines::bs()
# from the stored knots/degree/boundary.
.evaluate_step <- function(step, x) {
  if (step$type == "linear") {
    step$intercept + step$slope * x
  } else {
    B <- splines::bs(x, knots = step$knots, degree = step$degree,
                      Boundary.knots = step$boundary, intercept = TRUE)
    as.numeric(B %*% step$coefs)
  }
}
