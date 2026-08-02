# ── P-spline base learner (Eilers & Marx penalized B-splines) ───────────────
#
# A penalized-regression-spline alternative to the linear base learner,
# analogous to mboost's bbs(). The basis size is fixed (df_basis raw B-spline
# columns); a ridge penalty (2nd-order difference matrix) is tuned, once per
# predictor column before boosting starts, so the spline's EFFECTIVE degrees
# of freedom equal a fixed target_df -- this keeps RSS comparable to the
# linear learner's implicit df (intercept + slope = 2), so component-wise
# selection between "linear" and "spline" candidates for the same column
# isn't structurally biased toward the more flexible learner.
#
# Effective df (edf) of a penalized-ridge smoother only depends on the basis
# matrix B and penalty matrix K (not on the working response u), so the
# penalty search below is done once per column, ahead of the round loop.

# Builds a B-spline basis (via base-R splines::bs()) for a single numeric
# predictor column, with df_basis raw basis functions (interior knots placed
# at quantiles of xj) and boundary knots fixed to range(xj).
.make_bspline_basis <- function(xj, df_basis = 10, degree = 3) {
  n_interior <- df_basis - degree - 1L
  if (n_interior < 0L) {
    stop("df_basis must be at least degree + 1.", call. = FALSE)
  }

  boundary <- range(xj)
  if (diff(boundary) == 0) {
    # Degenerate (constant) column: widen the boundary slightly so bs() does
    # not error; the resulting basis contributes nothing useful, but fitting
    # must not crash.
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

# Effective degrees of freedom of the penalized-ridge smoother
# S = B (B'B + lambda*K)^-1 B', computed via the cheap ncol(B) x ncol(B)
# trace identity tr(S) = tr((B'B + lambda*K)^-1 B'B), which depends only on
# the basis/penalty structure (BtB, K), not on any particular response.
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

# Precomputes, once per column of `design`, the basis/penalty/df-equalized
# lambda needed to fit a P-spline candidate against any working response u in
# every subsequent boosting round (a df_basis x df_basis ridge solve per
# round, not a fresh root-find).
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

# Summarizes the spline steps in one parameter's step history (`coef_k`, as
# returned in .run_boosting_loop_general()'s `coef` field) into a small
# per-variable table (variable, n_steps, edf) for print()/summary() to
# report smooth terms that a single scalar coefficient can no longer
# describe. Returns NULL when `coef_k` is a flat numeric vector (no spline
# steps present, e.g. learner == "linear") so callers can omit the section
# entirely in that (default) case.
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

# Computes one variable's accumulated partial effect on the (standardised)
# linear predictor scale, across a grid spanning its observed standardised
# range. `selected_k`/`intercept_step_k` are flat vectors, `coef_k` is
# either a flat numeric vector (learner = "linear") or a list of step
# objects; either way, this reconstructs the sum of every step that ever
# selected `variable`, via .evaluate_step() (a synthetic linear step object
# is built on the fly for the flat-vector case so both representations share
# one code path). Because boosting's per-column additive structure means a
# variable's total effect never depends on any other variable's value, this
# partial effect is exact, not an approximation.
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

# Evaluates a stored step (linear or spline, as produced by
# .best_base_learner_general()/the boosting loop's step-scaling) at new
# x-values. For spline steps this exactly reconstructs splines::bs() from
# the stored knots/degree/boundary -- sufficient to reproduce predictions
# without needing the original training data.
.evaluate_step <- function(step, x) {
  if (step$type == "linear") {
    step$intercept + step$slope * x
  } else {
    B <- splines::bs(x, knots = step$knots, degree = step$degree,
                      Boundary.knots = step$boundary, intercept = TRUE)
    as.numeric(B %*% step$coefs)
  }
}
