# ── Generalized component-wise base learner selection ────────────────────────
#
# Fits candidate base learners to every column of `design` against target
# `u`, and returns the single (column, learner-type) combination with the
# lowest residual sum of squares. `learner` controls which candidate types
# are tried:
#   "linear" - one simple linear (intercept + slope) candidate per column
#              (identical in behavior to the package's original
#              .best_base_learner(); this is the only path boost_gaussian()
#              exercises by default, so it costs nothing extra)
#   "spline" - one df-equalized P-spline candidate per column (needs
#              spline_cache_k, as built by .build_spline_cache())
#   "auto"   - both candidates per column, true RSS-best-of-both selection
# The returned list always includes `fitted` (the candidate's raw, un-nu-
# scaled prediction on `design`'s rows), used by the boosting loop to update
# the linear predictor with the exact same "nu * (a + b*x)"-style expression
# the original implementation used, keeping the linear/default path
# numerically identical to before.
.best_base_learner_general <- function(design, u, learner = "linear",
                                        spline_cache_k = NULL) {
  best_rss <- Inf
  best     <- NULL

  if (learner %in% c("linear", "auto")) {
    a <- mean(u)
    for (j in seq_len(ncol(design))) {
      xj   <- design[[j]]
      ss   <- sum(xj^2)
      b    <- if (ss > 0) sum(xj * u) / ss else 0
      pred <- a + b * xj
      rss  <- sum((u - pred)^2)
      if (rss < best_rss) {
        best_rss <- rss
        best <- list(var = names(design)[j], type = "linear",
                     slope = b, intercept = a, fitted = pred)
      }
    }
  }

  if (learner %in% c("spline", "auto")) {
    for (j in seq_len(ncol(design))) {
      var_name <- names(design)[j]
      sc       <- spline_cache_k[[var_name]]
      coefs    <- as.numeric(solve(sc$BtB + sc$lambda * sc$K,
                                   crossprod(sc$basis$B, u)))
      fitted_j <- as.numeric(sc$basis$B %*% coefs)
      rss      <- sum((u - fitted_j)^2)
      if (rss < best_rss) {
        best_rss <- rss
        best <- list(var = var_name, type = "spline", coefs = coefs,
                     knots = sc$basis$knots, boundary = sc$basis$boundary,
                     degree = sc$basis$degree, edf = sc$edf, lambda = sc$lambda,
                     fitted = fitted_j)
      }
    }
  }

  best
}

# Bakes `nu` into a candidate's stored slope/intercept (linear) or basis
# coefficients (spline), mirroring how the original implementation always
# stored nu-scaled coefficients (coef_mu[round] <- nu_mu * best$slope) so
# that later reconstruction (predict(), validation stepping) is "just apply
# the stored numbers" with no separate nu bookkeeping.
.scale_step <- function(best, nu) {
  if (best$type == "linear") {
    list(type = "linear", var = best$var,
         slope = nu * best$slope, intercept = nu * best$intercept)
  } else {
    list(type = "spline", var = best$var, coefs = nu * best$coefs,
         knots = best$knots, boundary = best$boundary, degree = best$degree,
         edf = best$edf, lambda = best$lambda)
  }
}

# Aggregates one parameter's per-step history (`coef_k`, as returned in
# .run_boosting_loop_general()'s `coef` field) into the legacy net-coefficient
# representation consumers like boost_gaussian() expect: a named numeric
# vector (one entry per variable in `var_names`) plus the total intercept
# drift accumulated across steps. When `coef_k` is a flat numeric vector
# (learner == "linear", every step for this parameter was linear), this
# reproduces the original tapply()-based aggregation exactly. When `coef_k`
# is a list of step objects (any spline step present), variables that ever
# received a spline step get NA (a single scalar can no longer describe
# their effect -- see the smooth-terms reporting added in a later phase) and
# the intercept total only includes linear steps' intercepts.
.legacy_net_coef <- function(coef_k, intercept_step_k, selected_k, var_names) {
  if (!is.list(coef_k)) {
    net_std <- tapply(coef_k, selected_k, sum)
    net_std <- net_std[var_names]
    names(net_std) <- var_names
    net_std[is.na(net_std)] <- 0
    return(list(net_std = net_std,
                total_intercept_step = sum(intercept_step_k),
                smooth_vars = character(0)))
  }

  slope_sum   <- stats::setNames(rep(0, length(var_names)), var_names)
  smooth_vars <- character(0)
  total_intercept <- 0
  for (s in coef_k) {
    if (s$type == "linear") {
      slope_sum[s$var] <- slope_sum[s$var] + s$slope
      total_intercept  <- total_intercept + s$intercept
    } else {
      smooth_vars <- union(smooth_vars, s$var)
    }
  }
  if (length(smooth_vars) > 0L) slope_sum[smooth_vars] <- NA_real_

  list(net_std = slope_sum, total_intercept_step = total_intercept,
       smooth_vars = smooth_vars)
}

# ── Generalized boosting loop, driven by a family object ────────────────────
#
# designs/nus/mstops are named lists keyed by family$parameters (e.g.
# list(mu = X_std, sigma = Z_std) for a 2-parameter family, or
# list(mu = X_std) for a 1-parameter family) -- no parameter count is
# hardcoded anywhere in this function. Supports method = "cyclic" (every
# parameter is updated each round, until its own mstops[[k]] budget is
# exhausted) and "noncyclic" (each round updates only whichever parameter's
# candidate step yields the largest risk decrease). Optional validation
# tracking (val_designs + val_y) enables early stopping via `patience`,
# mirroring boost_gaussian()'s original .run_boosting_loop().
.run_boosting_loop_general <- function(designs, y, family, nus, mstops, method,
                                        val_designs = NULL, val_y = NULL,
                                        patience = NULL, learner = "linear") {
  .validate_family(family)
  P <- family$parameters
  track_val <- !is.null(val_y)

  init <- family$init(y)
  eta  <- stats::setNames(lapply(P, function(k) family$link[[k]](init[[k]])), P)
  par  <- stats::setNames(lapply(P, function(k) family$invlink[[k]](eta[[k]])), P)

  if (track_val) {
    n_val   <- length(val_y)
    val_eta <- stats::setNames(lapply(P, function(k) rep(eta[[k]][1], n_val)), P)
    val_par <- stats::setNames(lapply(P, function(k) family$invlink[[k]](val_eta[[k]])), P)
  }

  # spline_cache[[k]] is only built (once, before the round loop) when a
  # spline candidate can actually be selected for parameter k; left empty for
  # learner == "linear" so the linear/default path pays zero extra cost.
  spline_cache <- stats::setNames(vector("list", length(P)), P)
  if (learner %in% c("spline", "auto")) {
    for (k in P) spline_cache[[k]] <- .build_spline_cache(designs[[k]])
  }

  # `steps[[k]]` always holds the internal, uniform list-of-step-objects
  # representation (each a linear or spline step, as returned by
  # .scale_step()); `selected`/`roundv` stay flat (a step's chosen variable
  # name / round is always a scalar, regardless of learner type). At the very
  # end, `steps` is collapsed back into flat `coef`/`intercept_step` numeric
  # vectors whenever every step for a parameter is linear -- which is always
  # true for learner == "linear" -- reproducing the original flat-vector
  # field structure exactly for that (default) case.
  selected <- roundv <- steps <-
    stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    selected[[k]] <- character(0)
    roundv[[k]]   <- integer(0)
    steps[[k]]    <- list()
  }

  risk0 <- family$risk(y, par)
  if (track_val) {
    val_risk0     <- family$risk(val_y, val_par)
    best_val_risk <- Inf
    best_round    <- 0L
    no_improve    <- 0L
  }
  executed_rounds <- 0L

  log_submodel  <- character(0)
  log_variable  <- character(0)
  log_coef      <- numeric(0)
  log_intercept <- numeric(0)
  log_round     <- integer(0)

  # Fits and applies one base-learner step for parameter `k` at `round`,
  # updating eta/par/selected/steps/roundv (and the validation accumulators,
  # if tracked) and the shared step_log vectors, all via <<- into this call's
  # enclosing environment. The eta/val_eta updates use best$fitted (the raw,
  # un-scaled candidate prediction) multiplied by nu as a single expression
  # -- "nu * (a + b*x)" for a linear candidate -- exactly matching the
  # original implementation's arithmetic so the linear/default path is
  # numerically unchanged.
  .apply_step <- function(k, round) {
    u    <- family$ngradient[[k]](y, par)
    best <- .best_base_learner_general(designs[[k]], u, learner, spline_cache[[k]])
    step <- nus[[k]] * best$fitted
    eta[[k]] <<- eta[[k]] + step
    par[[k]] <<- family$invlink[[k]](eta[[k]])

    stored_step <- .scale_step(best, nus[[k]])
    selected[[k]] <<- c(selected[[k]], best$var)
    steps[[k]]    <<- c(steps[[k]], list(stored_step))
    roundv[[k]]   <<- c(roundv[[k]], round)

    if (track_val) {
      x_val    <- val_designs[[k]][, best$var]
      val_step <- if (best$type == "linear") {
        nus[[k]] * (best$intercept + best$slope * x_val)
      } else {
        B_val <- splines::bs(x_val, knots = best$knots, degree = best$degree,
                              Boundary.knots = best$boundary, intercept = TRUE)
        nus[[k]] * as.numeric(B_val %*% best$coefs)
      }
      val_eta[[k]] <<- val_eta[[k]] + val_step
      val_par[[k]] <<- family$invlink[[k]](val_eta[[k]])
    }

    log_submodel <<- c(log_submodel, k)
    log_variable <<- c(log_variable, best$var)
    log_round    <<- c(log_round, round)
    if (best$type == "linear") {
      log_coef      <<- c(log_coef, nus[[k]] * best$slope)
      log_intercept <<- c(log_intercept, nus[[k]] * best$intercept)
    } else {
      log_coef      <<- c(log_coef, NA_real_)
      log_intercept <<- c(log_intercept, NA_real_)
    }

    invisible(NULL)
  }

  if (method == "cyclic") {
    n_rounds <- max(unlist(mstops))
    risk     <- numeric(n_rounds)
    val_risk <- if (track_val) numeric(n_rounds) else NULL

    for (round in seq_len(n_rounds)) {
      for (k in P) {
        if (round <= mstops[[k]]) .apply_step(k, round)
      }

      risk[round]     <- family$risk(y, par)
      executed_rounds <- round

      if (track_val) {
        val_risk[round] <- family$risk(val_y, val_par)
        if (val_risk[round] < best_val_risk) {
          best_val_risk <- val_risk[round]
          best_round    <- round
          no_improve    <- 0L
        } else {
          no_improve <- no_improve + 1L
        }
        if (no_improve >= patience) break
      }
    }
  } else {
    n_rounds <- mstops[[P[1L]]]  # single shared budget across all parameters
    risk     <- numeric(n_rounds)
    val_risk <- if (track_val) numeric(n_rounds) else NULL

    for (round in seq_len(n_rounds)) {
      risk_current <- family$risk(y, par)

      candidates <- lapply(P, function(k) {
        u    <- family$ngradient[[k]](y, par)
        best <- .best_base_learner_general(designs[[k]], u, learner, spline_cache[[k]])
        eta_k_cand <- eta[[k]] + nus[[k]] * best$fitted
        par_cand <- par
        par_cand[[k]] <- family$invlink[[k]](eta_k_cand)
        list(best = best, eta_k_cand = eta_k_cand,
             risk_cand = family$risk(y, par_cand))
      })

      improvements <- risk_current - vapply(candidates, `[[`, numeric(1), "risk_cand")
      win_idx <- which.max(improvements)   # ties favor the earlier parameter, matching >=
      k_win   <- P[win_idx]
      chosen  <- candidates[[win_idx]]

      eta[[k_win]] <- chosen$eta_k_cand
      par[[k_win]] <- family$invlink[[k_win]](eta[[k_win]])

      stored_step <- .scale_step(chosen$best, nus[[k_win]])
      selected[[k_win]] <- c(selected[[k_win]], chosen$best$var)
      steps[[k_win]]    <- c(steps[[k_win]], list(stored_step))
      roundv[[k_win]]   <- c(roundv[[k_win]], round)

      if (track_val) {
        x_val <- val_designs[[k_win]][, chosen$best$var]
        val_step <- if (chosen$best$type == "linear") {
          nus[[k_win]] * (chosen$best$intercept + chosen$best$slope * x_val)
        } else {
          B_val <- splines::bs(x_val, knots = chosen$best$knots,
                                degree = chosen$best$degree,
                                Boundary.knots = chosen$best$boundary,
                                intercept = TRUE)
          nus[[k_win]] * as.numeric(B_val %*% chosen$best$coefs)
        }
        val_eta[[k_win]] <- val_eta[[k_win]] + val_step
        val_par[[k_win]] <- family$invlink[[k_win]](val_eta[[k_win]])
      }

      log_submodel <- c(log_submodel, k_win)
      log_variable <- c(log_variable, chosen$best$var)
      log_round    <- c(log_round, round)
      if (chosen$best$type == "linear") {
        log_coef      <- c(log_coef, nus[[k_win]] * chosen$best$slope)
        log_intercept <- c(log_intercept, nus[[k_win]] * chosen$best$intercept)
      } else {
        log_coef      <- c(log_coef, NA_real_)
        log_intercept <- c(log_intercept, NA_real_)
      }

      risk[round]     <- family$risk(y, par)
      executed_rounds <- round

      if (track_val) {
        val_risk[round] <- family$risk(val_y, val_par)
        if (val_risk[round] < best_val_risk) {
          best_val_risk <- val_risk[round]
          best_round    <- round
          no_improve    <- 0L
        } else {
          no_improve <- no_improve + 1L
        }
        if (no_improve >= patience) break
      }
    }
  }

  risk <- risk[seq_len(executed_rounds)]
  if (track_val) val_risk <- val_risk[seq_len(executed_rounds)]

  step_log <- data.frame(
    round          = log_round,
    submodel       = log_submodel,
    variable       = log_variable,
    coef           = log_coef,
    intercept_step = log_intercept,
    stringsAsFactors = FALSE
  )

  # Collapse `steps[[k]]` back into flat numeric `coef`/`intercept_step`
  # vectors whenever every step for parameter k is linear (always true when
  # learner == "linear", the only path boost_gaussian() exercises by
  # default) -- reproducing the original flat-vector field structure
  # exactly. When any step is a spline step, `coef[[k]]` is instead the full
  # list of step objects (mixed linear/spline, in "auto" mode) and
  # `intercept_step[[k]]` is NA (no longer a meaningful flat scalar; readers
  # needing per-step detail should use the step objects in `coef[[k]]`).
  coef <- intercept_step <- stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    steps_k <- steps[[k]]
    all_linear <- length(steps_k) == 0L ||
      all(vapply(steps_k, function(s) s$type == "linear", logical(1)))
    if (all_linear) {
      coef[[k]]           <- vapply(steps_k, function(s) s$slope, numeric(1))
      intercept_step[[k]] <- vapply(steps_k, function(s) s$intercept, numeric(1))
    } else {
      coef[[k]]           <- steps_k
      intercept_step[[k]] <- NA_real_
    }
  }

  list(
    eta = eta, par = par,
    selected = selected, coef = coef,
    intercept_step = intercept_step, round = roundv,
    n_rounds = n_rounds, step_log = step_log,
    risk = risk, risk0 = risk0,
    best_round = if (track_val) best_round else NULL,
    val_risk   = if (track_val) val_risk else NULL,
    val_risk0  = if (track_val) val_risk0 else NULL
  )
}
